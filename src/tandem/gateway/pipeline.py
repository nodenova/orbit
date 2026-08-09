"""The gateway pipeline (spec sec 8).

One path from a canonical request to a canonical result, shared by all three wire
protocols. Order matters and is fixed:

    compact -> replay-aware render -> prompt-cache probe -> constrain
            -> cascade (router: best-of-N, tier-1 rerank, escalation)
            -> repair -> bounded retry -> record replay -> cache store
            -> receipt + audit -> context-scale reported usage

Compaction goes first because everything downstream is measured against the prompt
that is actually sent. Context scaling goes last because it is a reporting lie and
must not reach the model, the cache key or the audit record — the audit log records
what really happened, and the harness gets the scaled number.

`run` and `stream` are the same sequence: `_begin` does everything up to the model,
`_finish` does everything after it, and the two differ only in whether the middle
emits deltas as they arrive (sec 7.3's TTFT budget) or produces the whole result
first (best-of-N, which cannot be streamed honestly).

One deliberate asymmetry in that middle, stated here because it is invisible at the
call site: the incrementally-streamed path does **not** go through `Cascade`. It
reads `Backend.stream` directly and reports a synthesised `CascadeInfo` with
`candidates_generated=1`. That is sound only because `_not_streamable` has already
excluded every turn the cascade would do anything for — a turn that streams carries
no tools, is not `code_change` and is not `plan`, so there is no best-of-N to run,
no rerank to ask for and no T2 escalation to consider. It also means the sec 7.3
pressure valve does not apply to those turns, which is the intent: they are the
turns whose whole point is first-token latency.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from tandem.attest.audit import AuditLog, AuditRecord, now, sha256_text
from tandem.attest.receipt import Receipt, Tier0Attestation, Tier1Attestation
from tandem.backends.base import Backend, Delta, render_default
from tandem.backends.resident_swap import ResidentSwapBackend
from tandem.config import Config
from tandem.eval.worktree import WorktreeRunner
from tandem.eval.worktree import from_config as build_worktree_runner
from tandem.gateway.cache.kv_disk import DiskKVCache, KVSnapshot, align_down
from tandem.gateway.cache.prompt_cache import CacheEntry, PromptCache, chunk_digests
from tandem.gateway.compaction import Compactor
from tandem.gateway.context_scale import ContextScaler
from tandem.gateway.toolcall.constrain import Constrainer, tool_call_schema
from tandem.gateway.toolcall.repair import looks_like_tool_intent, repair
from tandem.gateway.toolcall.replay import ReplayMap, render_call
from tandem.router.cascade import Cascade, CascadeInfo
from tandem.router.classify import Classification, classify
from tandem.tier1.verifier import Tier1Verifier
from tandem.types import (
    GenRequest,
    GenResult,
    KVState,
    Message,
    Role,
    Sampling,
    StopReason,
    ToolCall,
    TurnClass,
    Usage,
)

# Enough to cover a working session's recent history for `/tandem/trace/last` and
# the sec 10.5 measurement discipline, without growing without bound.
_MAX_TRACES = 512

_RETRY_INSTRUCTION = (
    "Your previous reply was not a valid tool call. Reply with a single JSON object "
    'of exactly this shape and nothing else: {"name": "<tool name>", "arguments": '
    "{...}}. No prose, no markdown fence, no XML."
)


@dataclass
class TurnTrace:
    """Everything one turn did, for diagnostics and the measurement discipline (sec 10.5)."""

    request_id: str = ""
    compaction: dict[str, Any] = field(default_factory=dict)
    cascade: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    toolcall: dict[str, Any] = field(default_factory=dict)
    stream: dict[str, Any] = field(default_factory=dict)
    ttft_s: float = 0.0
    total_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "compaction": self.compaction,
            "cascade": self.cascade,
            "cache": self.cache,
            "toolcall": self.toolcall,
            "stream": self.stream,
            "ttft_s": round(self.ttft_s, 3),
            "total_s": round(self.total_s, 3),
        }


@dataclass
class _Turn:
    """One turn's state between `_begin` and `_finish`."""

    req: GenRequest
    rendered: str
    trace: TurnTrace
    t0: float


def _served_test_runner(cfg: Config) -> WorktreeRunner | None:
    """The worktree runner the served path uses for T2 escalation, or None.

    Two gates, both deliberate. The config has to opt in, because running the
    repository's suite on every `code_change` turn is a latency decision and a
    decision to execute that repository's code on every turn. And the runner has to
    have a test command: one built from linters alone would answer "passed" to
    every escalation check, which suppresses T2 while looking wired up — worse than
    leaving it dormant, because it is indistinguishable from working.
    """
    if not cfg.eval.escalate_on_test_failure:
        return None
    runner = build_worktree_runner(cfg.eval)
    if runner is None or not runner.measures_tests:
        return None
    return runner


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        tier0: Backend,
        tier1: Backend | None = None,
        *,
        audit: AuditLog | None = None,
    ):
        self.cfg = cfg
        # Rung 2 evicts tier 0 to make room for the verifier (sec 5.5), so tier-0
        # requests have to wait for tier 0 to be the model that is actually in
        # memory. The guard goes on here rather than in `build_tier1` because this is
        # the served path — the only place two requests race — and because a caller
        # that builds a Pipeline gets the guard whether or not it knew to ask.
        if isinstance(tier1, ResidentSwapBackend):
            tier0 = tier1.guard_tier0(tier0)
        self.tier0 = tier0
        self.tier1 = tier1
        self.verifier = Tier1Verifier(tier1, timeout_s=cfg.tier1.request_timeout_s)
        # T2 failure escalation (sec 7.2) is only reachable when the host can
        # actually run the repository's tests. Without a runner `Cascade` leaves the
        # whole path dormant, which is the honest behaviour: escalating on a failure
        # nobody observed is theatre. Off unless the config opts in — turning it on
        # runs the repo's suite on every code_change turn.
        self.worktree = _served_test_runner(cfg)
        self.test_runner = (
            self.worktree.as_test_runner(
                escalate_on_apply_failure=cfg.eval.escalate_on_apply_failure
            )
            if self.worktree is not None
            else None
        )
        self.cascade = Cascade(
            tier0, self.verifier, cfg.router, test_runner=self.test_runner
        )
        self.compactor = Compactor(
            enabled=cfg.compaction.enabled,
            strip_schemas=cfg.compaction.strip_tool_schemas,
            keep_original=cfg.compaction.keep_original,
            count_tokens=tier0.count_tokens,
        )
        self.scaler = ContextScaler(
            enabled=cfg.context_scale.enabled,
            assumed_window=cfg.context_scale.assumed_window,
            real_window=cfg.context_scale.real_window,
        )
        self.prompt_cache = PromptCache(
            budget_bytes=cfg.cache.prompt_cache_bytes,
            chunk_bytes=max(64, cfg.cache.kv_chunk_tokens * 4),
        )
        self.disk_kv = (
            DiskKVCache(
                cfg.cache.disk_kv_dir, budget_bytes=cfg.cache.disk_kv_budget_bytes
            )
            if cfg.cache.disk_kv_enabled
            else None
        )
        self.replay = ReplayMap(max_size=cfg.toolcall.replay_map_size)
        self.constrainer = Constrainer(enabled=cfg.toolcall.constrain)
        self.audit = audit or AuditLog(cfg.attest.audit_log, fsync=cfg.attest.fsync)
        # Bounded. Only `/tandem/trace/last` and the measurement discipline read
        # this, both of which want recent turns; an unbounded list is a slow leak
        # in a process that is meant to stay resident for a working day.
        self.traces: deque[TurnTrace] = deque(maxlen=_MAX_TRACES)
        # Rolling tool-call tally, so `tandem gate toolcall` can report the sec 10.2
        # number from live traffic rather than only from a synthetic suite.
        self.tool_turns = 0
        self.tool_turns_wellformed = 0
        # Tool-bearing turns the model answered in prose without attempting a call.
        # Held apart from both sides of the ratio; see `_settle_tool_calls`.
        self.tool_turns_prose = 0
        # Disk-cache hits are counted separately from the in-memory ones: they are
        # the ones that prove the cache survives a restart, and folding them into
        # `PromptCache.stats` would hide a disk cache that never hits behind a
        # healthy in-process hit rate.
        self.disk_kv_hits = 0
        self.disk_kv_misses = 0

    # --- rendering ----------------------------------------------------------

    def render(self, req: GenRequest) -> str:
        """Render through the backend, routing tool calls through the replay map.

        A backend with a real chat template owns its own rendering — but it is
        *handed* the replay renderer rather than left to find one, because a
        backend that cannot reach the map has no way to put the model's own sampled
        bytes back into the prompt, and reconstructing them from parsed calls is
        exactly what breaks the byte prefix (sec 8.5.5). Tier 0 dropped tool calls
        entirely for as long as it had no renderer to route them through.

        Asked of the backend rather than of its type: a wrapper that delegates
        `render` (rung 2's `SwapGuard`) changes none of the bytes, and a type check
        would read it as a real chat template and quietly drop replay-aware
        rendering.
        """

        def renderer(call: ToolCall) -> str:
            return render_call(call, self.replay)

        if not self.tier0.renders_canonically():
            return self.tier0.render(req, renderer)
        return render_default(req, renderer)

    # --- main path ----------------------------------------------------------

    async def run(
        self, req: GenRequest, *, no_compact: bool = False
    ) -> tuple[GenResult, TurnTrace]:
        turn = self._begin(req, no_compact=no_compact)
        result = await self._produce(turn)
        return result, turn.trace

    async def stream(
        self, req: GenRequest, *, no_compact: bool = False
    ) -> AsyncIterator[Delta]:
        """Stream a turn, incrementally where that can be done honestly.

        Single-candidate turns carrying no tools — `chat`, and `read_only` without
        a tool inventory — stream token by token straight off `Backend.stream`, so
        TTFT is first-token latency and the sec 7.3 budget (`chat` < 2 s) is
        actually served rather than merely framed correctly.

        Everything else runs to completion and emits one delta. Three reasons, all
        of them structural rather than unfinished work:

        * A `code_change` turn under best-of-N has no tokens to emit until the
          verifier has chosen a candidate. Streaming candidate 0 and retracting it
          would be worse than a pause.
        * A `plan` turn's text is rewritten after generation, when the tier-1
          critique is appended.
        * A tool-bearing turn needs the complete text before the tool-call layer
          (sec 8.5) can repair or retry it, and under constrained decoding the
          "text" is a JSON envelope that is not prose at all.

        The first delta is a **prologue**: `done=False`, no text, and a `result`
        carrying nothing but the prompt-side usage, so a wire encoder can open its
        stream with a true input-token count before any tokens exist. Backends
        never emit one; it is the pipeline's own.
        """
        turn = self._begin(req, no_compact=no_compact)
        yield Delta(result=GenResult(usage=self._prompt_usage(turn)))

        cls = classify(turn.req)
        reason = self._not_streamable(turn.req, cls)
        turn.trace.stream = {"incremental": reason == "", "reason": reason}
        if reason:
            result = await self._produce(turn)
            if result.text:
                yield Delta(text=result.text)
            yield Delta(done=True, result=result)
            return

        info = CascadeInfo(
            turn=cls.turn, classification=cls.as_dict(), candidates_generated=1
        )
        chunks: list[str] = []
        final: GenResult | None = None
        async for delta in self.tier0.stream(turn.req):
            if delta.text:
                if not turn.trace.ttft_s:
                    turn.trace.ttft_s = time.perf_counter() - turn.t0
                chunks.append(delta.text)
                yield Delta(text=delta.text)
            if delta.done:
                final = delta.result
                break

        streamed = "".join(chunks)
        result = final if final is not None else GenResult(text=streamed)
        if result.text != streamed:
            if result.text.startswith(streamed):
                # A backend that streams a prefix and completes on the final delta.
                tail = result.text[len(streamed) :]
                yield Delta(text=tail)
            else:
                # A backend whose deltas do not reassemble into its own result has
                # broken its contract. The client already has `streamed`, so that is
                # what the receipt and the audit log must attest to (sec 9.2) — the
                # log records what happened, not what was meant to.
                turn.trace.stream["mismatch"] = True
                result.text = streamed
        if not result.usage.input_tokens and not result.usage.output_tokens:
            result.usage = Usage(
                input_tokens=self.tier0.count_tokens(turn.rendered),
                output_tokens=self.tier0.count_tokens(result.text),
            )
        turn.trace.stream["deltas"] = len(chunks)

        info.elapsed_s = time.perf_counter() - turn.t0
        turn.trace.cascade = info.as_dict()
        result, tool_info = await self._settle_tool_calls(turn.req, result)
        turn.trace.toolcall = tool_info
        yield Delta(done=True, result=await self._finish(turn, result, info))

    # --- shared stages ------------------------------------------------------

    def _begin(self, req: GenRequest, *, no_compact: bool) -> _Turn:
        """Everything up to the model: compaction, sampling, rendering, cache probe."""
        t0 = time.perf_counter()
        if not req.request_id:
            req = req.with_(request_id=uuid.uuid4().hex)
        trace = TurnTrace(request_id=req.request_id)

        req, comp = self.compactor.apply(req, force_off=no_compact)
        trace.compaction = comp.as_dict()

        # Constrain *after* the render and the cache probe, which is the order this
        # module's own docstring states. It used to run first, and while that was
        # benign for `render_default` — which reads neither sampling nor
        # `json_schema` — `Backend.render` is an override point, and a chat template
        # that serialised `response_format` would have put the constraining schema
        # into the rendered bytes and therefore into the cache key. Two turns
        # differing only in a schema the model never sees would then miss each
        # other's cached prefix. Ordering it here makes that unrepresentable rather
        # than merely unobserved.
        rendered = self.render(req)
        req, trace.cache = self._probe_cache(rendered, req)
        req = self._prepare_sampling(req)
        return _Turn(req=req, rendered=rendered, trace=trace, t0=t0)

    def count_prompt_tokens(self, req: GenRequest, *, no_compact: bool = False) -> int:
        """Prompt tokens for `req`, counted against the prompt that would be sent.

        The count endpoint has to run the same compaction the completion path runs,
        or it describes a request this gateway would never send: the raw harness
        prompt is the compaction multiplier larger than what reaches the model, and
        a harness that drives its context meter and its auto-compact threshold off
        this number will compact its own history far too aggressively on the
        strength of it.

        Not recorded in the compaction history. A count is a probe, not a served
        turn — Claude Code issues them continuously — so recording them would both
        skew the M1 gate's statistics and bury the last real request in the sec 8.2
        diff view behind a stream of probes.
        """
        req, _ = self.compactor.apply(req, force_off=no_compact, record=False)
        return self.tier0.count_tokens(self.render(req))

    async def _produce(self, turn: _Turn) -> GenResult:
        """The non-streaming middle: cascade, then settle tool calls."""
        result, cinfo = await self.cascade.produce(turn.req)
        turn.trace.cascade = cinfo.as_dict()
        result, tool_info = await self._settle_tool_calls(turn.req, result)
        turn.trace.toolcall = tool_info
        return await self._finish(turn, result, cinfo)

    async def _finish(
        self, turn: _Turn, result: GenResult, cinfo: CascadeInfo
    ) -> GenResult:
        """Everything after the model: cache, receipt, audit, reported usage."""
        # A cache store can never fail a turn. The model has already answered by
        # the time we get here, and `_remember` is the first thing `_finish` does,
        # so any exception out of it used to lose a completed answer *and* skip the
        # audit record below — a request that vanished from the sec 9.2 chain
        # entirely while returning a 500 to the harness. A cache is an
        # optimisation: the only correct behaviour on a store failure is to degrade
        # to a cache miss and carry on. Recorded in the trace so a cache that has
        # silently stopped storing is visible rather than merely slow.
        try:
            await self._remember(turn.rendered, turn.req, result)
        except Exception as exc:  # noqa: BLE001 - a cache must not be able to fail a request
            turn.trace.cache["store_error"] = f"{type(exc).__name__}: {exc}"

        receipt = self._build_receipt(turn.req, cinfo)
        result.receipt = (
            receipt.as_dict() if self.cfg.attest.attach_to_response else None
        )
        await self._write_audit(turn.req, turn.rendered, result, cinfo)

        # Reporting-only scaling (sec 8.3). Applied after the audit record so the
        # log holds true counts.
        result.usage = Usage(
            input_tokens=self.scaler.scale(result.usage.input_tokens),
            output_tokens=result.usage.output_tokens,
            cached_input_tokens=self.scaler.scale(result.usage.cached_input_tokens),
        )

        result.total_s = time.perf_counter() - turn.t0
        turn.trace.total_s = result.total_s
        # A turn that never streamed had its first byte at the end. Recording that
        # as a TTFT of zero would flatter the sec 7.3 measurement.
        turn.trace.ttft_s = turn.trace.ttft_s or result.total_s
        result.ttft_s = turn.trace.ttft_s
        self.traces.append(turn.trace)
        return result

    def _prompt_usage(self, turn: _Turn) -> Usage:
        """Prompt-side usage for the stream prologue, already context-scaled."""
        return Usage(
            input_tokens=self.scaler.scale(self.tier0.count_tokens(turn.rendered))
        )

    def _not_streamable(self, req: GenRequest, cls: Classification) -> str:
        """Why this turn cannot stream incrementally, or "" if it can."""
        if req.has_tools():
            return "tool-bearing turn: the tool-call layer needs the whole reply"
        if cls.turn is TurnClass.CODE_CHANGE:
            # True at N=1 as well: T2 can replace the whole patch after the tests
            # run, so even a single-candidate code_change turn has nothing it can
            # commit to mid-flight.
            return "code_change turn: no tokens to emit until best-of-N and T2 have settled"
        if cls.turn is TurnClass.PLAN:
            return "plan turn: the tier-1 critique is appended after generation"
        return ""

    # --- stages -------------------------------------------------------------

    def _prepare_sampling(self, req: GenRequest) -> GenRequest:
        """Cool tool-bearing turns and attach the constraining schema (sec 8.5)."""
        if not req.has_tools():
            return req
        sampling = Sampling(
            temperature=min(
                req.sampling.temperature, self.cfg.toolcall.tool_turn_temperature
            ),
            top_p=req.sampling.top_p,
            seed=req.sampling.seed,
            max_tokens=req.sampling.max_tokens,
            stop=req.sampling.stop,
        )
        schema = req.json_schema
        if schema is None and self.constrainer.available:
            schema = tool_call_schema(req.tools)
        return req.with_(sampling=sampling, json_schema=schema)

    def _probe_cache(
        self, rendered: str, req: GenRequest
    ) -> tuple[GenRequest, dict[str, Any]]:
        """Find the longest reusable prefix, in memory first, then on disk.

        Memory before disk because a hit there needs no read at all. Disk is what
        makes the first turn after a restart warm rather than a cold prefill
        (sec 8.4) — without it the cache is only ever as old as the process.

        Returns the request with any restored state attached, and the trace entry.
        """
        total = len(rendered.encode("utf-8"))
        hit = self.prompt_cache.lookup(rendered)
        if hit is not None:
            if hit.entry.replay:
                self.replay.load(hit.entry.replay)
            return req, {
                "prefix_hit": True,
                "source": "memory",
                "covered_bytes": hit.covered_bytes,
                "remaining_bytes": hit.remaining_bytes,
                "covered_fraction": round(hit.covered_bytes / max(1, total), 3),
            }

        restored = self._restore_from_disk(rendered, req)
        if restored is None:
            return req, {"prefix_hit": False, "source": None, "covered_bytes": 0}
        state, snap = restored
        self.replay.load(snap.replay)
        return req.with_(warm_state=state), {
            "prefix_hit": True,
            "source": "disk",
            "covered_bytes": state.prefix_bytes,
            "remaining_bytes": total - state.prefix_bytes,
            "covered_fraction": round(state.prefix_bytes / max(1, total), 3),
            "restored_tokens": state.n_tokens,
        }

    def _restore_from_disk(
        self, rendered: str, req: GenRequest
    ) -> tuple[KVState, KVSnapshot] | None:
        """Longest chunk-aligned prefix of `rendered` we hold a usable state for.

        Walks boundaries longest-first and stats before reading, so a miss costs a
        few `stat` calls rather than a deserialisation. A snapshot whose
        `state_key` does not match this backend, container and adapter is skipped:
        it belongs to a different model, and restoring it would continue the
        conversation in one that never saw its own prefix.
        """
        if self.disk_kv is None or not self.tier0.supports_state():
            return None
        want = self.tier0.state_key(req.adapter)
        for prefix_bytes, digest in reversed(
            chunk_digests(rendered, self.prompt_cache.chunk_bytes)
        ):
            if not self.disk_kv.has(digest):
                continue
            snap = self.disk_kv.get(digest)
            if snap is None or snap.state_key != want:
                continue
            state = KVState(
                key=snap.state_key,
                prefix_bytes=snap.prefix_bytes or prefix_bytes,
                token_ids=tuple(snap.token_ids),
                next_logits=snap.next_logits,
                blob=snap.state_blob,
            )
            if not self.tier0.accepts_state(state, req.adapter):
                continue
            self.disk_kv_hits += 1
            return state, snap
        self.disk_kv_misses += 1
        return None

    async def _settle_tool_calls(
        self, req: GenRequest, result: GenResult
    ) -> tuple[GenResult, dict[str, Any]]:
        """Repair, then bounded retry (sec 8.5.3-4)."""
        info: dict[str, Any] = {"tools_present": req.has_tools()}
        if not req.has_tools():
            return result, info

        self.tool_turns += 1
        if result.tool_calls:
            info["outcome"] = "wellformed"
            self.tool_turns_wellformed += 1
            self.replay.put_all(result.raw_blocks)
            return result, info

        if self.cfg.toolcall.repair and result.text:
            outcome = repair(result.text, req.tools)
            info["repair_strategy"] = outcome.strategy
            info["rejected"] = outcome.rejected
            if outcome.ok:
                result.tool_calls = outcome.calls
                result.text = outcome.residual_text
                result.stop_reason = StopReason.TOOL_USE
                result.repaired = True
                result.raw_blocks.update(outcome.raw_blocks)
                self.replay.put_all(outcome.raw_blocks)
                info["outcome"] = "repaired"
                self.tool_turns_wellformed += 1
                return result, info

        # Retry only when the model was *trying* to call a tool. Re-prompting a turn
        # that was legitimately prose costs a whole generation and makes the model
        # chattier, not more correct (sec 8.5.4).
        if not looks_like_tool_intent(result.text):
            info["outcome"] = "prose"
            # Counted, but into neither side of the sec 10.2 ratio. This turn
            # produced no tool call and was not attempting one, so it is not
            # evidence that the tool-call layer works — it used to increment
            # `wellformed`, which meant a model that answered every tool-bearing
            # turn with "Sure, I'll take a look!" scored a perfect rate. Nor is it
            # straightforwardly a failure: on live traffic a model may legitimately
            # answer a tool-bearing turn in prose, and scoring that against the
            # gate would understate a layer that is working. It is not measured,
            # which this codebase models as its own outcome rather than folding
            # into whichever side is convenient. (The synthetic gate is a different
            # case and treats prose as a failure: every one of its scenario steps
            # requires a call, so there prose *is* the failure being measured.)
            self.tool_turns_prose += 1
            return result, info

        retry_req = req
        for attempt in range(1, self.cfg.toolcall.max_retries + 1):
            retry_req = retry_req.with_(
                messages=[
                    *retry_req.messages,
                    Message(role=Role.ASSISTANT, content=result.text),
                    Message(role=Role.USER, content=_RETRY_INSTRUCTION),
                ]
            )
            result = await self.tier0.generate(retry_req)
            result.retries = attempt
            if result.tool_calls:
                info["outcome"] = f"retry_{attempt}"
                self.tool_turns_wellformed += 1
                self.replay.put_all(result.raw_blocks)
                return result, info
            outcome = repair(result.text, req.tools)
            if outcome.ok:
                result.tool_calls = outcome.calls
                result.text = outcome.residual_text
                result.stop_reason = StopReason.TOOL_USE
                result.repaired = True
                result.raw_blocks.update(outcome.raw_blocks)
                self.replay.put_all(outcome.raw_blocks)
                info["outcome"] = f"retry_{attempt}_repaired"
                self.tool_turns_wellformed += 1
                return result, info

        info["outcome"] = "failed"
        return result, info

    async def _remember(
        self, rendered: str, req: GenRequest, result: GenResult
    ) -> None:
        """Cache the turn's prefix in memory and, when possible, on disk (sec 8.4).

        Aligned down to a chunk boundary before saving: a state covering a partial
        chunk cannot be matched as a prefix later, so storing it would be dead
        weight against both budgets. Trimming also gives the BPE slack the spec
        asks for — a boundary that shifts by a token or two costs one chunk rather
        than the whole entry.
        """
        prefix = align_down(rendered, self.prompt_cache.chunk_bytes)
        if not prefix:
            return
        replay = dict(result.raw_blocks)
        prefix_bytes = len(prefix.encode("utf-8"))
        self.prompt_cache.store(
            prefix,
            CacheEntry(
                digest="",
                prefix_bytes=prefix_bytes,
                n_tokens=result.usage.input_tokens,
                size_bytes=prefix_bytes,
                replay=replay,
            ),
        )

        if self.disk_kv is None or not self.tier0.supports_state():
            return
        state = self.tier0.export_state(req, prefix, result)
        if state is None:
            return
        digest = chunk_digests(prefix, self.prompt_cache.chunk_bytes)[-1][1]
        # Off the event loop. `put` writes a temp file, fsyncs it, renames it and
        # may walk the cache directory to enforce the budget — tens of milliseconds
        # of blocking syscalls at a realistic entry count. On the loop that is not
        # this request's latency, it is every *concurrent* request's: a stream in
        # another task cannot emit a delta while this one is in `write`.
        await asyncio.to_thread(
            self.disk_kv.put,
            KVSnapshot(
                digest=digest,
                token_ids=list(state.token_ids),
                next_logits=state.next_logits,
                state_blob=state.blob,
                # The replay map rides with the state it belongs to: a restored
                # prefix whose tool calls re-render differently is not the prefix
                # that was prefilled (sec 8.5.5).
                replay=replay,
                prefix_bytes=prefix_bytes,
                state_key=state.key,
            ),
        )

    def _build_receipt(self, req: GenRequest, cinfo: CascadeInfo) -> Receipt:
        return Receipt(
            tier0=Tier0Attestation(
                container_blake3=self.tier0.container_hash(),
                adapter_blake3=self.tier0.adapter_hash(req.adapter),
                profile_blake3=self.tier0.profile_hash(req.adapter),
                adapter_name=req.adapter,
            ),
            tier1=Tier1Attestation(
                container_blake3=self.tier1.container_hash() if self.tier1 else None,
                rung=self.cfg.tier1.rung if self.tier1 else None,
                invoked=cinfo.tier1_invoked,
                call=cinfo.tier1_call,
                expert_cache_configured_bytes=(
                    self.cfg.tier1.expert_cache_bytes if self.tier1 else None
                ),
            ),
            compaction_template=req.compaction_template,
            sampling=req.sampling,
            candidates_generated=cinfo.candidates_generated,
            candidate_selected=cinfo.candidate_selected,
            escalated=cinfo.escalated,
        )

    async def _write_audit(
        self, req: GenRequest, rendered: str, result: GenResult, cinfo: CascadeInfo
    ) -> None:
        # Off the event loop for the same reason as the disk cache store, and more
        # so: the append takes a file lock, re-reads the chain tail under it and
        # optionally fsyncs. Unlike the cache this is *not* wrapped in a
        # degrade-to-nothing handler — an audit record that cannot be written is a
        # failed request, because a served turn missing from the sec 9.2 chain is
        # precisely what the chain exists to make impossible.
        await asyncio.to_thread(
            self.audit.append,
            AuditRecord(
                request_id=req.request_id,
                ts=now(),
                harness=req.harness,
                tier0_hash=self.tier0.container_hash(),
                adapter_hash=self.tier0.adapter_hash(req.adapter),
                tier1_hash=self.tier1.container_hash() if self.tier1 else None,
                prompt_sha256=sha256_text(rendered),
                output_sha256=sha256_text(result.text),
                tools_invoked=tuple(c.name for c in result.tool_calls),
                escalated=cinfo.escalated,
            ),
        )

    # --- reporting ----------------------------------------------------------

    def _disk_kv_stats(self) -> dict[str, Any] | None:
        if self.disk_kv is None:
            return None
        total = self.disk_kv_hits + self.disk_kv_misses
        return {
            **self.disk_kv.stats(),
            "hits": self.disk_kv_hits,
            "misses": self.disk_kv_misses,
            "hit_rate": round(self.disk_kv_hits / total, 3) if total else 0.0,
            "backend_supports_state": self.tier0.supports_state(),
        }

    def tool_call_rate(self) -> dict[str, Any]:
        """Live tool-call validity, against the sec 10.2 blocking gate.

        The denominator is tool-bearing turns on which the model *attempted* a
        call. Prose turns are reported alongside rather than folded in, so a rate
        of 1.0 over three attempts cannot look like a rate of 1.0 over three
        hundred — the same denominator discipline `apply_rate` exists to enforce in
        the merge eval.
        """
        attempted = self.tool_turns - self.tool_turns_prose
        if not attempted:
            return {
                "turns": self.tool_turns,
                "attempted": 0,
                "prose": self.tool_turns_prose,
                "rate": None,
                "pass": None,
                "threshold": 0.99,
            }
        rate = self.tool_turns_wellformed / attempted
        return {
            "turns": self.tool_turns,
            "attempted": attempted,
            "prose": self.tool_turns_prose,
            "wellformed": self.tool_turns_wellformed,
            "rate": round(rate, 4),
            "threshold": 0.99,
            "pass": rate >= 0.99,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "compaction": self.compactor.measure(),
            "prompt_cache": self.prompt_cache.stats(),
            "disk_kv": self._disk_kv_stats(),
            "replay": self.replay.stats(),
            "constrain": self.constrainer.status(),
            "toolcall": self.tool_call_rate(),
            "router": self.cascade.stats(),
            "escalation": (
                self.worktree.describe()
                if self.worktree is not None
                else {
                    "enabled": False,
                    "reason": "no test command configured (tandem.toml [eval])",
                }
            ),
            "context_scale": self.scaler.describe(),
        }

    async def close(self) -> None:
        await self.tier0.close()
        if self.tier1:
            await self.tier1.close()
