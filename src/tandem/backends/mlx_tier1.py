"""Tier 1 — streamed verifier, reached over a process boundary (spec sec 5.4).

v1 runs `mlx-optiq --stream-experts` as a **separate process** and talks to it over
its OpenAI-compatible endpoint. That is not laziness about integration: mlx-optiq's
source lives in a private monorepo, so it is unforkable and unauditable, and the
only defensible way to depend on it is as a black box behind a pinned version and a
process boundary (sec 13). Phase 2 replaces it with an in-house streaming loader on
mlx-lm and this file's interface does not change.

Two invariants hold on every tier-1 call, and both now live in `tier1_call.py`
because rung 4 is a second transport and a clamp that drifts between transports is
not a clamp:

* **Tier 1 never generates a patch** (sec 5.1). `max_tokens` is clamped to the call
  type's budget. A `review` cannot quietly become a 4,000-token generation that
  takes six minutes — the clamp makes that structurally impossible, not merely
  discouraged.
* **Output is schema-constrained.** 2-bit's specific documented weakness is broken
  JSON and invented schema fields (sec 5.2); constrained decoding is what makes
  that weakness not bite. If the endpoint cannot constrain, we validate and reject
  rather than pass malformed judgements upstream.

What is left here is the part that is specific to this transport: the client, the
process boundary, and the prefill instrumentation Gate B reads.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from tandem.attest.hashing import hash_artefact
from tandem.backends.base import Backend
from tandem.backends.tier1_call import (
    Tier1Unavailable,
    build_payload,
    read_completion,
    resolve_reasoning_control,
    validate_or_raise,
)
from tandem.thresholds import SPEC_GATE_B_PREFILL_TOK_PER_S, judge
from tandem.types import (
    GenRequest,
    GenResult,
    Message,
    Role,
    Sampling,
    StopReason,
    Usage,
)

# Fragments the filler is assembled from. Deliberately a wide vocabulary rather
# than one repeated line — see `prefill_filler`.
_FILLER_VERBS = (
    "resolve",
    "collect",
    "flatten",
    "annotate",
    "reconcile",
    "dispatch",
    "prune",
    "serialise",
    "bisect",
    "coalesce",
    "validate",
    "rehydrate",
    "quantise",
    "route",
    "materialise",
    "invalidate",
    "checkpoint",
    "escalate",
    "attribute",
    "compact",
)
_FILLER_NOUNS = (
    "manifest",
    "shard",
    "cursor",
    "envelope",
    "quorum",
    "digest",
    "lease",
    "watermark",
    "partition",
    "ledger",
    "snapshot",
    "backlog",
    "delta",
    "receipt",
    "adapter",
    "expert",
    "router",
    "verdict",
    "worktree",
    "corpus",
)
_FILLER_TYPES = ("bytes", "str", "int", "float", "dict", "list", "tuple", "bool")


def prefill_filler(target_chars: int) -> str:
    """Exactly `target_chars` characters of code-shaped filler, deliberately varied.

    The variety is the point, and it is specific to what Gate B measures. This
    prompt is prefilled by a *streamed* MoE, where the cost is the union of experts
    the chunk routes to, read off SSD once each. One line repeated to 16k tokens
    routes to almost no experts — identical tokens route identically, and on a model
    whose first MoE layers use hash routing they route identically by construction.
    The union collapses, the engine's expert cache serves the whole sweep out of
    RAM, and the measurement reports a throughput no real prompt will ever see.
    That is the one direction a floor test must not fail in: Gate B decides whether
    the in-house streaming loader is a three-week option or M-blocking, and a
    flattering number defers that discovery to month two.

    So the filler spans a wide identifier vocabulary, which is what a real 16k
    context of source actually looks like to a router. It is still deterministic —
    fixed seed, no wall clock — because a gate whose number moves between runs is
    not a gate.

    It does *not* claim to reproduce the expert distribution of the operator's own
    repository; nothing synthetic can. It claims only to stop understating the
    number of experts a chunk touches, which is the failure the old filler had.
    """
    rng = random.Random(20260731)
    out: list[str] = []
    total = 0
    i = 0
    while total < target_chars:
        verb = rng.choice(_FILLER_VERBS)
        noun = rng.choice(_FILLER_NOUNS)
        other = rng.choice(_FILLER_NOUNS)
        typ = rng.choice(_FILLER_TYPES)
        tag = f"{verb}_{noun}_{i}"
        # Literals are drawn per block rather than reused: constants, hex digests
        # and paths are a large share of the distinct tokens in real source, and
        # they are exactly what a repeated unit has none of.
        limit = rng.randrange(2, 99991)
        digest = f"{rng.randrange(1 << 44):011x}"
        shape = i % 4
        if shape == 0:
            block = (
                f"def {tag}({other}: {typ}, limit: int = {limit}) -> {typ}:\n"
                f'    """{verb.capitalize()} the {noun} against the {other}."""\n'
                f"    {noun} = {other}.{verb}(limit=limit, strict={rng.choice(('True', 'False'))})\n"
                f"    if not {noun}:\n"
                f'        raise ValueError(f"{other} produced no {noun} under {{limit}}")\n'
                f"    return {noun}\n\n"
            )
        elif shape == 1:
            block = (
                f"class {verb.capitalize()}{noun.capitalize()}{i}(Base{other.capitalize()}):\n"
                f"    __slots__ = ('{noun}_{i}', '{other}_{i}', 'cursor_{digest}')\n"
                f"    threshold = {limit}\n"
                f"    checksum = 0x{digest}\n\n"
                f"    def {verb}(self, {other}: {typ} | None = None) -> {typ}:\n"
                f"        return self.{noun}_{i} if {other} is None else {other}\n\n"
            )
        elif shape == 2:
            block = (
                f"{tag.upper()} = {{\n"
                f'    "{noun}": "src/{other}/{tag}.py",\n'
                f'    "{other}": {limit},\n'
                f'    "digest": "{digest}",\n'
                f'    "mode": "{rng.choice(_FILLER_VERBS)}-{rng.choice(_FILLER_NOUNS)}",\n'
                f"}}\n\n"
            )
        else:
            block = (
                f"# {verb} {noun} {i}: rebuilt when the {other} changes ({digest}).\n"
                f"@register('{tag}', priority={limit % 97})\n"
                f"async def {other}_{verb}_{i}(ctx: Context) -> {typ}:\n"
                f"    ctx.emit('{tag}', {other}={limit}, digest='{digest}')\n"
                f"    return await ctx.{verb}('{noun}')\n\n"
            )
        out.append(block)
        total += len(block)
        i += 1
    return "".join(out)[:target_chars]


def expert_cache_provenance(
    configured_bytes: int | None, *, engine_version: str = ""
) -> dict[str, Any]:
    """What a Gate B receipt may honestly say about the expert cache (sec 10.5).

    Sec 10.5 wants the cache size reported *with* the throughput, because streamed
    prefill is a function of how much of the expert set is already in RAM. The trap
    is that `tier1.expert_cache_bytes` looks like that number and is not.

    Two separate reasons, and the second is the one that bites:

    1. It is never sent. Nothing in this client puts it in a request payload — the
       engine is a process we do not launch, so its configuration is the operator's
       command line, which this process cannot see.
    2. **There is nothing to send it to.** `optiq serve --stream-experts-cache N` is
       accepted and threaded through `install_streaming_experts` → `load_streaming`
       → `StreamedSwitchLinear`, then dropped: `_ShardWeightReader.__init__` takes
       `cache_experts` and never stores it, so `read()` issues an `os.pread` per
       expert on every call. Read off the installed source at 0.4.18. The flag is
       also a *count*, not a byte budget, so even implemented it would not take this
       value. The only expert cache in the streamed path is the OS page cache, sized
       by free RAM, which no flag sets and no receipt can pin.

    So the receipt records the configured value as tandem-side config that did not
    reach the engine, names the mechanism that actually served the reads, and says
    the engine's own setting is unknown to this process. Printing a bare
    `expert_cache_bytes: 12884901888` beside a throughput would attest a 12 GB cache
    that does not exist — confidently wrong rather than merely silent, which is the
    failure shape HANDOFF §5 exists to prevent, and worse than reporting nothing.

    `claim_verified_for` is pinned deliberately: if the operator runs a newer engine
    that does implement the LRU, the receipt shows the version skew instead of
    repeating a stale claim as fact.
    """
    return {
        "configured_bytes": configured_bytes,
        "reached_engine": False,
        "engine_mechanism": "OS page cache; engine has no expert LRU",
        "engine_version": engine_version or "unknown",
        "claim_verified_for": "mlx-optiq 0.4.18",
        "note": (
            "tier1.expert_cache_bytes is not plumbed to the engine and has no engine "
            "counterpart: --stream-experts-cache is accepted, is a per-projection "
            "count rather than a byte budget, and is discarded before it reaches the "
            "shard reader. Treat streamed prefill here as page-cache-backed, and "
            "re-verify this note against the engine version above before quoting it."
        ),
    }


@dataclass
class PrefillSample:
    """One measurement of streamed prefill throughput — M0 Gate B (sec 11)."""

    input_tokens: int
    output_tokens: int
    prefill_s: float
    total_s: float

    @property
    def prefill_tok_per_s(self) -> float:
        return self.input_tokens / self.prefill_s if self.prefill_s > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "prefill_s": round(self.prefill_s, 3),
            "total_s": round(self.total_s, 3),
            "prefill_tok_per_s": round(self.prefill_tok_per_s, 1),
        }


class OptiqTier1Backend(Backend):
    """Streamed 122B verifier behind mlx-optiq's OpenAI endpoint."""

    name = "optiq-tier1"
    tier = 1

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        container_path: str | None = None,
        timeout_s: float = 180.0,
        expert_cache_bytes: int | None = None,
        pinned_version: str = "",
        reasoning_control: str = "auto",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.expert_cache_bytes = expert_cache_bytes
        self.pinned_version = pinned_version
        # Resolved once, at construction, so an unknown value is a startup error
        # rather than a failed verdict on the first reranked turn.
        self.reasoning_control = resolve_reasoning_control(reasoning_control, model)
        self._container_hash = hash_artefact(container_path)
        self._client = httpx.AsyncClient(timeout=timeout_s)
        # Every prefill measurement taken this run, for the M0 Gate B report.
        self.prefill_samples: list[PrefillSample] = []

    def container_hash(self) -> str | None:
        return self._container_hash

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> tuple[bool, str]:
        try:
            r = await self._client.get(f"{self.endpoint}/models", timeout=5.0)
        except httpx.HTTPError as exc:
            return False, f"tier 1 endpoint unreachable at {self.endpoint}: {exc}"
        if r.status_code != 200:
            return False, f"tier 1 endpoint returned {r.status_code}"
        return True, "ok"

    async def generate(self, req: GenRequest) -> GenResult:
        """Serve a canonical request. Always schema-constrained, always clamped."""
        payload = build_payload(
            req, model=self.model, reasoning_control=self.reasoning_control
        )

        t0 = time.perf_counter()
        try:
            resp = await self._client.post(
                f"{self.endpoint}/chat/completions", json=payload
            )
        except httpx.HTTPError as exc:
            raise Tier1Unavailable(f"tier 1 call failed: {exc}") from exc
        total_s = time.perf_counter() - t0
        if resp.status_code != 200:
            raise Tier1Unavailable(
                f"tier 1 returned {resp.status_code}: {resp.text[:200]}"
            )

        body = resp.json()
        text, in_tok, out_tok = read_completion(body, payload, self.count_tokens)

        # Attribute time between prefill and decode using the measured decode rate
        # if the endpoint reports one; otherwise treat the whole call as prefill,
        # which understates prefill throughput rather than flattering it.
        #
        # A reported decode at least as long as the whole call is an incoherent
        # measurement — engine-side and client-side clocks disagreeing, or a
        # response served from the engine's own cache. Subtracting it anyway would
        # floor `prefill_s` at 1e-6 and hand Gate B a throughput in the billions,
        # which is the one direction a floor test must never fail in. Discard the
        # reported figure and fall back to the conservative default above.
        decode_s = _reported_decode_seconds(body)
        if decode_s >= total_s:
            decode_s = 0.0
        prefill_s = max(1e-6, total_s - decode_s)
        self.prefill_samples.append(
            PrefillSample(in_tok, out_tok, prefill_s=prefill_s, total_s=total_s)
        )

        if req.json_schema is not None:
            validate_or_raise(text, req.json_schema)

        return GenResult(
            text=text,
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
            total_s=total_s,
        )

    async def measure_prefill(self, input_tokens: int) -> PrefillSample:
        """M0 Gate B: instrument streamed prefill at a given input size (sec 11, 14.3).

        Gate B is >=200 tok/s. Below that, batch-union prefill (sec 5.3) is not
        happening inside the engine and the in-house loader becomes M-blocking
        rather than optional — which is a three-week schedule fact, so it is worth
        measuring on day three rather than discovering in month two.
        """
        # ~4 chars/token of filler. Length is the variable Gate B is quoted at, and
        # it is sized in characters rather than in repeats of a unit: a repeat count
        # derived from the token count undershoots by ~4%, and a sample labelled 16k
        # that carried 15.3k tokens measures a different input size than the one
        # being reported.
        filler = prefill_filler(input_tokens * 4)
        req = GenRequest(
            messages=[Message(role=Role.USER, content=filler)],
            system="Reply with the single word: ok",
            sampling=Sampling(temperature=0.0, max_tokens=4),
        )
        await self.generate(req)
        return self.prefill_samples[-1]

    def gate_b_report(
        self, *, threshold_tok_per_s: float = SPEC_GATE_B_PREFILL_TOK_PER_S
    ) -> dict[str, Any]:
        """M0 Gate B, judged against `threshold_tok_per_s` and against sec 11's 200.

        The threshold is a parameter rather than a constant because a host may be
        arithmetically unable to reach the spec figure — this one is, by ~40x, and a
        rung that can never report anything but red teaches nothing. Relaxing it lets
        the streamed path run and the *next* failure become visible; `meets_spec` in
        the returned row is what stops that from reading as a passed Gate B.
        """
        if not self.prefill_samples:
            return {"pass": False, "reason": "no samples taken"}
        rates = [s.prefill_tok_per_s for s in self.prefill_samples]
        worst = min(rates)
        verdict = judge(
            worst,
            target=threshold_tok_per_s,
            spec_target=SPEC_GATE_B_PREFILL_TOK_PER_S,
            higher_is_better=True,
            digits=1,
        )
        return {
            "pass": verdict["pass"],
            "meets_spec": verdict["meets_spec"],
            "relaxed": verdict["relaxed"],
            "threshold_tok_per_s": threshold_tok_per_s,
            "spec_threshold_tok_per_s": SPEC_GATE_B_PREFILL_TOK_PER_S,
            "worst_tok_per_s": round(worst, 1),
            # Reported with the number, not alongside it (sec 10.5). Streamed
            # prefill throughput is a function of how much of the expert set the
            # cache already holds, so a rate quoted without the model and the cache
            # size is not a measurement anyone can reproduce or compare.
            "model": self.model,
            # Provenance, not a bare number: the configured byte budget never
            # reaches the engine and has no engine-side counterpart. See
            # `expert_cache_provenance`.
            "expert_cache": expert_cache_provenance(
                self.expert_cache_bytes, engine_version=self.pinned_version
            ),
            "samples": [s.as_dict() for s in self.prefill_samples],
            "note": (
                f"Below {SPEC_GATE_B_PREFILL_TOK_PER_S:.0f} tok/s the engine is not "
                "doing batch-union prefill; Phase 2 in-house loader becomes "
                "M-blocking (spec sec 5.4, 11). A `pass` against a lower "
                "`threshold_tok_per_s` says the host was judged against its own "
                "floor, not that the spec figure was met — read `meets_spec`. "
                "Filler is deterministic and identifier-diverse so the chunk's "
                "expert union is not artificially small; it does not reproduce a "
                "real repository's expert distribution."
            ),
        }


def _reported_decode_seconds(body: dict[str, Any]) -> float:
    """Best-effort decode-time extraction from engine-specific timing fields."""
    for key in ("timings", "timing", "metrics"):
        block = body.get(key)
        if isinstance(block, dict):
            for field_name in ("predicted_ms", "decode_ms", "generation_ms"):
                if field_name in block:
                    return float(block[field_name]) / 1000.0
            for field_name in ("predicted_s", "decode_s"):
                if field_name in block:
                    return float(block[field_name])
    return 0.0
