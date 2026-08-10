"""Tier 0 — resident adapted generator on MLX (spec sec 4).

Apple-Silicon-only. On any other machine, importing this module succeeds and
constructing the backend raises `BackendUnavailable` with the reason; nothing else
in the runtime depends on MLX being importable.

The load-bearing design decision here is §4.2: **adapters are never merged into the
base**. Merging is the obvious implementation and it destroys the product — one
merged copy per adapter is ~20 GB each instead of ~250 MB, multi-tenancy goes away,
and the receipt can no longer name which adapter produced a change because there is
no longer a separable adapter. So the forward stays

    y = xW + s * (x @ A) @ B

with the deltas resident and separate, and `MultiAdapterLinear` holds every mounted
adapter's (A, B) at once, selecting per request.

Selection is via a `contextvars.ContextVar`, not a module global. Under concurrent
requests a global is a race with a silent wrong-answer failure mode: request 2 sets
the adapter while request 1 is mid-decode, and request 1 finishes in the wrong
adapter with a receipt attesting to the right one. A ContextVar is per-task.

**Untested off-target.** Every line below runs only on the M4 Max. It is written to
be correct on inspection and gated by the isolation test in `tandem.eval.isolation`,
which is the thing that actually proves it.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from tandem.attest.hashing import hash_artefact
from tandem.backends import mlx_kv
from tandem.backends.base import Backend, BackendUnavailable, Delta, ToolCallRenderer
from tandem.types import GenRequest, GenResult, KVState, StopReason, Usage

# The active adapter for the current request. Default None = base model.
ACTIVE_ADAPTER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tandem_active_adapter", default=None
)


def _require_mlx() -> tuple[Any, Any]:
    try:
        import mlx.core as mx
        from mlx import nn
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise BackendUnavailable(
            "MLX is not available on this machine. Tier 0 requires Apple Silicon "
            "with mlx>=0.32. Use backend='mock' for development off-target."
        ) from exc
    return mx, nn


@dataclass
class AdapterSpec:
    """One mounted adapter: its weights, its hashes, its provenance."""

    name: str
    path: Path
    scale: float = 2.0  # alpha/r for r=32, alpha=64
    rank: int = 32
    adapter_hash: str | None = None
    profile_hash: str | None = None
    # layer key -> (A, B) as loaded. Kept int8 on disk; §E2 tests whether that is
    # lossless against bf16 deltas.
    weights: dict[str, tuple[Any, Any]] = field(default_factory=dict, repr=False)


def build_multi_adapter_linear(base_cls: Any, mx: Any, nn: Any) -> Any:
    """Construct the MultiAdapterLinear class against a live MLX.

    Built lazily so this module imports on a machine with no MLX.
    """

    class MultiAdapterLinear(nn.Module):  # type: ignore[misc]
        """A frozen base Linear plus N resident LoRA deltas, selected per request."""

        def __init__(self, base: Any):
            super().__init__()
            self.base = base
            # name -> (A [in, r], B [r, out], scale)
            self.deltas: dict[str, tuple[Any, Any, float]] = {}

        def mount(self, name: str, a: Any, b: Any, scale: float) -> None:
            self.deltas[name] = (a, b, scale)

        def __call__(self, x: Any) -> Any:
            y = self.base(x)
            active = ACTIVE_ADAPTER.get()
            if active is None:
                return y
            delta = self.deltas.get(active)
            if delta is None:
                # An adapter mounted on some layers but not this one is normal:
                # §4.3 targets top-25% routed experts, so most expert layers carry
                # no delta. Falling through to base is correct, not an error.
                return y
            a, b, scale = delta
            # Dequantise int8 deltas lazily; the cast is cheap against the matmul.
            return y + scale * ((x.astype(b.dtype) @ a.astype(b.dtype)) @ b)

    return MultiAdapterLinear


def build_logits_processor(
    token_filter: Any, mx: Any, id_bound: int | None = None
) -> Any:
    """Adapt a `TokenFilter` to `mlx_lm`'s `logits_processors` protocol (sec 8.5.1).

    `mlx_lm` calls `processor(tokens, logits)` with every token sampled so far and
    the logits for the next one, shaped `[1, vocab]`, and takes the return value as
    the new logits. Masking is additive `-inf` on the disallowed ids, applied before
    `generate_step` takes its log-softmax, so a forbidden token has zero probability
    rather than a small one.

    `tokens` carries the last prompt token followed by everything generated —
    `generate_step` prefills all but one token without invoking processors — and
    that is exactly the sequence LMFE wants. It advances its parser on the *last*
    token only and treats what came before as an opaque key, so the prompt never
    reaches the schema parser and generation is parsed from the first sampled token.

    Two guards, both for a mismatch that is otherwise silent:

    * **An empty allowed-set returns the logits untouched.** Masking everything
      makes the whole row `-inf`, and the log-softmax downstream turns that into
      NaN, which samples a garbage token instead of raising. Unconstrained output
      that repair then handles (sec 8.5.3) is a far better failure.
    * **Ids at or past the logit width are dropped.** The tokenizer counts the
      vocabulary; the model's output dimension is often padded to a different one,
      and scattering an out-of-range id writes outside the row. `id_bound` is
      `constrain.logit_width_bound` — one past the largest id the filter can name —
      and when the row is at least that wide **no id can be out of range, so the
      filter is provably dead and is skipped**. The guard did not go away; it stopped
      running once per token to re-decide a fixed property of (tokenizer, model).
      That is F1, and it was **7.39 ms/token** — `int()` on values LMFE already
      returns as `int`. Passing no bound keeps the old per-token behaviour, because
      a wrong `safe_width` scatters outside the row and that must be derived from the
      model's own logits rather than assumed.

    **The mask cache is keyed on content, and the identity-keyed one it replaces
    could not have hit.** Caching on `id()` — pinning the list so the address cannot
    be recycled — measured **27.1 tok/s against 27.6 without**, i.e. nothing. The
    reason was never that the saving is small: lm-format-enforcer >= 0.11 builds a
    fresh list on every call, so an identity key matched **0 of 42** consecutive
    positions. It was incapable of hitting. Content is identical 40% of the time,
    ~85% within a string run, and compares in 0.12 ms against 1.48 ms to rebuild the
    array — F2. One slot, because the access pattern is a run rather than a working
    set, and because the processor is per-request so the slot dies with the turn.

    **`prev is not ids` is not redundant with `prev == ids`.** It is what makes the
    key sound if LMFE ever returns the *same* list mutated in place: identity
    equality would compare the new contents against themselves, report a hit, and
    reuse a mask built from the old ones — a silently wrong constraint, which is the
    one failure mode sec 8.5.1 cannot tolerate. Today it costs nothing (0 of 42
    shared an identity) and it fails toward a rebuild.

    **The 21 ms/token gap is not where this docstring used to put it.** It blamed
    the sync `tokens.tolist()` forces. The sync is unavoidable — the mask depends on
    the token just sampled — but it is *free*: a synthetic decode loop that syncs
    and does no other work runs at 13.24 ms/token against 13.12 unconstrained. What
    costs is host work serialised against an idle GPU, and the per-token cost is
    bimodal: ~1-4 ms in the JSON skeleton, ~10-13 ms inside a string, ~26-29 ms on
    the transition into one. F1 and F2 remove the two largest host-side terms;
    `docs/CONSTRAINED_DECODE.md` §5 has the measurements, and F4 — restructuring the
    decode loop so what remains overlaps the forward pass — is deliberately not done
    here, because it means owning the loop `mlx_lm.stream_generate` currently owns.
    """
    cached: tuple[Any, int, Any, Any] | None = None

    def processor(tokens: Any, logits: Any) -> Any:
        nonlocal cached
        allowed = token_filter(tokens.tolist())
        if not allowed:
            return logits
        vocab = logits.shape[-1]
        if id_bound is not None and vocab >= id_bound:
            ids = allowed
        else:
            ids = [int(t) for t in allowed if 0 <= int(t) < vocab]
            if not ids:
                return logits
        if cached is not None:
            prev_ids, prev_vocab, prev_dtype, prev_mask = cached
            if (
                prev_vocab == vocab
                and prev_dtype == logits.dtype
                and prev_ids is not ids
                and prev_ids == ids
            ):
                return logits + prev_mask
        mask = mx.full((vocab,), -float("inf"), dtype=logits.dtype)
        mask[mx.array(ids, dtype=mx.int32)] = 0.0
        cached = (ids, vocab, logits.dtype, mask)
        return logits + mask

    return processor


class MLXTier0Backend(Backend):
    """Resident Qwen3.6-35B-A3B @4-bit with N adapters mounted."""

    name = "mlx-tier0"
    tier = 0

    def __init__(
        self,
        model_path: str,
        *,
        adapter_dir: str | None = None,
        mtp: bool = True,
        max_kv_tokens: int = 32_768,
        max_state_bytes: int = 1 << 30,
    ):
        self.mx, self.nn = _require_mlx()
        try:
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("mlx-lm is required for tier 0") from exc

        self.model_path = model_path
        self.mtp = mtp
        self.max_kv_tokens = max_kv_tokens
        self.max_state_bytes = max_state_bytes
        self.model, self.tokenizer = load(model_path)
        self._container_hash = hash_artefact(model_path)
        self._adapters: dict[str, AdapterSpec] = {}
        self._targets: dict[str, Any] = {}
        # LMFE's vocabulary preprocessing, built on first constrained turn and kept
        # for the life of the backend. Not built here: it costs ~1.1 s and ~0.6 GB
        # against this tokenizer, and a deployment serving only free-form turns
        # should not pay it. `unload()` drops it with the tokenizer it describes.
        self._constrain_vocab: Any = None
        self._constrain_vocab_failed = False

        self._wrap_targets()
        if adapter_dir:
            self.mount_all(adapter_dir)
        # Wire the base so it is never paged out from under a decode (sec 2.1
        # budgets 20 GB wired for tier 0).
        self._wire_base()

    # --- residency (sec 5.5 rung 2) -----------------------------------------
    #
    # `Occupant`: what fallback rung 2 needs of a model that has to leave unified
    # memory so the 80B verifier can be admitted, and come back afterwards.

    async def unload(self) -> None:
        """Drop the weights. After this the backend cannot serve until `load`.

        The adapter *specs* survive — their paths and hashes are what `load` re-mounts
        from. Only the tensors go, which is the whole point: at 4-bit a 35B is ~20 GB
        and rung 2 exists because that 20 GB is what the 80B needs.
        """
        self.model = None
        self.tokenizer = None
        self._targets = {}
        # The vocabulary describes the tokenizer that is going away. `load()` builds
        # a fresh tokenizer, and handing a stale prefix tree to a reloaded model
        # would constrain decoding against a vocabulary that is no longer the one
        # being sampled from — wrong tokens allowed, silently.
        self._constrain_vocab = None
        self._constrain_vocab_failed = False
        for spec in self._adapters.values():
            spec.weights = {}
        with contextlib.suppress(AttributeError):  # pragma: no cover - older mlx
            self.mx.clear_cache()

    async def load(self) -> None:
        """Re-admit the weights, with the same identity they had before.

        Same container, same adapters, same adapter hashes — a tier 0 that came back
        from a swap with a different adapter set would produce receipts naming what
        did not run. The hashes are recomputed from the same paths rather than
        trusted from before, so a container edited while the model was out of memory
        is a mount failure and not a quietly wrong receipt.
        """
        if self.model is not None:
            return
        from mlx_lm import load

        self.model, self.tokenizer = load(self.model_path)
        self._wrap_targets()
        for name, spec in list(self._adapters.items()):
            self.mount(name, spec.path)
        self._wire_base()

    # --- mounting -----------------------------------------------------------

    def _wrap_targets(self) -> None:
        """Replace every targeted Linear with a MultiAdapterLinear.

        Target set per §4.3: attention projections, router/gate and shared-expert
        MLP always; routed experts only where a profile selects them, which the
        adapter's own key set decides at mount time.
        """
        cls = build_multi_adapter_linear(self.nn.Linear, self.mx, self.nn)
        for key, module, attr, child in _walk_linears(self.model, self.nn):
            if not _is_target(key):
                continue
            wrapper = cls(child)
            setattr(module, attr, wrapper)
            self._targets[key] = wrapper

    def mount_all(self, adapter_dir: str) -> None:
        """Mount every adapter in a directory **at startup**.

        Loading mid-flight stalls every in-flight request (sec 4.2), so there is
        deliberately no lazy path. An adapter that appears on disk later is
        available at the next restart, and the receipt says which was mounted.
        """
        root = Path(adapter_dir)
        if not root.is_dir():
            return
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and (entry / "adapters.safetensors").exists():
                self.mount(entry.name, entry)

    def mount(self, name: str, path: Path) -> AdapterSpec:
        weights = self.mx.load(str(path / "adapters.safetensors"))
        cfg = _read_adapter_config(path)
        spec = AdapterSpec(
            name=name,
            path=path,
            scale=cfg.get("alpha", 64) / max(1, cfg.get("rank", 32)),
            rank=cfg.get("rank", 32),
            adapter_hash=hash_artefact(path),
            profile_hash=hash_artefact(path / "routing_profile.json"),
        )
        mounted = 0
        for key, wrapper in self._targets.items():
            a = weights.get(f"{key}.lora_a")
            b = weights.get(f"{key}.lora_b")
            if a is None or b is None:
                continue
            wrapper.mount(name, a, b, spec.scale)
            mounted += 1
        if mounted == 0:
            raise ValueError(
                f"adapter {name!r} at {path} matched no target layer — key naming "
                "mismatch between the trainer and the served model"
            )
        self._adapters[name] = spec
        return spec

    def mounted_adapters(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    # --- KV state (sec 8.4) -------------------------------------------------

    def supports_state(self) -> bool:
        return True

    def export_state(
        self, req: GenRequest, rendered_prefix: str, result: GenResult | None
    ) -> KVState | None:
        """Snapshot this turn's KV cache, or None.

        The cache arrives on the result (`GenResult.kv_handle`) and is consumed
        here, because this is the only place that knows how far back the gateway
        wants: it keys the entry on a chunk-aligned *byte* prefix, and the cache at
        this point also holds everything the turn generated.

        **Coverage is counted in tokens, never in the bytes it is keyed by.** The
        chunk boundary lands mid-token, so `rendered_prefix` re-encodes to
        something that agrees with the turn's own ids only up to that boundary. It
        is used to confirm the cache is a superset of what the key names; what the
        state *carries* is its own token ids, and `_warm_start` re-checks those
        against the next prompt before restoring anything.

        Two shapes of cache, because the baseline container is the second one:

        * **Trimmable** (`KVCache`, `RotatingKVCache`, `QuantizedKVCache`) — rewind
          to the keyed prefix. Best case: the state then serves any conversation
          sharing that prefix, not only this one's continuation.
        * **Not trimmable.** 30 of this container's 40 layers are linear attention
          carrying a recurrent state, and a recurrent state cannot be rewound —
          `ArraysCache.is_trimmable()` is False, so `can_trim_prompt_cache` is
          False for the whole hybrid. Refusing here would have made this feature a
          no-op on the only model it ships against, invisibly, because every cache
          the off-target fake could build was trimmable. So the state instead
          covers everything the cache holds, prompt *and* reply, which is a prefix
          of the next turn's prompt whenever the reply re-renders to the bytes it
          was sampled as — the same assumption sec 8.5.5's replay map already
          makes, and `_warm_start` refuses when it does not hold.

        Every refusal below costs one cold prefill, which is the failure this path
        exists to make rarer, never a wrong answer.
        """
        handle = getattr(result, "kv_handle", None) if result is not None else None
        if result is not None:
            result.kv_handle = None
        if not isinstance(handle, _TurnCache):
            return None

        from mlx_lm.models import cache as kv

        held = [*handle.tokens, *handle.generated]
        keyed = _common_prefix(self._encode(rendered_prefix), held)
        if keyed == 0:
            return None
        length = mlx_kv.cache_length(handle.cache)
        if length < keyed:
            return None

        covered = length
        if length > keyed and kv.can_trim_prompt_cache(handle.cache):
            if kv.trim_prompt_cache(handle.cache, length - keyed) != length - keyed:
                return None
            covered = keyed
        elif length != len(held):
            # Nothing was rewound, so the state has to name every token in the
            # cache — and here it cannot. A `stream_generate` that fed a token it
            # never yielded (or yielded one twice) leaves the tail unaccountable,
            # and a state claiming a length it does not hold restores the model
            # one token out of step with its own prompt.
            return None

        blob = mlx_kv.dumps(handle.cache, self.mx, max_bytes=self.max_state_bytes)
        if blob is None:
            return None
        return KVState(
            key=self.state_key(req.adapter),
            prefix_bytes=len(rendered_prefix.encode("utf-8")),
            token_ids=tuple(held[:covered]),
            blob=blob,
        )

    def _warm_start(
        self, req: GenRequest, tokens: list[int], kv: Any
    ) -> tuple[Any, int]:
        """The cache to decode against, and how many prompt tokens it already holds.

        A restored state is used only when its ids are a genuine prefix of *this*
        prompt's ids. The identity key is checked first and answers "same model,
        same adapter"; it says nothing about the bytes, and a state whose tokens
        have diverged would be prefilled as if it were this prompt.

        One token is always left to feed: `generate_step` rejects an empty prompt,
        so a repeat of a turn already cached in full restores all but its last
        token and prefills that.
        """
        state = req.warm_state
        if state is not None and state.blob and self.accepts_state(state, req.adapter):
            ids = list(state.token_ids)
            keep = min(len(ids), len(tokens) - 1)
            if keep > 0 and ids[:keep] == tokens[:keep]:
                cache = mlx_kv.loads(state.blob, self.mx, kv)
                if cache is not None:
                    length = mlx_kv.cache_length(cache)
                    if length == keep:
                        return cache, keep
                    if length > keep and kv.can_trim_prompt_cache(cache):
                        kv.trim_prompt_cache(cache, length - keep)
                        return cache, keep
        return kv.make_prompt_cache(self.model), 0

    def _wire_base(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover - best effort
            self.mx.eval(self.model.parameters())

    # --- attestation --------------------------------------------------------

    def container_hash(self) -> str | None:
        return self._container_hash

    def adapter_hash(self, adapter: str | None) -> str | None:
        spec = self._adapters.get(adapter) if adapter else None
        return spec.adapter_hash if spec else None

    def profile_hash(self, adapter: str | None) -> str | None:
        spec = self._adapters.get(adapter) if adapter else None
        return spec.profile_hash if spec else None

    # --- generation ---------------------------------------------------------

    def render(self, req: GenRequest, render_tool_call: ToolCallRenderer = None) -> str:
        """Render through the model's own chat template.

        Must be the same bytes the model prefills, because the disk KV cache keys
        on this (sec 8.4).

        **Tool calls go into the message content, not into a structured
        `tool_calls` field**, and that is the whole subtlety here. Handed a parsed
        call, a chat template serialises it *its* way — key order, separators,
        whitespace — and the template's way is not the bytes the model sampled. The
        prefix then diverges at the first tool call in the conversation and every
        turn after it rebuilds from scratch (sec 8.5.5). `render_tool_call` returns
        the model's own block, which a template passes through verbatim as assistant
        content, so the prefix reproduces exactly. With no renderer supplied the
        block is canonical instead: one cache miss, never a wrong prompt.

        Dropping them was the earlier bug and it was silent twice over — the model
        could not see what it had already called, and two conversations differing
        only in their tool history hashed to one cache key.
        """
        msgs: list[dict[str, Any]] = []
        if req.system:
            msgs.append({"role": "system", "content": req.system})
        for m in req.messages:
            blocks = [
                render_tool_call(c)
                if render_tool_call is not None
                else f"{c.name}\n{c.arguments_json()}"
                for c in m.tool_calls
            ]
            if m.content or blocks or not m.tool_results:
                body = "\n".join(part for part in (m.content, *blocks) if part)
                msgs.append({"role": m.role.value, "content": body})
            for res in m.tool_results:
                # A failed call must not render identically to a successful one that
                # happened to return the same text; the model's next move differs.
                prefix = "error: " if res.is_error else ""
                msgs.append(
                    {
                        "role": "tool",
                        "content": f"{prefix}{res.content}",
                        "tool_call_id": res.tool_call_id,
                    }
                )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in req.tools
        ] or None
        rendered = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, tools=tools
        )
        return cast(str, rendered)

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def _encode(self, text: str) -> list[int]:
        """Token ids for `text`, exactly as `stream_generate` would produce them.

        The special-token decision is copied from `mlx_lm.stream_generate`'s string
        branch rather than left to it, because tier 0 now hands it *ids*: a KV state
        covers a count of tokens, and one extra BOS on either side would shift every
        id in the prompt against the state stored for it (sec 8.4).
        """
        bos = getattr(self.tokenizer, "bos_token", None)
        add_special = bos is None or not text.startswith(bos)
        return list(self.tokenizer.encode(text, add_special_tokens=add_special))

    async def generate(self, req: GenRequest) -> GenResult:
        chunks: list[str] = []
        result: GenResult | None = None
        async for delta in self.stream(req):
            if delta.text:
                chunks.append(delta.text)
            if delta.done:
                result = delta.result
        assert result is not None
        result.text = "".join(chunks)
        return result

    def _logits_processors(self, req: GenRequest) -> list[Any]:
        """Constrained decoding for this request, or nothing (sec 8.5.1).

        **This is the half that used to be missing, and its absence was silent.**
        `Pipeline` attaches `json_schema` to every tool-bearing turn and this backend
        never read it, so on real hardware the schema was computed, carried the whole
        way down, and dropped — leaving tool-call correctness entirely to repair and
        retry. `MockBackend` honoured the same field, which put the mock *stricter*
        than the hardware and inverted the rule in CLAUDE.md: a green suite could not
        see the gap. Measured cost of the gap on this host, before the fix: 0 clean
        first-attempt tool calls in 100 turns.

        The import is deliberately local. `gateway.toolcall` sits above the backend
        line and `gateway/__init__` reaches `Pipeline`, which imports this package —
        a module-level import here closes that cycle at interpreter start.
        """
        if req.json_schema is None or self._constrain_vocab_failed:
            return []
        from tandem.gateway.toolcall.constrain import Constrainer, logit_width_bound

        constrainer = Constrainer()
        if self._constrain_vocab is None:
            self._constrain_vocab = constrainer.vocabulary(self.tokenizer)
            if self._constrain_vocab is None:
                # The enforcer is not installed. Remembered, because the answer
                # cannot change while this process lives and probing it on every
                # turn would import-and-fail once per request.
                self._constrain_vocab_failed = True
                return []
        token_filter = constrainer.token_filter(req.json_schema, self._constrain_vocab)
        if token_filter is None:
            return []
        return [
            build_logits_processor(
                token_filter, self.mx, logit_width_bound(self.tokenizer)
            )
        ]

    async def stream(self, req: GenRequest) -> AsyncIterator[Delta]:
        from mlx_lm import stream_generate
        from mlx_lm.models import cache as kv
        from mlx_lm.sample_utils import make_sampler

        token = ACTIVE_ADAPTER.set(req.adapter)
        try:
            prompt = self.render(req)
            tokens = self._encode(prompt)
            prompt_cache, cached = self._warm_start(req, tokens, kv)
            sampler = make_sampler(
                temp=req.sampling.temperature,
                top_p=req.sampling.top_p,
            )
            self.mx.random.seed(req.sampling.seed)

            out_tokens = 0
            text_parts: list[str] = []
            generated: list[int] = []
            stop_reason = StopReason.END_TURN
            for step in stream_generate(
                self.model,
                self.tokenizer,
                prompt=tokens[cached:],
                max_tokens=req.sampling.max_tokens,
                sampler=sampler,
                logits_processors=self._logits_processors(req),
                prompt_cache=prompt_cache,
            ):
                out_tokens += 1
                text_parts.append(step.text)
                generated.append(int(step.token))
                yield Delta(text=step.text)
                joined = "".join(text_parts)
                if any(s and joined.endswith(s) for s in req.sampling.stop):
                    stop_reason = StopReason.STOP_SEQUENCE
                    break
            else:
                if out_tokens >= req.sampling.max_tokens:
                    stop_reason = StopReason.MAX_TOKENS

            yield Delta(
                done=True,
                result=GenResult(
                    text="".join(text_parts),
                    stop_reason=stop_reason,
                    usage=Usage(
                        input_tokens=len(tokens),
                        output_tokens=out_tokens,
                        cached_input_tokens=cached,
                    ),
                    kv_handle=_TurnCache(
                        cache=prompt_cache, tokens=tokens, generated=generated
                    ),
                ),
            )
        finally:
            ACTIVE_ADAPTER.reset(token)


# --- helpers ----------------------------------------------------------------


@dataclass
class _TurnCache:
    """The KV cache a turn decoded against, and the prompt ids it was fed."""

    cache: list[Any]
    tokens: list[int]
    # Every token id `stream_generate` yielded, which for a cache that cannot be
    # rewound is the only record of what it holds past the prompt.
    generated: list[int] = field(default_factory=list)


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for left, right in zip(a, b):
        if left != right:
            break
        n += 1
    return n


_ALWAYS_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate",
    "router",
    "shared_expert",
)


def _is_target(key: str) -> bool:
    """§4.3 targeting: all-linear at r=32 for attention/router/shared expert.

    Routed experts are included only when the adapter itself carries weights for
    them, which the routing profile decided at training time — so the served model
    wraps them and mount() fills in whichever top-25% the profile picked.
    """
    tail = key.rsplit(".", 1)[-1]
    if tail in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return True
    return bool(
        any(
            t in key
            for t in ("router", "shared_expert", "gate_proj", "up_proj", "down_proj")
        )
    )


def _walk_linears(
    root: Any, nn: Any, prefix: str = ""
) -> Iterator[tuple[str, Any, str, Any]]:
    """Yield (dotted_key, parent_module, attr_name, linear) for every Linear."""
    children = getattr(root, "children", None)
    items = children().items() if callable(children) else []
    for attr, child in items:
        key = f"{prefix}.{attr}" if prefix else attr
        if isinstance(child, (list, tuple)):
            for i, sub in enumerate(child):
                yield from _walk_linears(sub, nn, f"{key}.{i}")
            continue
        if isinstance(child, nn.Linear) or type(child).__name__ in (
            "Linear",
            "QuantizedLinear",
        ):
            yield key, root, attr, child
        else:
            yield from _walk_linears(child, nn, key)


def _read_adapter_config(path: Path) -> dict[str, Any]:
    import json

    cfg_path = path / "adapter_config.json"
    if not cfg_path.exists():
        return {}
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    lora = raw.get("lora_parameters", raw)
    return {"rank": lora.get("rank", lora.get("r", 32)), "alpha": lora.get("alpha", 64)}
