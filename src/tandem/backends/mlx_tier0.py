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

import contextvars
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from ..attest.hashing import hash_artefact
from ..types import GenRequest, GenResult, StopReason, Usage
from .base import Backend, BackendUnavailable, Delta, ToolCallRenderer

# The active adapter for the current request. Default None = base model.
ACTIVE_ADAPTER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tandem_active_adapter", default=None
)


def _require_mlx() -> tuple[Any, Any]:
    try:
        import mlx.core as mx
        import mlx.nn as nn
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

    class MultiAdapterLinear(nn.Module):
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
    ):
        self.mx, self.nn = _require_mlx()
        try:
            from mlx_lm import load  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("mlx-lm is required for tier 0") from exc

        self.model_path = model_path
        self.mtp = mtp
        self.max_kv_tokens = max_kv_tokens
        self.model, self.tokenizer = load(model_path)
        self._container_hash = hash_artefact(model_path)
        self._adapters: dict[str, AdapterSpec] = {}
        self._targets: dict[str, Any] = {}

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
        for spec in self._adapters.values():
            spec.weights = {}
        try:
            self.mx.clear_cache()
        except AttributeError:  # pragma: no cover - older mlx
            pass

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
        from mlx_lm import load  # type: ignore

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
        """Not yet. Deliberately stated rather than inherited.

        The disk KV cache is wired end to end and tested against the mock; what is
        missing is serialising an `mlx_lm` prompt cache to bytes and back. Until
        that lands, this backend prefills cold after a restart — slow, but correct.

        Whoever implements it: `export_state` must return a state whose `blob`
        deserialises to a cache covering *exactly* `rendered_prefix`, and
        `state_key` must keep including the container and adapter. A state restored
        under a different adapter produces fluent output from the wrong model and a
        receipt attesting to the adapter that did not produce it.
        """
        return False

    def _wire_base(self) -> None:
        try:
            self.mx.eval(self.model.parameters())
        except Exception:  # pragma: no cover - best effort
            pass

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
                render_tool_call(c) if render_tool_call is not None
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
        tools = (
            [{"type": "function", "function": {"name": t.name, "description": t.description,
                                               "parameters": t.parameters}} for t in req.tools]
            or None
        )
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, tools=tools
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

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

    async def stream(self, req: GenRequest) -> AsyncIterator[Delta]:
        from mlx_lm import stream_generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore

        token = ACTIVE_ADAPTER.set(req.adapter)
        try:
            prompt = self.render(req)
            in_tokens = self.count_tokens(prompt)
            sampler = make_sampler(
                temp=req.sampling.temperature,
                top_p=req.sampling.top_p,
            )
            self.mx.random.seed(req.sampling.seed)

            out_tokens = 0
            text_parts: list[str] = []
            stop_reason = StopReason.END_TURN
            for step in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=req.sampling.max_tokens,
                sampler=sampler,
            ):
                out_tokens += 1
                text_parts.append(step.text)
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
                    usage=Usage(input_tokens=in_tokens, output_tokens=out_tokens),
                ),
            )
        finally:
            ACTIVE_ADAPTER.reset(token)


# --- helpers ----------------------------------------------------------------

_ALWAYS_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "router",
                   "shared_expert")


def _is_target(key: str) -> bool:
    """§4.3 targeting: all-linear at r=32 for attention/router/shared expert.

    Routed experts are included only when the adapter itself carries weights for
    them, which the routing profile decided at training time — so the served model
    wraps them and mount() fills in whichever top-25% the profile picked.
    """
    tail = key.rsplit(".", 1)[-1]
    if tail in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return True
    if any(t in key for t in ("router", "shared_expert", "gate_proj", "up_proj", "down_proj")):
        return True
    return False


def _walk_linears(root: Any, nn: Any, prefix: str = ""):
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
