"""Deterministic mock backend.

Not a toy. This is what lets the gateway, router, tool-call layer, cascade and
attestation be tested end to end on a machine with no Metal device — which is every
CI box and every developer who is not sitting at the target laptop.

Three properties it must have, and does:

* **Deterministic.** Output is a pure function of (rendered prompt, seed, adapter,
  temperature). The determinism gates (sec 9.3) and the adapter isolation test
  (sec 4.2) are written against real backends but must be *runnable* here, and they
  are only meaningful if the mock does not cheat.
* **Adapter-sensitive.** Different mounted adapters produce different output. An
  isolation test that passes because the mock ignores adapters proves nothing.
* **Faultable.** It can emit the exact malformed tool-call shapes the repair layer
  (sec 8.5) exists to fix, so that layer is tested against its real inputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from ..types import GenRequest, GenResult, KVState, StopReason, ToolCall, Usage
from .base import Backend, Delta

_WORDS = (
    "patch applies cleanly the helper already exists in utils so reuse it rather "
    "than adding a second one guard the empty case and keep the error type the "
    "module already raises tests cover the boundary"
).split()


class Fault:
    """Named malformed-output modes, for exercising the repair layer."""

    NONE = "none"
    XML_HYBRID = "xml_hybrid"  # XML wrapper around JSON args
    FENCED = "fenced"  # ```json fence around the call
    BARE_OBJECT = "bare_object"  # object with no call envelope
    TRAILING_COMMA = "trailing_comma"
    SMART_QUOTES = "smart_quotes"
    FUNCTION_SYNTAX = "function_syntax"  # name(arg=value)
    MANGLED_NAME = "mangled_name"  # name unrecognisable; must be inferred from keys
    UNKNOWN_TOOL = "unknown_tool"  # invented tool; must be rejected
    TRUNCATED_JSON = "truncated_json"


@dataclass
class MockBackend(Backend):
    """A seeded pseudo-model.

    `script` short-circuits everything for tests that need an exact string; it is
    consumed in order and falls back to generated text when exhausted.
    """

    name: str = "mock"
    tier: int = 0
    container: str = "mock-container-v1"
    adapters: tuple[str, ...] = ()
    # Emit a tool call when the request carries tools. Off for pure-text tests.
    use_tools: bool = True
    fault: str = Fault.NONE
    script: list[str] = field(default_factory=list)
    # Artificial per-token delay, for exercising the latency pressure valve (sec 7.3).
    token_delay_s: float = 0.0
    # Hook for tests that need full control: (req) -> GenResult | None.
    responder: Callable[[GenRequest], GenResult | None] | None = None

    calls: list[GenRequest] = field(default_factory=list, repr=False)

    # --- Backend ------------------------------------------------------------

    def mounted_adapters(self) -> tuple[str, ...]:
        return self.adapters

    def container_hash(self) -> str | None:
        return _digest(self.container)

    def adapter_hash(self, adapter: str | None) -> str | None:
        return _digest(f"adapter:{adapter}") if adapter else None

    def profile_hash(self, adapter: str | None) -> str | None:
        return _digest(f"profile:{adapter}") if adapter else None

    async def generate(self, req: GenRequest) -> GenResult:
        self.calls.append(req)
        if self.responder is not None:
            forced = self.responder(req)
            if forced is not None:
                return forced

        rng = self._rng(req)
        prompt = self.render(req)
        in_tokens = self.count_tokens(prompt)
        cached_tokens = self._warm_tokens(req, prompt)

        if req.json_schema is not None:
            instance = sample_schema(req.json_schema, rng)
            text = json.dumps(instance, separators=(",", ":"))
            # A constrained *tier-0* turn yields a parsed tool call, because that is
            # what a real backend under a tool-call grammar produces — the schema is
            # the tool-call envelope, not free-form JSON. A constrained tier-1 call
            # yields the verdict as text. Collapsing the two would let the tool-call
            # path go untested precisely when prevention is enabled.
            if req.tools and self.use_tools and _is_tool_call_instance(instance, req):
                call = ToolCall(
                    id="call_"
                    + hashlib.sha256(f"{text}{req.sampling.seed}".encode()).hexdigest()[:16],
                    name=instance["name"],
                    arguments=instance.get("arguments") or {},
                )
                return GenResult(
                    text="",
                    tool_calls=(call,),
                    stop_reason=StopReason.TOOL_USE,
                    usage=Usage(cached_input_tokens=cached_tokens, input_tokens=in_tokens, output_tokens=self.count_tokens(text)),
                    raw_blocks={call.id: text},
                )
            return GenResult(
                text=text,
                stop_reason=StopReason.END_TURN,
                usage=Usage(cached_input_tokens=cached_tokens, input_tokens=in_tokens, output_tokens=self.count_tokens(text)),
            )

        if self.script:
            text = self.script.pop(0)
            return GenResult(
                text=text,
                stop_reason=StopReason.END_TURN,
                usage=Usage(cached_input_tokens=cached_tokens, input_tokens=in_tokens, output_tokens=self.count_tokens(text)),
            )

        if req.tools and self.use_tools:
            return self._tool_result(req, rng, in_tokens, cached_tokens)

        text = self._prose(rng, req.sampling.max_tokens)
        if self.token_delay_s:
            await asyncio.sleep(self.token_delay_s * self.count_tokens(text))
        return GenResult(
            text=text,
            stop_reason=StopReason.END_TURN,
            usage=Usage(cached_input_tokens=cached_tokens, input_tokens=in_tokens, output_tokens=self.count_tokens(text)),
        )

    async def stream(self, req: GenRequest) -> AsyncIterator[Delta]:
        result = await self.generate(req)
        # Chunk on word boundaries so SSE consumers see a realistic delta cadence.
        buf = result.text
        step = max(1, len(buf) // 8) if buf else 1
        for i in range(0, len(buf), step):
            if self.token_delay_s:
                await asyncio.sleep(self.token_delay_s)
            yield Delta(text=buf[i : i + step])
        yield Delta(done=True, result=result)

    # --- KV state (sec 8.4) -------------------------------------------------
    #
    # A faithful stand-in, not a stub. The mock cannot hold real attention KV, so
    # the "state" is the prefix bytes it covers — but it behaves like the real
    # thing in the ways the gateway depends on: it round-trips through the disk
    # format, it is refused when the container or adapter changed, and restoring
    # it makes the prefix it covers cost nothing to prefill. That is enough to
    # test the whole loop, and it will not silently pass if the wiring breaks.

    def supports_state(self) -> bool:
        return True

    def export_state(self, req: GenRequest, rendered_prefix: str, result: GenResult) -> KVState:
        data = rendered_prefix.encode("utf-8")
        return KVState(
            key=self.state_key(req.adapter),
            prefix_bytes=len(data),
            token_ids=tuple(range(self.count_tokens(rendered_prefix))),
            next_logits=hashlib.sha256(data).digest(),
            blob=data,
        )

    def _warm_tokens(self, req: GenRequest, prompt: str) -> int:
        """Tokens a restored state covers, and so does not need prefilling.

        Verifies the state actually is a prefix of *this* prompt. A state whose
        bytes have diverged is a cache bug, and reporting its tokens as cached
        would report a saving that did not happen.
        """
        state = req.warm_state
        if state is None or not self.accepts_state(state, req.adapter):
            return 0
        if not prompt.encode("utf-8").startswith(state.blob):
            return 0
        return state.n_tokens

    # --- internals ----------------------------------------------------------

    def _rng(self, req: GenRequest) -> random.Random:
        """Seed from everything that legitimately changes the output.

        Temperature is in the seed so best-of-N at t=0.6 yields *different*
        candidates while greedy stays reproducible; the candidate index rides in on
        the request's seed, which the router varies per candidate.
        """
        material = "\x00".join(
            [
                self.render(req),
                str(req.sampling.seed),
                str(req.sampling.temperature),
                req.adapter or "",
                self.container,
            ]
        )
        return random.Random(hashlib.sha256(material.encode()).digest())

    def _prose(self, rng: random.Random, max_tokens: int) -> str:
        n = min(max(8, rng.randint(12, 40)), max(1, max_tokens))
        return " ".join(rng.choice(_WORDS) for _ in range(n))

    def _tool_result(
        self, req: GenRequest, rng: random.Random, in_tokens: int, cached_tokens: int = 0
    ) -> GenResult:
        tool = req.tools[rng.randrange(len(req.tools))]
        args = {name: _sample_param(tool, name, rng) for name in tool.required_names()} or {
            name: _sample_param(tool, name, rng) for name in tool.param_names()[:1]
        }
        call_id = "call_" + hashlib.sha256(
            f"{tool.name}{json.dumps(args, sort_keys=True)}{req.sampling.seed}".encode()
        ).hexdigest()[:16]

        raw = _render_fault(self.fault, tool.name, args)
        if self.fault in (Fault.NONE,):
            call = ToolCall(id=call_id, name=tool.name, arguments=args)
            return GenResult(
                text="",
                tool_calls=(call,),
                stop_reason=StopReason.TOOL_USE,
                usage=Usage(cached_input_tokens=cached_tokens, input_tokens=in_tokens, output_tokens=self.count_tokens(raw)),
                raw_blocks={call_id: raw},
            )
        # Faulted: emit as raw text with no parsed call. The repair layer's job.
        return GenResult(
            text=raw,
            stop_reason=StopReason.END_TURN,
            usage=Usage(cached_input_tokens=cached_tokens, input_tokens=in_tokens, output_tokens=self.count_tokens(raw)),
        )


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _is_tool_call_instance(instance: Any, req: GenRequest) -> bool:
    """Is this schema instance a call to one of the request's own tools?"""
    return (
        isinstance(instance, dict)
        and isinstance(instance.get("name"), str)
        and instance["name"] in {t.name for t in req.tools}
    )


def _sample_param(tool: Any, name: str, rng: random.Random) -> Any:
    schema = tool.parameters.get("properties", {}).get(name, {})
    return _sample_typed(schema, rng, name)


def _sample_typed(schema: dict[str, Any], rng: random.Random, hint: str = "") -> Any:
    # `const` and `anyOf` are how the tool-call schema pins a name and offers one
    # branch per tool (see constrain.tool_call_schema). Without them the mock
    # invents a name where a constrained model could not, which makes the mock
    # *easier* to satisfy than the real thing — the one property a stand-in must
    # never have.
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][rng.randrange(len(schema["enum"]))]
    branches = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(branches, list) and branches:
        return _sample_typed(branches[rng.randrange(len(branches))], rng, hint)
    # Bounds are honoured for the same reason `const` is: a constrained decoder
    # physically cannot emit a value outside them, so a mock that can is easier to
    # satisfy than the real thing and hides bugs instead of catching them. This is
    # load-bearing for `rerank_schema`, whose `maximum` is what makes an index that
    # names no candidate unrepresentable rather than merely detected.
    t = schema.get("type", "string")
    if t == "integer":
        low = int(schema.get("minimum", 0))
        high = int(schema.get("maximum", low + 3))
        return rng.randint(low, max(low, high))
    if t == "number":
        low = float(schema.get("minimum", 0.0))
        high = float(schema.get("maximum", low + 1.0))
        return round(rng.uniform(low, max(low, high)), 3)
    if t == "boolean":
        return bool(rng.getrandbits(1))
    if t == "array":
        item = schema.get("items", {"type": "string"})
        low = int(schema.get("minItems", 1))
        high = int(schema.get("maxItems", max(low, 2)))
        return [_sample_typed(item, rng, hint) for _ in range(rng.randint(low, max(low, high)))]
    if t == "object":
        props = schema.get("properties", {})
        return {k: _sample_typed(v, rng, k) for k, v in props.items()}
    value = f"{hint or 'value'}-{rng.randrange(1000):03d}"
    limit = schema.get("maxLength")
    return value[: int(limit)] if isinstance(limit, int) else value


def sample_schema(schema: dict[str, Any], rng: random.Random) -> Any:
    """Produce a valid instance of a JSON Schema.

    Used to stand in for constrained decoding: the mock's tier-1 output is
    schema-valid by construction, exactly as the real constrained path guarantees
    (sec 5.1), so the verifier plumbing can be tested without a 122B model.
    """
    return _sample_typed(schema, rng)


def _render_fault(fault: str, name: str, args: dict[str, Any]) -> str:
    body = json.dumps(args, sort_keys=True)
    if fault == Fault.XML_HYBRID:
        inner = "".join(f"<{k}>{v}</{k}>" for k, v in sorted(args.items()))
        return f"<tool_call>\n<tool_name>{name}</tool_name>\n<arguments>{inner}</arguments>\n</tool_call>"
    if fault == Fault.FENCED:
        return f'```json\n{{"name": "{name}", "arguments": {body}}}\n```'
    if fault == Fault.BARE_OBJECT:
        return json.dumps({"name": name, **args}, sort_keys=True)
    if fault == Fault.TRAILING_COMMA:
        return f'{{"name": "{name}", "arguments": {body[:-1]}{"," if len(body) > 2 else ""}}},'
    if fault == Fault.SMART_QUOTES:
        return (
            f'{{“name”: “{name}”, “arguments”: '
            + body.replace('"', "“", 1).replace('"', "”", 1)
            + "}"
        )
    if fault == Fault.FUNCTION_SYNTAX:
        inner = ", ".join(f"{k}={json.dumps(v)}" for k, v in sorted(args.items()))
        return f"{name}({inner})"
    if fault == Fault.MANGLED_NAME:
        return json.dumps({"name": "tool‑call‑???", "arguments": args}, sort_keys=True)
    if fault == Fault.UNKNOWN_TOOL:
        return json.dumps({"name": "definitely_not_a_real_tool", "arguments": args})
    if fault == Fault.TRUNCATED_JSON:
        return f'{{"name": "{name}", "arguments": {body}'
    return json.dumps({"name": name, "arguments": args}, sort_keys=True)
