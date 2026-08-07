"""Backend interface.

Everything above this line (gateway, router, tool-call layer, attestation) is pure
Python and runs anywhere. Everything below it is Apple-Silicon-specific. The split
is deliberate: the MLX backends cannot be exercised on a CI box, so the mock has to
be good enough that the rest of the system is fully testable without one.

A backend owes callers three things:

* ``render`` — the exact byte prefix a prompt turns into. The disk KV cache keys on
  the SHA-256 of this (sec 8.4), so it must be stable and it must be the same bytes
  the model actually sees. A backend that renders one way and prefills another
  silently corrupts every cache hit.
* ``generate`` / ``stream`` — the model.
* ``container_hash`` — what the receipt attests to (sec 9.1).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from ..types import GenRequest, GenResult, Message, Role, ToolCall


@dataclass(frozen=True, slots=True)
class Delta:
    """One streaming increment."""

    text: str = ""
    # Set on the final delta.
    done: bool = False
    # The completed result, on the final delta. `Pipeline.stream` also uses this on
    # a leading `done=False` delta to hand a wire encoder the prompt-side usage
    # before any token exists; backends never do that.
    result: GenResult | None = None


# The replay-aware tool-call renderer (sec 8.5.5): a call in, the exact bytes it
# should occupy in the prompt out. None means "render canonically" — correct, and a
# cache miss on that turn.
ToolCallRenderer = Callable[[ToolCall], str] | None


class BackendUnavailable(RuntimeError):
    """Raised when a backend's runtime requirements are not met on this machine.

    Carries the reason verbatim so `tandem doctor` can tell a user "no Metal device"
    rather than "import failed".
    """


class Backend(ABC):
    """A text-generation engine."""

    name: str = "backend"
    # Tier this backend serves: 0 (resident generator) or 1 (streamed verifier).
    tier: int = 0

    @abstractmethod
    async def generate(self, req: GenRequest) -> GenResult: ...

    async def stream(self, req: GenRequest) -> AsyncIterator[Delta]:
        """Default: generate, then emit as a single delta.

        Correct but not incremental. Backends that can stream override this; the
        gateway's SSE paths are written against the iterator either way so a
        non-streaming backend still serves a streaming client.
        """
        result = await self.generate(req)
        if result.text:
            yield Delta(text=result.text)
        yield Delta(done=True, result=result)

    def render(self, req: GenRequest, render_tool_call: ToolCallRenderer = None) -> str:
        """Canonical prompt rendering. Override to match a real chat template.

        `render_tool_call` is the replay-aware renderer (sec 8.5.5), supplied by the
        gateway. It is passed to *every* backend, not only to the ones using the
        default renderer: a backend with a real chat template still has to put the
        model's own sampled bytes back into the prompt, and it cannot do that with a
        renderer it was never handed. Ignoring the argument is allowed and costs a
        cache miss per tool call; there is no way to ignore it and be wrong.
        """
        return render_default(req, render_tool_call)

    def renders_canonically(self) -> bool:
        """True when `render` is the default renderer and nothing else.

        The gateway asks, because a backend using the default has to be rendered
        *through the replay map* (sec 8.5.5) and a backend with its own chat
        template renders itself. Asking the object rather than inspecting its type
        is what lets a wrapper answer for what it wraps: a delegating `render`
        overrides the method while changing none of the bytes, and a type check
        reads that as a real chat template and quietly drops replay-aware
        rendering — a cache-key bug with no symptom but a lower hit rate.
        """
        return type(self).render is Backend.render

    def count_tokens(self, text: str) -> int:
        """Token count. The default is a byte-based estimate.

        Deliberately an estimate, not a lie dressed as precision: context scaling
        (sec 8.3) and cache budgeting both tolerate a few percent, and a backend
        with a real tokenizer overrides this.
        """
        return max(1, len(text) // 4)

    def container_hash(self) -> str | None:
        return None

    def adapter_hash(self, adapter: str | None) -> str | None:
        return None

    def profile_hash(self, adapter: str | None) -> str | None:
        return None

    def mounted_adapters(self) -> tuple[str, ...]:
        return ()

    # --- KV state (sec 8.4) -------------------------------------------------
    #
    # A backend that can snapshot its KV cache lets the disk cache survive a
    # restart, so the first turn after a reload is not a cold prefill. A backend
    # that cannot says so, and the gateway degrades to prefilling — which is slow,
    # not wrong.

    def supports_state(self) -> bool:
        return False

    def state_key(self, adapter: str | None) -> str:
        """Identity a KV state must match to be reusable on this backend.

        Container and adapter are both in it because a state restored under either
        one changed is silently wrong rather than merely stale.
        """
        return f"{self.name}:{self.container_hash() or '-'}:{adapter or '-'}"

    def accepts_state(self, state: Any, adapter: str | None) -> bool:
        return bool(state) and state.key == self.state_key(adapter)

    def export_state(self, req: GenRequest, rendered_prefix: str, result: GenResult) -> Any:
        """Snapshot the KV cache covering `rendered_prefix`, or None.

        Returning None is the honest default: most backends cannot do this, and a
        fabricated snapshot would restore a state that does not match its prompt.
        """
        return None

    async def close(self) -> None:
        return None


# --- default rendering ------------------------------------------------------
#
# A plain, explicit, stable format. Not any particular model's chat template — real
# backends override `render` with the tokenizer's own — but the same function every
# time, which is what the cache key needs.

_ROLE_TAG = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
}


def render_message(msg: Message, render_tool_call: ToolCallRenderer = None) -> str:
    """Render one message.

    `render_tool_call` is the replay-aware renderer (sec 8.5.5). Passing it is what
    makes turn N's prompt reproduce turn N-1's bytes exactly; omitting it renders
    canonically and costs a cache miss.
    """
    parts = [f"<|{_ROLE_TAG[msg.role]}|>\n"]
    if msg.content:
        parts.append(msg.content)
        parts.append("\n")
    for call in msg.tool_calls:
        body = (
            render_tool_call(call)
            if render_tool_call is not None
            else f"{call.name}\n{call.arguments_json()}"
        )
        parts.append(f"<|tool_call|>{body}\n<|/tool_call|>\n")
    for res in msg.tool_results:
        tag = "tool_error" if res.is_error else "tool_result"
        parts.append(f"<|{tag}|>{res.tool_call_id}\n{res.content}\n<|/{tag}|>\n")
    parts.append(f"<|/{_ROLE_TAG[msg.role]}|>\n")
    return "".join(parts)


def render_tools(req: GenRequest) -> str:
    if not req.tools:
        return ""
    spec = [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in req.tools
    ]
    return "<|tools|>\n" + json.dumps(spec, sort_keys=True, separators=(",", ":")) + "\n<|/tools|>\n"


def render_default(req: GenRequest, render_tool_call: ToolCallRenderer = None) -> str:
    parts: list[str] = []
    if req.system:
        parts.append(render_message(Message(role=Role.SYSTEM, content=req.system)))
    parts.append(render_tools(req))
    for msg in req.messages:
        parts.append(render_message(msg, render_tool_call))
    parts.append("<|assistant|>\n")
    return "".join(parts)
