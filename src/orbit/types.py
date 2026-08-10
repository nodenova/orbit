"""Canonical internal representation.

All three wire protocols (spec sec 8.1) normalise into these types on the way in and
denormalise out of them on the way back. Nothing downstream of the gateway edge —
router, backends, tool-call layer — knows which *protocol* spoke: `GenRequest.protocol`
is set by the wire modules and read by none of them.

Harness identity is a separate thing and deliberately does travel. `GenRequest.harness`
is not set by the wire layer at all — it cannot be, because the protocol a harness
speaks does not identify it — but by `Compactor.apply`, which fingerprints the system
prompt against its templates and stamps the match. Attestation reads it: the audit
record names the harness a turn came from (sec 9.2), because "which harness produced
this prompt" is part of what a receipt has to answer. The invariant that holds is the
protocol one; the harness field is an intentional exception, and it is written by
exactly one module.

Everything here is frozen or explicitly copy-on-write. The gateway pipeline is a
chain of transforms over a GenRequest, and attestation (sec 9) requires that the
request handed to a backend be exactly reconstructible from the audit record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


class TurnClass(str, Enum):
    """Router turn classification (spec sec 7.1)."""

    CHAT = "chat"
    READ_ONLY = "read_only"
    CODE_CHANGE = "code_change"
    PLAN = "plan"


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str = ""
    # JSON Schema for the tool's input.
    parameters: dict[str, Any] = field(default_factory=dict)

    def param_names(self) -> tuple[str, ...]:
        props = self.parameters.get("properties")
        return tuple(props) if isinstance(props, dict) else ()

    def required_names(self) -> tuple[str, ...]:
        req = self.parameters.get("required")
        return tuple(req) if isinstance(req, list) else ()


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def arguments_json(self) -> str:
        # sort_keys so the same call always renders to the same bytes; the disk KV
        # cache is keyed on the rendered byte prefix (sec 8.4).
        return json.dumps(self.arguments, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    def is_empty(self) -> bool:
        return not self.content and not self.tool_calls and not self.tool_results


@dataclass(frozen=True, slots=True)
class Sampling:
    """Sampling parameters. Recorded verbatim in the receipt (sec 9.1)."""

    temperature: float = 0.7
    top_p: float = 1.0
    seed: int = 0
    max_tokens: int = 4096
    stop: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
        }


@dataclass(frozen=True, slots=True)
class KVState:
    """A serialisable snapshot of a backend's KV cache covering a prompt prefix.

    Opaque above the backend: the gateway owns its lifetime and its bytes, never
    its meaning.

    `key` is the load-bearing field. A KV state is only valid for the exact
    backend, container and adapter that produced it — restoring one built under a
    different adapter would continue a conversation in a model that never saw its
    own prefix, and the failure is silent: fluent output, wrong model, and a
    receipt attesting to the adapter that did *not* produce it. So every state
    carries the identity it belongs to, and `Backend.accepts_state` checks it
    before the state is ever attached to a request.
    """

    key: str
    prefix_bytes: int
    token_ids: tuple[int, ...] = ()
    next_logits: bytes = b""
    blob: bytes = b""

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class GenRequest:
    """A generation request in canonical form."""

    messages: list[Message] = field(default_factory=list)
    system: str | None = None
    tools: tuple[ToolDef, ...] = ()
    sampling: Sampling = field(default_factory=Sampling)
    stream: bool = False

    # Which adapter to mount for this request. Bound per-request, never global:
    # a global would race under concurrency (sec 4.2).
    adapter: str | None = None

    # When set, decoding is constrained to this JSON Schema (sec 5.1, 8.5).
    json_schema: dict[str, Any] | None = None

    request_id: str = ""
    # Harness fingerprint, filled in by compaction (sec 8.2). None = unrecognised.
    harness: str | None = None
    # Wire protocol the request arrived on: messages | chat_completions | responses.
    protocol: str = ""

    # Set by the gateway when compaction rewrote the system prompt, so the
    # receipt and the --no-compact diff view can both reach the original.
    original_system: str | None = None
    compaction_template: str | None = None

    # A restored KV state covering a prefix of this request's rendered prompt
    # (sec 8.4). Carried on the request rather than set on the backend for the same
    # reason the active adapter is (sec 4.2): backend-global state races under
    # concurrent requests, and the failure mode is a silently wrong answer.
    warm_state: KVState | None = None

    def with_(self, **kw: Any) -> GenRequest:
        """Copy-on-write update; keeps pipeline stages side-effect free."""
        return replace(self, **kw)

    def has_tools(self) -> bool:
        return len(self.tools) > 0


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    # Tokens served from the prompt cache rather than prefilled (sec 8.4).
    cached_input_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
        }


@dataclass
class GenResult:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: StopReason = StopReason.END_TURN
    usage: Usage = field(default_factory=Usage)

    # tool_id -> the exact text the model sampled for that call (sec 8.5.5).
    # Clients hand back normalised JSON; re-rendering it differently breaks the
    # byte prefix and forces a full rebuild of the turn.
    raw_blocks: dict[str, str] = field(default_factory=dict)

    # Attestation metadata (sec 9.1). Serialised into the wire response.
    receipt: dict[str, Any] | None = None

    # The live KV cache this turn built, backend-opaque and read only by the
    # backend's own `export_state` (sec 8.4). It rides on the result, not on the
    # backend, for the reason the active adapter and the restored state do: a
    # backend-global handle races under concurrency, and best-of-N runs N
    # generations at once. `export_state` clears it, so a cache the size of the
    # prompt is not held for the life of the response.
    kv_handle: Any = None

    # Diagnostics that never leave the process except via the audit log.
    ttft_s: float = 0.0
    total_s: float = 0.0
    repaired: bool = False
    retries: int = 0
