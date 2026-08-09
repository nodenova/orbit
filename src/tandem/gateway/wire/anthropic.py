"""`/v1/messages` — the Anthropic Messages wire shape (spec sec 8.1).

Serves Claude Code and OpenClaw.

Harness plurality is a product requirement, not a nice-to-have: Anthropic tightened
the Claude Code harness boundary twice in 2026, once with under 24 hours' notice, and
a product whose entire value depends on one closed client is one product decision
from zero. So this file is one of three, all normalising into the same canonical
types and sharing one model, one cache and one router.

Nothing here touches Anthropic credentials — pointing a harness at a local endpoint
is a supported feature and involves none.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from tandem.gateway.wire import check_bounds, size_of
from tandem.types import (
    GenRequest,
    GenResult,
    Message,
    Role,
    Sampling,
    StopReason,
    ToolCall,
    ToolDef,
    ToolResult,
)

_STOP_REASON = {
    StopReason.END_TURN: "end_turn",
    StopReason.MAX_TOKENS: "max_tokens",
    StopReason.STOP_SEQUENCE: "stop_sequence",
    StopReason.TOOL_USE: "tool_use",
}


def _text_of(content: Any) -> str:
    """Flatten a content field that may be a string or a block list."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out: list[str] = []
    for block in content:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "\n".join(p for p in out if p)


def to_canonical(body: dict[str, Any]) -> GenRequest:
    system = body.get("system")
    if isinstance(system, list):
        system = _text_of(system)

    raw_messages = body.get("messages") or []
    # Checked before the loop rather than after it: the whole point of a message-count
    # bound is not to do per-message work a million times first.
    check_bounds(items=len(raw_messages) if isinstance(raw_messages, list) else 0)

    messages: list[Message] = []
    for raw in raw_messages:
        role = (
            Role(raw.get("role", "user"))
            if raw.get("role") in ("user", "assistant")
            else Role.USER
        )
        content = raw.get("content")

        calls: list[ToolCall] = []
        results: list[ToolResult] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    calls.append(
                        ToolCall(
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("input") or {},
                        )
                    )
                elif btype == "tool_result":
                    results.append(
                        ToolResult(
                            tool_call_id=block.get("tool_use_id", ""),
                            content=_text_of(block.get("content")),
                            is_error=bool(block.get("is_error", False)),
                        )
                    )
        # A tool_result block arrives with role "user"; keeping it as a TOOL message
        # lets the renderer tag it correctly without the rest of the runtime having
        # to know the Anthropic block layout.
        if results and not _text_of(content):
            role = Role.TOOL

        messages.append(
            Message(
                role=role,
                content=_text_of(content),
                tool_calls=tuple(calls),
                tool_results=tuple(results),
            )
        )

    tools = tuple(
        ToolDef(
            name=t.get("name", ""),
            description=t.get("description", ""),
            parameters=t.get("input_schema") or {},
        )
        for t in body.get("tools", [])
        if isinstance(t, dict) and t.get("name")
    )

    stop = body.get("stop_sequences") or []
    max_tokens = int(body.get("max_tokens", 4096))
    check_bounds(chars=size_of(system, messages), max_tokens=max_tokens)

    return GenRequest(
        messages=messages,
        system=system if isinstance(system, str) else None,
        tools=tools,
        sampling=Sampling(
            temperature=float(body.get("temperature", 0.7)),
            top_p=float(body.get("top_p", 1.0)),
            seed=int(body.get("seed", 0)),
            max_tokens=max_tokens,
            stop=tuple(stop) if isinstance(stop, list) else (),
        ),
        stream=bool(body.get("stream", False)),
        # A harness that supports adapter selection can pass it through metadata;
        # nothing standard carries it, so this is the extension point.
        adapter=(body.get("metadata") or {}).get("adapter"),
        protocol="messages",
    )


def from_canonical(
    result: GenResult, *, model: str, request_id: str = ""
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if result.text:
        content.append({"type": "text", "text": result.text})
    for call in result.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
        )
    body: dict[str, Any] = {
        "id": request_id or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _STOP_REASON[result.stop_reason],
        "stop_sequence": None,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cache_read_input_tokens": result.usage.cached_input_tokens,
        },
    }
    if result.receipt is not None:
        # Non-standard field. Harnesses ignore unknown keys, and the receipt is the
        # product (sec 9.1) — a customer must be able to read it off the response
        # without going to the audit log.
        body["tandem_receipt"] = result.receipt
    return body


def stream_options(body: dict[str, Any]) -> dict[str, Any]:
    """`StreamEncoder` keyword arguments this protocol reads off the request body.

    Empty here: the Messages protocol always reports usage, on `message_start` and
    again on `message_delta`, so there is nothing for a client to opt into. The
    function exists anyway so `app.py` can build any protocol's encoder identically
    rather than branching on which one has options.
    """
    return {}


class StreamEncoder:
    """Incremental SSE encoder.

    `open` -> `delta`* -> `close`, so the gateway can emit events as tokens arrive
    on the turns that stream (sec 7.3) and the whole sequence at once on the turns
    that cannot. The text content block opens lazily on the first delta: a turn
    that produces only tool calls must not announce a text block it never fills.
    """

    def __init__(self, *, model: str, request_id: str = ""):
        self.model = model
        self.id = request_id or f"msg_{uuid.uuid4().hex[:24]}"
        self._index = 0
        self._text_open = False

    def _emit(self, event: str, data: dict[str, Any]) -> list[str]:
        return [f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"]

    def open(self, input_tokens: int = 0) -> list[str]:
        return self._emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        )

    def delta(self, text: str) -> list[str]:
        if not text:
            return []
        out: list[str] = []
        if not self._text_open:
            out += self._emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            self._text_open = True
        out += self._emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self._index,
                "delta": {"type": "text_delta", "text": text},
            },
        )
        return out

    def close(self, result: GenResult) -> list[str]:
        out: list[str] = []
        if self._text_open:
            out += self._emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._index},
            )
            self._text_open = False
            self._index += 1

        for call in result.tool_calls:
            out += self._emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": {},
                    },
                },
            )
            out += self._emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": call.arguments_json(),
                    },
                },
            )
            out += self._emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._index},
            )
            self._index += 1

        out += self._emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _STOP_REASON[result.stop_reason],
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": result.usage.output_tokens},
            },
        )
        out += self._emit("message_stop", {"type": "message_stop"})
        return out

    def fail(self, message: str, err_type: str = "api_error") -> list[str]:
        """Terminate a stream that has already sent its headers.

        The status line went out with the first event, so an error this late can
        only be an in-band event. Silence would leave the harness waiting for a
        `message_stop` that is never coming.

        An open text block is closed first. The Anthropic SDK's stream accumulator
        tracks blocks by index, and an `error` arriving on top of an unterminated
        block leaves it holding a half-built message — so the harness's own error
        handling runs against corrupt state rather than against a clean partial
        turn. No `message_stop` follows: that event means the turn completed, and
        this one did not.
        """
        out: list[str] = []
        if self._text_open:
            out += self._emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._index},
            )
            self._text_open = False
            self._index += 1
        out += self._emit("error", error(500, message, err_type))
        return out


def count_tokens_response(n: int) -> dict[str, Any]:
    return {"input_tokens": n}


def error(
    status: int, message: str, err_type: str = "invalid_request_error"
) -> dict[str, Any]:
    return {"type": "error", "error": {"type": err_type, "message": message}}


def _now() -> int:
    return int(time.time())
