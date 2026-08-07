"""`/v1/chat/completions` — the OpenAI Chat Completions wire shape (spec sec 8.1).

Serves OpenCode, Crush and every generic OpenAI-compatible client.

The shape difference that matters: tool-call arguments cross the wire as a JSON
*string*, not an object, and clients round-trip that string back on the next turn.
Re-serialising it differently — a space after a colon, a different key order — is
exactly the byte-prefix break the replay map exists to prevent (sec 8.5.5), so
arguments are emitted through `ToolCall.arguments_json()`, which is stable.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ...types import (
    GenRequest,
    GenResult,
    Message,
    Role,
    Sampling,
    StopReason,
    ToolCall,
    ToolDef,
    ToolResult,
    Usage,
)
from . import check_bounds, size_of

_FINISH_REASON = {
    StopReason.END_TURN: "stop",
    StopReason.MAX_TOKENS: "length",
    StopReason.STOP_SEQUENCE: "stop",
    StopReason.TOOL_USE: "tool_calls",
}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        )
    return ""


def to_canonical(body: dict[str, Any]) -> GenRequest:
    raw_messages = body.get("messages") or []
    # Checked before the loop rather than after it: the whole point of a message-count
    # bound is not to do per-message work a million times first.
    check_bounds(items=len(raw_messages) if isinstance(raw_messages, list) else 0)

    system_parts: list[str] = []
    messages: list[Message] = []

    for raw in raw_messages:
        role_name = raw.get("role", "user")
        content = _content_text(raw.get("content"))

        if role_name in ("system", "developer"):
            if content:
                system_parts.append(content)
            continue

        if role_name == "tool":
            messages.append(
                Message(
                    role=Role.TOOL,
                    tool_results=(
                        ToolResult(
                            tool_call_id=raw.get("tool_call_id", ""),
                            content=content,
                        ),
                    ),
                )
            )
            continue

        calls: list[ToolCall] = []
        for tc in raw.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            args: dict[str, Any] = {}
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str) and raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                    args = parsed if isinstance(parsed, dict) else {"value": parsed}
                except json.JSONDecodeError:
                    # A client that hands back unparseable arguments is a client
                    # bug, but dropping the call entirely loses the conversation's
                    # shape; keep it raw so the prompt still shows what happened.
                    args = {"_raw": raw_args}
            calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))

        role = Role.ASSISTANT if role_name == "assistant" else Role.USER
        messages.append(Message(role=role, content=content, tool_calls=tuple(calls)))

    tools: list[ToolDef] = []
    for t in body.get("tools", []) or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        tools.append(
            ToolDef(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters") or {},
            )
        )

    stop = body.get("stop")
    if isinstance(stop, str):
        stop_seqs: tuple[str, ...] = (stop,)
    elif isinstance(stop, list):
        stop_seqs = tuple(str(s) for s in stop)
    else:
        stop_seqs = ()

    max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 4096)

    schema = None
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema")

    system = "\n\n".join(system_parts) if system_parts else None
    check_bounds(chars=size_of(system, messages), max_tokens=max_tokens)

    return GenRequest(
        messages=messages,
        system=system,
        tools=tuple(tools),
        sampling=Sampling(
            temperature=float(body.get("temperature", 0.7) or 0.0),
            top_p=float(body.get("top_p", 1.0) or 1.0),
            seed=int(body.get("seed") or 0),
            max_tokens=max_tokens,
            stop=stop_seqs,
        ),
        stream=bool(body.get("stream", False)),
        json_schema=schema,
        adapter=body.get("adapter") or (body.get("metadata") or {}).get("adapter"),
        protocol="chat_completions",
    )


def from_canonical(result: GenResult, *, model: str, request_id: str = "") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.text or None}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json()},
            }
            for call in result.tool_calls
        ]
    body: dict[str, Any] = {
        "id": request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _FINISH_REASON[result.stop_reason],
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": result.usage.input_tokens,
            "completion_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.input_tokens + result.usage.output_tokens,
            "prompt_tokens_details": {"cached_tokens": result.usage.cached_input_tokens},
        },
    }
    if result.receipt is not None:
        body["tandem_receipt"] = result.receipt
    return body


def stream_options(body: dict[str, Any]) -> dict[str, Any]:
    """`StreamEncoder` keyword arguments this protocol reads off the request body.

    Every wire module exposes this so `app.py` can build any protocol's encoder the
    same way; only this one has anything to say. `stream_options.include_usage` is
    the client asking for the usage chunk, and without it sec 8.3's context scaling
    never reaches an OpenAI-compatible streaming client at all — the scaled number
    is the whole mechanism by which the harness decides to compact.
    """
    opts = body.get("stream_options")
    return {"include_usage": bool(opts.get("include_usage")) if isinstance(opts, dict) else False}


class StreamEncoder:
    """Incremental SSE encoder: `open` -> `delta`* -> `close`."""

    def __init__(self, *, model: str, request_id: str = "", include_usage: bool = False):
        self.model = model
        self.id = request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        # One `created` for the whole stream. A per-chunk timestamp would have the
        # same completion claiming several creation times.
        self.created = int(time.time())
        # Off unless the client set `stream_options.include_usage`. Not defaulted on:
        # a chunk with an empty `choices` array is a shape many OpenAI-compatible
        # clients only tolerate because they asked for it, and several index
        # `choices[0]` unconditionally otherwise.
        self.include_usage = include_usage
        self._input_tokens = 0

    def _chunk(self, delta: dict[str, Any], finish: str | None = None) -> list[str]:
        payload: dict[str, Any] = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if self.include_usage:
            # Part of the same contract: with usage requested, every chunk carries
            # the key and only the final usage-only chunk carries a value.
            payload["usage"] = None
        return [f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"]

    def _usage_chunk(self, usage: Usage) -> list[str]:
        """The final chunk: empty `choices`, whole-request usage, then [DONE]."""
        # The prologue's count is the fallback, not the preference: a backend that
        # reports its own prompt tokens knows better than a pre-generation estimate,
        # and a turn whose result carries none would otherwise report zero to the
        # harness's context meter — the exact silent failure sec 8.3 exists to avoid.
        prompt_tokens = usage.input_tokens or self._input_tokens
        payload = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": prompt_tokens + usage.output_tokens,
                "prompt_tokens_details": {"cached_tokens": usage.cached_input_tokens},
            },
        }
        return [f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"]

    def open(self, input_tokens: int = 0) -> list[str]:
        # Chat Completions has nowhere to put prompt usage until the stream ends, so
        # unlike the sibling protocols the count is held rather than emitted here.
        self._input_tokens = input_tokens
        return self._chunk({"role": "assistant", "content": ""})

    def delta(self, text: str) -> list[str]:
        return self._chunk({"content": text}) if text else []

    def close(self, result: GenResult) -> list[str]:
        out: list[str] = []
        for i, call in enumerate(result.tool_calls):
            out += self._chunk(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments_json()},
                        }
                    ]
                }
            )
        out += self._chunk({}, finish=_FINISH_REASON[result.stop_reason])
        if self.include_usage:
            out += self._usage_chunk(result.usage)
        out.append("data: [DONE]\n\n")
        return out

    def fail(self, message: str, err_type: str = "api_error") -> list[str]:
        """Terminate a stream whose headers already went out."""
        payload = json.dumps(error(500, message, err_type), separators=(",", ":"))
        return [f"data: {payload}\n\n", "data: [DONE]\n\n"]


def error(status: int, message: str, err_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "param": None, "code": None}}
