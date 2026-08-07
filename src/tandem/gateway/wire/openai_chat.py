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
)

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
    system_parts: list[str] = []
    messages: list[Message] = []

    for raw in body.get("messages", []):
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

    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or 4096

    schema = None
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema")

    return GenRequest(
        messages=messages,
        system="\n\n".join(system_parts) if system_parts else None,
        tools=tuple(tools),
        sampling=Sampling(
            temperature=float(body.get("temperature", 0.7) or 0.0),
            top_p=float(body.get("top_p", 1.0) or 1.0),
            seed=int(body.get("seed") or 0),
            max_tokens=int(max_tokens),
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


class StreamEncoder:
    """Incremental SSE encoder: `open` -> `delta`* -> `close`."""

    def __init__(self, *, model: str, request_id: str = ""):
        self.model = model
        self.id = request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        # One `created` for the whole stream. A per-chunk timestamp would have the
        # same completion claiming several creation times.
        self.created = int(time.time())

    def _chunk(self, delta: dict[str, Any], finish: str | None = None) -> list[str]:
        payload = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return [f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"]

    def open(self, input_tokens: int = 0) -> list[str]:
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
        out.append("data: [DONE]\n\n")
        return out

    def fail(self, message: str, err_type: str = "api_error") -> list[str]:
        """Terminate a stream whose headers already went out."""
        payload = json.dumps(error(500, message, err_type), separators=(",", ":"))
        return [f"data: {payload}\n\n", "data: [DONE]\n\n"]


def sse_events(result: GenResult, *, model: str, request_id: str = "") -> list[str]:
    enc = StreamEncoder(model=model, request_id=request_id)
    return [
        *enc.open(result.usage.input_tokens),
        *enc.delta(result.text),
        *enc.close(result),
    ]


def error(status: int, message: str, err_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "param": None, "code": None}}
