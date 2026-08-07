"""`/v1/responses` — the OpenAI Responses wire shape (spec sec 8.1).

Serves Codex.

Structurally different from Chat Completions in three ways that matter here:
`input` replaces `messages` and may be a bare string; `instructions` replaces the
system message; and the output is a flat list of typed items rather than a choice
with a message. Tools are flat (`{"type": "function", "name": ...}`) rather than
nested under a `function` key.
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

_STATUS = {
    StopReason.END_TURN: "completed",
    StopReason.MAX_TOKENS: "incomplete",
    StopReason.STOP_SEQUENCE: "completed",
    StopReason.TOOL_USE: "completed",
}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in (
                "input_text",
                "output_text",
                "text",
                "summary_text",
            ):
                out.append(part.get("text", ""))
        return "\n".join(p for p in out if p)
    return ""


def to_canonical(body: dict[str, Any]) -> GenRequest:
    messages: list[Message] = []
    raw_input = body.get("input")

    if isinstance(raw_input, str):
        messages.append(Message(role=Role.USER, content=raw_input))
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str):
                messages.append(Message(role=Role.USER, content=item))
                continue
            if not isinstance(item, dict):
                continue
            itype = item.get("type", "message")

            if itype == "function_call":
                args = item.get("arguments")
                parsed: dict[str, Any] = {}
                if isinstance(args, dict):
                    parsed = args
                elif isinstance(args, str) and args.strip():
                    try:
                        obj = json.loads(args)
                        parsed = obj if isinstance(obj, dict) else {"value": obj}
                    except json.JSONDecodeError:
                        parsed = {"_raw": args}
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        tool_calls=(
                            ToolCall(
                                id=item.get("call_id") or item.get("id", ""),
                                name=item.get("name", ""),
                                arguments=parsed,
                            ),
                        ),
                    )
                )
                continue

            if itype == "function_call_output":
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_results=(
                            ToolResult(
                                tool_call_id=item.get("call_id", ""),
                                content=_text_of(item.get("output")),
                            ),
                        ),
                    )
                )
                continue

            role_name = item.get("role", "user")
            role = {
                "assistant": Role.ASSISTANT,
                "system": Role.SYSTEM,
                "developer": Role.SYSTEM,
            }.get(role_name, Role.USER)
            messages.append(Message(role=role, content=_text_of(item.get("content"))))

    # `instructions` is the Responses system prompt; a system item in `input` folds
    # in after it so both routes reach the compactor's fingerprinter.
    system_parts = [body["instructions"]] if isinstance(body.get("instructions"), str) else []
    kept: list[Message] = []
    for msg in messages:
        if msg.role is Role.SYSTEM:
            if msg.content:
                system_parts.append(msg.content)
        else:
            kept.append(msg)

    tools: list[ToolDef] = []
    for t in body.get("tools", []) or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") not in (None, "function"):
            # Hosted tools (web_search, file_search, computer_use) are not ours to
            # serve; passing them to a local model would produce calls nothing can
            # execute. Out of scope for v1 (sec 12).
            continue
        name = t.get("name") or (t.get("function") or {}).get("name")
        if not name:
            continue
        params = t.get("parameters") or (t.get("function") or {}).get("parameters") or {}
        tools.append(
            ToolDef(name=name, description=t.get("description", ""), parameters=params)
        )

    schema = None
    text_cfg = body.get("text")
    if isinstance(text_cfg, dict):
        fmt = text_cfg.get("format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            schema = fmt.get("schema")

    return GenRequest(
        messages=kept,
        system="\n\n".join(system_parts) if system_parts else None,
        tools=tuple(tools),
        sampling=Sampling(
            temperature=float(body.get("temperature", 0.7) or 0.0),
            top_p=float(body.get("top_p", 1.0) or 1.0),
            seed=int(body.get("seed") or 0),
            max_tokens=int(body.get("max_output_tokens") or 4096),
        ),
        stream=bool(body.get("stream", False)),
        json_schema=schema,
        adapter=(body.get("metadata") or {}).get("adapter"),
        protocol="responses",
    )


def _message_item(text: str, item_id: str, *, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "message",
        "id": item_id,
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_item(call: ToolCall, item_id: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments_json(),
        "status": "completed",
    }


def _body(
    result: GenResult,
    *,
    model: str,
    rid: str,
    created: int,
    output: list[dict[str, Any]],
    output_text: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": rid,
        "object": "response",
        "created_at": created,
        "status": _STATUS[result.stop_reason],
        "model": model,
        "output": output,
        "output_text": output_text,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "input_tokens_details": {"cached_tokens": result.usage.cached_input_tokens},
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.input_tokens + result.usage.output_tokens,
        },
    }
    if result.stop_reason is StopReason.MAX_TOKENS:
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    if result.receipt is not None:
        body["tandem_receipt"] = result.receipt
    return body


def from_canonical(result: GenResult, *, model: str, request_id: str = "") -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if result.text:
        output.append(_message_item(result.text, f"msg_{uuid.uuid4().hex[:24]}"))
    output.extend(
        _function_item(call, f"fc_{uuid.uuid4().hex[:24]}") for call in result.tool_calls
    )
    return _body(
        result,
        model=model,
        rid=request_id or f"resp_{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        output=output,
        output_text=result.text,
    )


class StreamEncoder:
    """Incremental SSE encoder: `open` -> `delta`* -> `close`.

    Item ids are minted here rather than by `from_canonical`, because a Responses
    stream announces an item before it has content and every later event for that
    item has to carry the same `item_id`. Building the final body from ids the
    encoder already published is the only way those agree.
    """

    def __init__(self, *, model: str, request_id: str = ""):
        self.model = model
        self.id = request_id or f"resp_{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self._seq = 0
        self._index = 0
        self._message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._message_open = False
        self._text: list[str] = []

    def _emit(self, event: str, data: dict[str, Any]) -> list[str]:
        payload = {**data, "sequence_number": self._seq}
        self._seq += 1
        return [f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"]

    def _skeleton(self, status: str, input_tokens: int = 0) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "model": self.model,
            "output": [],
            "output_text": "",
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "total_tokens": input_tokens,
            },
        }

    def open(self, input_tokens: int = 0) -> list[str]:
        skeleton = self._skeleton("in_progress", input_tokens)
        return [
            *self._emit("response.created", {"type": "response.created", "response": skeleton}),
            *self._emit(
                "response.in_progress", {"type": "response.in_progress", "response": skeleton}
            ),
        ]

    def delta(self, text: str) -> list[str]:
        if not text:
            return []
        out: list[str] = []
        if not self._message_open:
            out += self._emit(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": self._index,
                    "item": _message_item("", self._message_id, status="in_progress"),
                },
            )
            self._message_open = True
        self._text.append(text)
        out += self._emit(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": self._message_id,
                "output_index": self._index,
                "content_index": 0,
                "delta": text,
            },
        )
        return out

    def close(self, result: GenResult) -> list[str]:
        out: list[str] = []
        items: list[dict[str, Any]] = []
        text = "".join(self._text)

        if self._message_open:
            item = _message_item(text, self._message_id)
            items.append(item)
            out += self._emit(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": self._message_id,
                    "output_index": self._index,
                    "content_index": 0,
                    "text": text,
                },
            )
            out += self._emit(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": self._index, "item": item},
            )
            self._index += 1

        for call in result.tool_calls:
            item = _function_item(call, f"fc_{uuid.uuid4().hex[:24]}")
            items.append(item)
            out += self._emit(
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": self._index, "item": item},
            )
            out += self._emit(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": self._index,
                    "delta": item["arguments"],
                },
            )
            out += self._emit(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": self._index,
                    "arguments": item["arguments"],
                },
            )
            out += self._emit(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": self._index, "item": item},
            )
            self._index += 1

        final = _body(
            result,
            model=self.model,
            rid=self.id,
            created=self.created,
            output=items,
            output_text=text,
        )
        out += self._emit("response.completed", {"type": "response.completed", "response": final})
        return out

    def fail(self, message: str, err_type: str = "api_error") -> list[str]:
        """Terminate a stream whose headers already went out."""
        return self._emit(
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    **self._skeleton("failed"),
                    "error": {"code": err_type, "message": message},
                },
            },
        )


def sse_events(result: GenResult, *, model: str, request_id: str = "") -> list[str]:
    enc = StreamEncoder(model=model, request_id=request_id)
    return [
        *enc.open(result.usage.input_tokens),
        *enc.delta(result.text),
        *enc.close(result),
    ]


def error(status: int, message: str, err_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "param": None, "code": None}}
