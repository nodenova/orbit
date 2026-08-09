"""Tool-call replay map (spec sec 8.5.5).

The subtlest bug in this whole layer, and the one most likely to be rediscovered
painfully by anyone who removes this file.

The model samples some exact text for a tool call. The harness parses it, executes
the tool, and sends the call back on the next turn — but as *normalised JSON*, its
own rendering, not the model's bytes. If the gateway re-renders that normalised
form into the prompt, the byte prefix no longer matches what was prefilled, the
prompt cache misses, and the entire turn is rebuilt from scratch. On a 25k-token
conversation that is the difference between a 2 s and a 60 s turn, and it happens
on *every* tool-using turn, which is all of them.

So: keep `tool_id -> exact sampled block`, bounded, and persist it inside the KV
file so it survives a restart alongside the state it belongs to.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from tandem.types import Message, ToolCall


class ReplayMap:
    """Bounded tool_id -> exact sampled text."""

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._map: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, tool_id: str) -> bool:
        return tool_id in self._map

    def put(self, tool_id: str, sampled: str) -> None:
        if not tool_id or not sampled:
            return
        with self._lock:
            if tool_id in self._map:
                self._map.move_to_end(tool_id)
            self._map[tool_id] = sampled
            while len(self._map) > self.max_size:
                self._map.popitem(last=False)

    def put_all(self, blocks: dict[str, str]) -> None:
        for tool_id, sampled in blocks.items():
            self.put(tool_id, sampled)

    def get(self, tool_id: str) -> str | None:
        with self._lock:
            sampled = self._map.get(tool_id)
            if sampled is not None:
                self._map.move_to_end(tool_id)
            return sampled

    def as_dict(self) -> dict[str, str]:
        with self._lock:
            return dict(self._map)

    def load(self, data: dict[str, str]) -> None:
        for tool_id, sampled in data.items():
            self.put(tool_id, sampled)

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self._map), "max_size": self.max_size}


def render_call(call: ToolCall, replay: ReplayMap | None = None) -> str:
    """The bytes to put in the prompt for one tool call.

    Prefers the model's own sampled text; falls back to a canonical rendering. This
    is the function `Backend.render` must route tool calls through — rendering them
    any other way is what breaks the byte prefix.

    A call we have no record for (from before a restart, or evicted from the bounded
    map) renders canonically and costs a cache miss on that turn, which is the
    correct degradation: a miss is slow, a wrong prompt is wrong.
    """
    if replay is not None:
        sampled = replay.get(call.id)
        if sampled is not None:
            return sampled
    return f"{call.name}\n{call.arguments_json()}"


def coverage(messages: Iterable[Message], replay: ReplayMap) -> tuple[int, int]:
    """(calls with sampled text on record, total calls) across a conversation.

    Full coverage means the whole prompt prefix reproduces byte-for-byte; anything
    less predicts exactly how far back the prompt cache can hit.
    """
    total = 0
    known = 0
    for msg in messages:
        for call in msg.tool_calls:
            total += 1
            if call.id in replay:
                known += 1
    return known, total
