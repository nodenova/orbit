"""Tool-call reliability (spec sec 8.5): belt and braces, because this is what breaks.

Five layers, in the order they engage:

1. **Prevent** — constrained decoding (`constrain.py`).
2. **Train** — the A0 harness adapter (`orbit.adapters.a0`), which attacks the cause.
3. **Repair** — recover malformed shapes (`repair.py`).
4. **Retry** — bounded, only when tool intent was detected (`pipeline.retry`).
5. **Replay** — exact sampled text preserved across turns (`replay.py`).
"""

from orbit.gateway.toolcall.constrain import Constrainer, tool_call_schema
from orbit.gateway.toolcall.repair import (
    RepairOutcome,
    looks_like_tool_intent,
    repair,
    resolve_name,
)
from orbit.gateway.toolcall.replay import ReplayMap, coverage, render_call

__all__ = [
    "Constrainer",
    "RepairOutcome",
    "ReplayMap",
    "coverage",
    "looks_like_tool_intent",
    "render_call",
    "repair",
    "resolve_name",
    "tool_call_schema",
]
