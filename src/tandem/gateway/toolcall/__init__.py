"""Tool-call reliability (spec sec 8.5): belt and braces, because this is what breaks.

Five layers, in the order they engage:

1. **Prevent** — constrained decoding (`constrain.py`).
2. **Train** — the A0 harness adapter (`tandem.adapters.a0`), which attacks the cause.
3. **Repair** — recover malformed shapes (`repair.py`).
4. **Retry** — bounded, only when tool intent was detected (`pipeline.retry`).
5. **Replay** — exact sampled text preserved across turns (`replay.py`).
"""

from .constrain import Constrainer, tool_call_schema
from .repair import RepairOutcome, looks_like_tool_intent, repair, resolve_name
from .replay import ReplayMap, coverage, render_call

__all__ = [
    "Constrainer",
    "tool_call_schema",
    "RepairOutcome",
    "looks_like_tool_intent",
    "repair",
    "resolve_name",
    "ReplayMap",
    "coverage",
    "render_call",
]
