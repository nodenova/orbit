"""A purpose-built agent loop for Agents-A1-4B, driven through ollama's `/api/chat`.

The companion arm to `tools/quality/agent_eval.py`, which drives the same task set
through Claude Code. This one exists to separate two questions that run together
there: how good the model is, and how much of its score was the harness around it.
It emits the same artifact schema, so `judge` and `report` consume both unchanged.

Everything about the tasks, the isolation and the grading is imported from the
Claude Code arm rather than reimplemented. What differs is declared in the artifact's
`harness.declared_deltas` and nowhere else — an undeclared difference between the two
arms is a bug, because the number this produces is only worth having if it is
comparable to the number it is compared against.
"""

from __future__ import annotations

VERSION = "0.1.0"
