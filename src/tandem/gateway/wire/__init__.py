"""Wire protocols (spec sec 8.1).

Three shapes, one process, one model, one cache, one router:

* `anthropic`        -> /v1/messages          (Claude Code, OpenClaw)
* `openai_chat`      -> /v1/chat/completions  (OpenCode, Crush, generic)
* `openai_responses` -> /v1/responses         (Codex)

Each module exposes the same four functions — `to_canonical`, `from_canonical`,
`sse_events`, `error` — so `app.py` handles all three identically and a fourth
protocol is a new module rather than a new branch.
"""

from . import anthropic, openai_chat, openai_responses

__all__ = ["anthropic", "openai_chat", "openai_responses"]
