"""Wire protocols (spec sec 8.1).

Three shapes, one process, one model, one cache, one router:

* `anthropic`        -> /v1/messages          (Claude Code, OpenClaw)
* `openai_chat`      -> /v1/chat/completions  (OpenCode, Crush, generic)
* `openai_responses` -> /v1/responses         (Codex)

Each module exposes the same four names — `to_canonical`, `from_canonical`,
`StreamEncoder`, `error` — so `app.py` handles all three identically and a fourth
protocol is a new module rather than a new branch. `StreamEncoder` is the streaming
half of that contract and is what `app.py` actually drives: `open` -> `delta`* ->
`close`, with `fail` for a stream that has to end mid-flight. There is deliberately
no whole-sequence-at-once helper — one existed, nothing called it, and a second
encoding path is a second place for the two to drift.

`anthropic` additionally exposes `count_tokens_response`, because only that protocol
has a token-counting endpoint.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from orbit.types import Message

# --- edge bounds ------------------------------------------------------------
#
# Every request that reaches a `to_canonical` is already fully parsed, so these are
# not a defence against a hostile allocation — that has to be a body-size limit
# above the JSON parse. They bound what the *rest* of the runtime is asked to do:
# rendering, tokenising and prefilling are all linear or worse in these numbers, and
# an unbounded `max_tokens` is a decode loop with no end. Generous on purpose: a
# real Claude Code turn at a 200k window is ~800k characters, so a legitimate
# request never comes close and anything that does is a bug or an attack.
#
# Defined above the submodule imports below because the submodules import them from
# here; the package module object already carries them by the time it imports them.
MAX_MESSAGES = 10_000
MAX_INPUT_CHARS = 8_000_000
MAX_OUTPUT_TOKENS = 262_144


def check_bounds(*, items: int = 0, chars: int = 0, max_tokens: int = 1) -> None:
    """Raise `ValueError` if a canonicalised request exceeds the edge bounds.

    Every argument defaults to something in range so a caller can check what it
    knows when it knows it: the item count is checkable before any per-message work
    is done, the character total only after. Splitting it into two functions would
    just move that decision somewhere less obvious.

    `ValueError` rather than a bespoke exception because `app.py` already turns one
    into the protocol's own 400 shape; a new exception type would need a new branch
    in the very handler this package exists to keep protocol-agnostic.
    """
    if items > MAX_MESSAGES:
        raise ValueError(f"too many messages: {items} > {MAX_MESSAGES}")
    if chars > MAX_INPUT_CHARS:
        raise ValueError(f"request too large: {chars} characters > {MAX_INPUT_CHARS}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    if max_tokens > MAX_OUTPUT_TOKENS:
        raise ValueError(f"max_tokens too large: {max_tokens} > {MAX_OUTPUT_TOKENS}")


def size_of(system: Any, messages: Iterable[Message]) -> int:
    """Characters of prompt text in a canonicalised request.

    Measured on the canonical messages rather than on the raw body: the raw body
    would need a recursive walk of arbitrary JSON, and it is the canonical text that
    goes on to be rendered and tokenised. Tool-call arguments are excluded — they are
    small next to file contents and counting them would mean serialising every call
    on a path that exists to be cheap.
    """
    total = len(system) if isinstance(system, str) else 0
    for msg in messages:
        total += len(msg.content)
        total += sum(len(r.content) for r in msg.tool_results)
    return total


from orbit.gateway.wire import anthropic, openai_chat, openai_responses

__all__ = [
    "MAX_INPUT_CHARS",
    "MAX_MESSAGES",
    "MAX_OUTPUT_TOKENS",
    "anthropic",
    "check_bounds",
    "openai_chat",
    "openai_responses",
    "size_of",
]
