"""Turn classification (spec sec 7.1).

    classify(request) -> {chat, read_only, code_change, plan}

Signals the spec names: which tools the turn is likely to invoke, whether the
previous turn produced a diff, prompt length, explicit user directive.

Deliberately heuristic and deliberately cheap. Its only job is deciding whether to
spend ~20 s of tier-1 rerank on this turn, and it is wrong in a bounded way: a
misclassified `code_change` costs latency, a misclassified `chat` costs a missed
quality gate. So the tie-breaks lean toward `code_change` when a turn carries
mutating tools, and toward `chat` when nothing suggests otherwise.

A learned router is explicitly out of scope for v1 (sec 7.2) — it adds a tuning
surface without being needed to test the thesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from orbit.types import GenRequest, Message, Role, TurnClass

# Tool names that mutate the working tree. Matched on substrings because harnesses
# name the same capability differently (edit_file / str_replace_editor / apply_patch).
_MUTATING = (
    "edit",
    "write",
    "create",
    "apply_patch",
    "patch",
    "replace",
    "insert",
    "delete",
    "move",
    "rename",
    "commit",
    "format",
)
_READ_ONLY = ("read", "cat", "view", "grep", "search", "glob", "list", "ls", "find")
# Shell tools are ambiguous: `pytest` is read-only in spirit, `sed -i` is not.
_SHELL = ("bash", "shell", "run_command", "terminal", "exec")

_PLAN_DIRECTIVE = re.compile(
    r"\b(plan|design|approach|architect|propose|outline|strategy|how (?:would|should) (?:we|i|you))\b",
    re.IGNORECASE,
)
_CODE_DIRECTIVE = re.compile(
    r"\b(fix|implement|add|refactor|rename|migrate|patch|change|update|remove|delete|"
    r"write (?:a|the|some)? ?(?:test|function|class|method)|make .{0,30}(?:pass|work))\b",
    re.IGNORECASE,
)
_READ_DIRECTIVE = re.compile(
    r"\b(what|where|why|how does|explain|show me|find|look at|review|summar[iy])\b",
    re.IGNORECASE,
)
_DIFF_MARKER = re.compile(r"^(?:diff --git |@@ |\+\+\+ |--- )", re.MULTILINE)

# A long prompt is a working-session signal, not a chat signal (sec 7.1).
_LONG_PROMPT_CHARS = 6_000


@dataclass
class Classification:
    turn: TurnClass
    confidence: float
    signals: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn.value,
            "confidence": round(self.confidence, 2),
            "signals": self.signals,
        }


def _last_user_text(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role is Role.USER and msg.content:
            return msg.content
    return ""


def previous_turn_produced_diff(messages: list[Message]) -> bool:
    for msg in reversed(messages):
        if msg.role is Role.ASSISTANT:
            return bool(_DIFF_MARKER.search(msg.content or ""))
        # A tool result carrying a diff counts too — that is how most harnesses
        # report a successful edit back into the conversation.
        for res in msg.tool_results:
            if _DIFF_MARKER.search(res.content or ""):
                return True
    return False


def classify(req: GenRequest) -> Classification:
    tool_names = [t.name.lower() for t in req.tools]
    has_mutating = any(any(m in n for m in _MUTATING) for n in tool_names)
    has_shell = any(any(s in n for s in _SHELL) for n in tool_names)
    has_read = any(any(r in n for r in _READ_ONLY) for n in tool_names)

    user_text = _last_user_text(req.messages)
    prompt_chars = sum(len(m.content or "") for m in req.messages) + len(
        req.system or ""
    )
    prior_diff = previous_turn_produced_diff(req.messages)

    signals: dict[str, Any] = {
        "mutating_tools": has_mutating,
        "shell_tools": has_shell,
        "read_tools": has_read,
        "prompt_chars": prompt_chars,
        "prior_turn_diff": prior_diff,
        "n_tools": len(tool_names),
    }

    # Explicit user directive outranks tool inventory: a harness ships the same tool
    # set on every turn, so the tools tell you what is *possible*, not what is asked.
    if _PLAN_DIRECTIVE.search(user_text) and not _CODE_DIRECTIVE.search(user_text):
        signals["directive"] = "plan"
        return Classification(TurnClass.PLAN, 0.8, signals)

    if _CODE_DIRECTIVE.search(user_text):
        signals["directive"] = "code"
        if has_mutating or has_shell:
            return Classification(TurnClass.CODE_CHANGE, 0.9, signals)
        return Classification(TurnClass.CODE_CHANGE, 0.6, signals)

    # Continuing a turn that already produced a diff is still a code_change turn:
    # the follow-up is usually "now fix the test it broke", which is exactly where
    # T2 escalation earns its keep.
    if prior_diff and (has_mutating or has_shell):
        signals["directive"] = "continuation"
        return Classification(TurnClass.CODE_CHANGE, 0.7, signals)

    if _READ_DIRECTIVE.search(user_text):
        signals["directive"] = "read"
        return Classification(TurnClass.READ_ONLY, 0.75, signals)

    if not req.tools:
        signals["directive"] = "none"
        return Classification(TurnClass.CHAT, 0.85, signals)

    if has_mutating and prompt_chars > _LONG_PROMPT_CHARS:
        signals["directive"] = "long working session"
        return Classification(TurnClass.CODE_CHANGE, 0.55, signals)

    if has_read:
        signals["directive"] = "tools available, no clear directive"
        return Classification(TurnClass.READ_ONLY, 0.5, signals)

    signals["directive"] = "fallthrough"
    return Classification(TurnClass.CHAT, 0.4, signals)
