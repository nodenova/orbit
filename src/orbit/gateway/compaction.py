"""Harness compaction — the largest single latency win (spec sec 8.2).

Claude Code sends a ~10-25k token harness prompt. Swapping it for a ~150-token
equivalent and stripping tool descriptions to name + parameter types was measured at
28x, prefill 60 s -> 2 s [V]. Nothing in the inference stack offers that multiplier.

Three things the spec insists on, and the reason each is here rather than in a
simpler implementation:

**Versioned templates.** `cc-2026.08@v3` is not decoration. Claude Code's system
prompt changes; a fingerprint pinned to an exact string silently stops matching and
the customer quietly loses the 28x with no error anywhere. So detection is *scored* —
a template declares markers and a minimum, and a partial match still compacts but
raises a staleness signal that `orbit doctor` and the audit log both surface.

**A --no-compact escape hatch and a diff view.** Compaction is lossy. A customer will
eventually need to see exactly what was sent, and "trust us" is not an answer for the
buyer this product targets.

**Measured, not assumed.** `CompactionResult` carries before/after token counts.
The spec's gate is >=10x on your own harness; `measure()` is what reports it.
"""

from __future__ import annotations

import difflib
import re
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from orbit.types import GenRequest, ToolDef


@dataclass(frozen=True)
class CompactionTemplate:
    """A compact equivalent of one harness's system prompt."""

    id: str  # e.g. "cc-2026.08@v3"
    harness: str  # e.g. "claude_code"
    # Distinctive phrases from the harness's own prompt. Order-insensitive.
    markers: tuple[str, ...]
    # How many markers must match to claim this harness.
    min_markers: int
    replacement: str
    # Below this fraction of markers, the match is still accepted but flagged as a
    # drifted harness prompt. Not "fewer than all": harness prompts legitimately
    # vary by platform, flags and version, so requiring a full house would mark
    # every real request stale and the signal would be ignored — which is exactly
    # the silent-degradation failure the spec warns about, arrived at from the
    # other direction. Half the markers missing is a real drift.
    stale_below_ratio: float = 0.5
    # Harness releases this template was authored against. Advisory: recorded in the
    # receipt so a regression can be tied to a harness upgrade.
    authored_for: tuple[str, ...] = ()

    def score(self, system: str) -> int:
        low = system.lower()
        return sum(1 for m in self.markers if m.lower() in low)


@dataclass
class CompactionResult:
    applied: bool
    template_id: str | None
    harness: str | None
    original_system: str | None
    compacted_system: str | None
    original_tokens: int
    compacted_tokens: int
    # True when the harness matched on fewer markers than the template declares —
    # the harness prompt has drifted and the template needs re-authoring.
    stale_fingerprint: bool = False
    matched_markers: int = 0
    total_markers: int = 0
    reason: str = ""

    @property
    def multiplier(self) -> float:
        if self.compacted_tokens <= 0:
            return 1.0
        return self.original_tokens / self.compacted_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "template_id": self.template_id,
            "harness": self.harness,
            "original_tokens": self.original_tokens,
            "compacted_tokens": self.compacted_tokens,
            "multiplier": round(self.multiplier, 2),
            "stale_fingerprint": self.stale_fingerprint,
            "matched_markers": f"{self.matched_markers}/{self.total_markers}",
            "reason": self.reason,
        }

    def diff(self) -> str:
        """Unified diff of what the harness sent vs what the model saw."""
        if self.original_system is None and self.compacted_system is None:
            # `keep_original=False` deliberately retains no prompt text. Returning an
            # empty diff here would read as "compaction changed nothing", which is
            # the opposite of the truth; say why there is nothing to show.
            return "(no prompt text retained: compaction.keep_original = false)\n"
        a = (self.original_system or "").splitlines(keepends=True)
        b = (self.compacted_system or "").splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                a, b, fromfile="harness-system", tofile="compacted-system", n=2
            )
        )


# --- the shared compact core ------------------------------------------------
#
# What every coding harness's multi-thousand-token preamble actually conveys, once
# the safety boilerplate, tone guidance and worked examples are removed. Kept short
# on purpose: the whole point is that the resident model is *already* a coding model
# and does not need to be told what a file is.

_CORE = """You are a coding agent working in a local repository.

- Use the provided tools to read and edit files; never guess file contents.
- Prefer small, targeted diffs that match the surrounding code's conventions.
- Run the repository's own tests and linters to check your work.
- When a tool call is needed, emit it as a well-formed call and nothing else.
- Stop when the task is done; do not narrate.
"""

TEMPLATES: tuple[CompactionTemplate, ...] = (
    CompactionTemplate(
        id="cc-2026.08@v3",
        harness="claude_code",
        markers=(
            "You are Claude Code",
            "Anthropic's official CLI",
            "IMPORTANT: Assist with authorized security testing",
            "# Harness",
            "file_path:line_number",
            "system-reminder",
        ),
        min_markers=2,
        replacement=_CORE
        + "\nReference code as file_path:line_number. Match the surrounding style.\n",
        authored_for=("2026.08",),
    ),
    CompactionTemplate(
        id="opencode-2026.06@v1",
        harness="opencode",
        markers=("You are opencode", "opencode", "sst/opencode"),
        min_markers=1,
        replacement=_CORE,
        authored_for=("2026.06",),
    ),
    CompactionTemplate(
        id="codex-2026.05@v1",
        harness="codex",
        markers=("You are Codex", "codex CLI", "OpenAI Codex"),
        min_markers=1,
        replacement=_CORE,
        authored_for=("2026.05",),
    ),
    CompactionTemplate(
        id="crush-2026.04@v1",
        harness="crush",
        markers=("You are Crush", "charmbracelet"),
        min_markers=1,
        replacement=_CORE,
        authored_for=("2026.04",),
    ),
    CompactionTemplate(
        id="openclaw-2026.07@v1",
        harness="openclaw",
        markers=("You are OpenClaw", "openclaw"),
        min_markers=1,
        replacement=_CORE,
        authored_for=("2026.07",),
    ),
)


def detect(
    system: str | None, templates: Iterable[CompactionTemplate] = TEMPLATES
) -> tuple[CompactionTemplate | None, int]:
    """Best-matching template and its marker score, or (None, 0)."""
    if not system:
        return None, 0
    best: CompactionTemplate | None = None
    best_score = 0
    for t in templates:
        s = t.score(system)
        if s >= t.min_markers and s > best_score:
            best, best_score = t, s
    return best, best_score


# --- tool-schema stripping --------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def one_line(text: str, limit: int = 120) -> str:
    """First sentence of a description, collapsed and truncated."""
    if not text:
        return ""
    flat = " ".join(text.split())
    first = _SENTENCE_END.split(flat, maxsplit=1)[0]
    if len(first) > limit:
        first = first[: limit - 1].rstrip() + "…"
    return first


def _type_of(schema: dict[str, Any]) -> str:
    if "enum" in schema:
        return "enum"
    t = schema.get("type")
    if isinstance(t, list):
        return "|".join(str(x) for x in t)
    if t == "array":
        item = schema.get("items", {})
        inner = _type_of(item) if isinstance(item, dict) else "any"
        return f"{inner}[]"
    return str(t or "any")


def strip_tool(tool: ToolDef) -> ToolDef:
    """Reduce a tool to name + parameter types + one-line description (sec 8.2).

    Keeps `required` and enum values: dropping `required` makes the model omit
    mandatory arguments, and dropping enums makes it invent values — both of which
    show up directly as tool-call failures on the sec 10.2 gate. Everything else
    (nested descriptions, examples, defaults, long prose) goes.
    """
    props = tool.parameters.get("properties")
    if not isinstance(props, dict):
        return ToolDef(
            name=tool.name,
            description=one_line(tool.description),
            parameters=tool.parameters,
        )

    slim: dict[str, Any] = {}
    for name, schema in props.items():
        if not isinstance(schema, dict):
            slim[name] = {"type": "any"}
            continue
        entry: dict[str, Any] = {"type": _type_of(schema)}
        if "enum" in schema:
            entry["enum"] = schema["enum"]
        slim[name] = entry

    params: dict[str, Any] = {"type": "object", "properties": slim}
    req = tool.parameters.get("required")
    if isinstance(req, list) and req:
        params["required"] = req
    return ToolDef(
        name=tool.name, description=one_line(tool.description), parameters=params
    )


# --- the pipeline stage -----------------------------------------------------


# A served gateway runs for days. `history` exists for the M1 gate and the diff
# view, both of which want a recent sample rather than every turn since boot, and a
# Claude Code system prompt is 40-100 KB — an unbounded list of them is a gigabyte
# after ten thousand turns. Bounded rather than cleared, so the gate still has a
# population to take a median over; the running totals below keep `measure()`
# honest about how many turns that median was drawn from.
DEFAULT_HISTORY_LIMIT = 256


@dataclass
class Compactor:
    enabled: bool = True
    strip_schemas: bool = True
    # Retain the harness's original prompt on the request so --no-compact and the
    # diff view can reach it. Off only if a deployment considers the raw prompt
    # itself sensitive enough not to hold in memory — in which case it must not be
    # held in `history` either, which is the other place it would otherwise live.
    keep_original: bool = True
    templates: tuple[CompactionTemplate, ...] = TEMPLATES
    # Injected so the multiplier is measured in the serving model's own tokens
    # rather than a character-count proxy.
    count_tokens: Callable[[str], int] = field(default=lambda s: max(1, len(s) // 4))
    # How many recent results to retain. See DEFAULT_HISTORY_LIMIT.
    history_limit: int = DEFAULT_HISTORY_LIMIT
    # The most recent `history_limit` results, for the M1 gate report and the diff
    # view. A deque so retention is enforced by the container rather than by every
    # call site remembering to trim.
    history: deque[CompactionResult] = field(default_factory=deque, repr=False)
    # Population counts over *every* turn, not just the retained window: a median
    # taken from 256 samples out of 10k turns is fine, but reporting n=256 would
    # understate the evidence behind the gate.
    turns_seen: int = 0
    turns_applied: int = 0
    turns_stale: int = 0

    def __post_init__(self) -> None:
        limit = max(1, self.history_limit)
        if self.history.maxlen != limit:
            self.history = deque(self.history, maxlen=limit)

    def _record(self, result: CompactionResult, record: bool) -> CompactionResult:
        """Append to the bounded history and update the population counters."""
        if not record:
            # `record=False` is a probe rather than a served turn — the token-count
            # endpoint compacts to answer honestly, but a probe is not a request the
            # model ran. Counting it would skew the M1 gate's multiplier statistics,
            # and appending it would hide the last real turn behind a probe in the
            # sec 8.2 diff view — Claude Code probes far more often than it completes.
            return result
        if not self.keep_original:
            # The flag has to bite here and not only on the outgoing GenRequest:
            # `history` outlives the request, and holding the raw prompt in it is
            # exactly what the deployment asked us not to do.
            result.original_system = None
            result.compacted_system = None
        self.history.append(result)
        self.turns_seen += 1
        if result.applied:
            self.turns_applied += 1
            if result.stale_fingerprint:
                self.turns_stale += 1
        return result

    def apply(
        self, req: GenRequest, *, force_off: bool = False, record: bool = True
    ) -> tuple[GenRequest, CompactionResult]:
        original = req.system
        tmpl, score = detect(original, self.templates)
        harness = tmpl.harness if tmpl else None

        original_tokens = self._tokens_for(original, req.tools)

        if force_off or not self.enabled:
            result = self._record(
                CompactionResult(
                    applied=False,
                    template_id=None,
                    harness=harness,
                    original_system=original,
                    compacted_system=original,
                    original_tokens=original_tokens,
                    compacted_tokens=original_tokens,
                    reason="compaction disabled",
                ),
                record,
            )
            return req.with_(harness=harness), result

        if tmpl is None:
            # An unrecognised harness is compacted not at all. Guessing at a prompt
            # we do not recognise is how you mis-strip a tool.
            result = self._record(
                CompactionResult(
                    applied=False,
                    template_id=None,
                    harness=None,
                    original_system=original,
                    compacted_system=original,
                    original_tokens=original_tokens,
                    compacted_tokens=original_tokens,
                    reason="no template matched this system prompt",
                ),
                record,
            )
            return req, result

        tools = (
            tuple(strip_tool(t) for t in req.tools) if self.strip_schemas else req.tools
        )
        compacted_tokens = self._tokens_for(tmpl.replacement, tools)
        stale = score < tmpl.stale_below_ratio * len(tmpl.markers)

        result = self._record(
            CompactionResult(
                applied=True,
                template_id=tmpl.id,
                harness=tmpl.harness,
                original_system=original,
                compacted_system=tmpl.replacement,
                original_tokens=original_tokens,
                compacted_tokens=compacted_tokens,
                stale_fingerprint=stale,
                matched_markers=score,
                total_markers=len(tmpl.markers),
                reason=(
                    f"harness prompt matched only {score}/{len(tmpl.markers)} markers for "
                    f"{tmpl.id}; the harness has been upgraded and the template needs "
                    "re-authoring before it mis-strips something"
                    if stale
                    else "ok"
                ),
            ),
            record,
        )

        new_req = req.with_(
            system=tmpl.replacement,
            tools=tools,
            harness=tmpl.harness,
            original_system=original if self.keep_original else None,
            compaction_template=tmpl.id,
        )
        return new_req, result

    def _tokens_for(self, system: str | None, tools: Iterable[ToolDef]) -> int:
        import json

        n = self.count_tokens(system) if system else 0
        for t in tools:
            n += self.count_tokens(
                json.dumps(
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                    sort_keys=True,
                )
            )
        return n

    def measure(self) -> dict[str, Any]:
        """M1 gate (sec 11): >=10x measured compaction on your own harness.

        The multiplier statistics come from the retained window; the counts come
        from the running totals. Reporting the window's length as `n` would claim
        a 256-turn sample when the process has served ten thousand.
        """
        if not self.turns_applied:
            return {"pass": False, "reason": "compaction never applied", "n": 0}
        applied = [r for r in self.history if r.applied]
        if not applied:
            return {
                "pass": False,
                "reason": "no compacted turn inside the retained window",
                "n": self.turns_applied,
            }
        mults = sorted(r.multiplier for r in applied)
        median = mults[len(mults) // 2]
        return {
            "pass": median >= 10.0,
            "threshold": 10.0,
            "median_multiplier": round(median, 2),
            "min_multiplier": round(mults[0], 2),
            "max_multiplier": round(mults[-1], 2),
            "n": self.turns_applied,
            "sampled": len(applied),
            "window": self.history.maxlen,
            "stale_fingerprints": self.turns_stale,
        }
