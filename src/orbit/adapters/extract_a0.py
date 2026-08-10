"""A0 — harness adapter from synthetic tool-call traces (spec sec 6.1).

**A0 first, and it is the sleeper win.** Local models emit XML/JSON tool-call
hybrids; the harness sees no valid call, re-prompts, the model garbles identically,
and you get infinite "let me do that" loops. The incumbent fix is regex recovery
plus retries — which this runtime also ships (sec 8.5.3) — but that treats the
symptom. Training the exact tool-call shape into an adapter attacks the cause.

A0 is also the cheapest adapter to build and the only one that generalises across
every customer, because it encodes the *harness's* format, not the repository's
conventions.

Provenance: `SYNTHETIC_HARNESS` (sec 9.4). Traces are generated from a grammar in
this file — not sampled from another model. That distinction is the difference
between a corpus we can attest to an auditor and one we cannot.

The generator is seeded and deterministic, so a corpus hash pins an exact dataset.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbit.types import ToolDef

# A representative coding-agent tool surface. Names are drawn from the shapes real
# harnesses use so the adapter learns the format against realistic identifiers.
DEFAULT_TOOLS: tuple[ToolDef, ...] = (
    ToolDef(
        name="read_file",
        description="Read a file from the repository.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    ),
    ToolDef(
        name="write_file",
        description="Write contents to a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    ),
    ToolDef(
        name="edit_file",
        description="Replace an exact string in a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    ),
    ToolDef(
        name="run_bash",
        description="Run a shell command.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    ),
    ToolDef(
        name="grep",
        description="Search file contents by regex.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
            },
            "required": ["pattern"],
        },
    ),
    ToolDef(
        name="glob",
        description="Find files by glob pattern.",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    ),
)

_PATHS = (
    "src/click/core.py",
    "src/click/_termui_impl.py",
    "tests/test_options.py",
    "README.md",
    "pyproject.toml",
    "src/utils/retry.py",
    "internal/server/handler.go",
    "lib/parser.rs",
    "app/models/user.rb",
    "packages/core/src/index.ts",
)
_PATTERNS = (
    r"def \w+\(",
    "TODO",
    r"class \w+",
    "retry",
    r"import \w+",
    "deprecated",
    r"raise \w+Error",
    "async def",
)
_COMMANDS = (
    "pytest -q",
    "pytest tests/test_options.py -x",
    "ruff check .",
    "mypy src",
    "go test ./...",
    "cargo test",
    "npm run build",
    "git diff --stat",
)
_GLOBS = ("**/*.py", "src/**/*.ts", "**/test_*.py", "**/*.go", "docs/**/*.md")

_TASKS = (
    "Find where the retry backoff is implemented.",
    "Read the option parsing code and tell me how nargs is handled.",
    "The test for shell completion is failing — look into it.",
    "Add a timeout parameter to the HTTP client.",
    "Rename `parse_args` to `parse_arguments` across the package.",
    "Check whether the deprecation warning still fires on Python 3.13.",
    "Search for any remaining uses of the old logging helper.",
    "Run the test suite and report what fails.",
    "Fix the type error mypy reports in the parser module.",
    "Show me the README section on custom parameter types.",
)

_ACKS = (
    "",
    "",
    "",
    "",  # weighted toward no preamble: the target behaviour is a bare call
    "I'll check that.",
    "Let me look at the file.",
    "Searching for that now.",
)


@dataclass
class A0Trace:
    """One synthetic turn: a task, optional tool results, and the correct call."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {"messages": self.messages}


def _sample_args(tool: ToolDef, rng: random.Random) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for name in tool.param_names():
        required = name in tool.required_names()
        # Optional parameters appear ~30% of the time, so the adapter learns they
        # are optional rather than learning to always emit every key.
        if not required and rng.random() > 0.3:
            continue
        if name == "path":
            args[name] = rng.choice(_PATHS)
        elif name == "pattern":
            args[name] = (
                rng.choice(_PATTERNS) if tool.name == "grep" else rng.choice(_GLOBS)
            )
        elif name == "command":
            args[name] = rng.choice(_COMMANDS)
        elif name == "glob":
            args[name] = rng.choice(_GLOBS)
        elif name in ("old_string", "new_string", "content"):
            args[name] = rng.choice(
                ["retries = 3", "retries = 5", "timeout: float = 30.0", "return None"]
            )
        elif name in ("offset", "limit", "timeout"):
            args[name] = rng.choice([10, 30, 100, 200])
        elif name == "replace_all":
            args[name] = bool(rng.getrandbits(1))
        else:
            args[name] = "value"
    return args


def _render_call(tool: ToolDef, args: dict[str, Any]) -> str:
    """The one canonical shape the adapter is being trained to emit.

    Exactly what `repair.py` recognises as already-correct, and exactly what
    `constrain.tool_call_schema` enforces — the three layers agree on one target, or
    they teach the model three different things.
    """
    return json.dumps({"name": tool.name, "arguments": args}, ensure_ascii=False)


def _tool_result_text(tool: ToolDef, rng: random.Random) -> str:
    if tool.name == "run_bash":
        return rng.choice(
            [
                "3 passed, 1 failed in 2.14s",
                "All checks passed.",
                "error: type mismatch at line 42",
            ]
        )
    if tool.name in ("grep", "glob"):
        return "\n".join(rng.sample(_PATHS, k=min(3, len(_PATHS))))
    return "  1  def parse_args(self, ctx, args):\n  2      ...\n"


def generate(
    n: int = 2000,
    *,
    tools: tuple[ToolDef, ...] = DEFAULT_TOOLS,
    seed: int = 0,
    multi_step_fraction: float = 0.35,
) -> list[A0Trace]:
    """Generate `n` synthetic tool-call traces.

    A third are multi-step — a call, its result, then a follow-up call — because the
    failure this adapter targets is a loop, and a loop only appears once there is a
    second turn to get wrong.
    """
    rng = random.Random(seed)
    traces: list[A0Trace] = []
    for _ in range(n):
        tool = tools[rng.randrange(len(tools))]
        args = _sample_args(tool, rng)
        ack = rng.choice(_ACKS)
        assistant = (f"{ack}\n" if ack else "") + _render_call(tool, args)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": rng.choice(_TASKS)},
            {"role": "assistant", "content": assistant},
        ]

        if rng.random() < multi_step_fraction:
            follow_tool = tools[rng.randrange(len(tools))]
            follow_args = _sample_args(follow_tool, rng)
            messages.append({"role": "user", "content": _tool_result_text(tool, rng)})
            messages.append(
                {"role": "assistant", "content": _render_call(follow_tool, follow_args)}
            )

        traces.append(A0Trace(messages=messages))
    return traces


def write_jsonl(traces: list[A0Trace], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.writelines(
            json.dumps(trace.as_record(), ensure_ascii=False) + "\n" for trace in traces
        )
    return p


def report(traces: list[A0Trace]) -> dict[str, Any]:
    multi = sum(1 for t in traces if len(t.messages) > 2)
    names: dict[str, int] = {}
    for trace in traces:
        for msg in trace.messages:
            if msg["role"] != "assistant":
                continue
            for line in msg["content"].splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        names_key = json.loads(line)["name"]
                    except (json.JSONDecodeError, KeyError):
                        continue
                    names[names_key] = names.get(names_key, 0) + 1
    return {
        "traces": len(traces),
        "multi_step": multi,
        "multi_step_fraction": round(multi / len(traces), 3) if traces else 0.0,
        "calls_by_tool": dict(sorted(names.items())),
        "source_kind": "synthetic_harness",
        "note": "Generated from a grammar in this module. Never sampled from another model (sec 9.4).",
    }
