"""Seven named tools, each a fixed set of parameter slots that expands to one command.

The Claude Code arm gave the model twenty tools, half of them duplicated capability —
`Bash(cat)` beside `Read`, `Bash(grep)` beside `Grep`, `Bash(find)` beside `Glob`.
Small models degrade as the tool count grows, so this is the same capability in seven,
and the model never writes a shell string except through `run`, which is allowlisted on
argv[0] and argv-split rather than shell-evaluated.

Every path resolves inside the task's worktree and an escape is a tool error rather
than a write. Every observation is capped before it enters the history: uncapped
`pytest` output is how a context window disappears in one turn.
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

READ_LINE_BUDGET = 400
SEARCH_MATCH_BUDGET = 60
LIST_ENTRY_BUDGET = 200
RUN_LINE_BUDGET = 80
RUN_TIMEOUT_S = 900.0
DIFFSTAT_LINE_BUDGET = 24

_ALLOWED_GIT = ("diff", "status", "log", "show")
# `pytest`, `ruff` and `mypy` are run as `sys.executable -m`, never bare on PATH. The
# Claude Code arm reached them through a shell with the venv active; a subprocess from
# here does not, and a `FileNotFoundError` recorded as a failing command is
# indistinguishable from a model that wrote code failing lint — a lesson the check
# runner in the other arm already carries.
_AS_MODULE = {"pytest": "pytest", "ruff": "ruff", "mypy": "mypy"}


class PathEscape(Exception):
    """A path that resolved outside the worktree. Never executed, always reported."""


@dataclass(slots=True)
class ToolResult:
    text: str
    ok: bool = True
    evidence: bool = False
    truncated: bool = False


# Which argument identifies the call in the log below. `by_name` counters say the model
# made 71 `read_file` calls and nothing says which files, so a missed anchor could not be
# attributed: an answer omitting a figure that was in a file it read is a synthesis
# failure, and one omitting a figure it never opened is a navigation failure, and those
# take opposite fixes.
_PRIMARY_ARGUMENT = {
    "list_files": "pattern",
    "search": "query",
    "read_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "run": "command",
}

# The eval's own task module is *in* the worktree at the pinned revision, and it carries
# every task's anchors and rubric — "3 = compaction first, context scaling last" is a line
# of it. An agent that greps the question's own phrasing can find the answer key, and one
# episode did: it searched "wire protocol request path", read `agent_eval_tasks.py` and
# scored 2/2. So a score can be inflated by the tree it is measured in, and both halves are
# counted separately because they are different sizes of problem: reading the module hands
# over the anchors and the rubric, while a search hit that merely quotes its name or one
# task prompt hands over nothing. `Criterion(` and `anchors=(` appear in the answer key and
# nowhere else in the tree.
#
# Counting it is not enough, because a plain `search` is the leak's main route rather than
# an unlucky read: one line of that module is
# `anchors=(("backends/base.py", "base.py"), ("Backend",), ("MockBackend",)),` and the two
# lines under it are the rubric, so `search Backend` returns the whole graded answer to
# low-04. `search mmap` returns mid-05's anchor tuple the same way. Both `read_file` and
# `search` therefore drop these two modules and count what they dropped. It is a declared
# delta because the Claude Code arm has no equivalent and its stored numbers were produced
# with the leak open.
_ANSWER_KEY_MODULES = ("agent_eval.py", "agent_eval_tasks.py")
_ANSWER_KEY_MARKERS = ("Criterion(", "anchors=(", "rubric=(")

# Read-only and deterministic between writes, so an identical repeat is answerable from
# the history the model already has. `run` is excluded because `pytest` after an edit is
# the same command with a different answer, which is the point of running it.
_REPEATABLE = frozenset({"search", "list_files", "read_file"})
_REPEAT_LIMIT = 2


@dataclass(slots=True)
class ToolCounters:
    """Typed counters, because "the tool call could not be parsed" is a headline number.

    `parse_failures` is the calls the harness could not execute as issued — an unknown
    function or a missing required parameter. A turn of prose with no call at all is
    `no_tool_call` and kept separate: that is the model choosing not to act, not the
    native format failing to survive the wire, and merging the two would hide the one
    number this experiment exists to test.
    """

    by_name: Counter[str] = field(default_factory=Counter)
    unknown_tool: int = 0
    missing_parameter: int = 0
    denied: int = 0
    path_escapes: int = 0
    errors: int = 0
    truncated: int = 0
    salvaged: int = 0
    no_tool_call: int = 0
    repeats_blocked: int = 0

    @property
    def parse_failures(self) -> int:
        return self.unknown_tool + self.missing_parameter

    @property
    def total(self) -> int:
        return sum(self.by_name.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "by_name": dict(sorted(self.by_name.items())),
            "total": self.total,
            "parse_failures": self.parse_failures,
            "unknown_tool": self.unknown_tool,
            "missing_parameter": self.missing_parameter,
            "salvaged": self.salvaged,
            "denied": self.denied,
            "path_escapes": self.path_escapes,
            "errors": self.errors,
            "truncated": self.truncated,
            "no_tool_call": self.no_tool_call,
            "repeats_blocked": self.repeats_blocked,
        }


def _definition(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}

TOOL_NAMES = (
    "list_files",
    "search",
    "read_file",
    "write_file",
    "edit_file",
    "run",
    "finish",
)


def definitions() -> list[dict[str, Any]]:
    """Passed as `tools:` so the model's own template renders the format instructions.

    The syntax is never restated in the prompt pack: the model was trained against the
    template's exact wording and a paraphrase is a distribution shift for no gain.
    Descriptions stay short because each definition costs roughly 66 prompt tokens.
    """
    return [
        _definition(
            "list_files",
            "List files in the repository matching a glob pattern.",
            {
                "pattern": _STRING | {"description": "e.g. '*.py' or 'src/**/*.toml'"},
                "path": _STRING,
            },
            ["pattern"],
        ),
        _definition(
            "search",
            "Search file contents for a string or regular expression.",
            {"query": _STRING, "path": _STRING, "glob": _STRING},
            ["query"],
        ),
        _definition(
            "read_file",
            "Read a file, with line numbers. Give start/end to read one region.",
            {"path": _STRING, "start": _INTEGER, "end": _INTEGER},
            ["path"],
        ),
        _definition(
            "write_file",
            "Create a new file. To change a file that already exists, use edit_file.",
            {"path": _STRING, "content": _STRING},
            ["path", "content"],
        ),
        _definition(
            "edit_file",
            "Replace one exact occurrence of a string in a file.",
            {"path": _STRING, "old": _STRING, "new": _STRING},
            ["path", "old", "new"],
        ),
        _definition(
            "run",
            "Run one allowlisted command: pytest, ruff, mypy, python3, "
            "or git diff/status/log/show.",
            {"command": _STRING},
            ["command"],
        ),
        _definition(
            "finish",
            "End the task and return your final answer.",
            {"answer": _STRING},
            ["answer"],
        ),
    ]


def search_binary() -> str:
    """`rg` is not installed on the reference machine, and its absence is silent.

    The interactive shell's `rg` is a shim belonging to another tool, so it resolves for
    a person and raises `FileNotFoundError` for a subprocess. BSD grep is GNU-compatible
    for the two flags used here, and the artifact records which binary answered so two
    runs are never compared across search backends without anyone noticing.
    """
    return shutil.which("rg") or shutil.which("grep") or "grep"


def search_backend() -> str:
    return Path(search_binary()).name


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """`**` crossing directories, `*` and `?` not. `find -path` has no `**` at all.

    Enumeration is still `find` — no `fd` fast path, because `fd` is not installed on
    the reference machine — and only the matching happens here.
    """
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(parts) + r"\Z")


def _int_or_none(value: Any) -> int | None:
    """Arguments arrive as ints from the parsed path and as strings from salvage."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


class Toolbox:
    """The tool surface for one episode, bound to one worktree."""

    def __init__(
        self,
        root: Path,
        *,
        truncation_note: str,
        read_lines: int = READ_LINE_BUDGET,
        search_matches: int = SEARCH_MATCH_BUDGET,
        run_lines: int = RUN_LINE_BUDGET,
        hide_answer_key: bool = True,
    ) -> None:
        self.root = root.resolve()
        self.truncation_note = truncation_note
        self.read_lines = read_lines
        self.search_matches = search_matches
        self.run_lines = run_lines
        self.counters = ToolCounters()
        self.evidence = 0
        self.writes = 0
        self.created: set[str] = set()
        self._repeats: Counter[tuple[str, str]] = Counter()
        self.hide_answer_key = hide_answer_key
        self.answer_key_reads = 0
        self.answer_key_mentions = 0
        self.answer_key_blocked = 0
        self.log: list[dict[str, Any]] = []
        self.finish_answer: str | None = None
        self._find = shutil.which("find") or "find"
        self.search_binary = search_binary()

    @property
    def search_backend(self) -> str:
        return Path(self.search_binary).name

    @staticmethod
    def _is_answer_key(relative: str) -> bool:
        return Path(relative).name in _ANSWER_KEY_MODULES

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        resolved = (
            candidate if candidate.is_absolute() else self.root / candidate
        ).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise PathEscape(raw)
        return resolved

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)) if path != self.root else "."

    def _git(self, *argv: str) -> str:
        done = subprocess.run(
            ["git", *argv],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        return done.stdout

    @property
    def tree_changed(self) -> bool:
        """Whether the worktree holds anything the grader would collect as a diff."""
        return bool(self._git("status", "--porcelain", "-uall").strip())

    def tree_summary(self) -> str:
        """`git diff --stat`, plus the paths git does not track yet.

        Handed back on every successful write rather than left for the model to ask
        for. `task_patch.md` already *instructs* a `git diff` after editing and the
        prescription was ignored — `run` calls went 7 to 0 between packs — while the
        three episodes that did produce a diff are the three that ran commands
        unprompted. An observation arrives whether or not the model thought to look.

        Untracked paths are listed separately because `git diff` cannot see them, and
        a new test file is the whole deliverable on two of the patch tasks.
        """
        stat = self._git("diff", "--stat").strip().splitlines()
        created = [
            line[3:]
            for line in self._git("status", "--porcelain", "-uall").splitlines()
            if line.startswith("??")
        ]
        if not stat and not created:
            return "git diff --stat: the working tree is unchanged."
        blocks = []
        if stat:
            shown = stat[:DIFFSTAT_LINE_BUDGET]
            if len(stat) > DIFFSTAT_LINE_BUDGET:
                shown.append(f"[... {len(stat) - DIFFSTAT_LINE_BUDGET} more lines ...]")
            blocks.append("git diff --stat:\n" + "\n".join(shown))
        if created:
            blocks.append("new files, not yet tracked by git:\n" + "\n".join(created))
        return "\n".join(blocks)

    def _within(self, line: str) -> str:
        """A `find` hit as a repo-relative path, or "" if it landed outside the tree."""
        try:
            return str(Path(line).relative_to(self.root))
        except ValueError:
            return ""

    def _truncated(
        self, body: str, *, shown: int, total: int, unit: str, path: str
    ) -> str:
        self.counters.truncated += 1
        note = self.truncation_note.format(
            shown=shown, total=total, unit=unit, path=path
        )
        return f"{body}\n{note}"

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = self._repeat_guard(name, arguments) or self._dispatch(name, arguments)
        if any(marker in result.text for marker in _ANSWER_KEY_MARKERS):
            self.answer_key_reads += 1
        elif any(module in result.text for module in _ANSWER_KEY_MODULES):
            self.answer_key_mentions += 1
        argument = str(arguments.get(_PRIMARY_ARGUMENT.get(name, ""), ""))
        # Relative, because the worktree prefix is a 90-character temp path and truncating
        # the whole thing to a fixed budget cut off the filename — the only part that says
        # what was read. Resolved on both sides rather than compared as strings: the model
        # is handed `/var/folders/...` from `tempfile.mkdtemp` while `self.root` is the
        # resolved `/private/var/folders/...`, so a prefix test on the raw text never
        # matched and every absolute-path call in the last full run logged as the same
        # 40-character temp directory with the filename cut off.
        if argument.startswith("/"):
            with contextlib.suppress(ValueError, OSError):
                argument = str(Path(argument).resolve().relative_to(self.root)) or "."
        self.log.append(
            {
                "name": name,
                "arg": argument[:120],
                "ok": result.ok,
                "evidence": result.evidence,
            }
        )
        return result

    def _repeat_guard(self, name: str, arguments: dict[str, Any]) -> ToolResult | None:
        """Break an identical read-only call that has already been answered twice.

        Greedy decoding makes this a hard loop rather than a tendency: the observation
        coming back is byte-identical, so the next turn's prompt differs only by that
        identical block and the model re-emits the same call. Measured on `high-01`,
        which issued `search minLength` **81 times** in 93 searches and spent 79 of its
        100 turns on it. The way out is to make the observation different, which is why
        this replaces the result rather than suppressing it silently.

        Only the read-only tools, and the counters reset on a write: `run pytest` after
        an edit is a repeat whose answer legitimately changes, and so is re-reading a
        file that has just been modified.
        """
        if name not in _REPEATABLE:
            return None
        key = (name, repr(sorted(arguments.items())))
        self._repeats[key] += 1
        if self._repeats[key] <= _REPEAT_LIMIT:
            return None
        self.counters.by_name[name] += 1
        self.counters.repeats_blocked += 1
        return ToolResult(
            f"You have already run this exact {name} {self._repeats[key] - 1} times "
            f"and the result has not changed; running it again will not change it. "
            f"Ask something different, read_file the file you want to see, or make the "
            f"edit the task asks for.",
            ok=False,
        )

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = {
            "list_files": self._list_files,
            "search": self._search,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "run": self._run,
        }.get(name)
        if handler is None:
            self.counters.unknown_tool += 1
            return ToolResult(
                f"No tool named {name!r}. The tools are: {', '.join(TOOL_NAMES)}.",
                ok=False,
            )
        self.counters.by_name[name] += 1
        try:
            result = handler(arguments)
        except PathEscape as exc:
            self.counters.path_escapes += 1
            self.counters.errors += 1
            return ToolResult(
                f"{exc.args[0]!r} is outside the repository. Use paths relative to its root.",
                ok=False,
            )
        except KeyError as exc:
            self.counters.missing_parameter += 1
            self.counters.errors += 1
            return ToolResult(f"{name} needs a {exc.args[0]} parameter.", ok=False)
        except (OSError, subprocess.SubprocessError) as exc:
            self.counters.errors += 1
            return ToolResult(f"{name} failed: {type(exc).__name__}: {exc}", ok=False)
        if result.evidence:
            self.evidence += 1
        return result

    def _list_files(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = str(arguments["pattern"])
        base = self._resolve(str(arguments.get("path") or "."))
        # A bare `*.py` almost always means "anywhere", and a 4 B model will not think
        # to write `**/*.py`. Anchored patterns keep their meaning.
        matcher = _glob_to_regex(pattern if "/" in pattern else f"**/{pattern}")
        found = subprocess.run(
            [self._find, str(base), "-type", "f", "-not", "-path", "*/.git/*"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        names = sorted(
            relative
            for line in found.stdout.splitlines()
            if line and (relative := self._within(line)) and matcher.match(relative)
        )
        if not names:
            return ToolResult(
                f"No files match {pattern!r} under {self._relative(base)}."
            )
        body = "\n".join(names[:LIST_ENTRY_BUDGET])
        if len(names) > LIST_ENTRY_BUDGET:
            return ToolResult(
                self._truncated(
                    body,
                    shown=LIST_ENTRY_BUDGET,
                    total=len(names),
                    unit="paths",
                    path=pattern,
                ),
                truncated=True,
            )
        return ToolResult(body)

    def _search(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments["query"])
        base = self._resolve(str(arguments.get("path") or "."))
        glob = arguments.get("glob")
        if self.search_backend == "rg":
            command = [
                self.search_binary,
                "--line-number",
                "--no-heading",
                "--color",
                "never",
            ]
            if glob:
                command += ["--glob", str(glob)]
        else:
            command = [self.search_binary, "-rn", "--exclude-dir=.git"]
            if glob:
                command += [f"--include={glob}"]
        command += ["-e", query, str(base)]
        found = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=300
        )
        # Exit 1 is "no matches" for both binaries and is an answer, not a failure.
        if found.returncode > 1:
            self.counters.errors += 1
            return ToolResult(f"search failed: {found.stderr.strip()[:300]}", ok=False)
        lines = []
        for line in found.stdout.splitlines():
            if not line:
                continue
            relative = line.replace(f"{self.root}/", "", 1)
            if self.hide_answer_key and self._is_answer_key(relative.split(":", 1)[0]):
                self.answer_key_blocked += 1
                continue
            lines.append(relative)
        if not lines:
            return ToolResult(f"No matches for {query!r}.", evidence=True)
        if len(lines) > self.search_matches:
            return ToolResult(
                self._truncated(
                    "\n".join(lines[: self.search_matches]),
                    shown=self.search_matches,
                    total=len(lines),
                    unit="matches",
                    path=query,
                ),
                evidence=True,
                truncated=True,
            )
        return ToolResult("\n".join(lines), evidence=True)

    def _read_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self._resolve(str(arguments["path"]))
        if self.hide_answer_key and self._is_answer_key(self._relative(path)):
            self.answer_key_blocked += 1
            return ToolResult(
                f"{self._relative(path)} is the evaluation harness's own module and is "
                f"not readable during a task. Answer from the repository itself.",
                ok=False,
            )
        if not path.is_file():
            self.counters.errors += 1
            return ToolResult(f"{self._relative(path)} is not a file.", ok=False)
        lines = path.read_text(errors="replace").splitlines()
        start = max(1, _int_or_none(arguments.get("start")) or 1)
        end = _int_or_none(arguments.get("end")) or len(lines)
        end = min(end, len(lines))
        window = lines[start - 1 : end]
        if not window:
            return ToolResult(
                f"{self._relative(path)} has {len(lines)} lines; nothing at {start}-{end}.",
                evidence=True,
            )
        truncated = len(window) > self.read_lines
        shown = window[: self.read_lines]
        body = "\n".join(
            f"{start + offset:>6}\t{text}" for offset, text in enumerate(shown)
        )
        if truncated:
            return ToolResult(
                self._truncated(
                    body,
                    shown=self.read_lines,
                    total=len(lines),
                    unit="lines",
                    path=self._relative(path),
                ),
                evidence=True,
                truncated=True,
            )
        return ToolResult(body or f"{self._relative(path)} is empty.", evidence=True)

    def _write_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self._resolve(str(arguments["path"]))
        content = str(arguments["content"])
        # Whole-file replacement is refused on a file this episode did not create, and
        # this is measured rather than cautious. Asked to *add* a test to
        # `tests/fake_mlx.py`, the model called `write_file` on it and replaced 856
        # lines with 19 — deleting the module every MLX test imports, along with the
        # docstring that states what the stand-in proves. One call turned `pytest`,
        # `ruff` and `mypy` from green to red, and the answer describing it was
        # otherwise honest. A 4 B model cannot reproduce a file it is editing, so the
        # tool that lets it try is a footgun; `edit_file` expresses the same intent
        # without the blast radius. Re-writing a file the episode created itself is
        # still allowed, because iterating on your own new test is not this failure.
        relative = self._relative(path)
        if path.is_file() and relative not in self.created:
            self.counters.denied += 1
            return ToolResult(
                f"{relative} already exists and write_file would replace all "
                f"{len(path.read_text(errors='replace').splitlines())} lines of it. "
                f"Use edit_file to change part of it, or write_file to a new path.",
                ok=False,
            )
        if not path.is_file():
            self.created.add(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.writes += 1
        self._repeats.clear()
        return ToolResult(
            f"Wrote {self._relative(path)}: {len(content.splitlines())} lines, "
            f"{len(content.encode())} bytes.\n\n{self.tree_summary()}"
        )

    def _edit_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self._resolve(str(arguments["path"]))
        old, new = str(arguments["old"]), str(arguments["new"])
        if not path.is_file():
            self.counters.errors += 1
            return ToolResult(f"{self._relative(path)} is not a file.", ok=False)
        body = path.read_text(errors="replace")
        hits = body.count(old)
        if hits == 0:
            self.counters.errors += 1
            return ToolResult(
                f"That exact text is not in {self._relative(path)}. Read it and copy the "
                "text you want to replace.",
                ok=False,
            )
        if hits > 1:
            self.counters.errors += 1
            return ToolResult(
                f"That text appears {hits} times in {self._relative(path)}. Include enough "
                "surrounding lines to make it unique.",
                ok=False,
            )
        path.write_text(body.replace(old, new, 1))
        self.writes += 1
        self._repeats.clear()
        return ToolResult(f"Edited {self._relative(path)}.\n\n{self.tree_summary()}")

    def _run(self, arguments: dict[str, Any]) -> ToolResult:
        raw = str(arguments["command"]).strip()
        try:
            argv = shlex.split(raw)
        except ValueError as exc:
            self.counters.errors += 1
            return ToolResult(f"Could not parse that command: {exc}", ok=False)
        if not argv:
            self.counters.errors += 1
            return ToolResult("No command given.", ok=False)

        head = Path(argv[0]).name
        if head in _AS_MODULE:
            command = [sys.executable, "-m", _AS_MODULE[head], *argv[1:]]
        elif head == "python3":
            command = [sys.executable, *argv[1:]]
        elif head == "git" and len(argv) > 1 and argv[1] in _ALLOWED_GIT:
            command = ["git", *argv[1:]]
        else:
            self.counters.denied += 1
            return ToolResult(
                f"{head!r} is not allowed. Allowed: pytest, ruff, mypy, python3, and "
                f"git {'/'.join(_ALLOWED_GIT)}.",
                ok=False,
            )

        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.root / "src")
        try:
            done = subprocess.run(
                command,
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=RUN_TIMEOUT_S,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            self.counters.errors += 1
            return ToolResult(f"{raw}: {type(exc).__name__}: {exc}", ok=False)

        lines = (done.stdout + done.stderr).strip().splitlines()
        header = f"exit {done.returncode}"
        if len(lines) <= self.run_lines:
            return ToolResult(f"{header}\n" + "\n".join(lines))
        half = self.run_lines // 2
        middle = f"[... {len(lines) - 2 * half} lines elided ...]"
        body = "\n".join([header, *lines[:half], middle, *lines[-half:]])
        self.counters.truncated += 1
        return ToolResult(body, truncated=True)
