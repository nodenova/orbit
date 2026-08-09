"""Applying a patch and running the repository's own checks (spec sec 7.2, 10.1).

Two callers want the same machine:

* **The merge eval (sec 10.1)** cannot report `test pass` or `convention
  conformance` without it — two of its five metrics, and without them
  `compare_arms` correctly refuses to call anything an M3 pass.
* **T2 failure escalation (sec 7.2)** cannot fire without it. `Cascade` takes a
  `test_runner`; with none, escalation returns immediately and the whole path is
  dead code in the served product.

Both need: take a patch, put it somewhere isolated, run the repo's own commands,
report what happened. Neither may touch the user's checkout — the eval runs
hundreds of patches, and the served path runs while somebody is working in that
tree — so everything happens in a detached `git worktree` under a scratch
directory, removed afterwards.

**Linting happens inside the patched worktree, not the source checkout.** Running
the linter over the repo's current files measures the repository, not the patch,
and would score every arm identically.

Running a repository's test command executes code from that repository. That is
what measuring a patch means and it is not something this module can sandbox away,
so it is opt-in: with no `test_command` configured `tests_passed` stays None —
"not measured", which is what `summarise` and `compare_arms` are built to
propagate — and T2 escalation stays dormant rather than escalating on a failure
nobody observed.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Same neutralisation as `adapters.gitwalk`: a customer with `diff.external` set
# would otherwise silently produce nonsense.
#
# `core.hooksPath` is pointed at a path that holds no hooks because `git worktree
# add` runs the repository's `post-checkout` hook, and this module's premise is
# that the *test command* is the one piece of opt-in code execution. Without this,
# a repo whose post-checkout installs dependencies runs it hundreds of times per
# eval — and it runs even when no `test_command` is configured at all, which is the
# configuration an operator chose precisely to avoid executing repository code.
_GIT = [
    "git",
    "-c",
    "core.quotepath=false",
    "-c",
    "diff.external=",
    "-c",
    "core.hooksPath=/dev/null",
    "--no-pager",
]

# Every git invocation here is bounded. `WorktreeRunner._lock` serialises the whole
# evaluation, so one `worktree add` blocked on a hook, an NFS stall or a
# credential prompt stalls every remaining case with no diagnostic at all.
_GIT_TIMEOUT_S = 120.0

_FENCE = re.compile(r"```[^\n`]*\n(.*?)(?:```|\Z)", re.DOTALL)
_DIFF_FILE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_UNIFIED_HEAD = re.compile(r"^--- \S[^\n]*\n\+\+\+ \S", re.MULTILINE)


# --- patch text -------------------------------------------------------------


def touched_files(diff: str) -> set[str]:
    return {m.group(2) for m in _DIFF_FILE.finditer(diff)}


def looks_like_diff(text: str) -> bool:
    return bool(_DIFF_FILE.search(text) or _UNIFIED_HEAD.search(text))


def _reject_option(value: str, what: str) -> str:
    """Refuse a caller-supplied revision or path that git would read as an option.

    `git log`/`git apply` accept `--output=<file>` and create-and-truncate it, so a
    `base_rev` arriving from `tandem.toml` is one hyphen away from writing a file of
    the caller's choosing. `--` separates revisions from paths but does *not* stop
    option parsing before it, and `--end-of-options` only exists from git 2.24.
    Rejecting the leading hyphen outright works on every version and costs nothing:
    no revision or path this module legitimately handles starts with one.
    """
    if value.startswith("-"):
        raise ValueError(
            f"refusing to pass {what} {value!r} to git: it would be read as an option"
        )
    return value


def extract_diff(text: str) -> str:
    """Pull the unified diff out of a model reply.

    Models wrap patches in prose and fences; `git apply` accepts neither. Fenced
    blocks win when there are any, and every diff-looking block is kept rather than
    just the first — a reply that emits one fence per file is a patch in pieces,
    not several competing patches.

    Returns "" when the reply carries no patch at all, which callers must treat as
    "nothing to test" rather than as a failure: a `code_change` turn that answers a
    question in prose has not produced a patch that failed.
    """
    if not text or not text.strip():
        return ""
    blocks = [
        b for b in (m.group(1) for m in _FENCE.finditer(text)) if looks_like_diff(b)
    ]
    if blocks:
        return "\n".join(b.strip("\n") for b in blocks) + "\n"
    # Unfenced: take from the first diff header to the end. Trailing prose after a
    # bare diff is rare and `git apply` ignores what it cannot parse at the tail.
    match = _DIFF_FILE.search(text) or _UNIFIED_HEAD.search(text)
    if match is None:
        return ""
    return text[match.start() :].strip("\n") + "\n"


# --- outcomes ---------------------------------------------------------------


@dataclass
class PatchOutcome:
    """What happened to one patch.

    `None` on `lint_clean` / `tests_passed` means *not measured* and never *passed*.
    The distinction is the whole reason the M3 gate can refuse to return a verdict.
    """

    applied: bool = False
    lint_clean: bool | None = None
    tests_passed: bool | None = None
    output: str = ""
    # The reply carried no patch. Distinct from a patch that failed to apply.
    empty: bool = False
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "lint_clean": self.lint_clean,
            "tests_passed": self.tests_passed,
            "empty": self.empty,
            "duration_s": round(self.duration_s, 2),
            "output": self.output,
        }


# text -> (tests passed, output). The shape `Cascade` and the merge eval both take.
TestRunner = Callable[[str], Awaitable[tuple[bool, str]]]


class _BoundedTail:
    """The last `limit` characters of a stream, discarded as they are read.

    `subprocess.run(capture_output=True)` buffers the *whole* output before anyone
    can truncate it. A test command or trainer that emits a progress line per step
    for six hours is bounded only by RAM, and the machine OOMs with no adapter, no
    report and no diagnostic — the exact run that was most expensive to lose.
    Truncating at read time bounds it by construction instead.
    """

    __slots__ = ("_buf", "dropped", "limit")

    def __init__(self, limit: int):
        self.limit = max(0, limit)
        self._buf = ""
        self.dropped = 0

    def feed(self, text: str) -> None:
        if not text:
            return
        self._buf += text
        excess = len(self._buf) - self.limit
        if excess > 0:
            self.dropped += excess
            self._buf = self._buf[excess:]

    def text(self) -> str:
        """Keep the end. A test runner puts its summary there."""
        if not self.dropped:
            return self._buf.strip()
        return f"[...{self.dropped} chars truncated...]\n" + self._buf.strip()


class WorktreeRunner:
    """Applies patches in throwaway git worktrees and runs the repo's checks."""

    def __init__(
        self,
        repo: str | Path,
        *,
        linters: Sequence[Sequence[str]] = (),
        test_command: Sequence[str] = (),
        setup_command: Sequence[str] = (),
        base_rev: str = "HEAD",
        test_timeout_s: float = 600.0,
        lint_timeout_s: float = 120.0,
        setup_timeout_s: float = 600.0,
        git_timeout_s: float = _GIT_TIMEOUT_S,
        scratch_dir: str | Path | None = None,
        output_limit: int = 8_000,
    ):
        self.repo = Path(repo).resolve()
        self.linters = [list(l) for l in linters if l]
        self.test_command = list(test_command)
        self.setup_command = list(setup_command)
        self.base_rev = _reject_option(base_rev, "base revision")
        self.test_timeout_s = test_timeout_s
        self.lint_timeout_s = lint_timeout_s
        self.setup_timeout_s = setup_timeout_s
        self.git_timeout_s = git_timeout_s
        # Resolved against tandem's working directory, never against the repo under
        # test. A scratch tree inside that repo is a directory the repo's own test
        # runner will happily collect and its linters will happily lint — a full
        # copy of the suite, discovered recursively, once per case.
        #
        # Saying that is not enough to make it true: a *relative* default resolves
        # against whatever the process's cwd happens to be, and for `tandem eval
        # merge --repo .` that cwd is the repo under test. So the default is an
        # absolute path outside any repository, and a configured one that still
        # lands inside `repo` is moved out rather than honoured — the alternative is
        # a customer watching `?? var/` for hours while their own pytest collects a
        # full second checkout of their suite once per case.
        self.scratch = self._safe_scratch(scratch_dir)
        self.output_limit = output_limit
        # One patch at a time. Two suites racing on one machine measure each other's
        # contention, and the latency numbers are published (sec 10.5).
        self._lock = asyncio.Lock()

    def _safe_scratch(self, scratch_dir: str | Path | None) -> Path:
        """A scratch root that is guaranteed to sit outside the repo under test."""
        default = Path(tempfile.gettempdir()) / "tandem-worktrees"
        self.scratch_relocated_from: str = ""
        if scratch_dir is None:
            return default
        candidate = Path(scratch_dir).expanduser().resolve()
        if candidate == self.repo or candidate.is_relative_to(self.repo):
            self.scratch_relocated_from = str(candidate)
            return default
        return candidate

    # --- reporting ----------------------------------------------------------

    @property
    def measures_tests(self) -> bool:
        return bool(self.test_command)

    @property
    def measures_lint(self) -> bool:
        return bool(self.linters)

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "enabled": True,
            "repo": str(self.repo),
            "base_rev": self.base_rev,
            "linters": [" ".join(l) for l in self.linters],
            "test_command": " ".join(self.test_command),
            "setup_command": " ".join(self.setup_command),
            "scratch_dir": str(self.scratch),
            "measures": {"tests": self.measures_tests, "lint": self.measures_lint},
        }
        if self.scratch_relocated_from:
            # Named, not silent: the operator configured a path and did not get it.
            out["scratch_relocated_from"] = self.scratch_relocated_from
            out["scratch_relocated_because"] = (
                "the configured worktree directory is inside the repository under "
                "test; the repo's own test runner and linters would collect it"
            )
        return out

    # --- entry points -------------------------------------------------------

    async def evaluate(self, diff: str, *, base_rev: str | None = None) -> PatchOutcome:
        """Apply `diff` to a fresh worktree and run the configured checks."""
        if not diff or not diff.strip():
            # A reply with no patch has not passed the suite, and it is not
            # convention-clean either. Leaving either unmeasured would let an arm
            # that answers in prose skip the metric it was about to lose — and
            # `convention_rate` is an *average over the measured cases*, so an arm
            # that only produces a patch three times in a hundred would otherwise be
            # scored on those three and could beat an arm scored on all hundred. The
            # served path never reaches here — the `as_test_runner` adapter takes
            # "no patch" as "nothing to test", which is a different question with a
            # different answer.
            return PatchOutcome(
                empty=True,
                tests_passed=False if self.measures_tests else None,
                lint_clean=False if self.measures_lint else None,
                output="reply carried no patch",
            )
        async with self._lock:
            return await asyncio.to_thread(
                self._evaluate, diff, base_rev or self.base_rev
            )

    async def verify_base(self, *, base_rev: str | None = None) -> PatchOutcome:
        """Run the checks on an unpatched worktree.

        A repository whose suite already fails at the base makes `test_pass_rate`
        meaningless — every arm scores zero and the metric silently stops
        discriminating. Worth one run before an eval that takes hours.
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._evaluate, None, base_rev or self.base_rev
            )

    def as_test_runner(
        self, *, base_rev: str | None = None, escalate_on_apply_failure: bool = False
    ) -> TestRunner:
        """Adapt to the `Cascade.test_runner` shape (sec 7.2).

        Returning True means "do not escalate", so everything this runner did not
        actually observe returns True: a reply with no patch, a patch that could not
        be applied when the caller has not opted into treating that as a failure, a
        host with no test command. Escalation costs a tier-1 review plus a tier-0
        regeneration, and spending that on a failure nobody saw is theatre.
        """

        async def run(text: str) -> tuple[bool, str]:
            diff = extract_diff(text)
            if not diff:
                return True, "no patch in the reply; nothing to test"
            outcome = await self.evaluate(diff, base_rev=base_rev)
            if not outcome.applied:
                return (not escalate_on_apply_failure), outcome.output
            if outcome.tests_passed is None:
                return True, outcome.output
            return outcome.tests_passed, outcome.output

        return run

    # --- the work -----------------------------------------------------------

    def _evaluate(self, diff: str | None, base_rev: str) -> PatchOutcome:
        t0 = time.perf_counter()
        out = PatchOutcome()
        # 0o700: the default scratch root lives under the system temp directory,
        # which on a shared machine is world-readable, and a worktree is a full copy
        # of the customer's source.
        self.scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
        work = self.scratch / f"tandem-{uuid.uuid4().hex[:12]}"
        patch_file = work.with_suffix(".patch")

        added, detail = self._add_worktree(work, base_rev)
        if not added:
            # `worktree add` can fail after creating the directory; leaving it
            # behind would make the next run's `git worktree list` wrong.
            self._remove_worktree(work)
            out.output = detail
            out.duration_s = time.perf_counter() - t0
            return out

        sections: list[str] = []
        try:
            if diff is None:
                out.applied = True
            else:
                patch_file.write_text(diff, encoding="utf-8")
                out.applied, detail = self._apply(work, patch_file)
                if not out.applied:
                    # A patch that does not apply cannot pass the suite, and it
                    # cannot conform to the repository's conventions either. Both
                    # are a measured False, not an unmeasured None — but only where
                    # a suite or a linter exists to have failed. Scoring the tests
                    # and leaving conventions unmeasured is what let an arm with a
                    # 3% apply rate win `convention_rate` against an arm measured on
                    # every case.
                    out.tests_passed = False if self.measures_tests else None
                    out.lint_clean = False if self.measures_lint else None
                    out.output = detail
                    return out

            if self.setup_command:
                code, text = self._run(self.setup_command, work, self.setup_timeout_s)
                if code != 0:
                    # Both metrics, on the same denominator: a setup command that
                    # fails under this patch is a failure of this patch until
                    # `verify_base` says the harness was already broken.
                    out.tests_passed = False if self.measures_tests else None
                    out.lint_clean = False if self.measures_lint else None
                    out.output = f"setup command failed (exit {code}):\n{text}"
                    return out

            if self.linters and diff is not None:
                out.lint_clean, text = self._lint(work, diff)
                if out.lint_clean is False:
                    sections.append(text)

            if self.test_command:
                code, text = self._run(self.test_command, work, self.test_timeout_s)
                out.tests_passed = code == 0
                sections.append(text)
            return out
        finally:
            out.output = out.output or "\n\n".join(s for s in sections if s)
            out.duration_s = time.perf_counter() - t0
            patch_file.unlink(missing_ok=True)
            self._remove_worktree(work)

    def _add_worktree(self, work: Path, base_rev: str) -> tuple[bool, str]:
        try:
            base_rev = _reject_option(base_rev, "base revision")
        except ValueError as exc:
            return False, str(exc)
        proc = self._git(
            self.repo, "worktree", "add", "--detach", "--force", str(work), base_rev
        )
        if proc.returncode != 0:
            return False, (
                f"could not create a worktree at {base_rev!r} in {self.repo}: "
                f"{self._tail(proc.stderr)}"
            )
        return True, ""

    def _remove_worktree(self, work: Path) -> None:
        self._git(self.repo, "worktree", "remove", "--force", str(work))
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        # Removal can leave administrative files behind when the directory went
        # first, so the leftovers are cleaned up here — by path, for *this* worktree.
        #
        # NOT `git worktree prune`. Prune takes no path: it walks the whole registry
        # of the repository it is pointed at, which is the **user's** repository, and
        # deregisters every entry it currently considers unreachable. A colleague's
        # worktree on an unmounted volume is exactly that. Its files survive the
        # deregistration but its index and HEAD do not, so `git status` there answers
        # `fatal: not a git repository` after remount and staged work is
        # unrecoverable until somebody knows to run `git worktree repair`. This
        # function runs once per patch — hundreds of times in one eval — and said
        # nothing about it in the output.
        self._remove_worktree_entry(work)

    def _remove_worktree_entry(self, work: Path) -> None:
        """Delete the administrative directory git kept for `work`, and only that.

        Git records each linked worktree under `<git-common-dir>/worktrees/<name>`
        with a `gitdir` file naming the worktree's own `.git`. Matching on that path
        is how "prune tandem's leftover" is expressed without touching an entry
        tandem did not create.
        """
        proc = self._git(self.repo, "rev-parse", "--git-common-dir")
        if proc.returncode != 0:
            return
        common = Path(proc.stdout.strip() or ".git")
        if not common.is_absolute():
            common = (self.repo / common).resolve()
        admin_root = common / "worktrees"
        if not admin_root.is_dir():
            return
        mine = {str(work), str(work / ".git")}
        for entry in admin_root.iterdir():
            try:
                recorded = (entry / "gitdir").read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            if recorded.strip() in mine:
                shutil.rmtree(entry, ignore_errors=True)

    def _apply(self, work: Path, patch_file: Path) -> tuple[bool, str]:
        """Three attempts, weakest assumption last.

        Plain first. `--3way` next, which recovers from context drift by falling
        back to the blobs the index lines name — usually available, because the
        patch is against a commit in this very repository. `-p0` last, for a model
        that emitted paths without the `a/` `b/` prefixes.
        """
        last = ""
        for extra in ([], ["--3way"], ["-p0"]):
            # `--` before the patch file: `git apply` takes options anywhere before
            # it, and the separator is what makes a path that starts with a hyphen
            # structurally unable to become one. The name is tandem's own today; the
            # separator costs nothing and removes the question.
            proc = self._git(
                work, "apply", "--whitespace=nowarn", *extra, "--", str(patch_file)
            )
            if proc.returncode == 0:
                return True, ""
            last = proc.stderr.strip() or proc.stdout.strip()
        return False, f"patch did not apply: {self._tail(last)}"

    def _lint(self, work: Path, diff: str) -> tuple[bool | None, str]:
        """Run each linter over the files the patch touches, inside the worktree.

        None when nothing was measurable — no linter, no touched file that exists,
        or a linter that could not be run. An honest "not measured" rather than a
        free pass that would flatter every arm equally.

        **The file names come from the model's own diff, so they are validated here
        rather than trusted.** Until now the only thing keeping `../../etc/passwd`
        or `--fix` out of the linter's argv was that `git apply` refuses to write
        outside the worktree, so no such file would exist to be passed on — a real
        defence, but an implicit one belonging to a different program. It breaks the
        moment a linter is configured as a formatter (`black` without `--check`
        writes), or a path that is *not* created by the patch happens to exist in
        the tree. `_contained` makes the requirement local and explicit.
        """
        files = sorted(f for f in touched_files(diff) if self._contained(work, f))
        if not files:
            return None, ""
        problems: list[str] = []
        measured = False
        for linter in self.linters:
            code, text = self._run([*linter, *files], work, self.lint_timeout_s)
            if code == 127:
                # The linter is not installed here. Not the patch's fault.
                continue
            measured = True
            if code != 0:
                problems.append(f"$ {' '.join(linter)}\n{text}")
        if not measured:
            return None, ""
        return (not problems), "\n\n".join(problems)

    def _contained(self, work: Path, relpath: str) -> bool:
        """Is `relpath` a real file inside the worktree, safe to put in an argv?"""
        if not relpath or relpath.startswith("-") or Path(relpath).is_absolute():
            return False
        candidate = work / relpath
        try:
            resolved = candidate.resolve()
        except OSError:
            return False
        # `resolve()` follows symlinks, so a patch that adds `x -> /etc` and then
        # names `x/passwd` is caught here and not by the linter's own idea of a path.
        if not (resolved == work.resolve() or resolved.is_relative_to(work.resolve())):
            return False
        return candidate.is_file()

    def _run(self, cmd: Sequence[str], cwd: Path, timeout: float) -> tuple[int, str]:
        """Run a command, keeping only the tail of what it writes.

        Output is drained on a reader thread and truncated as it arrives rather than
        after the fact, so a command that never stops talking costs a fixed number
        of bytes instead of the machine.
        """
        try:
            proc = subprocess.Popen(
                list(cmd),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            # 127 is "command not found" by convention, and `_lint` reads it to tell
            # a missing linter apart from a failing one.
            return 127, f"could not run `{' '.join(cmd)}`: {exc}"

        tail = _BoundedTail(self.output_limit)
        reader = threading.Thread(target=_drain, args=(proc.stdout, tail), daemon=True)
        reader.start()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            reader.join(timeout=5.0)
            note = f"`{' '.join(cmd)}` timed out after {timeout:.0f}s"
            captured = tail.text()
            return 124, f"{note}\n{captured}" if captured else note
        reader.join(timeout=10.0)
        return code, tail.text()

    def _git(
        self, cwd: Path, *args: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run git, bounded in time and pinned to UTF-8.

        `text=True` alone decodes with the *process locale*: under `LC_ALL=C` the
        child encoding is ASCII and a non-ASCII path or commit subject comes back
        mangled, differently on a workstation and on a CI runner.
        """
        try:
            return subprocess.run(
                [*_GIT, "-C", str(cwd), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.git_timeout_s if timeout is None else timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            limit = self.git_timeout_s if timeout is None else timeout
            return subprocess.CompletedProcess(
                args=list(args),
                returncode=124,
                stdout="",
                stderr=f"`git {' '.join(args[:3])}` timed out after {limit:.0f}s",
            )
        except OSError as exc:
            return subprocess.CompletedProcess(
                args=list(args),
                returncode=127,
                stdout="",
                stderr=f"could not run git: {exc}",
            )

    def _tail(self, text: str) -> str:
        """Keep the end. A test runner puts its summary there."""
        text = text.strip()
        if len(text) <= self.output_limit:
            return text
        return "[...truncated...]\n" + text[-self.output_limit :]


def _drain(stream: Any, tail: _BoundedTail) -> None:
    """Read a pipe to EOF, keeping only the tail. Never raises into the thread."""
    try:
        with stream:
            while True:
                chunk = stream.read(65_536)
                if not chunk:
                    return
                tail.feed(chunk)
    except (OSError, ValueError):
        return


def from_config(cfg: Any, *, repo: str | Path | None = None) -> WorktreeRunner | None:
    """Build a runner from an `EvalConfig`, or None when it would measure nothing.

    None rather than a runner that runs no commands: `Cascade` reads
    `test_runner is None` as "this host cannot test", and handing it a runner that
    always answers "passed" would suppress escalation while looking wired up.
    """
    if not cfg.test_command and not cfg.linters:
        return None
    return WorktreeRunner(
        repo if repo is not None else cfg.repo,
        linters=cfg.linters,
        test_command=cfg.test_command,
        setup_command=cfg.setup_command,
        base_rev=cfg.base_rev,
        test_timeout_s=cfg.test_timeout_s,
        lint_timeout_s=cfg.lint_timeout_s,
        setup_timeout_s=cfg.setup_timeout_s,
        scratch_dir=cfg.worktree_dir,
    )
