"""Patch application and the repo's own checks (spec sec 7.2, 10.1).

Real git repositories rather than mocks, for the same reason the adapter tests
build them: every failure mode that matters here — a patch that will not apply, a
worktree that outlives its run, a linter that reads the wrong copy of a file —
lives in git's actual behaviour.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from orbit.config import EvalConfig
from orbit.eval.worktree import (
    WorktreeRunner,
    extract_diff,
    from_config,
    touched_files,
)

# A "suite" that passes only when the patch was applied, and a "linter" that fails
# on a marker. Both read files from their own working directory, which is what
# makes them able to tell the patched worktree apart from the source checkout.
PASSES_ONLY_WHEN_PATCHED = [
    sys.executable,
    "-c",
    "import sys; sys.exit(0 if open('mod.py').read().strip() == 'VALUE = 2' else 1)",
]
REJECTS_A_MARKER = [
    sys.executable,
    "-c",
    "import sys; sys.exit(1 if any('NOLINT' in open(p).read() for p in sys.argv[1:]) else 0)",
]

PATCH = (
    "diff --git a/mod.py b/mod.py\n"
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1 +1 @@\n"
    "-VALUE = 1\n"
    "+VALUE = 2\n"
)
UNLINTABLE_PATCH = (
    "diff --git a/mod.py b/mod.py\n"
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1 +1 @@\n"
    "-VALUE = 1\n"
    "+VALUE = 2  # NOLINT\n"
)
DOES_NOT_APPLY = (
    "diff --git a/mod.py b/mod.py\n"
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1 +1 @@\n"
    "-VALUE = 99\n"
    "+VALUE = 2\n"
)


def _git(repo, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "orbit@example.com")
    _git(r, "config", "user.name", "orbit")
    (r / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "initial")
    return r


@pytest.fixture
def runner(repo, tmp_path):
    return WorktreeRunner(
        repo,
        linters=[REJECTS_A_MARKER],
        test_command=PASSES_ONLY_WHEN_PATCHED,
        scratch_dir=tmp_path / "scratch",
        test_timeout_s=60.0,
    )


# --- pulling a patch out of a reply -----------------------------------------


def test_extract_diff_from_a_fenced_reply():
    reply = (
        f"Here is the change:\n\n```diff\n{PATCH}```\n\nIt keeps the existing style."
    )
    assert extract_diff(reply) == PATCH


def test_extract_diff_keeps_every_fence():
    """A reply with one fence per file is a patch in pieces, not rival patches."""
    other = PATCH.replace("mod.py", "other.py")
    reply = f"```diff\n{PATCH}```\nand\n```diff\n{other}```"
    out = extract_diff(reply)
    assert touched_files(out) == {"mod.py", "other.py"}


def test_extract_diff_from_a_bare_diff():
    assert extract_diff(PATCH).strip() == PATCH.strip()


def test_extract_diff_ignores_prose_and_unrelated_fences():
    assert extract_diff("I would change the retry count.") == ""
    assert extract_diff("```python\nVALUE = 2\n```") == ""
    assert extract_diff("") == ""


def test_extract_diff_survives_a_truncated_fence():
    """max_tokens can cut the closing fence off. The patch is still there."""
    assert "VALUE = 2" in extract_diff(f"sure:\n```diff\n{PATCH}")


# --- applying and checking ---------------------------------------------------


@pytest.mark.asyncio
async def test_a_patch_that_applies_runs_the_suite(runner):
    outcome = await runner.evaluate(PATCH)
    assert outcome.applied
    assert outcome.tests_passed is True
    assert outcome.lint_clean is True


@pytest.mark.asyncio
async def test_the_linter_reads_the_patched_worktree_not_the_checkout(runner, repo):
    """The point of the worktree.

    Linting the source checkout would score the repository, which is identical for
    every arm. This patch is clean on disk and dirty once applied.
    """
    outcome = await runner.evaluate(UNLINTABLE_PATCH)
    assert outcome.applied
    assert outcome.lint_clean is False
    assert "NOLINT" not in (repo / "mod.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_patch_that_does_not_apply_is_a_measured_failure(runner):
    outcome = await runner.evaluate(DOES_NOT_APPLY)
    assert not outcome.applied
    # Measured False, not unmeasured None: a patch that will not apply cannot pass.
    assert outcome.tests_passed is False
    assert "did not apply" in outcome.output


@pytest.mark.asyncio
async def test_a_reply_with_no_patch_cannot_have_passed_the_suite(runner, tmp_path):
    """Scoring it unmeasured would let a prose answer dodge the metric."""
    outcome = await runner.evaluate("")
    assert outcome.empty
    assert outcome.tests_passed is False

    untested = WorktreeRunner(runner.repo, scratch_dir=tmp_path / "scratch")
    assert (await untested.evaluate("")).tests_passed is None


@pytest.mark.asyncio
async def test_no_test_command_means_not_measured(repo, tmp_path):
    lint_only = WorktreeRunner(
        repo, linters=[REJECTS_A_MARKER], scratch_dir=tmp_path / "scratch"
    )
    outcome = await lint_only.evaluate(PATCH)
    assert outcome.applied
    assert outcome.tests_passed is None
    assert outcome.lint_clean is True


@pytest.mark.asyncio
async def test_a_missing_linter_is_unmeasured_rather_than_failed(repo, tmp_path):
    absent = WorktreeRunner(
        repo, linters=[["orbit-no-such-linter"]], scratch_dir=tmp_path / "scratch"
    )
    outcome = await absent.evaluate(PATCH)
    assert outcome.applied
    assert outcome.lint_clean is None


@pytest.mark.asyncio
async def test_the_checkout_is_untouched_and_no_worktree_survives(
    runner, repo, tmp_path
):
    await runner.evaluate(PATCH)
    assert (repo / "mod.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert _git(repo, "status", "--porcelain") == ""
    listed = _git(repo, "worktree", "list", "--porcelain")
    assert "scratch" not in listed
    assert not list((tmp_path / "scratch").glob("orbit-*"))


@pytest.mark.asyncio
async def test_a_test_command_that_times_out_fails_rather_than_hangs(repo, tmp_path):
    slow = WorktreeRunner(
        repo,
        test_command=[sys.executable, "-c", "import time; time.sleep(30)"],
        test_timeout_s=0.5,
        scratch_dir=tmp_path / "scratch",
    )
    outcome = await slow.evaluate(PATCH)
    assert outcome.tests_passed is False
    assert "timed out" in outcome.output


@pytest.mark.asyncio
async def test_an_unknown_base_revision_reports_rather_than_raises(runner):
    outcome = await runner.evaluate(PATCH, base_rev="deadbeefdeadbeefdeadbeefdeadbeef")
    assert not outcome.applied
    assert "worktree" in outcome.output


@pytest.mark.asyncio
async def test_verify_base_runs_the_suite_unpatched(runner):
    """The suite only passes when patched, so an honest base check must say so."""
    outcome = await runner.verify_base()
    assert outcome.applied
    assert outcome.tests_passed is False


# --- the Cascade adapter (sec 7.2) ------------------------------------------


@pytest.mark.asyncio
async def test_test_runner_reports_failure_for_a_failing_patch(repo, tmp_path):
    never_passes = WorktreeRunner(
        repo,
        test_command=[sys.executable, "-c", "raise SystemExit(1)"],
        scratch_dir=tmp_path / "scratch",
    )
    passed, output = await never_passes.as_test_runner()(f"```diff\n{PATCH}```")
    assert passed is False
    assert isinstance(output, str)


@pytest.mark.asyncio
async def test_test_runner_does_not_escalate_on_a_reply_with_no_patch(runner):
    passed, output = await runner.as_test_runner()("I would raise the retry count.")
    # True means "do not escalate". A prose answer has not produced a patch that
    # failed, and escalating costs a tier-1 review plus a regeneration.
    assert passed is True
    assert "nothing to test" in output


@pytest.mark.asyncio
async def test_apply_failure_only_escalates_when_asked(runner):
    passed, _ = await runner.as_test_runner()(DOES_NOT_APPLY)
    assert passed is True
    passed, _ = await runner.as_test_runner(escalate_on_apply_failure=True)(
        DOES_NOT_APPLY
    )
    assert passed is False


# --- config -----------------------------------------------------------------


def test_from_config_is_none_when_it_would_measure_nothing():
    assert from_config(EvalConfig()) is None


def test_from_config_builds_a_runner_when_a_command_is_declared(tmp_path):
    cfg = EvalConfig(test_command=["pytest", "-q"], repo=str(tmp_path))
    runner = from_config(cfg)
    assert runner is not None
    assert runner.measures_tests
    assert not runner.measures_lint
