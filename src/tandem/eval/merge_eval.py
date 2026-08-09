"""Repo-held-out merge eval (spec sec 10.1).

**The primary metric, and the one that decides whether there is a product.**

Benchmarks are the wrong instrument. The thesis is about merge quality — METR's
standing finding is that ~half of SWE-bench-passing PRs would not be merged by
maintainers, a ~24 pp gap — so the eval measures merge quality on the customer's own
repository, not a leaderboard.

Hold out the most recent K merged PRs (never used in training). For each: give the
task description and parent-commit context, generate a patch, score five ways.

    test pass              repo's own test suite passes
    convention conformance linter/formatter clean; matches import, naming,
                           error-handling patterns
    diff proximity         normalised edit distance to the merged diff
    blast radius           files and lines touched vs the merged diff
    review-comment proxy   would this have drawn a review comment?

Report four bars: base, +A1, +A1+A2, full cascade with tier-1 rerank. **If the
adapter doesn't beat the base model on the customer's own repo, there is no
product** — and this is what says so, at M3, week 8, before tier 1 is built.

Methodology reference: RepoPeftBench (arXiv:2606.06492), 604 Python repos. Use its
shape; substitute customer repos.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Explicit re-export form (`as`): mypy runs with `no_implicit_reexport`, under
# which a plain import makes these private to this module — see the note below on
# why callers import them from here.
from tandem.eval.worktree import WorktreeRunner
from tandem.eval.worktree import extract_diff as extract_diff  # noqa: PLC0414
from tandem.eval.worktree import touched_files as touched_files  # noqa: PLC0414
from tandem.types import GenRequest, Message, Role, Sampling

_HUNK_LINE = re.compile(r"^[+-](?![+-])", re.MULTILINE)


@dataclass
class EvalCase:
    """One held-out merged change."""

    sha: str
    prompt: str
    reference_diff: str
    ts: int = 0
    # Revision the generated patch is applied to. Empty means the change's own
    # first parent, which is what "generate this change from the state before it"
    # means and what extraction walked to build the case.
    base_rev: str = ""

    def base(self) -> str:
        return self.base_rev or f"{self.sha}^1"


@dataclass
class CaseScore:
    sha: str
    arm: str
    test_pass: bool | None = None
    convention_clean: bool | None = None
    diff_proximity: float = 0.0
    blast_radius_files: float = 0.0
    blast_radius_lines: float = 0.0
    review_comment_proxy: float | None = None
    latency_s: float = 0.0
    generated_diff: str = ""
    # None when nothing tried to apply the patch. False is a real failure: an arm
    # whose patches do not apply is not doing well, and averaging that away with
    # the arms that were never tested would hide it.
    applied: bool | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sha": self.sha,
            "arm": self.arm,
            "applied": self.applied,
            "test_pass": self.test_pass,
            "convention_clean": self.convention_clean,
            "diff_proximity": round(self.diff_proximity, 4),
            "blast_radius_files": round(self.blast_radius_files, 3),
            "blast_radius_lines": round(self.blast_radius_lines, 3),
            "review_comment_proxy": self.review_comment_proxy,
            "latency_s": round(self.latency_s, 2),
            "error": self.error,
        }


# --- metrics ----------------------------------------------------------------
#
# `touched_files` and `extract_diff` live in `worktree` and are re-exported here:
# the runner needs them to decide what to lint, and the metrics need them to score,
# and one definition of "which files does this patch touch" is the only version of
# that question worth having.


def changed_line_count(diff: str) -> int:
    return len(_HUNK_LINE.findall(diff))


def changed_line_set(diff: str) -> set[str]:
    out: set[str] = set()
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "diff --git ", "index ", "@@")):
            continue
        if line and line[0] in "+-":
            body = line[1:].strip()
            if body:
                out.add(line[0] + body)
    return out


def diff_proximity(generated: str, reference: str) -> float:
    """1.0 = identical change, 0.0 = disjoint.

    Jaccard over changed lines rather than character edit distance: two patches that
    make the same edits at different offsets are the same patch, and a character
    metric would score that as distant. Also linear, which matters when the eval
    runs over hundreds of cases.
    """
    g, r = changed_line_set(generated), changed_line_set(reference)
    if not g and not r:
        return 1.0
    if not g or not r:
        return 0.0
    return len(g & r) / len(g | r)


def blast_radius(generated: str, reference: str) -> tuple[float, float]:
    """Files and lines touched, as a ratio to the merged diff.

    1.0 is the reference's own footprint. Above 1.0 the patch is broader than what
    was merged — the single most common reason a working patch draws review
    comments — and below 1.0 it may be incomplete. Reported as a ratio rather than
    a score because the direction of the error matters and averaging it away would
    hide it.
    """
    gf, rf = len(touched_files(generated)), len(touched_files(reference))
    gl, rl = changed_line_count(generated), changed_line_count(reference)
    return (gf / rf if rf else float(gf)), (gl / rl if rl else float(gl))


# --- the harness ------------------------------------------------------------

# (prompt, adapter) -> generated diff. Each arm supplies one.
Generator = Callable[[GenRequest], Awaitable[str]]
# (case, diff) -> probability this patch would draw a review comment, or None for
# "this one was not scored" — a human pass that covered forty of sixty cases has
# measured forty, and rounding the rest to zero would be a fabrication.
ReviewProxy = Callable[["EvalCase", str], Awaitable[float | None]]

# How much of a review-comment probability each part of a tier-1 verdict accounts
# for. Calibrated by argument rather than by data, and it should be recalibrated
# against real review history once A2 exists — which is the point of keeping the
# numbers in one visible table instead of inline.
_VERDICT_PRIOR = {"accept": 0.1, "revise": 0.6, "reject": 0.9}
_SEVERITY_WEIGHT = {"blocking": 0.4, "major": 0.2, "minor": 0.05}


def tier1_review_proxy(
    verifier: Any, *, conventions: str = "", seed: int = 0
) -> ReviewProxy:
    """Score "would this draw a review comment?" with the tier-1 verifier.

    **This proxy is biased in favour of the cascade arm and must be read that
    way.** The cascade arm uses tier 1 to pick its candidate and is then scored by
    tier 1, so a win it records on this metric is partly the verifier agreeing with
    itself. The tier-0 arms (base, +A1, +A1+A2) are scored by a model that had no
    hand in producing them, so the comparison that decides M3 — base against +A1 —
    is unaffected. Prefer real review history (`--review-proxy file:...`, and A2
    once it exists) wherever it is available.

    A failed verdict scores None rather than a neutral 0.5: the verifier declining
    to answer is not evidence that the patch is middling.
    """

    async def proxy(case: EvalCase, diff: str) -> float | None:
        if not diff.strip():
            # No patch at all reliably draws a comment. Not a verifier call.
            return 1.0
        verdict = await verifier.review(
            diff, case.prompt, conventions=conventions, seed=seed
        )
        if not verdict.ok:
            return None
        score = _VERDICT_PRIOR.get(str(verdict.data.get("verdict", "revise")), 0.6)
        for issue in verdict.data.get("issues", []) or []:
            score += _SEVERITY_WEIGHT.get(str(issue.get("severity", "minor")), 0.05)
        return min(1.0, round(score, 4))

    return proxy


def scored_review_proxy(scores: dict[str, float]) -> ReviewProxy:
    """Replay review-comment scores from a file, keyed by the case's commit sha.

    This is the human pass, and the A2 pass after it: run the eval once to produce
    `per_case` with every generated diff in it, score them outside this process,
    and feed the scores back. Cases with no score stay unmeasured.
    """

    async def proxy(case: EvalCase, _diff: str) -> float | None:
        value = scores.get(case.sha)
        return None if value is None else float(value)

    return proxy


@dataclass
class Arm:
    """One configuration under test.

    The four bars the spec asks for: base, +A1, +A1+A2, full cascade.
    """

    name: str
    generate: Generator
    adapter: str | None = None


@dataclass
class ArmSummary:
    arm: str
    n: int = 0
    # Not one of the five gate metrics: it is the diagnostic that says whether the
    # other two worktree metrics mean anything. A 20% apply rate makes
    # `test_pass_rate` a statement about eighty percent nothing.
    apply_rate: float | None = None
    test_pass_rate: float | None = None
    convention_rate: float | None = None
    mean_proximity: float = 0.0
    median_proximity: float = 0.0
    mean_blast_files: float = 0.0
    mean_blast_lines: float = 0.0
    # Mean of the *per-case* blast accuracy. Held separately from the two means
    # above because it cannot be recovered from them: see `_blast_accuracy`.
    blast_accuracy: float | None = None
    mean_review_proxy: float | None = None
    mean_latency_s: float = 0.0
    errors: int = 0
    # How many cases each rate was actually computed over. A rate is a fraction and
    # a fraction hides its denominator, which is how an arm scored on three cases
    # came to be compared against an arm scored on a hundred. `None` means "this
    # summary was not built by `summarise`" — the hand-assembled ones in the tests —
    # and switches the coverage check off rather than failing them all closed.
    applied_n: int | None = None
    tested_n: int | None = None
    linted_n: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "n": self.n,
            "apply_rate": self.apply_rate,
            "applied_n": self.applied_n,
            "test_pass_rate": self.test_pass_rate,
            "tested_n": self.tested_n,
            "convention_rate": self.convention_rate,
            "linted_n": self.linted_n,
            "mean_proximity": round(self.mean_proximity, 4),
            "median_proximity": round(self.median_proximity, 4),
            "mean_blast_files": round(self.mean_blast_files, 3),
            "mean_blast_lines": round(self.mean_blast_lines, 3),
            "blast_accuracy": _round(self.blast_accuracy),
            "mean_review_proxy": self.mean_review_proxy,
            "mean_latency_s": round(self.mean_latency_s, 2),
            "errors": self.errors,
        }


# The five metrics the M3 gate counts (sec 11): A1 must beat base on >=3 of 5.
METRIC_KEYS = (
    "test_pass_rate",
    "convention_rate",
    "mean_proximity",
    "blast_radius_accuracy",
    "review_proxy",
)


async def run_arm(
    arm: Arm,
    cases: Sequence[EvalCase],
    *,
    runner: WorktreeRunner | None = None,
    review_proxy: ReviewProxy | None = None,
    max_tokens: int = 2048,
) -> list[CaseScore]:
    scores: list[CaseScore] = []
    for case in cases:
        score = CaseScore(sha=case.sha, arm=arm.name)
        req = GenRequest(
            messages=[Message(role=Role.USER, content=case.prompt)],
            adapter=arm.adapter,
            # Greedy. The eval measures the model, not the sampler's variance, and a
            # temperature here would make two runs of the same arm disagree.
            sampling=Sampling(
                temperature=0.0, top_p=1.0, seed=0, max_tokens=max_tokens
            ),
        )
        t0 = time.perf_counter()
        try:
            raw = await arm.generate(req)
        except Exception as exc:  # noqa: BLE001 - one bad case must not void the run
            score.error = str(exc)
            score.latency_s = time.perf_counter() - t0
            scores.append(score)
            continue
        score.latency_s = time.perf_counter() - t0

        # Score the patch, not the reply it arrived in. A model that explains its
        # change in prose around a fenced diff has produced the same patch as one
        # that emitted it bare, and the prose lines would otherwise count as
        # changed lines against the reference.
        generated = extract_diff(raw)
        score.generated_diff = generated

        score.diff_proximity = diff_proximity(generated, case.reference_diff)
        score.blast_radius_files, score.blast_radius_lines = blast_radius(
            generated, case.reference_diff
        )
        if runner is not None:
            outcome = await runner.evaluate(generated, base_rev=case.base())
            score.applied = outcome.applied
            score.convention_clean = outcome.lint_clean
            score.test_pass = outcome.tests_passed
        if review_proxy is not None:
            score.review_comment_proxy = await review_proxy(case, generated)
        scores.append(score)
    return scores


def summarise(arm_name: str, scores: Sequence[CaseScore]) -> ArmSummary:
    valid = [s for s in scores if not s.error]
    summary = ArmSummary(arm=arm_name, n=len(valid), errors=len(scores) - len(valid))
    if not valid:
        return summary

    applied = [s for s in valid if s.applied is not None]
    summary.applied_n = len(applied)
    if applied:
        summary.apply_rate = round(
            sum(1 for s in applied if s.applied) / len(applied), 4
        )
    tested = [s for s in valid if s.test_pass is not None]
    summary.tested_n = len(tested)
    if tested:
        summary.test_pass_rate = round(
            sum(1 for s in tested if s.test_pass) / len(tested), 4
        )
    linted = [s for s in valid if s.convention_clean is not None]
    summary.linted_n = len(linted)
    if linted:
        summary.convention_rate = round(
            sum(1 for s in linted if s.convention_clean) / len(linted), 4
        )
    proxied = [
        s.review_comment_proxy for s in valid if s.review_comment_proxy is not None
    ]
    if proxied:
        summary.mean_review_proxy = round(sum(proxied) / len(proxied), 4)

    prox = [s.diff_proximity for s in valid]
    summary.mean_proximity = sum(prox) / len(prox)
    summary.median_proximity = statistics.median(prox)
    summary.mean_blast_files = sum(s.blast_radius_files for s in valid) / len(valid)
    summary.mean_blast_lines = sum(s.blast_radius_lines for s in valid) / len(valid)
    # Score each case, then average. Averaging the *ratios* first and taking the
    # distance afterwards is what the metric's own docstring forbids, and it is not
    # a rounding-level difference: an arm that touches nothing on half the cases and
    # twice too much on the other half averages to exactly 1.0 and scores a perfect
    # 1.0000, beating an arm that is within 10% on every single case. Since
    # blast radius is one of only two metrics always measured, that is a free win on
    # a fifth of the M3 gate available to the arm with the *least* consistent
    # footprint.
    summary.blast_accuracy = sum(
        case_blast_accuracy(s.blast_radius_files, s.blast_radius_lines) for s in valid
    ) / len(valid)
    summary.mean_latency_s = sum(s.latency_s for s in valid) / len(valid)
    return summary


def case_blast_accuracy(files_ratio: float, lines_ratio: float) -> float:
    """Closeness of one patch's blast radius to the merged diff's own footprint.

    Distance from 1.0 in either direction, so a patch that touches twice as much and
    one that touches half as much are both penalised — averaging the raw ratio would
    let those cancel.
    """
    err = abs(files_ratio - 1.0) + abs(lines_ratio - 1.0)
    return 1.0 / (1.0 + err)


def _blast_accuracy(summary: ArmSummary) -> float:
    """The arm's blast-radius accuracy, per case where that is available.

    `summarise` records the per-case mean, which is the honest number. The fallback
    on the arm's mean ratios exists only for an `ArmSummary` assembled by hand — it
    is the cancelling average this metric is defined to avoid, and it cannot be
    recovered from the two means, so it is used and not preferred.
    """
    if summary.blast_accuracy is not None:
        return summary.blast_accuracy
    return case_blast_accuracy(summary.mean_blast_files, summary.mean_blast_lines)


# A rate computed over fewer than this fraction of an arm's cases is not a
# statement about the arm. It is reported (the summary carries both the rate and
# its denominator) but it is not offered to `compare_arms`, because the comparison
# is between arms and there is no reason to believe the minority of cases one arm
# managed to produce a patch for is the same population as the cases the other was
# scored on. Set at a half rather than tuned: the failure it exists to stop is an
# arm scoring 1.0 on three of a hundred cases.
_MIN_RATE_COVERAGE = 0.5


def _covered(rate: float | None, denominator: int | None, n: int) -> float | None:
    if rate is None or denominator is None or n <= 0:
        return rate
    return rate if denominator >= _MIN_RATE_COVERAGE * n else None


def comparable_metrics(summary: ArmSummary) -> dict[str, float | None]:
    return {
        "test_pass_rate": _covered(summary.test_pass_rate, summary.tested_n, summary.n),
        "convention_rate": _covered(
            summary.convention_rate, summary.linted_n, summary.n
        ),
        "mean_proximity": summary.mean_proximity,
        "blast_radius_accuracy": _blast_accuracy(summary),
        # Lower is better for the raw proxy (probability of drawing a comment), so
        # it is inverted here to make "higher is better" uniform across metrics.
        "review_proxy": (
            None
            if summary.mean_review_proxy is None
            else 1.0 - summary.mean_review_proxy
        ),
    }


def compare_arms(baseline: ArmSummary, candidate: ArmSummary) -> dict[str, Any]:
    """M3 gate (sec 11): A1 beats base on >=3 of 5 merge-eval metrics."""
    base_m = comparable_metrics(baseline)
    cand_m = comparable_metrics(candidate)
    wins: list[str] = []
    losses: list[str] = []
    unmeasured: list[str] = []
    for key in METRIC_KEYS:
        b, c = base_m.get(key), cand_m.get(key)
        if b is None or c is None:
            unmeasured.append(key)
            continue
        (wins if c > b else losses).append(key)
    measured = len(wins) + len(losses)
    return {
        "baseline": baseline.arm,
        "candidate": candidate.arm,
        "wins": wins,
        "losses": losses,
        "unmeasured": unmeasured,
        "measured_metrics": measured,
        "threshold": 3,
        "pass": len(wins) >= 3,
        "note": (
            f"Only {measured}/5 metrics measured; a pass on fewer than 3 measurable "
            "metrics is not a pass. Wire up the test hook and linters "
            "(tandem.toml [eval]) before treating this as the M3 gate."
            if measured < 3
            else "M3 gate: A1 must beat base on >=3 of 5 metrics"
        ),
        "baseline_metrics": {k: _round(v) for k, v in base_m.items()},
        "candidate_metrics": {k: _round(v) for k, v in cand_m.items()},
    }


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


@dataclass
class MergeEvalReport:
    repo: str = ""
    n_cases: int = 0
    arms: list[ArmSummary] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    per_case: list[dict[str, Any]] = field(default_factory=list)
    # Which of the five metrics this run was equipped to measure at all. Recorded
    # on the report so a stored result cannot later be read as though the
    # unmeasured ones had been measured and come out even.
    measured: dict[str, bool] = field(default_factory=dict)
    # Outcome of `verify_base`: did the repo's own suite pass before any patch?
    base_health: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "n_cases": self.n_cases,
            "measured": self.measured,
            "base_health": self.base_health,
            "arms": [a.as_dict() for a in self.arms],
            "comparisons": self.comparisons,
            "per_case": self.per_case,
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    def table(self) -> str:
        """The four bars, as text. An em dash is "not measured", never "zero"."""
        head = (
            f"{'arm':<24} {'n':>4} {'applied':>8} {'tests':>7} {'lint':>7} "
            f"{'prox':>7} {'blast_f':>8} {'blast_l':>8}"
        )
        lines = [head, "-" * len(head)]
        for a in self.arms:
            lines.append(
                f"{a.arm:<24} {a.n:>4} {_pct(a.apply_rate):>8} "
                f"{_pct(a.test_pass_rate):>7} {_pct(a.convention_rate):>7} "
                f"{a.mean_proximity:>7.3f} {a.mean_blast_files:>8.2f} {a.mean_blast_lines:>8.2f}"
            )
        return "\n".join(lines)


def _pct(v: float | None) -> str:
    return "  —" if v is None else f"{v * 100:.1f}%"


async def run(
    cases: Sequence[EvalCase],
    arms: Sequence[Arm],
    *,
    repo: Path | None = None,
    runner: WorktreeRunner | None = None,
    review_proxy: ReviewProxy | None = None,
) -> MergeEvalReport:
    report = MergeEvalReport(
        repo=str(repo or (runner.repo if runner else "")), n_cases=len(cases)
    )
    report.measured = {
        "tests": bool(runner and runner.measures_tests),
        "lint": bool(runner and runner.measures_lint),
        "review_proxy": review_proxy is not None,
    }
    summaries: list[ArmSummary] = []
    for arm in arms:
        scores = await run_arm(arm, cases, runner=runner, review_proxy=review_proxy)
        report.per_case.extend(s.as_dict() for s in scores)
        summaries.append(summarise(arm.name, scores))
    report.arms = summaries

    # Every arm against the first, which is the base model by convention.
    if summaries:
        baseline = summaries[0]
        for candidate in summaries[1:]:
            report.comparisons.append(compare_arms(baseline, candidate))
    return report


def cases_from_holdout(pairs: Sequence[Any]) -> list[EvalCase]:
    """Build eval cases from `extract_a1`'s holdout split.

    The base revision is the change's first parent. Extraction walks first-parent
    history (sec 6.2), so `sha^1` is exactly the tree the merged change was written
    against — which is what a patch generated from that task description has to
    apply to.
    """
    return [
        EvalCase(
            sha=p.sha,
            prompt=p.prompt,
            reference_diff=p.completion,
            ts=p.ts,
            base_rev=f"{p.sha}^1",
        )
        for p in pairs
    ]
