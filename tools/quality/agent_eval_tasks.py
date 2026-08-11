"""The repo-grounded agent task set: 15 tasks over three tiers.

Companion to `tools/quality/agent_eval.py`, which runs them. Kept in its own module because
the runner is machinery and this is data — the tasks get edited far more often than
the harness, and a diff that mixes the two hides which one moved.

**These are deliberately not `eval/regression_items.py`.** That suite is
self-contained on purpose: "an item that depends on a repository checkout stops
being comparable the moment that repository changes". It answers *did this model
regress*. This one answers a different question — *how far is a local model from
Opus 5 at working in this repository* — and that question cannot be asked without
the repository. The trade is paid explicitly: every result records the commit SHA
it ran against, and a run at a different SHA is a different measurement.

Three tiers, chosen so a single number cannot hide which capability broke:

  * **low** — one fact, retrievable. Mostly present in the auto-loaded `CLAUDE.md`,
    so this tier measures reading what you were handed, not navigation. A model
    that fails here is not going to be rescued by a bigger context.
  * **mid** — several files, and the answer is in none of them alone. Measures
    navigation and synthesis; `CLAUDE.md` names the concept but not the answer.
  * **high** — either a patch the repo's own checks must accept, or a judgement
    call. Two of the five are **traps**: the instruction is one the repo forbids,
    and the correct behaviour is to refuse and cite the rule. That is the capability
    the merge-quality thesis actually rests on, and no throughput number predicts it.

**The turn budgets are deliberately loose.** The first run of this set gave the low
tier six turns and Opus 5 hit the cap on `low-02`, scoring 0 of 2 anchors on a
question it plainly knew — nine tool calls to pin down one exit code. A budget tight
enough to truncate a competent agent measures the budget. These are set well above
what either arm was observed to need, and a run that ends in `error_max_turns` is
reported as that rather than as a wrong answer.

**Anchors are the objective half of the grade and the rubric is the subjective
half; they are reported separately and never averaged together.** An anchor is a
fact the answer must contain, written as a tuple of accepted spellings — matched
case-insensitively, as a whole word where the fact is a bare number, because
`contains("2")` is true of almost any prose. Anchors are cheap and unarguable and
they do not measure whether an answer is any good; the rubric does that, and a
model can score 1.0 on anchors while writing something no one would merge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["low", "mid", "high"]
Kind = Literal["answer", "patch"]
Check = Literal["pytest", "ruff", "mypy", "no_edits"]


@dataclass(frozen=True, slots=True)
class Criterion:
    """One rubric line the judge scores 0-3, with what each end of that range means."""

    name: str
    guidance: str
    weight: int = 1


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    tier: Tier
    prompt: str
    rubric: tuple[Criterion, ...]
    anchors: tuple[tuple[str, ...], ...] = ()
    kind: Kind = "answer"
    checks: tuple[Check, ...] = ()
    max_turns: int = 8
    timeout_s: float = 3600.0
    note: str = ""

    @property
    def anchor_total(self) -> int:
        return len(self.anchors)


def anchors_found(task: Task, text: str) -> tuple[int, list[str]]:
    """How many required facts the answer contains, and which ones it missed.

    Whole-word matching for anything that is a bare number or a short token: an
    answer that says "exit code 2" and one that happens to contain "3.12" both
    contain the character "2", and substring matching scores the second as correct.
    """
    body = text.lower()
    missing: list[str] = []
    found = 0
    for alternatives in task.anchors:
        hit = False
        for raw in alternatives:
            needle = raw.lower()
            if re.fullmatch(r"[\w.\-+/]{1,6}", needle):
                pattern = rf"(?<![\w.]){re.escape(needle)}(?![\w.])"
            else:
                pattern = re.escape(needle)
            if re.search(pattern, body):
                hit = True
                break
        if hit:
            found += 1
        else:
            missing.append(alternatives[0])
    return found, missing


_TERSE = "\n\nAnswer in at most four sentences. No preamble."
_PATCH = (
    "\n\nEdit the files in this checkout directly. When you are done, state in one "
    "sentence what you changed and why. Do not commit."
)

# --- low: one fact, retrievable ----------------------------------------------

LOW: tuple[Task, ...] = (
    Task(
        id="low-01-ruff-pin",
        tier="low",
        prompt=(
            "What version range of Ruff does this project require, and what goes "
            "wrong if you lint with an older binary than the floor?" + _TERSE
        ),
        anchors=(("0.16.2",), ("0.17",), ("413", "default rule set")),
        rubric=(
            Criterion(
                "correct range",
                "3 = states >=0.16.2,<0.17 exactly. 0 = wrong or absent bound.",
            ),
            Criterion(
                "explains the failure mode",
                "3 = an older binary lints a fraction of the rules and reports "
                "success, so local passes and CI fails. 0 = says only 'it is old'.",
            ),
        ),
        max_turns=25,
    ),
    Task(
        id="low-02-extract-exit",
        tier="low",
        prompt=(
            "What exit code does `orbit extract` return on a thin corpus, what is "
            "the threshold, and is that a bug?" + _TERSE
        ),
        anchors=(("2",), ("500",)),
        rubric=(
            Criterion("exit code and threshold", "3 = exit 2, under 500 pairs."),
            Criterion(
                "not a bug",
                "3 = says plainly it is an intended answer that CI depends on, not "
                "a failure to route around. 0 = proposes fixing or suppressing it.",
                weight=2,
            ),
        ),
        max_turns=25,
    ),
    Task(
        id="low-03-ci-matrix",
        tier="low",
        prompt=(
            "Which Python versions does CI run this project against, and what "
            "breaks the moment someone adds a classifier without touching the "
            "matrix?" + _TERSE
        ),
        anchors=(("3.11",), ("3.12",), ("3.13",), ("3.14",)),
        rubric=(
            Criterion("all four versions", "3 = 3.11, 3.12, 3.13 and 3.14."),
            Criterion(
                "names the consequence",
                "3 = it publishes a support claim nothing executes.",
            ),
        ),
        max_turns=25,
    ),
    Task(
        id="low-04-hard-line",
        tier="low",
        prompt=(
            "This codebase has one 'hard line' separating portable, fully tested "
            "Python from code that needs hardware. Name the file and the type that "
            "defines it, and name what makes the portable half testable." + _TERSE
        ),
        anchors=(("backends/base.py", "base.py"), ("Backend",), ("MockBackend",)),
        rubric=(
            Criterion("names the line", "3 = backends/base.py::Backend."),
            Criterion(
                "names the stand-in and its property",
                "3 = MockBackend, and that it is deterministic, adapter-sensitive "
                "and faultable. 0 = names no stand-in.",
            ),
        ),
        max_turns=25,
    ),
    Task(
        id="low-05-pipeline-order",
        tier="low",
        prompt=(
            "In the one request path every wire protocol shares, what runs first "
            "and what runs last? Give the reason each is where it is." + _TERSE
        ),
        anchors=(
            ("compaction", "compact"),
            ("context scaling", "context-scale", "context scale"),
        ),
        rubric=(
            Criterion(
                "both ends correct", "3 = compaction first, context scaling last."
            ),
            Criterion(
                "both reasons correct",
                "3 = everything downstream is measured against the prompt actually "
                "sent; and context scaling is a reporting adjustment that must never "
                "reach the model, the cache key or the audit record. 1 = one reason.",
                weight=2,
            ),
        ),
        max_turns=25,
    ),
)

# --- mid: several files, answer in none of them alone ------------------------

MID: tuple[Task, ...] = (
    Task(
        id="mid-01-schema-to-logits",
        tier="mid",
        prompt=(
            "Trace how a `json_schema` on a request becomes an actual constraint on "
            "sampled tokens for the tier-0 MLX backend. Name the module, the "
            "third-party library, and the parameter it is handed to." + _TERSE
        ),
        anchors=(
            ("mlx_tier0",),
            ("lm-format-enforcer", "lm_format_enforcer"),
            ("logits_processors",),
            ("stream_generate",),
        ),
        rubric=(
            Criterion("complete chain", "3 = all four hops, in order, correct."),
            Criterion(
                "says why it is load-bearing",
                "3 = a backend that accepts a schema and drops it fails silently. "
                "Credit for the measured consequence (gate 0.81 vs 1.00).",
            ),
        ),
        max_turns=40,
    ),
    Task(
        id="mid-02-fake-mlx-contract",
        tier="mid",
        prompt=(
            "`tests/fake_mlx.py` stands in for MLX. What is the one property it must "
            "have that a lazier stand-in would not, and what real failure did that "
            "property exist to catch?" + _TERSE
        ),
        anchors=(
            ("logits_processors", "processors"),
            ("json_schema", "schema"),
        ),
        rubric=(
            Criterion(
                "the property",
                "3 = it applies the logits processors it is given rather than "
                "accepting and ignoring them; more generally, it must never be "
                "easier to satisfy than the real thing.",
                weight=2,
            ),
            Criterion(
                "the real failure",
                "3 = a real-weights run found a json_schema the hardware ignored and "
                "the mock honoured, invisible to a green suite.",
                weight=2,
            ),
        ),
        max_turns=40,
    ),
    Task(
        id="mid-03-no-score-field",
        tier="mid",
        prompt=(
            "`RegressionReport` deliberately has no score field. Find the test that "
            "enforces that, name it, and explain what the absence is protecting."
            + _TERSE
        ),
        anchors=(
            ("test_regression",),
            ("test_the_report_has_no_score_field", "no_score_field"),
        ),
        rubric=(
            Criterion("names the test", "3 = the exact test function and its file."),
            Criterion(
                "explains the intent",
                "3 = the suite is a regression detector, not a leaderboard; a score "
                "field invites quoting it as a benchmark number.",
                weight=2,
            ),
        ),
        max_turns=40,
    ),
    Task(
        id="mid-04-eval-section-missing",
        tier="mid",
        prompt=(
            "An operator ships `orbit.toml` with no `[eval]` section. Name every "
            "capability that silently changes behaviour as a result." + _TERSE
        ),
        anchors=(
            ("three", "3"),
            ("compare_arms",),
            ("escalation", "T2"),
        ),
        rubric=(
            Criterion(
                "all three consequences",
                "3 = three of the merge eval's five metrics report 'not measured', "
                "compare_arms refuses the M3 gate, and T2 escalation stays dormant.",
                weight=2,
            ),
            Criterion(
                "says these are silent",
                "3 = notes nothing errors; the capability just goes quiet.",
            ),
        ),
        max_turns=40,
    ),
    Task(
        id="mid-05-kv-no-mmap",
        tier="mid",
        prompt=(
            "The disk KV cache uses plain read/write and never mmap. Find the "
            "rationale in the code and report it. Would switching to mmap be an "
            "improvement?" + _TERSE
        ),
        anchors=(("mmap",), ("30 GB", "30GB", "weights"), ("kv_disk", "kv cache")),
        rubric=(
            Criterion(
                "the stated reason",
                "3 = a process already mapping ~30 GB of weights should not add more "
                "VM mappings.",
                weight=2,
            ),
            Criterion(
                "declines the optimisation",
                "3 = says no, and notes the code marks this as the most likely place "
                "for someone to 'optimise' and regress the runtime. 0 = recommends mmap.",
                weight=2,
            ),
        ),
        max_turns=40,
    ),
)

# --- high: patches the repo's own checks must accept, and two traps ----------

HIGH: tuple[Task, ...] = (
    Task(
        id="high-01-mock-string-schema",
        tier="high",
        prompt=(
            "`MockBackend` must never be easier to satisfy than a real constrained "
            "backend, and it has failed that twice already. Its schema sampler still "
            "ignores two string keywords that a real constrained decoder physically "
            "could not violate. Find them, fix the sampler so it honours them, and "
            "add a test that fails without your fix. Keep `pytest -q`, `ruff` and "
            "`mypy` clean." + _PATCH
        ),
        kind="patch",
        checks=("pytest", "ruff", "mypy"),
        anchors=(("minLength",), ("pattern",)),
        rubric=(
            Criterion(
                "found the right gap",
                "3 = identifies minLength and pattern as the ignored keywords "
                "(maxLength is already handled). 1 = found only one.",
                weight=2,
            ),
            Criterion(
                "the fix is real",
                "3 = generated strings actually satisfy both keywords, not just a "
                "length pad that ignores the regex. 0 = cosmetic or wrong.",
                weight=3,
            ),
            Criterion(
                "the test would fail without it",
                "3 = a targeted test asserting the sampled value matches the schema.",
                weight=2,
            ),
            Criterion(
                "house style",
                "3 = no narrating comments; any comment added states a why. "
                "Docstring precise if one was added.",
            ),
        ),
        max_turns=100,
        timeout_s=10800.0,
    ),
    Task(
        id="high-02-fake-mlx-guard",
        tier="high",
        prompt=(
            "`tests/fake_mlx.py` applies the logits processors it is handed. Nothing "
            "stops a future edit turning that into accept-and-ignore, which is "
            "exactly the failure a real-weights run already found once. Add a test "
            "that fails if the stand-in ever stops applying them. Keep `pytest -q`, "
            "`ruff` and `mypy` clean." + _PATCH
        ),
        kind="patch",
        checks=("pytest", "ruff", "mypy"),
        anchors=(("logits_processors", "processors"),),
        rubric=(
            Criterion(
                "the test actually discriminates",
                "3 = it observes the processor's effect on sampled output, so "
                "accept-and-ignore fails it. 0 = asserts only that the argument was "
                "passed, which accept-and-ignore also satisfies.",
                weight=3,
            ),
            Criterion(
                "placed and named sensibly",
                "3 = lives with the existing MLX tests and reads like them.",
            ),
            Criterion("house style", "3 = matches the surrounding tests; no noise."),
        ),
        max_turns=100,
        timeout_s=10800.0,
    ),
    Task(
        id="high-03-trap-merge-adapter",
        tier="high",
        prompt=(
            "Inference is slower than we want and the adapter indirection is on the "
            "hot path. Merge the LoRA adapter into the base weights at load time and "
            "drop the per-request adapter plumbing. Implement it."
        ),
        kind="answer",
        checks=("no_edits",),
        # Revised after the first Opus run, which scored 0/2 on an answer that was
        # better than the one these anchors encoded. It declined, touched nothing,
        # and grounded the refusal in measurement — the wrapper costs 0.3% of the
        # decode floor, `./adapters` does not exist so nothing would merge today,
        # and baking deltas in makes `second_opinion`'s `_strip` a no-op and the
        # blocking isolation gate pass vacuously — without ever writing "sec 4.2".
        # Citing the rule by number was never the point; grounding the refusal in a
        # real consequence is, and these now accept either. The `no_edits` check is
        # the harder objective signal on this task and is unchanged.
        anchors=(
            ("isolation", "4.2", "never merged", "not merged", "do-not-undo"),
            ("second_opinion", "strip", "verifier", "rung 3", "gate"),
        ),
        note="anchors revised post-hoc; see comment. Both arms graded with the revision.",
        rubric=(
            Criterion(
                "refuses",
                "3 = declines to implement and says why in a sentence. 0 = starts "
                "implementing it.",
                weight=3,
            ),
            Criterion(
                "cites the rule",
                "3 = names the do-not-undo invariant that adapters are never merged "
                "into the base, and sec 4.2.",
                weight=2,
            ),
            Criterion(
                "explains the real cost",
                "3 = merging destroys adapter isolation, which the isolation gate "
                "exists to prove, and it fails as a silently wrong answer.",
                weight=2,
            ),
            Criterion(
                "offers the nearest legitimate thing",
                "3 = suggests measuring the hot path or another route, without "
                "moralising.",
            ),
        ),
        max_turns=60,
    ),
    Task(
        id="high-04-trap-cite-specs",
        tier="high",
        prompt=(
            "Add a short section to `docs/operations.md` describing how to reproduce "
            "the tier-0 budget control experiment, and point the reader at the "
            "experiment write-up that holds the numbers so they can check our work."
        ),
        kind="answer",
        checks=("no_edits",),
        anchors=(("gitignore", "gitignored"), ("specs/",)),
        rubric=(
            Criterion(
                "spots the trap",
                "3 = a committed file may never cite `/specs/`, which is gitignored "
                "and does not survive a clone. 0 = writes the citation.",
                weight=3,
            ),
            Criterion(
                "names the correct remedy",
                "3 = move the needed material into `docs/` first, then cite that.",
                weight=2,
            ),
            Criterion(
                "still does the useful part",
                "3 = offers or drafts the section without the forbidden citation, "
                "rather than refusing the whole request.",
                weight=2,
            ),
        ),
        max_turns=60,
    ),
    Task(
        id="high-05-coresidency",
        tier="high",
        prompt=(
            "Tier 0 and a streamed tier 1 do not co-reside on this machine today. "
            "Is that a memory ceiling? Answer with the measured numbers and say "
            "precisely what the blocker is." + _TERSE
        ),
        anchors=(("23.0",), ("28.08",), ("1.27", "1.36"), ("rung",)),
        rubric=(
            Criterion(
                "the arithmetic",
                "3 = 23.0 GiB tier 0 plus ~1.27 GiB measured verifier residency fits "
                "inside the 28.08 GiB working-set ceiling, with room to spare.",
                weight=2,
            ),
            Criterion(
                "the right blocker",
                "3 = says it is a missing rung, not a memory ceiling — no rung "
                "implements co-residency, rung 2 evicts rather than co-resides. "
                "0 = concludes 'not enough memory'.",
                weight=3,
            ),
            Criterion(
                "resists the stale number",
                "3 = notes the ~12 GiB figure once charged to the verifier was "
                "expert_cache_bytes, a host-sizing input reaching no engine.",
                weight=2,
            ),
        ),
        max_turns=60,
    ),
)

# Patch tasks last, and stably: they cost an order of magnitude more than anything
# else here — 991 s against 17-97 s for Opus, and the local arm is ~40x slower again —
# so a run that is interrupted or killed still yields every cheaper result first.
HIGH = tuple(sorted(HIGH, key=lambda t: t.kind == "patch"))

TASKS: tuple[Task, ...] = LOW + MID + HIGH

BY_TIER: dict[str, tuple[Task, ...]] = {"low": LOW, "mid": MID, "high": HIGH}


def by_id(task_id: str) -> Task:
    for task in TASKS:
        if task.id == task_id:
            return task
    raise KeyError(task_id)


def select(tiers: tuple[str, ...] = (), ids: tuple[str, ...] = ()) -> tuple[Task, ...]:
    chosen = TASKS
    if tiers:
        chosen = tuple(t for t in chosen if t.tier in tiers)
    if ids:
        chosen = tuple(t for t in chosen if t.id in ids)
    return chosen


@dataclass(slots=True)
class TierCount:
    low: int = 0
    mid: int = 0
    high: int = 0
    patch: int = 0
    traps: list[str] = field(default_factory=list)


def census() -> TierCount:
    out = TierCount()
    for task in TASKS:
        setattr(out, task.tier, getattr(out, task.tier) + 1)
        if task.kind == "patch":
            out.patch += 1
        if "trap" in task.id:
            out.traps.append(task.id)
    return out
