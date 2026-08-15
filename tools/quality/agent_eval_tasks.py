"""The repo-grounded agent task set: 28 tasks over three tiers and six families.

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

**Tier is not the only axis, and for fifteen tasks it was the only one recorded.**
Every one of those asked the same *kind* of question — read this repository and
explain it — so a tier that reported 14/14 said nothing about whether the arm could
diagnose a symptom, derive a document from source, edit one in place, or report that
it could not reach a file. Each task therefore also carries a `family` and a
`deliver` label, `select` and `report` work on both axes, and the set was extended
into the four families it did not contain at all.

The taxonomy and the reason for each addition come from a study of ~45k production
agent sessions (2026-08-14) that is **not in this repository and is not reproducible
from it** — the same standing as a `sec N.M` reference. Four of its findings each
named a capability nothing here was measuring:

  * **changing an existing artefact fails more often than writing a new one**, on
    documents and on code alike, and creation is all this set had;
  * **bounded tasks fail by misreading a short instruction or by not reaching
    something**, not by running out of steam — the opposite of how hard tasks fail,
    and the tasks here are all long, fully-specified prompts;
  * **a stored prompt replayed against a different subject** is its own failure mode,
    worst on comprehension work;
  * **half of all requests expect prose back and never touch a write tool**, which is
    why `deliver` is recorded next to `family`.

`target` — what the work acts on — is that study's third axis and is deliberately
absent: in a single-repository eval it would read `own_codebase` for nearly
everything, and it was the weakest of the three axes where it was measured.

**The set still over-weights comprehension**: 12 of 28 tasks against ~31% of that
corpus, because nothing was removed. Every existing task has run history under
`var/`, and a task deleted to fix a ratio takes its comparisons with it. Read the
per-family rows as coverage rather than as a distribution, and a family of one or
two tasks as direction rather than magnitude.

The six most recent tasks close the classes that had no representative at all, each
one named by that study and sized by its share of the tier it sits in:
`explain_general_tech` (26.5% of `low`), `explain_business_term` (10.3% of `low`),
`ops_setup`, `review_code`, `review_doc`, and `write_code` — the last of which is
the create-side control the set could not previously run, so the corpus finding that
**creating is easier than modifying** (`write_code` 8.3% unhappy against `edit_code`
11.7%) had no form here that could confirm or refute it.

Three of that study's classes are still absent, and deliberately:

  * `write_business_text`, `translate` and `offtask_personal` — real and large there,
    but a repo-grounded form would measure the model's prose rather than its work in
    this repository, which is the same reason the `prose` and `offtask` families are
    not in `Family`;
  * `analyze_data` — every candidate artefact lives under `var/`, which is gitignored
    and therefore absent from the detached worktree every run uses. A task whose
    subject does not survive `git worktree add` grades the checkout, not the model;
  * `summarize` — 1.4% and the thinnest class in the source gold set, and in a
    single-repository eval every draft of it collapsed into `understand_own_system`
    or `generate_doc_from_source`, both of which are already covered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["low", "mid", "high"]
Kind = Literal["answer", "patch"]
Check = Literal["pytest", "ruff", "mypy", "no_edits", "docs_only"]

# What kind of work is being asked for, and what the requester expects back. Both
# vocabularies are the production study's, minus the classes with no repo-grounded
# form: `prose` and `offtask` are real and large there and would be measuring the
# model's writing rather than its work in this repository.
Family = Literal[
    "understand", "diagnose", "change_code", "documents", "operate", "decide"
]
Deliverable = Literal["answer", "inline_content", "file_artifact", "code_change"]


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
    # No default, so a new task cannot be added without being classified. The whole
    # point of the axis is that the set drifted into one family unnoticed.
    family: Family
    prompt: str
    rubric: tuple[Criterion, ...]
    deliver: Deliverable = "answer"
    # A stored prompt, rule file or pipeline step supplying the instructions rather
    # than the user. A delivery mechanism, not a task — every attempt to make it a
    # family swallowed the underlying job into it — so it is its own flag.
    template_driven: bool = False
    anchors: tuple[tuple[str, ...], ...] = ()
    kind: Kind = "answer"
    checks: tuple[Check, ...] = ()
    max_turns: int = 8
    timeout_s: float = 3600.0
    note: str = ""

    @property
    def anchor_total(self) -> int:
        return len(self.anchors)

    @property
    def expects_diff(self) -> bool:
        """Whether an edit is part of what was asked for, so the tree must be diffed.

        `kind` is how a task is run; `deliver` is what the requester wanted back, and
        the two came apart on `mid-08`. It is `deliver="code_change"` held at
        `kind="answer"` on purpose — the prompt is 34 characters and stating a
        procedure would destroy the ambiguity it exists to measure — so diff
        collection returned early, `_run_checks` never ran, and the judge was shown
        prose only. Both arms edited the tree anyway, two files and four, and none of
        it was recorded. A `no_edits` trap is excluded because there the correct diff
        is the empty one.
        """
        return self.kind == "patch" or (
            self.deliver == "code_change" and "no_edits" not in self.checks
        )


def anchors_found(task: Task, text: str) -> tuple[int, list[str]]:
    """How many required facts the answer contains, and which ones it missed.

    Whole-word matching for anything that is a bare number or a short token: an
    answer that says "exit code 2" and one that happens to contain "3.12" both
    contain the character "2", and substring matching scores the second as correct.

    **The trailing boundary excludes a following word character, and a period only
    when a word character follows it.** The earlier form excluded any following period
    and therefore failed on a sentence-final version number: two harness runs whose
    answers read ">=0.16.2,<0.17." scored 0 on the `0.17` anchor while a third, whose
    only difference was ">=0.16.2,<0.17, as specified in", scored 1. It cost `exit
    code 2.` and `Python 3.12.` the same way, so it under-counted every arm including
    the control. `0.17.3` and `0.170` are still correctly rejected, which is what the
    boundary was for.
    """
    body = text.lower()
    missing: list[str] = []
    found = 0
    for alternatives in task.anchors:
        hit = False
        for raw in alternatives:
            needle = raw.lower()
            if re.fullmatch(r"[\w.\-+/]{1,6}", needle):
                pattern = rf"(?<![\w.]){re.escape(needle)}(?!\w)(?!\.\w)"
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
        family="understand",
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
        family="understand",
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
        family="understand",
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
        family="understand",
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
        family="understand",
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
    Task(
        id="low-06-cheap-doctor",
        tier="low",
        family="operate",
        prompt=(
            "I need this checkout's offline posture and whether constrained decoding "
            "is available, on a machine that must not load a model right now. Give me "
            "the exact invocation." + _TERSE
        ),
        anchors=(("doctor",), ("mock",), ("--config",)),
        rubric=(
            Criterion(
                "the escape, not the command",
                "3 = `orbit --config <file> doctor` against a config whose backend is "
                '"mock". 0 = says to run `orbit doctor`.',
                weight=2,
            ),
            Criterion(
                "where the flag goes",
                "3 = `--config` is global and goes before the subcommand.",
            ),
            Criterion(
                "what it costs otherwise",
                "3 = doctor on the mlx backend loads 23.0 GiB eagerly, because "
                "`MLXTier0Backend.__init__` calls `mlx_lm.load()`; it reads like `git "
                "status` and behaves like a model load.",
                weight=2,
            ),
        ),
        max_turns=25,
    ),
    Task(
        id="low-07-green-local-red-ci",
        tier="low",
        family="diagnose",
        prompt=(
            "`ruff format` and `ruff check --fix` are clean on every file I touched, "
            "`pytest` and `mypy` pass, and CI is red on the format step. What did I "
            "miss, and what should I have run?" + _TERSE
        ),
        anchors=(
            ("markdown", ".md", "docs"),
            ("ruff format",),
            ("--check", "whole tree", "no path"),
        ),
        rubric=(
            Criterion(
                "names the surface",
                "3 = `.md` files: ruff formats Python inside ```python fences, so docs "
                "are in the format check and 'touched paths' gets read as the .py ones. "
                "0 = blames the ruff version or the config.",
                weight=3,
            ),
            Criterion(
                "names what to run",
                "3 = the whole-tree `ruff format --check` CI runs, not the per-file "
                "command the PostToolUse hook runs.",
                weight=2,
            ),
            Criterion(
                "reads the step, not the job",
                "3 = notes the lint job splits check and format into two steps so the "
                "failing step names the cause, and that this cost seven consecutive "
                "red builds with every other job green.",
            ),
        ),
        max_turns=30,
    ),
    Task(
        id="low-08-prefill-vs-decode",
        tier="low",
        family="understand",
        # The production corpus's largest `low` class by a distance: 26.5% of `low` is
        # a general technical question, against 2.6% that touches code at all. The
        # class is not "what is prefill" — that needs no repository — it is a general
        # question asked *of a codebase*, and the failure it exists to catch is an
        # answer that is textbook-correct and says nothing about this checkout.
        prompt=(
            "I know roughly what prefill and decode are. Why does the gap between "
            "them decide the whole design here, rather than being an implementation "
            "detail?" + _TERSE
        ),
        anchors=(("prefill",), ("decode",), ("40x", "40×", "~40")),
        rubric=(
            Criterion(
                "gets the direction right",
                "3 = prefill is the cheap direction and reading is prefill-bound, so a "
                "model too big to generate with is still affordable to read with. "
                "0 = has the asymmetry backwards or treats the two as interchangeable.",
                weight=3,
            ),
            Criterion(
                "connects it to this design",
                "3 = it is why the streamed tier is a verifier and never a generator: "
                "the resident model writes, the larger one only judges. 0 = a correct "
                "general explanation with nothing in it specific to this repository.",
                weight=3,
            ),
            Criterion(
                "carries the number",
                "3 = ~40x on measured hardware, and says it was measured here rather "
                "than quoting a figure from the literature.",
                weight=2,
            ),
        ),
        max_turns=25,
    ),
    Task(
        id="low-09-what-is-a-rung",
        tier="low",
        family="understand",
        # `explain_business_term` is 10.3% of `low` and 2.8% of `mid` — a vocabulary
        # lookup, where the word is the organisation's rather than the industry's.
        # "Rung" is this repository's: it appears in `CLAUDE.md` without a definition,
        # which is exactly the position a newcomer is in.
        prompt=(
            "Someone told me tier 1 is 'at rung 3 on this machine'. What is a rung, "
            "how many are there, and what does rung 3 actually mean?" + _TERSE
        ),
        # Not an anchor on "4" or "rung 4": any answer on this topic contains both,
        # so it would score without discriminating — the `low-06` defect. `5.5` and a
        # second rung name each require having found the ladder rather than the word.
        anchors=(
            ("second_opinion",),
            ("adapter",),
            ("5.5", "ladder", "fallback"),
            ("resident_swap", "remote", "streamed"),
        ),
        rubric=(
            Criterion(
                "defines the term",
                "3 = the sec 5.5 fallback ladder — which of four ways the tier-1 "
                "verifier role is filled when the streamed 122B is not available. "
                "0 = guesses from the word, or conflates a rung with a tier.",
                weight=3,
            ),
            Criterion(
                "rung 3 specifically",
                "3 = `second_opinion`: tier 0 itself with its adapter unmounted, so it "
                "needs no second model and costs no memory, and it is weaker than the "
                "streamed verifier while still catching adapter overfit.",
                weight=3,
            ),
            Criterion(
                "names the others without padding",
                "3 = 1 streamed, 2 `resident_swap` (evicts tier 0, ~10 s each way), "
                "4 `remote` (leaves the machine, breaks the offline claim). Credit for "
                "brevity — this is a vocabulary question, not a tour.",
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
        family="understand",
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
        family="understand",
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
        family="understand",
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
        family="understand",
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
        family="understand",
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
    Task(
        id="mid-06-tier1-budgets",
        tier="mid",
        family="documents",
        deliver="inline_content",
        prompt=(
            "Derive from the code a markdown table of every tier-1 call type with its "
            "output ceiling in tokens, and add a final row for what a call whose "
            "schema title matches none of them gets. Table only, no prose."
        ),
        anchors=(
            ("rerank",),
            ("plan_critique",),
            ("128",),
            ("512",),
            ("640",),
        ),
        # The discriminator is the last row, and it is the one thing reading the
        # constants gets wrong: `DEFAULT_BUDGET = 256` is unreachable, because
        # `call_type_of` maps an unrecognised title to `review` before the `.get`
        # default can apply. 256 is what a model that read `CALL_BUDGETS` answers;
        # 512 is what a model that followed the call answers.
        note="the fallback row separates reading the constants from tracing the call",
        rubric=(
            Criterion(
                "the table is complete",
                "3 = three call types with 128 / 512 / 640, no invented row. "
                "1 = a type missing or a ceiling wrong.",
                weight=2,
            ),
            Criterion(
                "the fallback row is right",
                "3 = an unrecognised schema title is classified as `review` by "
                "`call_type_of` and gets 512, so `DEFAULT_BUDGET = 256` is never "
                "reached. 1 = answers 256 from the constant alone.",
                weight=3,
            ),
            Criterion(
                "delivered as asked",
                "3 = a markdown table and nothing else. 0 = prose with the numbers "
                "buried in it.",
                weight=2,
            ),
        ),
        max_turns=40,
    ),
    Task(
        id="mid-07-template-mismatch",
        tier="mid",
        family="operate",
        template_driven=True,
        prompt=(
            "Run our standard pre-flight from the team's stored procedure before a "
            "measurement:\n"
            '  1. `curl -s localhost:11434/api/ps` must come back `{"models":[]}`\n'
            "  2. `lsof -nP -iTCP:8081 -sTCP:LISTEN` must come back empty\n"
            "  3. read headroom as `total - active`; abort under ~27 GB\n"
            "  4. pilot one call, measure, then eight, then the hundred\n\n"
            "Apply it to `orbit audit verify` and tell me whether we are clear to run."
            + _TERSE
        ),
        anchors=(
            ("audit verify",),
            (
                "no weights",
                "loads no",
                "does not load",
                "never load",
                "without loading",
            ),
        ),
        note=(
            "a stored prompt replayed against the wrong subject — the largest "
            "over-represented failure at this complexity, and worst on comprehension"
        ),
        rubric=(
            Criterion(
                "notices the procedure does not apply",
                "3 = `orbit audit verify` builds no backend and loads no weights — it "
                "is on operations.md §2's free list — so the memory ladder and the "
                "GPU-ownership checks constrain nothing here. 0 = works through the "
                "four steps as written.",
                weight=3,
            ),
            Criterion(
                "still answers the question asked",
                "3 = says plainly that it is clear to run. 0 = only critiques the "
                "procedure, or refuses.",
                weight=2,
            ),
            Criterion(
                "invents no reading",
                "3 = reports no `ps` output, headroom figure or pilot result it did "
                "not obtain; `curl` and `lsof` are not available to it. 0 = states a "
                "measurement it never took.",
                weight=3,
            ),
        ),
        max_turns=30,
    ),
    Task(
        id="mid-08-terse-raise-kv",
        tier="mid",
        family="change_code",
        deliver="code_change",
        # 34 characters, deliberately. The median request at this complexity is 86,
        # and the failure that over-indexes hardest there is the agent answering a
        # short instruction it has under-specified evidence for. Every other prompt
        # in this set states its own success criteria; this one states none.
        prompt="max_kv_tokens is too low. Raise it.",
        # **The rubric below was corrected on 2026-08-15 after the run and before the
        # judge, and the first version graded the wrong thing.** It required "prefill
        # time, not KV memory: 131k is ~136 s at the measured 963 tok/s", which is
        # `orbit.toml`'s own comment copied in as the answer — the `mid-02` failure
        # (§17.7) in a new place: the graded answer was a quotation, not a derivation.
        # Two things sit under it, both verified in the tree:
        #
        #   * `self.max_kv_tokens` occurs **once** in all of `src/`, at its own
        #     assignment (`mlx_tier0.py:228`). `_warm_start` builds the cache as
        #     `kv.make_prompt_cache(self.model)` with no `max_kv_size`, and
        #     `tests/fake_mlx.py:481` carries that parameter unused. The knob is inert,
        #     so raising it is a no-op at runtime and an answer arguing only about the
        #     number is arguing about a dead control. §19.7 already recorded that the
        #     control said this and that "the task rewards none of that".
        #   * 963 tok/s is `platform.md:191`, the **32 k** row. Prefill rate falls with
        #     context, so extrapolating it to 131k makes ~136 s a floor rather than an
        #     estimate. The repository's comment holds a rate constant that it should not.
        #
        # `platform.md` §1.2's score-matrix ceiling is the sharper constraint of its
        # kind, but it is measured on `Qwen3-Coder-Next` and this is a tier-0 knob, so
        # it is credited as reasoning and never required as fact.
        anchors=(
            ("131072", "131,072", "131k"),
            ("peak", "get_peak_memory"),
            ("prefill",),
            ("never read", "not read", "inert", "no-op", "no op", "unused", "dead"),
        ),
        rubric=(
            Criterion(
                "notices the knob does nothing",
                "3 = `max_kv_tokens` is stored at `mlx_tier0.py:228` and read nowhere, "
                "and the prompt cache is built with no `max_kv_size`, so raising it "
                "changes no behaviour and the request as posed cannot be satisfied by "
                "editing the value. 0 = treats the number as live.",
                weight=3,
            ),
            Criterion(
                "finds the constraint that would bind if it were live",
                "3 = a prefill-side ceiling rather than KV bytes, and does not present "
                "~136 s as settled: 963 tok/s is the 32 k rate and prefill falls with "
                "context. Full credit also for the single-allocation score matrix "
                "(§1.2), noting it is measured on another model. 1 = quotes "
                "`orbit.toml`'s comment back. 0 = argues from KV memory alone.",
                weight=2,
            ),
            Criterion(
                "names what would authorise it",
                "3 = an `mx.get_peak_memory()` reading on a loaded model, which "
                "`orbit.toml` asks for by name and nothing here can take — with credit "
                "for noticing it would measure a value the code never applies.",
                weight=2,
            ),
            Criterion(
                "handles the under-specified instruction",
                "3 = states the value it would raise to and on what evidence, or asks. "
                "0 = changes the number with no reason given.",
                weight=2,
            ),
        ),
        max_turns=30,
    ),
    Task(
        id="mid-09-absent-adapters",
        tier="mid",
        family="operate",
        prompt=(
            "List the adapters this checkout has trained and give the rung each one "
            "is registered at." + _TERSE
        ),
        anchors=(("gitignore", "gitignored"), ("adapter_dir", "orbit.toml")),
        note="the file it names is absent and the premise it carries is wrong",
        rubric=(
            Criterion(
                "reports the limit instead of filling it in",
                "3 = there are none: `/adapters/` is gitignored, so it never survives "
                "a clone and a fresh worktree has no such directory. 0 = names an "
                "adapter, a path or a count that does not exist.",
                weight=3,
            ),
            Criterion(
                "corrects the premise",
                "3 = a rung is a tier-1 property (`[tier1].rung`, sec 5.5), not "
                "something an adapter is registered at — adapter choice rides on the "
                "request, never on the backend.",
                weight=2,
            ),
            Criterion(
                "says where they would be",
                "3 = `adapter_dir` under `[tier0]`, default `adapters`, and what "
                "produces one.",
            ),
        ),
        max_turns=30,
    ),
    Task(
        id="mid-10-edit-doc-in-place",
        tier="mid",
        family="documents",
        deliver="file_artifact",
        prompt=(
            "`docs/operations.md` §2 lists the commands that load weights as a table "
            "and the ones that do not as a run-on sentence, so the two cannot be read "
            "against each other. Make it one table with a column saying which. Do not "
            "add, drop or reword a command, and change nothing else in the file."
            + _PATCH
        ),
        kind="patch",
        checks=("ruff", "docs_only"),
        anchors=(("operations.md",),),
        # An in-place edit of an existing artefact, which fails more often than
        # writing a new one — and its three characteristic failures are all visible
        # here: a command silently dropped, a table left malformed, and a file the
        # answer claims to have written that the worktree has no record of
        # (`claimed_edits_without_diff` in the runner catches the third).
        note="the create-from-scratch counterpart is high-04; this one modifies",
        rubric=(
            Criterion(
                "every command survived",
                "3 = each command from both halves appears exactly once, none "
                "invented, none reworded. 0 = anything dropped.",
                weight=3,
            ),
            Criterion(
                "in place, not appended",
                "3 = §2 is replaced; no leftover duplicate of either list.",
                weight=2,
            ),
            Criterion(
                "the file still reads",
                "3 = one well-formed table, and §2.1 and the surrounding prose intact.",
                weight=2,
            ),
            Criterion(
                "keeps the reasons",
                "3 = the existing 'Why' column survives as written rather than being "
                "re-invented in the model's own words.",
            ),
        ),
        max_turns=60,
        timeout_s=7200.0,
    ),
    Task(
        id="mid-11-ops-constrain-missing",
        tier="mid",
        family="operate",
        # `ops_setup` is 2.8% of `mid` and its complaint mix is dominated by the agent
        # answering the general form of the question. The trap is that the obvious
        # answer — install the extra — is right and incomplete: this repository draws a
        # distinction between prevention and repair that a stock answer will not carry.
        prompt=(
            "I set this up with `pip install -e '.[dev]'` and everything passes, but a "
            "colleague says my tool-call numbers are not comparable to CI's. What is "
            "different about my environment, and what do I run?" + _TERSE
        ),
        anchors=(
            ("constrain",),
            ("lm-format-enforcer", "lm_format_enforcer"),
            ("repair",),
        ),
        rubric=(
            Criterion(
                "names the missing extra",
                "3 = `pip install -e '.[dev,constrain]'`, which is what CI installs; "
                "`lm-format-enforcer` is the package it pulls in.",
                weight=2,
            ),
            Criterion(
                "says what actually differs",
                "3 = without it the tool-call path exercises *repair* rather than "
                "*prevention* — the schema never reaches the logits, so a malformed "
                "call is fixed after the fact instead of being impossible. 0 = says "
                "only that a dependency is missing.",
                weight=3,
            ),
            Criterion(
                "the numbers are the point",
                "3 = the two configurations are graded by the same blocking gate and "
                "do not score the same, so a local pass proves less than CI's. Credit "
                "for `orbit gate toolcall --runs 100` or `orbit doctor` as the check.",
                weight=2,
            ),
        ),
        max_turns=30,
    ),
    Task(
        id="mid-12-review-adapter-patch",
        tier="mid",
        family="change_code",
        # `review_code` is 2.8% and is the one `change_code` task where the deliverable
        # is prose. The diff is in the prompt on purpose: a review task supplies the
        # artefact, and making the model find it first would measure navigation again.
        # What it must supply is the invariant, which is nowhere in the diff.
        prompt=(
            "Review this patch before I push it.\n\n"
            "```diff\n"
            "--- a/src/orbit/backends/mlx_tier0.py\n"
            "+++ b/src/orbit/backends/mlx_tier0.py\n"
            "@@\n"
            "     def __init__(self, cfg: Tier0Config) -> None:\n"
            "         self._model, self._tokenizer = mlx_lm.load(cfg.model)\n"
            "+        self._active_adapter: str | None = None\n"
            "+\n"
            "+    def set_adapter(self, adapter: str | None) -> None:\n"
            "+        self._active_adapter = adapter\n"
            "\n"
            "     async def generate(self, req: GenRequest) -> GenResult:\n"
            "-        adapter = req.adapter\n"
            "+        adapter = self._active_adapter\n"
            "         ...\n"
            "```\n\n"
            "It removes a per-request lookup from the hot path and the suite is green."
            + _TERSE
        ),
        anchors=(
            ("concurren", "race", "races"),
            ("GenRequest", "req.adapter", "per-request"),
            ("4.2", "isolation"),
        ),
        rubric=(
            Criterion(
                "rejects it, for the right reason",
                "3 = adapter choice is bound per request and never on the backend, "
                "because backend-global state races under concurrency: two in-flight "
                "requests wanting different adapters get whichever was set last. "
                "0 = approves it, or objects on style or naming.",
                weight=3,
            ),
            Criterion(
                "says why green tests do not clear it",
                "3 = the failure is a silently wrong answer, not an exception, and a "
                "suite that issues one request at a time cannot see it. 0 = treats "
                "the passing suite as evidence either way.",
                weight=3,
            ),
            Criterion(
                "grounds it in the rule",
                "3 = names the do-not-undo invariant (sec 4.2 / adapter isolation) "
                "rather than deriving the objection from first principles alone.",
                weight=2,
            ),
            Criterion(
                "answers the premise it was given",
                "3 = engages with the stated motive — says whether the indirection is "
                "actually on the hot path, or what would have to be measured to know. "
                "0 = ignores the performance claim entirely.",
            ),
        ),
        max_turns=30,
    ),
    Task(
        id="mid-13-review-onboarding-doc",
        tier="mid",
        family="documents",
        deliver="answer",
        # `review_doc` is 4.2% corrected and 8.8% unhappy. Three errors are planted and
        # each is checkable against exactly one place in the tree; a fourth statement is
        # correct and unremarkable, because a review that flags everything is not a
        # review. The `mid-06` lesson applies — the anchors measure how many were found,
        # and whether the correct line survived is the rubric's job.
        prompt=(
            "A new joiner wrote this section for our onboarding page. Check it against "
            "the repository and tell me what to fix.\n\n"
            "> ## Before your first PR\n"
            # Deliberately `[dev,constrain]` and not `[dev]`: the latter is a real
            # deficiency (it exercises repair rather than prevention) and is what
            # `mid-11` grades, so an arm that knows the repository would correctly
            # flag it and be marked down by the criterion that wants a correct line
            # left alone. A control line has to be one with nothing wrong with it.
            "> Install with `pip install -e '.[dev,constrain]'`, then run `pytest -q`.\n"
            "> We pin `ruff>=0.15` so everyone lints the same rule set.\n"
            "> Run `orbit doctor --config orbit.toml` to check your runtime.\n"
            "> If `orbit extract` finds too few pairs it exits 1, so treat a 1 as "
            "'corpus not ready yet' and move on.\n"
            # No `_TERSE` here, unlike every other answer task at this tier. The
            # deliverable is a list of findings with a reason each, and a four-sentence
            # cap would make the task measure compression instead of review — the arm
            # that finds all three and explains them would be graded against the same
            # ceiling as one that finds two.
        ),
        anchors=(("0.16.2",), ("before",), ("exit 2", "exits 2", "exit code 2")),
        rubric=(
            Criterion(
                "finds all three",
                "3 = the ruff floor is `>=0.16.2,<0.17`, `--config` is global and goes "
                "before the subcommand, and `extract` exits 2. 2 = two of three.",
                weight=3,
            ),
            Criterion(
                "says why each matters",
                "3 = 0.16 moved the default rule set from 59 rules to 413, so an older "
                "binary lints a twentieth of what CI does and reports success; the "
                "exit code is depended on by CI and is an answer rather than a "
                "failure. 0 = corrects the values with no consequence attached.",
                weight=3,
            ),
            Criterion(
                "leaves the correct line alone",
                "3 = does not invent a fault in the install or `pytest -q` line. "
                "0 = rewrites the whole section, or flags a correct statement.",
                weight=2,
            ),
        ),
        max_turns=40,
    ),
    Task(
        id="mid-14-write-scaler-inverse",
        tier="mid",
        family="change_code",
        deliver="code_change",
        # The create-side control for `mid-08`. The production corpus puts `write_code`
        # at 8.3% unhappy against `edit_code`'s 11.7% — creating is measurably easier
        # than modifying — and until this task the set had no mid-tier creation at all,
        # so that asymmetry could not be reproduced or refuted here. Deliberately a
        # *new* function rather than a change to `headroom_tokens`, which would make it
        # a second edit task.
        prompt=(
            "`ContextScaler` can turn a real token count into the inflated one we "
            "report to the harness. Nothing can go the other way, and the receipt code "
            "needs it: given a usage figure a harness was shown, how many real tokens "
            "did that request actually spend? Add it, and the tests that pin it. Keep "
            "`pytest -q`, `ruff` and `mypy` clean." + _PATCH
        ),
        kind="patch",
        checks=("pytest", "ruff", "mypy"),
        # One anchor, on the one thing that separates a correct inverse from an
        # obvious one. `scale` rounds up with `math.ceil`, so the round trip is not
        # exact and the inverse cannot simply divide; an arm that never says so has
        # not read the function it is inverting.
        anchors=(("ceil", "round", "exact"),),
        rubric=(
            Criterion(
                "the inverse is correct",
                "3 = divides by the same factor and handles the disabled scaler, where "
                "the factor is 1.0 and the figure passes through unchanged. 0 = wrong "
                "direction, or divides by a hardcoded constant.",
                weight=3,
            ),
            Criterion(
                "handles the rounding it inherits",
                "3 = notices `scale` rounds up, so the round trip recovers a range "
                "rather than a value, and either documents the bound or rounds in the "
                "direction that cannot under-report. 0 = presents it as exact.",
                weight=3,
            ),
            Criterion(
                "the tests would fail without it",
                "3 = covers the round trip, the disabled scaler, and a zero or "
                "out-of-range input. 0 = asserts one happy-path value.",
                weight=2,
            ),
            Criterion(
                "house style",
                "3 = lives in `context_scale.py` beside `scale`, no narrating "
                "comments, docstring precise and only where a caller needs it.",
            ),
        ),
        max_turns=100,
        timeout_s=10800.0,
    ),
)

# Patch tasks last, for the reason given under HIGH below.
MID = tuple(sorted(MID, key=lambda t: t.kind == "patch"))

# --- high: patches the repo's own checks must accept, and two traps ----------

HIGH: tuple[Task, ...] = (
    Task(
        id="high-01-mock-string-schema",
        tier="high",
        family="change_code",
        deliver="code_change",
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
        family="change_code",
        deliver="code_change",
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
        family="change_code",
        deliver="code_change",
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
        family="documents",
        deliver="file_artifact",
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
        family="decide",
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


def select(
    tiers: tuple[str, ...] = (),
    ids: tuple[str, ...] = (),
    families: tuple[str, ...] = (),
) -> tuple[Task, ...]:
    chosen = TASKS
    if tiers:
        chosen = tuple(t for t in chosen if t.tier in tiers)
    if ids:
        chosen = tuple(t for t in chosen if t.id in ids)
    if families:
        chosen = tuple(t for t in chosen if t.family in families)
    return chosen


@dataclass(slots=True)
class TierCount:
    low: int = 0
    mid: int = 0
    high: int = 0
    patch: int = 0
    template: int = 0
    traps: list[str] = field(default_factory=list)
    families: dict[str, int] = field(default_factory=dict)


def census() -> TierCount:
    out = TierCount()
    for task in TASKS:
        setattr(out, task.tier, getattr(out, task.tier) + 1)
        out.families[task.family] = out.families.get(task.family, 0) + 1
        if task.kind == "patch":
            out.patch += 1
        if task.template_driven:
            out.template += 1
        if "trap" in task.id:
            out.traps.append(task.id)
    return out
