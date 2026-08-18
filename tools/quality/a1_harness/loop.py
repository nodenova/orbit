"""The episode loop: a pinned prefix, append-only history, and an evidence gate.

Two failures in the Claude-Code-driven arm are structural rather than diagnostic, and
this loop is shaped to make both impossible rather than rare:

  * **The template raises `No user query found in messages`** whenever its backward
    scan finds no user-role message outside a `<tool_response>` wrapper. That killed one
    task 69 turns into a 100-turn budget, consistent with compaction summarising the
    original query away. Here the task prompt is message [1], never evicted and never
    rewritten, so the exception is unreachable by construction and observations are what
    get dropped instead.
  * **Nothing was reused across turns** — 11.0 M input tokens at zero cached — because
    the prompt was rebuilt rather than extended. Here every turn's prompt is a strict
    prefix extension of the previous one, which is the precondition for KV prefix reuse;
    an eviction is the one thing that breaks it and is therefore counted and flagged.

The model is a research agent by training, told in its own default prompt not to use
tools for coding help and to stop investigating early. The evidence gate is the direct
countermeasure and it is a knob, because measuring it both ways is the point.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.quality.a1_harness import VERSION, tools
from tools.quality.a1_harness.transport import (
    DEFAULT_MODEL,
    SINGLE_PASS_PREFILL_TOK_S,
    ContextOverflow,
    HostState,
    ToolCall,
    Transport,
    TransportError,
    TruncatedToolCall,
)
from tools.quality.agent_eval import Run, _collect_patch, drop_worktree, make_worktree
from tools.quality.agent_eval_tasks import Task, anchors_found

ARM = "a1-harness"
LABEL = "Agents-A1-4B-Q8 (a1-harness)"


def label_for(model: str) -> str:
    """The judge prints this verbatim, so a run against another tag must not say A1."""
    if model == DEFAULT_MODEL:
        return LABEL
    name, _, tag = model.rsplit("/", 1)[-1].partition(":")
    stem = name.removesuffix("-GGUF")
    return f"{stem}-{tag} (a1-harness)" if tag else f"{stem} (a1-harness)"


PACK_ROOT = Path(__file__).resolve().parent / "prompts"
# Each template's variables are fixed and checked at load, so a typo is a startup error
# rather than a silently blank substitution in a prompt nobody reads again.
PACK_VARIABLES: dict[str, frozenset[str]] = {
    "system.md": frozenset({"tool_names", "worktree", "max_turns"}),
    "task_answer.md": frozenset({"task_prompt", "worktree", "max_turns"}),
    "task_patch.md": frozenset({"task_prompt", "worktree", "max_turns"}),
    "observation_truncated.md": frozenset({"shown", "total", "unit", "path"}),
    "nudge_no_tool_call.md": frozenset({"tool_names", "turns_used", "max_turns"}),
    "nudge_truncated.md": frozenset(
        {"tool_names", "turns_used", "max_turns", "max_output_tokens"}
    ),
    "final_answer.md": frozenset({"evidence_count"}),
    "finish_blocked.md": frozenset({"evidence_count", "min_evidence"}),
}

# Loaded only when `--answer-review` asks for it, so adding the gate does not change the
# hash of a pack that predates it — a pack sha is how a score stays attached to the prompt
# that produced it, and silently moving v1's would orphan every number recorded against it.
OPTIONAL_PACK_VARIABLES: dict[str, frozenset[str]] = {
    "answer_review.md": frozenset(),
    "handed_context.md": frozenset({"path"}),
    "patch_blocked.md": frozenset({"turns_used", "max_turns"}),
    "no_changes_yet.md": frozenset({"turns_used", "max_turns"}),
}

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class PromptPack:
    """A directory of prompts, hashed. A score without its prompt is not a measurement."""

    name: str
    sha: str
    files: dict[str, str]

    @classmethod
    def load(cls, spec: str) -> PromptPack:
        path = Path(spec) if "/" in spec or Path(spec).is_dir() else PACK_ROOT / spec
        if not path.is_dir():
            raise ValueError(f"no prompt pack at {path}")
        files: dict[str, str] = {}
        for name, allowed in {**PACK_VARIABLES, **OPTIONAL_PACK_VARIABLES}.items():
            source = path / name
            if not source.is_file():
                if name in OPTIONAL_PACK_VARIABLES:
                    continue
                raise ValueError(f"prompt pack {path.name} is missing {name}")
            text = source.read_text()
            used = {
                field_name
                for _, field_name, _, _ in string.Formatter().parse(text)
                if field_name
            }
            if unknown := used - allowed:
                raise ValueError(
                    f"{name} uses {sorted(unknown)}, which {path.name} does not define; "
                    f"allowed here: {sorted(allowed)}"
                )
            files[name] = text
        digest = hashlib.sha256()
        for name in sorted(files):
            digest.update(name.encode())
            digest.update(files[name].encode())
        return cls(name=path.name, sha=digest.hexdigest()[:16], files=files)

    def render(self, name: str, **values: Any) -> str:
        return self.files[name].format(**values).strip()


@dataclass
class HarnessRun(Run):
    """`Run` plus what the other arm's artifact could not tell us.

    Declared as fields rather than set on an instance because `dataclasses.asdict`
    silently drops anything undeclared — which would produce an artifact missing every
    one of these, with no error anywhere.
    """

    turn_stats: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: dict[str, Any] = field(default_factory=dict)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    prefix_reuse: dict[str, Any] = field(default_factory=dict)
    finish_blocked: int = 0
    answer_key_reads: int = 0
    answer_key_mentions: int = 0
    answer_key_blocked: int = 0
    answer_reviews: int = 0
    answer_revised: bool = False
    draft_answer: str = ""
    handed_context_chars: int = 0
    nudges: int = 0
    stall_streak_max: int = 0
    think_off_from_turn: int = 0
    patch_blocked: int = 0
    idle_notices: int = 0
    length_capped: int = 0
    truncated_turns: int = 0
    truncated_tool_calls: int = 0
    final_answer_forced: bool = False
    evictions: int = 0
    observations_elided: int = 0
    thinking_chars: int = 0
    end_reason: str = ""
    answer_source: str = ""


# A delta evaluated slower than this cannot be a delta evaluation: the observed
# single-pass band on this host is 732-1,188 tok/s, and a re-prefill of a 15 k prompt
# carrying 66 new tokens reads as 3.3 tok/s on its delta. The gap is three orders of
# magnitude, so the threshold is not a delicate one.
DELTA_RATE_FLOOR_TOK_S = 200.0

# ...but only for a model that prefills like A1 does, and the floor above is a sixth of
# that model's single-pass rate rather than a property of reuse. Qwen3.8-27B prefills at
# ~140 tok/s, so a *perfectly* reused turn evaluates its delta at ~113-240 tok/s and the
# constant above scores it as a re-prefill: measured, it called "no reuse" on 28 of 28
# episodes while the server log showed `sim = 0.930` and a restored context checkpoint on
# every turn. The floor therefore scales with the model, and the ratio is what carries
# over — 200/1200 — so the A1 default is arithmetically unchanged.
DELTA_RATE_FLOOR_FRACTION = DELTA_RATE_FLOOR_TOK_S / SINGLE_PASS_PREFILL_TOK_S

# What the forced final answer gets when the episode has already spent its wall budget.
FORCED_ANSWER_TIMEOUT_S = 120.0


def prefix_reuse(
    turn_stats: list[dict[str, Any]],
    *,
    single_pass_tok_s: float = SINGLE_PASS_PREFILL_TOK_S,
) -> dict[str, Any]:
    """Per turn: was only the new text evaluated, or the whole prompt over again?

    Two earlier forms of this were wrong in opposite directions and both would have
    been believed. `prompt_eval_count` collapsing to the per-turn delta is not the
    signal — ollama reports the full prompt length every turn, cached or not. Nor is a
    *flat duration* against a rising count, which only holds when each turn appends
    little: an agent turn appends a whole file, so the duration rightly grows with the
    delta while the prefix is still being reused. That form reported "no reuse" on a run
    whose per-turn numbers show reuse on four turns out of five.

    What holds in general is the rate the delta was evaluated at. Under reuse the
    duration is explained by the new tokens alone; under a re-prefill the same duration
    covers the whole prompt, so the delta's apparent rate collapses. The test is weak
    when a turn's delta is most of its prompt — `delta_fraction` says when to distrust
    it — and decisive in the case that matters, a small delta on a long prompt.
    """
    rows = [
        (int(t["prompt_eval_count"]), float(t["prompt_eval_ms"]))
        for t in turn_stats
        if t.get("prompt_eval_ms")
    ]
    if len(rows) < 2:
        return {"turns": len(rows), "verdict": "not measured"}

    floor = single_pass_tok_s * DELTA_RATE_FLOOR_FRACTION
    per_turn: list[dict[str, Any]] = []
    re_prefilled: list[int] = []
    for index, ((previous, _), (count, ms)) in enumerate(itertools.pairwise(rows), 2):
        delta = count - previous
        delta_rate = delta / (ms / 1000) if ms else 0.0
        whole_rate = count / (ms / 1000) if ms else 0.0
        # Either test alone misreads one end of the range: the delta rate is weak when
        # the delta is most of the prompt, and the whole-prompt rate is weak when the
        # turn appends almost nothing. A re-prefill fails both — it evaluates the whole
        # prompt at the single-pass rate by definition.
        reused = delta_rate >= floor or whole_rate >= 2.0 * single_pass_tok_s
        if not reused:
            re_prefilled.append(index)
        per_turn.append(
            {
                "turn": index,
                "count": count,
                "delta": delta,
                "ms": round(ms, 1),
                "delta_tok_s": round(delta_rate, 1),
                "whole_tok_s": round(whole_rate, 1),
                "delta_fraction": round(delta / count, 2) if count else 0.0,
                "reused": reused,
            }
        )
    kept = len(per_turn) - len(re_prefilled)
    return {
        "turns": len(rows),
        "compared": len(per_turn),
        "reused_turns": kept,
        "re_prefilled_turns": re_prefilled,
        "delta_rate_floor_tok_s": round(floor, 1),
        "single_pass_reference_tok_s": single_pass_tok_s,
        "per_turn": per_turn,
        "verdict": (
            f"reused on {kept} of {len(per_turn)} turns"
            if kept
            else "no reuse detected"
        ),
    }


def _strip_tool_calls(content: str) -> str:
    """Never send a `<tool_call>` block back inside `content`.

    On the salvage path the block is text the harness parsed itself; returning it in
    `content` alongside the structured call renders it twice, and the second copy is
    prose the template forbids after a call.
    """
    return _TOOL_CALL_BLOCK.sub("", content).strip()


def _assistant_message(turn_content: str, calls: list[ToolCall]) -> dict[str, Any]:
    """The assistant turn as it goes back — reasoning excluded, always.

    Reasoning kept in its own field is dropped by the template and costs nothing.
    Inlined into `content` it is re-rendered into the prompt every turn for the rest of
    the episode: 42 prompt tokens as a field against 703 inlined, measured on the same
    three-message array. The pinned user message at [1] is what makes it *every* turn.
    """
    message: dict[str, Any] = {
        "role": "assistant",
        "content": _strip_tool_calls(turn_content),
    }
    if calls:
        message["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in calls
        ]
    return message


def _evict(messages: list[dict[str, Any]]) -> int:
    """Drop the oldest un-elided tool observation. Returns the lines given up.

    Observations, oldest first, never the pinned prompt: this is the one operation that
    breaks the strict prefix, so it is counted and the episode carrying it is flagged
    when its throughput is read.
    """
    for message in messages[2:]:
        if message.get("role") != "tool":
            continue
        body = str(message.get("content") or "")
        if body.startswith("[observation elided"):
            continue
        lines = len(body.splitlines())
        name = message.get("tool_name") or "a tool"
        message["content"] = f"[observation elided: {lines} lines from {name}]"
        return lines
    return 0


def run_task(
    task: Task,
    transport: Transport,
    pack: PromptPack,
    *,
    sha: str,
    min_evidence: int = 1,
    max_nudges: int = 2,
    answer_reviews: int = 0,
    review_think: bool = False,
    handed: tuple[str, ...] = (),
    hide_answer_key: bool = True,
    evict_at: float = 0.75,
    wrapper_override: str = "",
    patch_gate: int = 3,
    idle_notice_every: int = 10,
    max_idle_notices: int = 3,
    think_off_after: int = 0,
    single_pass_tok_s: float = SINGLE_PASS_PREFILL_TOK_S,
    verbose: bool = True,
) -> HarnessRun:
    run = HarnessRun(task_id=task.id, tier=task.tier, arm=ARM, family=task.family)
    run.anchors_total = task.anchor_total
    worktree = make_worktree(sha, f"a1h-{task.id}")
    box = tools.Toolbox(
        worktree,
        truncation_note=pack.files["observation_truncated.md"],
        hide_answer_key=hide_answer_key,
    )
    # Normally the wrapper follows `kind`, and the override exists for one experiment:
    # a patch task run under the answer wrapper. `task_patch.md` prescribes editing in
    # five numbered steps and says the diff is the deliverable, and the arm under it
    # calls neither write tool — while the same model under `task_answer.md`, which
    # says nothing about editing, edits three files and verifies with `git diff`
    # unprompted. Wrapper and task both differed there, so it was a confound until this
    # flag could hold the task fixed.
    wrapper = wrapper_override or (
        "task_patch.md" if task.kind == "patch" else "task_answer.md"
    )
    task_message = pack.render(
        wrapper,
        task_prompt=task.prompt,
        worktree=str(worktree),
        max_turns=task.max_turns,
    )
    if handed:
        # The Claude Code arm auto-loads `CLAUDE.md` for *both* its arms — that is stated
        # in its own module docstring, and the low tier is designed around it. This arm
        # replaced the whole 26.7 k preamble with a 2 k prompt and dropped the repository's
        # instruction file with it, which was never on the declared-delta list. So it was
        # an undeclared difference in the model's context, not a design choice, and the
        # tier it costs most is the one whose answers are written down in that file: the
        # `lm-format-enforcer` name, the 0.81-against-1.00 gate figure and the property
        # `tests/fake_mlx.py` exists to hold are all in it and nowhere in the file the
        # question points at.
        #
        # It is concatenated rather than `format`-substituted because the file contains
        # brace-bearing JSON (`{"models":[]}`) and `str.format` would raise on it.
        blocks = [pack.render("handed_context.md", path=", ".join(handed))]
        for relative in handed:
            body = (worktree / relative).read_text()
            blocks.append(f"--- {relative} ---\n{body}\n--- end {relative} ---")
            run.handed_context_chars += len(body)
        task_message = "\n\n".join([*blocks, task_message])
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": pack.render(
                "system.md",
                tool_names=", ".join(tools.TOOL_NAMES),
                worktree=str(worktree),
                max_turns=task.max_turns,
            ),
        },
        {"role": "user", "content": task_message},
    ]
    definitions = tools.definitions()
    started = time.perf_counter()
    deadline = started + task.timeout_s
    # Applying a fixed checklist to text already written is not a reasoning task, and the
    # turn after a review is the one turn where that is known in advance. Left to think,
    # the review turn is what made the gate cost 30% of the wall clock for four rubric
    # points; with thinking off this model answers directly.
    think_next: bool | None = None
    # Consecutive stalled turns, reset by any turn that calls a tool. The budget below
    # is `max_nudges` against *this*, not against the episode total, and the difference
    # decided five of 28 episodes in the last full run. Every one of them ended
    # `nudges=2, capped=3` — and the three capped turns were spread out and recovered
    # from, `high-04`'s at turns 30, 47 and 58 of 59 with 17 and 11 productive turns
    # between them. §6.2 bounds this to stop "a loop burning a 100-turn budget arguing
    # with itself"; a stall the model climbs out of twice is not that loop, and counting
    # the episode total killed `high-01` at turn 15 of 100 and `high-02` at turn 10.
    # `low-07`, whose caps fell on turns 3, 4 and 5, is the case the bound was written
    # for and a consecutive budget still catches it.
    stalls = 0
    # Thinking off for the rest of the episode once the cap has cut this many turns off
    # empty. **Measured over two full `change_code` runs and rejected: the default is 0
    # and this is kept only as a knob.** It was proposed against `high-02` spending
    # 167,936 of 182,347 output tokens on turns that emitted nothing, and it does stop
    # that — but it buys a worse loop. Across the 12 episodes of those two runs the
    # correlation is near-total: all 8 where it never tripped ended on a real `finish`
    # in 7-23 turns, and all 4 where it did ran long, 3 of them to the 100-turn cap,
    # producing `search minLength` x81, `read_file` x81 and `run` x76. That is §17.5's
    # finding — with no reasoning channel the model does not reason less — arriving
    # through the tools instead of through `content`, and it cost `mid-14` a patch that
    # had passed `pytest`, `ruff` and `mypy` on the two runs before it.
    think_locked = False

    def patch_missing() -> bool:
        """Whether the deliverable is a diff, there is none, and the gate has budget.

        Keyed on `kind`, never on `expects_diff`. `mid-08` is `deliver="code_change"`
        held at `kind="answer"` and its rubric's top criterion is *noticing the knob is
        inert*, so the answer it wants may correctly touch nothing — the same
        distinction `_collect_patch` draws before asserting `patch_produced`.
        """
        return (
            task.kind == "patch"
            and run.patch_blocked < patch_gate
            and not box.tree_changed
        )

    def force_final_answer() -> str:
        """One turn with thinking off, asked to conclude. Returns what it said."""
        messages.append(
            {
                "role": "tool",
                "tool_name": "harness",
                "content": pack.render("final_answer.md", evidence_count=box.evidence),
            }
        )
        forced = transport.chat(
            messages,
            definitions,
            # Floored, never the bare remaining wall budget: this also runs at the turn
            # and wall caps, where the remainder is zero or negative and the call would
            # fail instead of answering. Overrunning by a minute to record what an
            # episode found is the trade every other exit here already takes.
            timeout=max(FORCED_ANSWER_TIMEOUT_S, deadline - time.perf_counter()),
            think=False,
        )
        run.turns += 1
        run.turn_stats.append(forced.stats())
        run.input_tokens += forced.prompt_eval_count
        run.output_tokens += forced.eval_count
        # The forced turn may answer either way, and reading only `content` threw away a
        # correct answer once: asked to stop and conclude, the model called `finish` with
        # 193 tokens and the rescue recorded an empty string over the top of it. Tools
        # stay on the call because that `finish` is the contract working — and because
        # dropping them rewrites the system block and re-prefills the whole prompt.
        said = forced.content.strip()
        for call in forced.tool_calls:
            if call.name == "finish":
                box.counters.by_name["finish"] += 1
                said = str(call.arguments.get("answer") or "").strip() or said
                break
        return said

    while run.turns < task.max_turns:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            run.end_reason = f"wall cap at {task.timeout_s:.0f}s"
            break
        try:
            turn = transport.chat(
                messages,
                definitions,
                timeout=remaining,
                think=False if think_locked else think_next,
            )
            think_next = None
        except ContextOverflow as overflow:
            given_up = _evict(messages)
            if not given_up:
                run.end_reason = f"context exhausted: {overflow}"
                break
            run.evictions += 1
            run.observations_elided += 1
            continue
        except TruncatedToolCall as exc:
            # Retried once with thinking off rather than counted as a failure. Under
            # greedy sampling an identical retry reproduces the identical truncated
            # call, so the retry has to change something, and thinking is what the cap
            # was spent on — the same reasoning as the forced final answer below.
            run.truncated_tool_calls += 1
            if think_next is False:
                run.error = str(exc)
                run.end_reason = "tool call truncated twice"
                break
            think_next = False
            continue
        except TransportError as exc:
            run.error = str(exc)
            run.end_reason = "transport error"
            break

        run.turns += 1
        run.turn_stats.append(turn.stats())
        run.input_tokens += turn.prompt_eval_count
        run.output_tokens += turn.eval_count
        run.thinking_chars += len(turn.thinking)
        box.counters.salvaged += turn.salvaged
        if turn.done_reason == "length":
            run.length_capped += 1
        if not run.ttft_s:
            run.ttft_s = turn.ttft_s
        if run.turns == 1:
            transport.verify_keep_alive()

        if not turn.tool_calls:
            box.counters.no_tool_call += 1
            answer = turn.content.strip()
            if answer and box.evidence >= min_evidence:
                if patch_missing():
                    run.patch_blocked += 1
                    stalls = 0
                    messages.append(_assistant_message(turn.content, []))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": "harness",
                            "content": pack.render(
                                "patch_blocked.md",
                                turns_used=run.turns,
                                max_turns=task.max_turns,
                            ),
                        }
                    )
                    continue
                # The gate has to sit on this path too. With thinking on, this model
                # frequently answers in prose and never calls `finish` — 3 of 10 episodes
                # in the first review run, including the worst-scoring one — so a gate
                # hung only on the tool call reviews the episodes that needed it least.
                if run.answer_reviews < answer_reviews:
                    run.answer_reviews += 1
                    run.draft_answer = answer
                    messages.append(_assistant_message(turn.content, []))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": "harness",
                            "content": pack.render("answer_review.md"),
                        }
                    )
                    think_next = review_think
                    continue
                run.answer, run.answer_source = answer, "implicit"
                run.answer_revised = bool(
                    run.draft_answer and answer != run.draft_answer
                )
                run.end_reason = "answered without calling finish"
                break
            # A turn the cap cut off mid-reasoning is not prose and not a refusal to act:
            # `content` is empty because every token went to `thinking`. Nudging it about
            # "prose with no tool call" describes something that did not happen, and
            # recording its empty content as the answer throws the episode away — which is
            # exactly what happened to one trap task that had made 27 successful tool
            # calls and passed `no_edits`, scored 0/2 on an empty string.
            cut_off = turn.done_reason == "length" and not answer
            if cut_off:
                run.truncated_turns += 1
                # The same remedy the truncated-tool-call path already applies, for the
                # same reason: the cap was spent on reasoning, so reasoning is what the
                # next turn has to change. Measured across the last full run these turns
                # cost 12,288 of `high-01`'s 18,061 output tokens (68%) and 12,288 of
                # `high-02`'s 14,774 (83%), and every one of them emitted nothing at all.
                # One turn only — `think_next` is cleared as soon as it is spent, because
                # §17.5 measured a whole episode with thinking off as a trap: the model
                # does not reason less, it reasons in `content`.
                think_next = False
                if think_off_after and run.truncated_turns >= think_off_after:
                    if not think_locked:
                        run.think_off_from_turn = run.turns
                    think_locked = True
            stalls += 1
            run.stall_streak_max = max(run.stall_streak_max, stalls)
            if stalls > max_nudges and patch_missing():
                # Forcing an answer here is what produced the run's only confabulation:
                # asked to conclude with no work done, the model reported the test it had
                # been planning as written. On a patch task the tree is checkable, so say
                # what it holds instead of asking for a summary of what it does not.
                run.patch_blocked += 1
                stalls = 0
                messages.append(_assistant_message(turn.content, []))
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": "harness",
                        "content": pack.render(
                            "patch_blocked.md",
                            turns_used=run.turns,
                            max_turns=task.max_turns,
                        ),
                    }
                )
                continue
            if stalls > max_nudges:
                # One forced attempt with thinking off before giving up. The model has
                # already demonstrated it will spend the whole cap reasoning; with
                # thinking disabled it answers directly, measured at 31 tokens.
                if not run.final_answer_forced and box.evidence >= min_evidence:
                    run.final_answer_forced = True
                    messages.append(_assistant_message(turn.content, []))
                    answer = force_final_answer()
                run.answer, run.answer_source = (
                    answer,
                    (
                        "forced final answer"
                        if run.final_answer_forced
                        else "nudges exhausted"
                    ),
                )
                run.end_reason = (
                    "forced a final answer after the cap cut the turn off"
                    if run.final_answer_forced
                    else f"no tool call after {max_nudges} consecutive nudges"
                )
                break
            run.nudges += 1
            messages.append(_assistant_message(turn.content, []))
            # A `tool` message, not a `user` one, and this is measured rather than
            # tasteful. The template picks the last user-role message as the point after
            # which assistant turns are rendered differently, and it skips anything
            # wrapped as a tool response — so a user-role nudge moves that point and
            # invalidates the prefix from there on. It cost a full re-prefill: 66 new
            # tokens took 19.8 s against a 15,185-token prompt, a rate of 3.3 tok/s on
            # the delta where the surrounding turns ran at 732-1,188.
            if cut_off:
                nudge = pack.render(
                    "nudge_truncated.md",
                    tool_names=", ".join(tools.TOOL_NAMES),
                    turns_used=run.turns,
                    max_turns=task.max_turns,
                    max_output_tokens=transport.max_output_tokens,
                )
            else:
                nudge = pack.render(
                    "nudge_no_tool_call.md",
                    tool_names=", ".join(tools.TOOL_NAMES),
                    turns_used=run.turns,
                    max_turns=task.max_turns,
                )
            messages.append({"role": "tool", "tool_name": "harness", "content": nudge})
            continue

        messages.append(_assistant_message(turn.content, turn.tool_calls))
        stalls = 0
        finished = False
        for call in turn.tool_calls:
            if call.name == "finish":
                box.counters.by_name["finish"] += 1
                if box.evidence < min_evidence:
                    run.finish_blocked += 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": "finish",
                            "content": pack.render(
                                "finish_blocked.md",
                                evidence_count=box.evidence,
                                min_evidence=min_evidence,
                            ),
                        }
                    )
                    continue
                # The mirror of the evidence gate, on the other end of the episode.
                # `min_evidence` refuses an answer with nothing read behind it; this
                # refuses one with nothing written behind it, and it is bounded for the
                # same reason — an agent that cannot do the work must still be allowed
                # to say so rather than spend a 100-turn budget being told no.
                if patch_missing():
                    run.patch_blocked += 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": "finish",
                            "content": pack.render(
                                "patch_blocked.md",
                                turns_used=run.turns,
                                max_turns=task.max_turns,
                            ),
                        }
                    )
                    continue
                draft = str(call.arguments.get("answer") or "").strip()
                # One bounded revision, and the checklist it carries is the same text the
                # system prompt already holds. The prompt-only form of it did not take:
                # low-01 read the pyproject comment that says "from 59 rules to 413",
                # answered "a twentieth of the rules" from the next line of the same
                # comment, and dropped the figure. This delivers it as the last thing in
                # the context before the answer is written instead of the first.
                if draft and run.answer_reviews < answer_reviews:
                    run.answer_reviews += 1
                    run.draft_answer = draft
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": "finish",
                            "content": pack.render("answer_review.md"),
                        }
                    )
                    think_next = review_think
                    continue
                # Never worse than the draft: a revision turn that comes back empty must
                # not overwrite an answer the model had already written, which is the same
                # failure the forced-answer path was built for.
                run.answer = draft or run.draft_answer
                run.answer_revised = bool(
                    run.draft_answer and draft and draft != run.draft_answer
                )
                run.answer_source = "finish" if draft else "draft after review"
                run.end_reason = "finish"
                finished = True
                break
            result = box.call(call.name, call.arguments)
            messages.append(
                {"role": "tool", "tool_name": call.name, "content": result.text}
            )
        if finished:
            break

        # An observation, not a procedure. §16.1 and §17.11 both measured the
        # prescription being ignored — `task_patch.md` numbers these steps and `run`
        # calls went 7 to 0 between packs — while the three episodes that produced a
        # diff all reached a write tool by call 8 or 9 and the two that produced none
        # never called one at all. So the tree is reported rather than requested, on a
        # schedule, and bounded so a long healthy episode is not talked at.
        if (
            task.kind == "patch"
            and run.idle_notices < max_idle_notices
            and run.turns >= idle_notice_every * (run.idle_notices + 1)
            and not box.tree_changed
        ):
            run.idle_notices += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_name": "harness",
                    "content": pack.render(
                        "no_changes_yet.md",
                        turns_used=run.turns,
                        max_turns=task.max_turns,
                    ),
                }
            )

        # Proactive eviction, so the common case does not pay a wasted round trip. The
        # typed 400 on overflow is the backstop that makes a threshold miscalculation
        # survivable rather than fatal, and it is why this can be a cheap estimate:
        # last turn's own count, which is exactly the prompt that was just accepted.
        while turn.prompt_eval_count > evict_at * transport.num_ctx:
            if not _evict(messages):
                break
            run.evictions += 1
            run.observations_elided += 1
            turn.prompt_eval_count = int(turn.prompt_eval_count * 0.9)

    if not run.end_reason:
        run.end_reason = f"turn cap at {task.max_turns}"
    if not run.answer and run.draft_answer:
        run.answer, run.answer_source = run.draft_answer, "draft after review"
    # The same rescue as the stall path, on the exit it was missing from. `high-02` spent
    # its whole 100-turn budget investigating, never emitted prose and never called
    # `finish`, and the episode was recorded with a 0-character answer — no anchor, no
    # rubric, nothing to read. An episode with evidence behind it has something to say at
    # the cap, and the cap is just another way of running out.
    if (
        not run.answer
        and not run.error
        and not run.final_answer_forced
        and box.evidence >= min_evidence
    ):
        run.final_answer_forced = True
        run.answer = force_final_answer()
        run.answer_source = "forced final answer at the cap"
        run.end_reason = f"{run.end_reason}, forced a final answer"
    if not run.answer and run.answer_source != "finish":
        run.answer_source = run.answer_source or "none"

    run.wall_s = time.perf_counter() - started
    run.tool_uses = box.counters.total
    run.tool_calls = box.counters.as_dict()
    run.tool_log = box.log
    run.answer_key_reads = box.answer_key_reads
    run.answer_key_mentions = box.answer_key_mentions
    run.answer_key_blocked = box.answer_key_blocked
    run.prefix_reuse = prefix_reuse(run.turn_stats, single_pass_tok_s=single_pass_tok_s)
    run.ok = bool(run.answer.strip()) and not run.error
    run.anchors_hit, run.anchors_missing = anchors_found(task, run.answer)
    _collect_patch(task, run, worktree)
    drop_worktree(worktree)

    if verbose:
        print(
            f"  {ARM} {task.id:<26} {'ok ' if run.ok else 'ERR'} "
            f"{run.wall_s:7.1f}s  ttft {run.ttft_s:5.1f}s  "
            f"{run.turns:>2}t {run.tool_uses:>2}tc  out {run.output_tokens:>6}  "
            # Answer length, because with `think: false` this model reasons in `content`
            # instead of not reasoning: one mid-tier answer came back as 9,107 characters
            # of unresolved deliberation that scored both anchors on substring matches
            # while failing the task's four-sentence instruction outright.
            f"ans {len(run.answer):>5}c  "
            f"diff {len(run.diff):>5}B  "
            f"anchors {run.anchors_hit}/{run.anchors_total}  "
            f"reuse {run.prefix_reuse.get('verdict', '?')}  {run.end_reason}"
            + (f"  {run.error[:60]}" if run.error else ""),
            flush=True,
        )
    return run


def declared_deltas(
    *,
    sampling_mode: str,
    min_evidence: int,
    search_backend: str,
    think: bool | str,
    max_output_tokens: int,
    answer_reviews: int = 0,
    handed: tuple[str, ...] = (),
    hide_answer_key: bool = True,
    wrapper_override: str = "",
    max_nudges: int = 2,
    patch_gate: int = 0,
    idle_notice_every: int = 10,
    max_idle_notices: int = 3,
    think_off_after: int = 0,
) -> list[str]:
    """Every deliberate difference from the Claude Code arm. Anything absent is a bug."""
    deltas = [
        "no TodoWrite tool",
        "seven named tools replace the Claude Code allowlist",
        "compact system prompt instead of the Claude Code preamble",
        "observation truncation",
        f"per-turn output cap of {max_output_tokens} tokens",
        (
            f"{max_nudges} consecutive stalled turns end the episode; a tool call "
            f"resets the count"
        ),
        "thinking is disabled for the one turn after the cap cuts a turn off empty",
        "one forced final answer with thinking off when the episode ends without one",
        "write_file and edit_file results carry git diff --stat and the untracked paths",
        (
            "write_file refuses a file the episode did not create; edit_file is the "
            "only way to change one that was already there"
        ),
        (
            "an identical search, list_files or read_file is answered with a repeat "
            "notice after the second time, until a write resets it"
        ),
    ]
    if patch_gate:
        deltas.append(
            f"on a patch task, finish is refused while the working tree is unchanged, "
            f"up to {patch_gate} times"
        )
    if think_off_after:
        deltas.append(
            f"thinking is disabled for the rest of the episode once the cap has cut "
            f"{think_off_after} turns off empty"
        )
    if max_idle_notices:
        deltas.append(
            f"on a patch task, an unchanged working tree is reported as an observation "
            f"every {idle_notice_every} turns, up to {max_idle_notices} times"
        )
    if sampling_mode == "greedy":
        deltas.append("sampling diverges from the model card (greedy)")
    if min_evidence > 0:
        deltas.append(
            f"finish is blocked until {min_evidence} read_file/search calls succeed"
        )
    if answer_reviews > 0:
        deltas.append(
            f"{answer_reviews} bounded answer review(s), taken with thinking off, "
            f"before the first answer is recorded"
        )
    if hide_answer_key:
        deltas.append(
            "read_file and search hide the eval's own task module, which is in the "
            "worktree and whose anchor tuples a plain search returns"
        )
    deltas.append(
        f"{', '.join(handed)} handed in the pinned prefix, as Claude Code auto-loads it"
        if handed
        else "no repository instruction file in the prefix, where Claude Code auto-loads "
        "CLAUDE.md for both of its arms"
    )
    if search_backend != "rg":
        deltas.append(
            f"search runs {search_backend}; ripgrep is not installed on this host"
        )
    if not think:
        deltas.append("thinking disabled")
    elif isinstance(think, str):
        deltas.append(
            f"reasoning_effort pinned to {think}, where the model template defaults "
            f"it to xhigh"
        )
    if wrapper_override:
        deltas.append(
            f"every task wrapped in {wrapper_override} regardless of its kind, so a "
            f"patch task is not shown the five-step patch procedure"
        )
    return deltas


def artifact(
    runs: list[dict[str, Any]],
    *,
    sha: str,
    transport: Transport,
    pack: PromptPack,
    host: HostState,
    min_evidence: int,
    max_nudges: int,
    answer_reviews: int,
    handed: tuple[str, ...],
    hide_answer_key: bool,
    search_backend: str,
    wrapper_override: str = "",
    patch_gate: int = 0,
    idle_notice_every: int = 10,
    max_idle_notices: int = 3,
    think_off_after: int = 0,
    single_pass_tok_s: float = SINGLE_PASS_PREFILL_TOK_S,
) -> dict[str, Any]:
    """Schema-identical to the other arm's `run` output, plus a `harness` block."""
    return {
        "mode": "run",
        "arm": ARM,
        "label": label_for(transport.model),
        "sha": sha,
        "harness": {
            "version": VERSION,
            "endpoint": transport.endpoint,
            "model": transport.model,
            "model_digest": host.model_digest,
            "model_digest_source": host.model_digest_source,
            "ollama_version": host.ollama_version,
            "trained_context": host.trained_context,
            "prompt_pack": pack.name,
            "prompt_pack_sha": pack.sha,
            "sampling_mode": transport.sampling.mode,
            "sampling": transport.sampling.options(),
            "num_ctx": transport.num_ctx,
            "max_output_tokens": transport.max_output_tokens,
            "think": transport.think,
            "min_evidence": min_evidence,
            "max_nudges": max_nudges,
            "patch_gate": patch_gate,
            "idle_notice_every": idle_notice_every,
            "max_idle_notices": max_idle_notices,
            "think_off_after": think_off_after,
            "single_pass_prefill_tok_s": single_pass_tok_s,
            "answer_reviews": answer_reviews,
            "handed_context": list(handed),
            "hide_answer_key": hide_answer_key,
            "search_backend": search_backend,
            "wrapper_override": wrapper_override,
            "env": {
                **host.env,
                "keep_alive": transport.keep_alive,
                "keep_alive_verified": transport.keep_alive_verified,
            },
            "warnings": host.warnings,
            "declared_deltas": declared_deltas(
                sampling_mode=transport.sampling.mode,
                min_evidence=min_evidence,
                search_backend=search_backend,
                think=transport.think,
                max_output_tokens=transport.max_output_tokens,
                answer_reviews=answer_reviews,
                handed=handed,
                hide_answer_key=hide_answer_key,
                wrapper_override=wrapper_override,
                max_nudges=max_nudges,
                patch_gate=patch_gate,
                idle_notice_every=idle_notice_every,
                max_idle_notices=max_idle_notices,
                think_off_after=think_off_after,
            ),
        },
        "runs": runs,
    }
