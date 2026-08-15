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


def prefix_reuse(turn_stats: list[dict[str, Any]]) -> dict[str, Any]:
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

    per_turn: list[dict[str, Any]] = []
    re_prefilled: list[int] = []
    for index, ((previous, _), (count, ms)) in enumerate(itertools.pairwise(rows), 2):
        delta = count - previous
        delta_rate = delta / (ms / 1000) if ms else 0.0
        reused = delta_rate >= DELTA_RATE_FLOOR_TOK_S
        if not reused:
            re_prefilled.append(index)
        per_turn.append(
            {
                "turn": index,
                "count": count,
                "delta": delta,
                "ms": round(ms, 1),
                "delta_tok_s": round(delta_rate, 1),
                "whole_tok_s": round(count / (ms / 1000), 1) if ms else 0.0,
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
        "delta_rate_floor_tok_s": DELTA_RATE_FLOOR_TOK_S,
        "single_pass_reference_tok_s": SINGLE_PASS_PREFILL_TOK_S,
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
    wrapper = "task_patch.md" if task.kind == "patch" else "task_answer.md"
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

    while run.turns < task.max_turns:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            run.end_reason = f"wall cap at {task.timeout_s:.0f}s"
            break
        try:
            turn = transport.chat(
                messages, definitions, timeout=remaining, think=think_next
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
            if run.nudges >= max_nudges:
                # One forced attempt with thinking off before giving up. The model has
                # already demonstrated it will spend the whole cap reasoning; with
                # thinking disabled it answers directly, measured at 31 tokens.
                if not run.final_answer_forced and box.evidence >= min_evidence:
                    run.final_answer_forced = True
                    messages.append(_assistant_message(turn.content, []))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": "harness",
                            "content": pack.render(
                                "final_answer.md", evidence_count=box.evidence
                            ),
                        }
                    )
                    forced = transport.chat(
                        messages,
                        definitions,
                        timeout=max(1.0, deadline - time.perf_counter()),
                        think=False,
                    )
                    run.turns += 1
                    run.turn_stats.append(forced.stats())
                    run.input_tokens += forced.prompt_eval_count
                    run.output_tokens += forced.eval_count
                    # The forced turn may answer either way, and reading only `content`
                    # threw away a correct answer once: asked to stop and conclude, the
                    # model called `finish` with 193 tokens and the rescue recorded an
                    # empty string over the top of it. Tools stay on the call because
                    # that `finish` is the contract working — and because dropping them
                    # rewrites the system block and re-prefills the whole prompt.
                    answer = forced.content.strip()
                    for call in forced.tool_calls:
                        if call.name == "finish":
                            box.counters.by_name["finish"] += 1
                            answer = (
                                str(call.arguments.get("answer") or "").strip()
                                or answer
                            )
                            break
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
                    else f"no tool call after {max_nudges} nudges"
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
    if not run.answer and run.answer_source != "finish":
        run.answer_source = run.answer_source or "none"

    run.wall_s = time.perf_counter() - started
    run.tool_uses = box.counters.total
    run.tool_calls = box.counters.as_dict()
    run.tool_log = box.log
    run.answer_key_reads = box.answer_key_reads
    run.answer_key_mentions = box.answer_key_mentions
    run.answer_key_blocked = box.answer_key_blocked
    run.prefix_reuse = prefix_reuse(run.turn_stats)
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
    think: bool,
    max_output_tokens: int,
    answer_reviews: int = 0,
    handed: tuple[str, ...] = (),
    hide_answer_key: bool = True,
) -> list[str]:
    """Every deliberate difference from the Claude Code arm. Anything absent is a bug."""
    deltas = [
        "no TodoWrite tool",
        "seven named tools replace the Claude Code allowlist",
        "compact system prompt instead of the Claude Code preamble",
        "observation truncation",
        f"per-turn output cap of {max_output_tokens} tokens",
        "one forced final answer with thinking off when the cap cuts a turn off empty",
    ]
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
) -> dict[str, Any]:
    """Schema-identical to the other arm's `run` output, plus a `harness` block."""
    return {
        "mode": "run",
        "arm": ARM,
        "label": LABEL,
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
            "answer_reviews": answer_reviews,
            "handed_context": list(handed),
            "hide_answer_key": hide_answer_key,
            "search_backend": search_backend,
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
            ),
        },
        "runs": runs,
    }
