"""Local model vs Opus 5, same Claude Code, same repository (`agent_eval_tasks.py`).

Everything measured about a local model here so far reads it through a harness Orbit
wrote — `qcn_quality.py` posts to the engine directly, so it measures the model and
none of the agent loop around it. The owner's question is the other one: **put the
local model behind the coding agent people actually use, give it the same tasks in
the same repository, and see how far it is from Opus 5.**

**The comparison is honest because only one thing differs.** Both arms run the same
`claude` binary, the same prompt, the same tools, in a fresh detached worktree at the
same commit. The local arm changes `ANTHROPIC_BASE_URL` and nothing else. Three
consequences worth stating, because each would otherwise be mistaken for a model
difference:

  * **Project settings are off** (`--setting-sources user`). The repo's
    `SessionStart` hook runs `uv pip install -e .` against `CLAUDE_PROJECT_DIR`,
    which in an eval worktree repoints the editable install at a directory this tool
    then deletes. The `PostToolUse` hook is the other reason: it runs `ruff format`
    and `ruff check --fix` on every file the agent writes, so with it on, the lint
    metric measures the hook and scores every arm identically.
  * **`CLAUDE.md` is still loaded**, because only `--bare` suppresses it and both
    arms get it. It is most of the low tier's answer, which is that tier's point.
  * **Tools are allowlisted rather than bypassed**, identically for both arms. A
    headless session auto-denies anything outside the list, and a denied call is a
    tool error the model has to cope with — which is itself part of the measurement.

**Timing, and what this tool can and cannot see.** `--include-partial-messages` gives
per-turn event boundaries, and the intent was to split prefill from decode — per turn,
`message_start` to the first `content_block_delta` is prefill, that delta to
`message_stop` is decode. **That works for an incrementally streamed transport and
not for the local arm.** optiq generates the whole turn and then writes the event
sequence, so the two timestamps land together: a 599 s local session recorded a 29 ms
decode window, which reads as 25,602 tok/s. `Run.stream_observable` is the guard, and
the reported rate is **output tokens over wall clock** for both arms — honest, and
dominated by prefill on the local arm, which is what an agent turn there actually
costs. For the model's own prefill and decode rates, measured directly off a
transport that does stream, use `tools/quality/qcn_quality.py throughput`.

**Grading is two numbers that are never averaged into one.** Anchors are objective
and unarguable: facts the answer must contain, checked in `agent_eval_tasks`. The
rubric is an LLM judge, blind to which arm wrote which answer and with position
randomised per task. Patch tasks add a third, hardest number: the repository's own
`pytest`, `ruff` and `mypy`, run inside the patched worktree.

**The judge is Opus and one arm is Opus.** That is a self-preference bias and it is
not corrected here, only bounded: the anchor score and the check results are
judge-free, so a rubric that disagrees with them is visible rather than silent. Read
the three columns together; a claim resting on the rubric alone is not supported by
this tool.

    python tools/quality/agent_eval.py run --arm opus  --out var/agent-eval/opus.json
    python tools/quality/agent_eval.py run --arm local --out var/agent-eval/local.json
    python tools/quality/agent_eval.py judge \\
        var/agent-eval/opus.json var/agent-eval/local.json --out var/agent-eval/judged.json
    python tools/quality/agent_eval.py report var/agent-eval/judged.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# The repo root rather than this directory, so the module is
# `tools.quality.agent_eval_tasks` under both the interpreter and mypy. Importing it
# bare resolves at runtime and not under mypy, whose `mypy_path` is `src` alone and
# deliberately so.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.quality.agent_eval_tasks import Task, anchors_found, by_id, census, select

REPO = Path(__file__).resolve().parents[2]

# Identical for both arms. A headless session auto-denies anything absent here, so
# widening this changes what is being measured -- keep it in step across arms or the
# comparison is between two different agents.
ALLOWED_TOOLS = (
    "Read Grep Glob Edit Write TodoWrite "
    "Bash(pytest:*) Bash(ruff:*) Bash(mypy:*) Bash(python3:*) "
    "Bash(git diff:*) Bash(git status:*) Bash(git log:*) Bash(git show:*) "
    "Bash(ls:*) Bash(cat:*) Bash(rg:*) Bash(grep:*) Bash(find:*) Bash(head:*) Bash(tail:*)"
)
DISALLOWED_TOOLS = "WebSearch WebFetch"

# An arm carrying `base_url` is served locally: the env below repoints Claude Code at
# it and nothing else about the run changes. `opus` has none and goes to the real API.
#
# The two local arms reach Claude Code by different routes, and neither is a proxy for
# the other. `local` is mlx-optiq behind `tools/serve/qcn_cc_proxy.py`, which exists
# because optiq's Anthropic writer never closes the socket (operations.md §7.6).
# `a1` is ollama, which serves `/v1/messages` natively, but still needs the proxy for a
# different reason: Claude Code puts a system-role message inside `messages`, and this
# model's Jinja refuses that with a 500 before the model runs (`--hoist-system`).
ARMS: dict[str, dict[str, str]] = {
    "opus": {"model": "opus", "label": "Opus 5 (claude.ai)"},
    "local": {
        "model": "qwen3-coder-next",
        "label": "Qwen3-Coder-Next-4bit (optiq)",
        "base_url": "http://127.0.0.1:8082",
    },
    "a1": {
        # `num_ctx` is not settable per request through /v1/messages, and the published
        # model sets none, so it would serve ollama's small default against a ~26.7k
        # token Claude Code preamble. This tag is the same blobs with the window
        # raised — see the Modelfile step in the run notes.
        "model": "a1-eval",
        "label": "Agents-A1-4B-Q8 (ollama)",
        "base_url": "http://127.0.0.1:8086",
    },
}


@dataclass
class TurnTiming:
    prefill_s: float = 0.0
    decode_s: float = 0.0


@dataclass
class Run:
    """One (arm, task) session and everything measurable about it."""

    task_id: str
    tier: str
    arm: str
    ok: bool = False
    error: str = ""
    answer: str = ""
    wall_s: float = 0.0
    ttft_s: float = 0.0
    turns: int = 0
    tool_uses: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    cost_usd: float = 0.0
    anchors_hit: int = 0
    anchors_total: int = 0
    anchors_missing: list[str] = field(default_factory=list)
    diff: str = ""
    files_touched: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def stream_observable(self) -> bool:
        """Whether this arm's transport actually streamed, or only looked like it.

        optiq's Anthropic writer generates the full turn and *then* emits the event
        sequence, so `message_start` and the first delta land together and the whole
        decode window collapses — measured at 29 ms against a 599 s session, which
        reads as 25,602 tok/s. The split below is real for an incrementally streamed
        transport and meaningless for a buffered one, so it is reported only when
        the events were spread across a meaningful part of the session.
        """
        return self.decode_s > 0.05 * self.wall_s

    @property
    def decode_tok_per_s(self) -> float:
        """Decode rate, or 0.0 where the transport buffered and it is unmeasurable."""
        if not self.stream_observable or not self.decode_s:
            return 0.0
        return self.output_tokens / self.decode_s

    @property
    def session_tok_per_s(self) -> float:
        """Output tokens over the whole session — comparable across both arms.

        Dominated by prefill on the local arm, which is the point: it is what the
        agent loop actually costs, not what the model does once it is generating.
        `tools/quality/qcn_quality.py throughput` measures the model's own prefill
        and decode rates directly, and that is where a clean tok/s comes from.
        """
        return self.output_tokens / self.wall_s if self.wall_s else 0.0

    @property
    def anchor_rate(self) -> float:
        return self.anchors_hit / self.anchors_total if self.anchors_total else 0.0


# --- worktree ---------------------------------------------------------------


def head_sha(rev: str = "HEAD") -> str:
    """Resolve a revision to a full sha.

    Pinnable rather than always HEAD because the two arms must run against the
    same tree to be comparable, and HEAD moves underneath a long run — this repo
    took two commits between the Opus arm finishing and the local arm restarting,
    one of them from a different session working in the same checkout.
    """
    out = subprocess.run(
        ["git", "rev-parse", rev],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def make_worktree(sha: str, tag: str) -> Path:
    """A detached worktree at `sha`, so a patch task cannot touch the real checkout.

    `core.hooksPath` is neutralised for the same reason `eval/worktree.py` does it:
    `git worktree add` runs `post-checkout`, and this module's premise is that no
    repository code runs except the checks it invokes deliberately.
    """
    root = Path(tempfile.mkdtemp(prefix=f"orbit-agenteval-{tag}-"))
    path = root / "wt"
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "worktree",
            "add",
            "--detach",
            "--quiet",
            str(path),
            sha,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return path


def drop_worktree(path: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    shutil.rmtree(path.parent, ignore_errors=True)


# --- one session ------------------------------------------------------------


def build_env(arm: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("CLAUDE_CODE_SIMPLE", None)
    base_url = ARMS[arm].get("base_url")
    if not base_url:
        return env
    env.pop("ANTHROPIC_API_KEY", None)
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_AUTH_TOKEN"] = "sk-optiq-local"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    # Nothing is emitted during prefill, so time-to-first-byte *equals* prefill time
    # and the client's idle timeout is what kills a long first turn. The effective
    # value is max(env, 300000); a 128k prefill at ~175 tok/s is ~745 s
    # (operations.md §7.6), which the floor alone would not survive.
    env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] = "2400000"
    return env


def _usage_from(node: dict[str, Any]) -> dict[str, int]:
    usage = node.get("usage") or {}
    return {
        "input": int(usage.get("input_tokens") or 0),
        "output": int(usage.get("output_tokens") or 0),
        "cache_read": int(usage.get("cache_read_input_tokens") or 0),
    }


def run_one(task: Task, arm: str, sha: str, *, verbose: bool = True) -> Run:
    run = Run(task_id=task.id, tier=task.tier, arm=arm)
    run.anchors_total = task.anchor_total
    worktree = make_worktree(sha, f"{arm}-{task.id}")
    cmd = [
        "claude",
        "-p",
        task.prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model",
        ARMS[arm]["model"],
        "--max-turns",
        str(task.max_turns),
        "--setting-sources",
        "user",
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        ALLOWED_TOOLS,
        "--disallowedTools",
        DISALLOWED_TOOLS,
    ]
    env = build_env(arm)
    # Without this the worktree has no editable install (the SessionStart hook that
    # would provide one is deliberately off), so the agent's own `pytest` would
    # import `orbit` from the real checkout and test the wrong tree.
    env["PYTHONPATH"] = str(worktree / "src")

    started = time.perf_counter()
    turn = TurnTiming()
    turn_open = False
    saw_delta = False
    text_parts: list[str] = []
    watchdog: threading.Timer | None = None
    expired = threading.Event()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=worktree,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # A watchdog rather than a deadline checked in the loop below: iterating
        # `proc.stdout` blocks inside readline, and the local arm emits nothing for
        # minutes at a time while it prefills, so an in-loop check never runs when it
        # is needed. `proc.wait(timeout=...)` was the only enforcement and it happens
        # after the stream closes — high-03 ran 68 minutes against a 60-minute cap
        # and was stopped by Claude Code's idle timeout, not by this tool.
        def expire() -> None:
            expired.set()
            with contextlib.suppress(OSError):
                proc.kill()

        watchdog = threading.Timer(task.timeout_s, expire)
        watchdog.daemon = True
        watchdog.start()

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            now = time.perf_counter() - started
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")

            if kind == "stream_event":
                inner = event.get("event") or {}
                itype = inner.get("type")
                if itype == "message_start":
                    turn = TurnTiming(prefill_s=now)
                    turn_open, saw_delta = True, False
                elif itype == "content_block_delta" and turn_open and not saw_delta:
                    saw_delta = True
                    run.prefill_s += now - turn.prefill_s
                    turn.decode_s = now
                    if not run.ttft_s:
                        run.ttft_s = now
                elif itype == "message_stop" and turn_open:
                    if saw_delta:
                        run.decode_s += now - turn.decode_s
                    turn_open = False

            elif kind == "assistant":
                message = event.get("message") or {}
                for block in message.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(str(block.get("text") or ""))
                    elif block.get("type") == "tool_use":
                        run.tool_uses += 1

            elif kind == "result":
                run.wall_s = time.perf_counter() - started
                run.turns = int(event.get("num_turns") or 0)
                run.cost_usd = float(event.get("total_cost_usd") or 0.0)
                totals = _usage_from(event)
                run.input_tokens = totals["input"]
                run.output_tokens = totals["output"]
                run.cache_read_tokens = totals["cache_read"]
                run.ok = not event.get("is_error")
                if event.get("subtype") not in (None, "success"):
                    run.error = str(event.get("subtype"))
                final = event.get("result")
                if isinstance(final, str) and final.strip():
                    # On a failed session `result` carries Claude Code's own error
                    # text, not the model's answer — "The model's tool call could
                    # not be parsed (retry also failed)" is what the local arm
                    # returns when it emits a malformed call twice. Storing that as
                    # `answer` hands the judge an error string to grade as prose and
                    # scores a harness-visible failure as a bad reply.
                    if run.ok:
                        run.answer = final
                    else:
                        run.error = f"{run.error + ': ' if run.error else ''}{final}"
        proc.wait(timeout=120)
        stderr = (proc.stderr.read() if proc.stderr else "") or ""
        if proc.returncode != 0 and not run.answer:
            run.ok = False
            run.error = run.error or f"exit {proc.returncode}: {stderr[-400:]}"
    except subprocess.TimeoutExpired:
        proc.kill()
        run.ok = False
        run.error = "did not exit within 120s of the stream closing"
    except (OSError, ValueError) as exc:
        run.ok = False
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        if watchdog is not None:
            watchdog.cancel()

    if expired.is_set():
        run.ok = False
        run.error = f"killed at the {task.timeout_s:.0f}s cap"

    if not run.wall_s:
        run.wall_s = time.perf_counter() - started
    if not run.answer:
        run.answer = "\n".join(p for p in text_parts if p.strip())

    run.anchors_hit, run.anchors_missing = anchors_found(task, run.answer)
    _collect_patch(task, run, worktree)
    drop_worktree(worktree)
    if verbose:
        print(
            f"  {arm:>5} {task.id:<26} {'ok ' if run.ok else 'ERR'} "
            f"{run.wall_s:7.1f}s  ttft {run.ttft_s:6.1f}s  "
            f"{run.turns:>2}t {run.tool_uses:>2}tc  "
            f"out {run.output_tokens:>6}  {run.session_tok_per_s:6.2f} tok/s  "
            f"anchors {run.anchors_hit}/{run.anchors_total}"
            + (f"  {run.error[:60]}" if run.error else ""),
            flush=True,
        )
    return run


def _collect_patch(task: Task, run: Run, worktree: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    # Not `.stdout.strip()`: porcelain status is a two-character code then a space,
    # and an unstaged modification writes the code as " M", so stripping the whole
    # output eats the first line's leading space and the path loses its first
    # character — recorded as "rc/orbit/backends/mock.py" once, and only ever on
    # the first line, which is what made it look like a path rather than a bug.
    run.files_touched = sorted(
        entry for line in status.splitlines() if (entry := line[2:].strip())
    )

    if "no_edits" in task.checks:
        # A trap task is failed by *doing* the thing, so an untouched tree is the
        # pass. Recorded as a check rather than an anchor because it is a property
        # of the worktree, not of the prose.
        run.checks["no_edits"] = {"passed": not run.files_touched}

    if task.kind != "patch":
        return

    subprocess.run(["git", "add", "-A"], cwd=worktree, capture_output=True, check=False)
    run.diff = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    ).stdout

    _run_checks(task, run, worktree)


def _run_checks(task: Task, run: Run, worktree: Path) -> None:
    """The repository's own commands, inside the patched worktree.

    Every tool is invoked through `sys.executable -m`, never bare on PATH. A session
    restart dropped the venv off PATH and `ruff` and `mypy` both came back
    `FileNotFoundError` recorded as `passed: False` — which in the result file is
    indistinguishable from a model that wrote code failing lint and type checks.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    for check in task.checks:
        if check == "pytest":
            run.checks["pytest"] = _shell(
                [sys.executable, "-m", "pytest", "-q"], worktree, env
            )
        elif check == "ruff":
            fmt = _shell(
                [sys.executable, "-m", "ruff", "format", "--check", "."], worktree, env
            )
            lint = _shell([sys.executable, "-m", "ruff", "check", "."], worktree, env)
            run.checks["ruff"] = {
                "passed": fmt["passed"] and lint["passed"],
                "format": fmt,
                "lint": lint,
            }
        elif check == "mypy":
            run.checks["mypy"] = _shell([sys.executable, "-m", "mypy"], worktree, env)


def _shell(cmd: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    try:
        out = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {"passed": False, "tail": f"{type(exc).__name__}: {exc}"}
    tail = (out.stdout + out.stderr).strip().splitlines()
    return {
        "passed": out.returncode == 0,
        "returncode": out.returncode,
        "tail": "\n".join(tail[-12:]),
    }


# --- judge ------------------------------------------------------------------

_JUDGE_INSTRUCTIONS = """You are grading two answers to the same task, written by two \
different AI coding agents working in the same repository. You do not know which \
model wrote which, and you must not guess or speculate about it.

Score each answer on every rubric criterion, 0-3:
  3 = fully meets the criterion as described
  2 = mostly, with a real gap
  1 = partially, or correct but for a wrong reason
  0 = does not meet it, or is wrong

Judge only against the criterion text. Do not reward length, confidence or polish. An \
answer that is short and correct beats a long one that buries the answer. If an answer \
was cut off or errored, score what is there.

Output ONE JSON object and nothing else, in exactly this shape:
{"A": {"<criterion name>": <0-3>, ...}, "B": {"<criterion name>": <0-3>, ...}, \
"note": "<one sentence on the clearest difference>"}"""


def judge_pair(
    task: Task, left: Run, right: Run, model: str, *, verbose: bool = True
) -> dict[str, Any]:
    """Blind rubric scoring of two answers, position randomised per task id."""
    rng = random.Random(task.id)
    flipped = rng.random() < 0.5
    a, b = (right, left) if flipped else (left, right)

    criteria = "\n".join(
        f"- {c.name} (weight {c.weight}): {c.guidance}" for c in task.rubric
    )
    prompt = (
        f"{_JUDGE_INSTRUCTIONS}\n\n## Task given to both agents\n\n{task.prompt}\n\n"
        f"## Rubric\n\n{criteria}\n\n"
        f"## Answer A\n\n{a.answer[:12000] or '(empty)'}\n\n"
        f"## Answer B\n\n{b.answer[:12000] or '(empty)'}\n"
    )
    if task.kind == "patch":
        prompt += (
            f"\n## Diff written by A\n\n```diff\n{a.diff[:12000] or '(none)'}\n```\n"
            f"\n## Diff written by B\n\n```diff\n{b.diff[:12000] or '(none)'}\n```\n"
        )

    out = subprocess.run(
        [
            "claude",
            "-p",
            prompt,
            "--model",
            model,
            "--output-format",
            "json",
            "--setting-sources",
            "user",
            "--allowedTools",
            "",
            "--disallowedTools",
            "WebSearch WebFetch Bash Read Edit Write Glob Grep",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    scores = _parse_judge(out.stdout)
    if scores is None:
        if verbose:
            print(f"  judge {task.id}: unparsable verdict", flush=True)
        return {"task_id": task.id, "error": "unparsable", "raw": out.stdout[-600:]}

    a_key, b_key = ("B", "A") if flipped else ("A", "B")
    return {
        "task_id": task.id,
        "flipped": flipped,
        "note": str(scores.get("note") or ""),
        "left": _weighted(task, scores.get(a_key) or {}),
        "right": _weighted(task, scores.get(b_key) or {}),
    }


def _parse_judge(stdout: str) -> dict[str, Any] | None:
    try:
        envelope = json.loads(stdout)
        body = envelope.get("result") if isinstance(envelope, dict) else None
    except ValueError:
        body = stdout
    if not isinstance(body, str):
        body = stdout
    match = re.search(r"\{.*\}", body, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _weighted(task: Task, raw: dict[str, Any]) -> dict[str, Any]:
    per: dict[str, int] = {}
    earned = possible = 0
    for c in task.rubric:
        value = raw.get(c.name)
        score = int(value) if isinstance(value, (int, float)) else 0
        score = max(0, min(3, score))
        per[c.name] = score
        earned += score * c.weight
        possible += 3 * c.weight
    return {
        "per_criterion": per,
        "earned": earned,
        "possible": possible,
        "rate": earned / possible if possible else 0.0,
    }


# --- report -----------------------------------------------------------------


def _checks_passed(run: dict[str, Any]) -> tuple[int, int]:
    checks = run.get("checks") or {}
    total = len(checks)
    passed = sum(1 for v in checks.values() if isinstance(v, dict) and v.get("passed"))
    return passed, total


def report(judged: dict[str, Any]) -> str:
    left_arm = judged["arms"]["left"]
    right_arm = judged["arms"]["right"]
    rows = judged["rows"]
    lines = [
        (
            f"commit {judged['sha'][:12]}   {len(rows)} tasks   "
            f"left={left_arm}  right={right_arm}"
        ),
        "",
        (
            f"{'task':<26} {'tier':<5} "
            f"{'anchors L/R':<12} {'rubric L/R':<12} {'checks L/R':<11} "
            f"{'wall L/R (s)':<16} {'tok/s L/R':<12}"
        ),
        "-" * 100,
    ]
    agg: dict[str, dict[str, float]] = {
        side: dict.fromkeys(
            (
                "anchor_hit",
                "anchor_tot",
                "rub_e",
                "rub_p",
                "chk_p",
                "chk_t",
                "wall",
                "out",
            ),
            0.0,
        )
        for side in ("left", "right")
    }
    for row in rows:
        left, right = row["left"], row["right"]
        rub = row.get("rubric") or {}
        lr = (rub.get("left") or {}).get("rate")
        rr = (rub.get("right") or {}).get("rate")
        lp, lt = _checks_passed(left)
        rp, rt = _checks_passed(right)
        for side, run in (("left", left), ("right", right)):
            p, t = _checks_passed(run)
            agg[side]["anchor_hit"] += run["anchors_hit"]
            agg[side]["anchor_tot"] += run["anchors_total"]
            agg[side]["chk_p"] += p
            agg[side]["chk_t"] += t
            agg[side]["wall"] += run["wall_s"]
            agg[side]["out"] += run["output_tokens"]
            scored = rub.get(side) or {}
            agg[side]["rub_e"] += scored.get("earned", 0)
            agg[side]["rub_p"] += scored.get("possible", 0)
        lines.append(
            f"{row['task_id']:<26} {left['tier']:<5} "
            f"{left['anchors_hit']}/{left['anchors_total']}"
            f" vs {right['anchors_hit']}/{right['anchors_total']}".ljust(52)
            + f"{_pct(lr)} vs {_pct(rr)}".ljust(13)
            + f"{lp}/{lt} vs {rp}/{rt}".ljust(12)
            + f"{left['wall_s']:.0f} vs {right['wall_s']:.0f}".ljust(17)
            + f"{_tps(left)} vs {_tps(right)}"
        )
    lines.append("-" * 100)
    for side, arm in (("left", left_arm), ("right", right_arm)):
        a = agg[side]
        lines.append(
            f"{arm:<24} anchors {a['anchor_hit']:.0f}/{a['anchor_tot']:.0f}"
            f"  rubric {_pct(a['rub_e'] / a['rub_p'] if a['rub_p'] else None)}"
            f"  checks {a['chk_p']:.0f}/{a['chk_t']:.0f}"
            f"  total wall {a['wall'] / 60:.1f} min"
            f"  output {a['out']:.0f} tok"
        )
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "  --" if value is None else f"{value * 100:3.0f}%"


def _tps(run: dict[str, Any]) -> str:
    """End-to-end, not decode: see `Run.stream_observable` for why."""
    wall = run.get("wall_s") or 0.0
    return f"{run['output_tokens'] / wall:.2f}" if wall else "--"


# --- cli --------------------------------------------------------------------


def _run_mode(args: argparse.Namespace) -> int:
    """Run an arm, checkpointing after every task so a lost machine costs one task.

    The local arm takes hours, and this host kernel-panics under sustained MLX GPU
    load — `completeMemory() prepare count underflow` at `IOGPUMemory.cpp:550`,
    twice in three days, a driver refcount bug rather than anything about headroom.
    The first local run lost thirteen completed tasks to it because results were
    only written at the end. They are written after each task now, and `--resume`
    reads them back.
    """
    tasks = select(tuple(args.tier), tuple(args.task))
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2
    sha = head_sha(args.at)
    c = census()

    done: dict[str, dict[str, Any]] = {}
    if args.resume and args.out and args.out.exists():
        prior = json.loads(args.out.read_text())
        if prior.get("sha") != sha:
            print(
                f"refusing to resume: {args.out} ran at {str(prior.get('sha'))[:12]}, "
                f"HEAD is {sha[:12]} — a different commit is a different measurement",
                file=sys.stderr,
            )
            return 2
        if prior.get("arm") != args.arm:
            print(
                f"refusing to resume: {args.out} holds arm {prior.get('arm')!r}",
                file=sys.stderr,
            )
            return 2
        done = {r["task_id"]: r for r in prior["runs"]}

    pending = [t for t in tasks if t.id not in done]
    print(
        f"arm={args.arm} ({ARMS[args.arm]['label']})  commit={sha[:12]}  "
        f"{len(pending)} to run of {len(tasks)} selected "
        f"({c.low + c.mid + c.high} in the set)"
        + (f", {len(done)} resumed" if done else ""),
        flush=True,
    )
    print()

    def checkpoint() -> None:
        ordered = [done[t.id] for t in select() if t.id in done]
        _write(
            args.out,
            {
                "mode": "run",
                "arm": args.arm,
                "label": ARMS[args.arm]["label"],
                "sha": sha,
                "runs": ordered,
            },
            quiet=True,
        )

    for task in pending:
        done[task.id] = asdict(run_one(task, args.arm, sha, verbose=True))
        checkpoint()

    checkpoint()
    finished = [done[t.id] for t in tasks if t.id in done]
    ok = sum(1 for r in finished if r["ok"])
    hit = sum(r["anchors_hit"] for r in finished)
    tot = sum(r["anchors_total"] for r in finished)
    print(
        f"\n{ok}/{len(finished)} sessions completed, anchors {hit}/{tot}, "
        f"wall {sum(r['wall_s'] for r in finished) / 60:.1f} min"
    )
    if args.out:
        print(f"wrote {args.out}")
    return 0


def _regrade_mode(args: argparse.Namespace) -> int:
    """Recompute anchor scores from stored answers, without re-running any model.

    An anchor set is a hypothesis about what a correct answer contains, and this
    one has already been wrong once: `high-03` required the phrase "sec 4.2" and
    scored 0/2 against a refusal that argued from measured cost instead. Re-running
    a model to re-grade text already on disk would be absurd, and on the local arm
    it would cost hours — so grading is separable from generation, and both arms
    are always regraded together.
    """
    changed = 0
    for path in args.files:
        payload = json.loads(path.read_text())
        for run in payload["runs"]:
            task = by_id(run["task_id"])
            hit, missing = anchors_found(task, run["answer"])
            if hit != run["anchors_hit"]:
                print(
                    f"  {path.name}:{run['task_id']} "
                    f"{run['anchors_hit']}/{run['anchors_total']} -> "
                    f"{hit}/{task.anchor_total}"
                )
                changed += 1
            run["anchors_hit"] = hit
            run["anchors_total"] = task.anchor_total
            run["anchors_missing"] = missing
        path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"regraded {len(args.files)} file(s), {changed} score(s) changed")
    return 0


def _recheck_mode(args: argparse.Namespace) -> int:
    """Re-run the repository's checks against stored diffs, with no model calls.

    Same principle as `regrade`: the expensive half is generating the patch, and
    verifying it is cheap and repeatable. It exists because the checks have already
    been wrong once for a reason that had nothing to do with any model — `ruff` and
    `mypy` were invoked bare on PATH, a session restart dropped the venv, and both
    recorded `passed: False` from `FileNotFoundError`.
    """
    for path in args.files:
        payload = json.loads(path.read_text())
        sha = payload["sha"]
        for record in payload["runs"]:
            task = by_id(record["task_id"])
            if task.kind != "patch":
                continue
            if not record.get("diff"):
                # "No patch produced" and "patch failed lint" are different results
                # and were being reported as the same F. A session killed before it
                # wrote anything leaves stale check entries behind; replace them.
                record["checks"] = {
                    "patch_produced": {"passed": False, "tail": "no diff produced"}
                }
                print(f"  {path.name}:{record['task_id']}  no diff — checks cleared")
                continue
            worktree = make_worktree(sha, f"recheck-{record['task_id']}")
            try:
                applied = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    cwd=worktree,
                    input=record["diff"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if applied.returncode != 0:
                    record["checks"]["apply"] = {
                        "passed": False,
                        "tail": applied.stderr[-400:],
                    }
                    print(f"  {path.name}:{record['task_id']} diff did not apply")
                    continue
                stub = Run(task_id=task.id, tier=task.tier, arm=payload["arm"])
                stub.checks = dict(record.get("checks") or {})
                _run_checks(task, stub, worktree)
                record["checks"] = stub.checks
                summary = "  ".join(
                    f"{k}={'P' if v.get('passed') else 'F'}"
                    for k, v in stub.checks.items()
                )
                print(f"  {path.name}:{record['task_id']}  {summary}")
            finally:
                drop_worktree(worktree)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {path}")
    return 0


def _judge_mode(args: argparse.Namespace) -> int:
    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    if left["sha"] != right["sha"]:
        print(
            f"refusing: arms ran at different commits "
            f"({left['sha'][:12]} vs {right['sha'][:12]})",
            file=sys.stderr,
        )
        return 2
    by_id_left = {r["task_id"]: r for r in left["runs"]}
    by_id_right = {r["task_id"]: r for r in right["runs"]}
    shared = [t for t in select() if t.id in by_id_left and t.id in by_id_right]
    print(f"judging {len(shared)} paired tasks with --model {args.judge_model}\n")

    rows = []
    for task in shared:
        lrun = Run(**by_id_left[task.id])
        rrun = Run(**by_id_right[task.id])
        verdict = judge_pair(task, lrun, rrun, args.judge_model)
        rows.append(
            {
                "task_id": task.id,
                "tier": task.tier,
                "kind": task.kind,
                "left": by_id_left[task.id],
                "right": by_id_right[task.id],
                "rubric": verdict,
                "judge_note": verdict.get("note", ""),
            }
        )
        rub = verdict.get("left"), verdict.get("right")
        print(
            f"  {task.id:<26} rubric "
            f"{_pct((rub[0] or {}).get('rate'))} vs {_pct((rub[1] or {}).get('rate'))}"
            f"   {verdict.get('note', '')[:70]}",
            flush=True,
        )

    payload = {
        "mode": "judge",
        "sha": left["sha"],
        "judge_model": args.judge_model,
        "judge_caveat": (
            "The judge is Opus and one arm is Opus; self-preference bias is bounded "
            "by the anchor and check columns, which are judge-free, not corrected."
        ),
        "arms": {"left": left["label"], "right": right["label"]},
        "rows": rows,
    }
    _write(args.out, payload)
    print("\n" + report(payload))
    return 0


def _report_mode(args: argparse.Namespace) -> int:
    print(report(json.loads(args.judged.read_text())))
    return 0


def _write(path: Path | None, payload: dict[str, Any], *, quiet: bool = False) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a checkpoint runs after every task, and a machine that
    # dies mid-write would otherwise leave a truncated file where the completed
    # results were — losing exactly what the checkpoint exists to protect.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
    if not quiet:
        print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="run one arm over the task set")
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument(
        "--tier", action="append", default=[], choices=("low", "mid", "high")
    )
    r.add_argument("--task", action="append", default=[])
    r.add_argument("--out", type=Path)
    r.add_argument(
        "--at",
        default="HEAD",
        help="revision to run against; pin it so both arms share a tree",
    )
    r.add_argument(
        "--resume",
        action="store_true",
        help="keep results already in --out and run only what is missing",
    )
    r.set_defaults(fn=_run_mode)

    j = sub.add_parser("judge", help="blind rubric scoring of two arms")
    j.add_argument("left", type=Path)
    j.add_argument("right", type=Path)
    j.add_argument("--judge-model", default="opus")
    j.add_argument("--out", type=Path)
    j.set_defaults(fn=_judge_mode)

    c = sub.add_parser("recheck", help="re-run repo checks against stored diffs")
    c.add_argument("files", type=Path, nargs="+")
    c.set_defaults(fn=_recheck_mode)

    g = sub.add_parser("regrade", help="recompute anchors from stored answers")
    g.add_argument("files", type=Path, nargs="+")
    g.set_defaults(fn=_regrade_mode)

    p = sub.add_parser("report", help="print the table from a judged file")
    p.add_argument("judged", type=Path)
    p.set_defaults(fn=_report_mode)

    args = ap.parse_args()
    result: int = args.fn(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
