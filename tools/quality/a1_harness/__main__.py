"""CLI. The pilot ladder is the interface: `preflight`, then `probe`, then `run --task`.

    python -m tools.quality.a1_harness preflight
    python -m tools.quality.a1_harness probe --tools --prompt 'read pyproject.toml'
    python -m tools.quality.a1_harness run --task low-01-ruff-pin \\
        --out var/agent-eval/a1-harness.json

Never open with the full task set. One call, then one task, then a tier, then the
fifteen — each measurement is what authorises the next step, and this host wedges rather
than failing cleanly when unified memory is overcommitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.quality.a1_harness import tools
from tools.quality.a1_harness.loop import (
    ARM,
    PromptPack,
    artifact,
    label_for,
    run_task,
)
from tools.quality.a1_harness.transport import (
    DEFAULT_HOST,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    SAMPLINGS,
    THINK_EFFORTS,
    Transport,
    TransportError,
)
from tools.quality.agent_eval import _write, head_sha
from tools.quality.agent_eval_tasks import census, select


def _think(choice: str) -> bool | str:
    """`on`/`off` stay booleans; an effort level goes to the wire as ollama's string."""
    return {"on": True, "off": False}.get(choice, choice)


def _transport(args: argparse.Namespace) -> Transport:
    return Transport(
        host=args.host,
        model=args.model,
        num_ctx=args.num_ctx,
        think=_think(args.think),
        sampling=SAMPLINGS[args.sampling],
        keep_alive=args.keep_alive,
        openai_compat=args.openai_compat,
        max_output_tokens=args.max_output_tokens,
    )


def _preflight_mode(args: argparse.Namespace) -> int:
    transport = _transport(args)
    try:
        host = transport.preflight()
    except TransportError as exc:
        print(f"preflight refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(host), indent=2))
    for entry in transport.resident():
        print(
            f"resident: {entry.get('name')} "
            f"{float(entry.get('size') or 0) / 2**30:.2f} GiB "
            f"ctx={entry.get('context_length')} expires={entry.get('expires_at')}"
        )
    return 0


def _probe_mode(args: argparse.Namespace) -> int:
    """One turn, printed as wire facts. The rung between preflight and a whole task."""
    transport = _transport(args)
    try:
        host = transport.preflight()
    except TransportError as exc:
        print(f"preflight refused: {exc}", file=sys.stderr)
        return 2
    definitions = tools.definitions() if args.tools else None
    messages: list[dict[str, Any]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    try:
        turn = transport.chat(messages, definitions, timeout=args.timeout)
    except TransportError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1
    transport.verify_keep_alive()
    print(
        json.dumps(
            {
                "endpoint": transport.endpoint,
                "ollama_version": host.ollama_version,
                "tools_passed": len(definitions or []),
                "keep_alive_verified": transport.keep_alive_verified,
                "stats": turn.stats(),
                "thinking_chars": len(turn.thinking),
                "content": turn.content[:1200],
                "tool_calls": [
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "salvaged": call.salvaged,
                    }
                    for call in turn.tool_calls
                ],
                "trailing_prose_dropped": turn.trailing_prose_dropped,
            },
            indent=2,
        )
    )
    for entry in transport.resident():
        print(
            f"resident: {float(entry.get('size') or 0) / 2**30:.2f} GiB "
            f"ctx={entry.get('context_length')}",
            file=sys.stderr,
        )
    return 0


def _run_mode(args: argparse.Namespace) -> int:
    """Checkpoint after every task: a lost machine should cost one task, not a run."""
    tasks = select(tuple(args.tier), tuple(args.task), tuple(args.family))
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2
    try:
        pack = PromptPack.load(args.prompt_pack)
    except ValueError as exc:
        print(f"prompt pack: {exc}", file=sys.stderr)
        return 2
    for flag, required in (
        ("--answer-review", "answer_review.md" if args.answer_review else ""),
        ("--handed-context", "handed_context.md" if args.handed_context else ""),
        ("--patch-gate", "patch_blocked.md" if args.patch_gate else ""),
        ("--idle-notices", "no_changes_yet.md" if args.idle_notices else ""),
    ):
        if required and required not in pack.files:
            print(
                f"{flag} needs {required}, which pack {pack.name} does not carry",
                file=sys.stderr,
            )
            return 2

    transport = _transport(args)
    try:
        host = transport.preflight()
    except TransportError as exc:
        print(f"preflight refused: {exc}", file=sys.stderr)
        return 2
    for warning in host.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    sha = head_sha(args.at)
    done: dict[str, dict[str, Any]] = {}
    if args.resume and args.out and args.out.exists():
        prior = json.loads(args.out.read_text())
        if prior.get("sha") != sha:
            print(
                f"refusing to resume: {args.out} ran at {str(prior.get('sha'))[:12]}, "
                f"asked for {sha[:12]} — a different commit is a different measurement",
                file=sys.stderr,
            )
            return 2
        if prior.get("arm") != ARM:
            print(
                f"refusing to resume: {args.out} holds arm {prior.get('arm')!r}",
                file=sys.stderr,
            )
            return 2
        prior_model = str((prior.get("harness") or {}).get("model") or "")
        if prior_model and prior_model != transport.model:
            print(
                f"refusing to resume: {args.out} ran {prior_model}, asked for "
                f"{transport.model} — a different model is a different arm",
                file=sys.stderr,
            )
            return 2
        done = {record["task_id"]: record for record in prior["runs"]}

    pending = [task for task in tasks if task.id not in done]
    print(
        f"arm={ARM} ({label_for(transport.model)})  commit={sha[:12]}  "
        f"pack={pack.name}@{pack.sha}  "
        f"num_ctx={transport.num_ctx}  think={transport.think}  "
        f"sampling={transport.sampling.mode}  min_evidence={args.min_evidence}  "
        f"nudges={args.nudges}  patch_gate={args.patch_gate}  "
        f"idle_notices={args.idle_notices}/{args.idle_notice_every}t  "
        f"think_off_after={args.think_off_after}  "
        f"answer_review={args.answer_review}  "
        f"handed={','.join(args.handed_context) or 'none'}\n"
        f"{len(pending)} to run of {len(tasks)} selected"
        + (f", {len(done)} resumed" if done else ""),
        flush=True,
    )
    print()

    search_backend = tools.search_backend()

    def checkpoint() -> None:
        ordered = [done[task.id] for task in select() if task.id in done]
        _write(
            args.out,
            artifact(
                ordered,
                sha=sha,
                transport=transport,
                pack=pack,
                host=host,
                min_evidence=args.min_evidence,
                max_nudges=args.nudges,
                answer_reviews=args.answer_review,
                handed=tuple(args.handed_context),
                hide_answer_key=not args.show_answer_key,
                search_backend=search_backend,
                wrapper_override=args.wrapper,
                patch_gate=args.patch_gate,
                idle_notice_every=args.idle_notice_every,
                max_idle_notices=args.idle_notices,
                think_off_after=args.think_off_after,
            ),
            quiet=True,
        )

    for task in pending:
        run = run_task(
            task,
            transport,
            pack,
            sha=sha,
            min_evidence=args.min_evidence,
            max_nudges=args.nudges,
            answer_reviews=args.answer_review,
            handed=tuple(args.handed_context),
            hide_answer_key=not args.show_answer_key,
            wrapper_override=args.wrapper,
            patch_gate=args.patch_gate,
            idle_notice_every=args.idle_notice_every,
            max_idle_notices=args.idle_notices,
            think_off_after=args.think_off_after,
        )
        done[task.id] = asdict(run)
        checkpoint()

    checkpoint()
    finished = [done[task.id] for task in tasks if task.id in done]
    ok = sum(1 for record in finished if record["ok"])
    hit = sum(record["anchors_hit"] for record in finished)
    total = sum(record["anchors_total"] for record in finished)
    print(
        f"\n{ok}/{len(finished)} episodes completed, anchors {hit}/{total}, "
        f"wall {sum(record['wall_s'] for record in finished) / 60:.1f} min"
    )
    if args.out:
        print(f"wrote {args.out}")
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument(
        "--think",
        choices=("on", "off", *THINK_EFFORTS),
        default="on",
        help="an effort level is Qwen3.8's reasoning_effort, whose template default "
        "is xhigh, so `on` is the most expensive setting and not a neutral one",
    )
    parser.add_argument("--sampling", choices=sorted(SAMPLINGS), default="greedy")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="per-turn decode cap; uncapped, greedy ran to 13.5k tokens on one question",
    )
    parser.add_argument(
        "--openai-compat",
        action="store_true",
        help="portability check only: /v1 re-serialises tool arguments to a JSON string",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser(
        "preflight", help="host state, and refuse if something else is resident"
    )
    _common(p)
    p.set_defaults(fn=_preflight_mode)

    b = sub.add_parser("probe", help="one turn, printed as wire facts")
    _common(b)
    b.add_argument("--prompt", required=True)
    b.add_argument("--system", default="")
    b.add_argument("--tools", action="store_true")
    b.add_argument("--timeout", type=float, default=900.0)
    b.set_defaults(fn=_probe_mode)

    r = sub.add_parser("run", help="drive the task set through this harness")
    _common(r)
    r.add_argument(
        "--tier", action="append", default=[], choices=("low", "mid", "high")
    )
    r.add_argument(
        "--family",
        action="append",
        default=[],
        choices=sorted(census().families),
        help="what kind of work, independent of how hard it is",
    )
    r.add_argument("--task", action="append", default=[])
    r.add_argument("--out", type=Path)
    r.add_argument(
        "--at", default="HEAD", help="revision to run against; pin it across arms"
    )
    r.add_argument("--resume", action="store_true")
    # v5 rather than v1: `--patch-gate` and `--idle-notices` are on by default and the
    # templates they render exist only from v5 on, so a v1 default would make the
    # documented invocation fail at startup. An older pack still runs, with the two
    # flags set to 0, and every artifact records which pack answered.
    r.add_argument("--prompt-pack", default="v5")
    r.add_argument(
        "--min-evidence",
        type=int,
        default=1,
        help="successful read_file/search calls required before finish is accepted; 0 disables",
    )
    r.add_argument(
        "--nudges",
        type=int,
        default=2,
        help="consecutive stalled turns tolerated before the episode ends; any tool "
        "call resets the count",
    )
    r.add_argument(
        "--patch-gate",
        type=int,
        default=3,
        help="times finish is refused on a patch task while git diff is empty; "
        "0 disables, needs patch_blocked.md in the pack",
    )
    r.add_argument(
        "--idle-notices",
        type=int,
        default=3,
        help="times an unchanged working tree is reported on a patch task; 0 disables, "
        "needs no_changes_yet.md in the pack",
    )
    r.add_argument("--idle-notice-every", type=int, default=10, metavar="TURNS")
    r.add_argument(
        "--think-off-after",
        type=int,
        default=0,
        help="capped-empty turns after which thinking is disabled for the rest of the "
        "episode; measured over two change_code runs and rejected, so 0 by default — "
        "every episode that tripped it ran long and three of four hit the turn cap",
    )
    r.add_argument(
        "--answer-review",
        type=int,
        default=0,
        help="bounded revisions of the first finish against the pack's checklist; "
        "needs answer_review.md in the pack",
    )
    r.add_argument(
        "--show-answer-key",
        action="store_true",
        help="stop hiding the eval's own task module from read_file and search; it is in "
        "the worktree and a plain search returns the anchors of the question being asked",
    )
    r.add_argument(
        "--handed-context",
        action="append",
        default=[],
        metavar="PATH",
        help="repo-relative file quoted into the pinned prefix, the way Claude Code "
        "auto-loads CLAUDE.md for both of its arms; needs handed_context.md in the pack",
    )
    r.add_argument(
        "--wrapper",
        choices=("task_answer.md", "task_patch.md"),
        default="",
        help="force one task wrapper regardless of task kind, instead of letting kind "
        "choose; the A/B for whether the five-step patch procedure suppresses the write "
        "tools it prescribes",
    )
    r.set_defaults(fn=_run_mode)

    args = ap.parse_args()
    result: int = args.fn(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
