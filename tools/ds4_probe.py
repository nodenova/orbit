"""What DeepSeek-V4-Flash can actually do on this host, driven without the server.

`optiq serve` aborts the process on the first request to this model (`PROCESSES.md` §6),
and the reason is the server's threading rather than the model: the identical call chain
— `stream_generate` -> `deepseek_v4` -> `moe_stream` — runs to completion on the main
thread. So every number here is taken by driving `load_streaming` + `stream_generate`
directly, single-threaded, which is the only configuration in which this model has ever
produced a token.

Ladder it. Prefill is the expensive half at ~1 chunk of expert sweep per 8k tokens, and
the KV cache is what decides the context ceiling, so each frontier authorises the next
(`real-weights` §2). Run with `--frontiers` ascending and stop where the report says to.

Prompt content is this repository's own source, not synthetic filler: the question is
what a *verifier* does with a real diff, and identifier-diverse real code is what a
streamed MoE routes through many experts rather than a few.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

REVIEW_TASK = """You are reviewing a proposed change to the file above.

The change replaces this line:

    for token, _ in generate_step(prompt, model, max_tokens=max_tokens):

with this line:

    for token, _ in generate_step(prompt, model, max_tokens=max_tokens, \
prefill_step_size=512):

Answer in at most three sentences: does this change alter the *output* of a greedy \
generation, or only its speed? Say which and why."""


def headroom_gb() -> float:
    out = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, check=False
    ).stdout
    for line in out.splitlines():
        if "Pages active" in line:
            active = int(line.split(":")[1].strip().rstrip(".")) * 16384
            return (36 * (1 << 30) - active) / 1e9
    return 0.0


def pageouts() -> int:
    out = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, check=False
    ).stdout
    for line in out.splitlines():
        if "Pageouts" in line:
            return int(line.split(":")[1].strip().rstrip("."))
    return 0


def repo_source(target_chars: int) -> str:
    """Real code, in a fixed order so a frontier is reproducible."""
    root = Path(__file__).resolve().parent.parent / "src" / "tandem"
    parts: list[str] = []
    total = 0
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        parts.append(f"# --- {path.name} ---\n{text}")
        total += len(text)
        if total >= target_chars:
            break
    return "\n".join(parts)[:target_chars]


def measure(
    model: Any,
    tok: Any,
    *,
    target_tokens: int,
    decode_tokens: int,
    task: str,
    thinking: bool = True,
) -> dict[str, Any]:
    from mlx_lm.generate import stream_generate

    # ~3 chars/token on code, corrected after the first frontier reports its real count.
    content = repo_source(max(target_tokens * 3, 200)) if target_tokens else ""
    body = f"{content}\n\n{task}" if content else task
    # This model ships its template in `chat_template.jinja` rather than in
    # `tokenizer_config.json`, and that template defaults `enable_thinking` to true —
    # so a verifier gets reasoning unless it asks not to, which is the request-side
    # half `tier1.reasoning_control = "deepseek_v4"` exists to send.
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": body}],
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    mx.reset_peak_memory()
    po_before = pageouts()
    t0 = time.perf_counter()
    first_token_s = 0.0
    pieces: list[str] = []
    for step in stream_generate(model, tok, prompt, max_tokens=decode_tokens):
        if not pieces:
            first_token_s = time.perf_counter() - t0
        pieces.append(step.text)
    wall = time.perf_counter() - t0

    n_in = len(prompt)
    decoded = len(pieces)
    decode_s = wall - first_token_s
    return {
        "input_tokens": n_in,
        "output_tokens": decoded,
        "prefill_s": round(first_token_s, 2),
        "prefill_tok_per_s": round(n_in / first_token_s, 1) if first_token_s else None,
        "decode_tok_per_s": round((decoded - 1) / decode_s, 2)
        if decoded > 1 and decode_s > 0
        else None,
        "total_s": round(wall, 1),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "headroom_gb": round(headroom_gb(), 1),
        "pageouts_delta": pageouts() - po_before,
        "text": "".join(pieces),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="/Users/vmehera/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-0731-OptiQ-2bit/snapshots/0edd7d3e70d562a0fc1d1574943ca4fe2b2c1e36",
    )
    ap.add_argument("--frontiers", default="0,2000,8000")
    ap.add_argument("--decode-tokens", type=int, default=8)
    ap.add_argument(
        "--headroom-floor-gb",
        type=float,
        default=8.0,
        help="stop the ladder rather than take the next frontier below this",
    )
    ap.add_argument(
        "--no-thinking",
        action="store_true",
        help="send enable_thinking=false, as tier1.reasoning_control does",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from optiq.runtime.moe_stream import load_streaming

    print(f"headroom {headroom_gb():.1f} GB", flush=True)
    t0 = time.perf_counter()
    model, tok = load_streaming(args.model, verbose=True)
    print(
        f"loaded in {time.perf_counter() - t0:.1f} s, "
        f"peak {mx.get_peak_memory() / 1e9:.2f} GB, headroom {headroom_gb():.1f} GB",
        flush=True,
    )

    # The first generation after a load pays kernel compilation (real-weights §5).
    measure(
        model,
        tok,
        target_tokens=0,
        decode_tokens=2,
        task="Say ok",
        thinking=not args.no_thinking,
    )

    report: dict[str, Any] = {
        "model": args.model,
        "thinking": not args.no_thinking,
        "frontiers": [],
    }
    for raw in args.frontiers.split(","):
        target = int(raw)
        free = headroom_gb()
        if free < args.headroom_floor_gb:
            print(f"STOPPING: headroom {free:.1f} GB below floor", flush=True)
            report["stopped_at"] = target
            break
        row = measure(
            model,
            tok,
            target_tokens=target,
            decode_tokens=args.decode_tokens,
            task=REVIEW_TASK,
            thinking=not args.no_thinking,
        )
        report["frontiers"].append(row)
        shown = dict(row)
        text = shown.pop("text")
        print(f"  {shown}", flush=True)
        print(f"    -> {text[:300]!r}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
