"""Rung 1 for the constrained-decode fixes: do they hold against real weights?

`tools/constrained_decode_bench.py` decomposes the cost without weights and concludes
that most of it is removable in Python (`docs/constrained-decoding.md`). This
script is the confirmation that decomposition owes: six processor variants sharing
one loaded model, so the comparison is apples-to-apples.

  A  unconstrained                  the free-form reference
  E  null processor                 returns logits untouched; never reads `tokens`
  F  null + tokens.tolist()         E plus the sync, and no mask work at all
  B  current processor              `mlx_tier0.build_logits_processor`, verbatim
  C  no redundant int() conversion  LMFE already returns list[int]
  D  C + content-keyed mask cache   id() is impossible; see the `identity` subcommand

E and F exist because A-vs-D measured a 22 ms/token gap where the rung-0
decomposition accounts for ~4. A rung-0 bench prices host work; it cannot price a
step that got slower *because* a processor is attached. A→E→F splits that residual
into plumbing, sync, and mask work, and F is the one the F4 decision turns on:
what F costs over A is what restructuring the loop could hide.

Loads tier 0 (~20.6 GiB) -- read `docs/operations.md` §1 and the `real-weights` skill
first, and start at `--runs 1`. It reports pageins per variant and voids its own
timings if the weights are being re-read from disk.

**`tools/mlxbench.py` must report ~247 GB/s before this means anything.** At the
23 GB/s of `operations.md` §3.1 every variant reads the same, because a 15x-slow
GPU swamps the host-side differences this measures.

    python tools/constrained_decode_realweights.py --runs 1 --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKING_SET_CEILING_GIB = 28.08

TOOLS_JSON: list[dict[str, Any]] = [
    {
        "name": "edit_file",
        "description": "Replace an exact string in a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "repo-relative path"},
                "old": {"type": "string", "description": "exact text to replace"},
                "new": {"type": "string", "description": "replacement text"},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run the test suite",
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
    },
]

PROMPT = (
    "In src/orbit/gateway/pipeline.py the call `compact(req)` must pass the token "
    "budget through as a keyword argument named `budget`. Make that edit."
)


def _vm_stat(field: str) -> int:
    out = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, check=False
    ).stdout
    m = re.search(rf"{field}:\s+(\d+)", out)
    return int(m.group(1)) if m else -1


def headroom_gb() -> float:
    """`total - active`, never `Pages free` -- platform.md §1.1."""
    active = _vm_stat("Pages active") * 16384
    total = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    )
    return (total - active) / 2**30


def default_model() -> Path:
    snaps = (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--mlx-community--Qwen3.6-35B-A3B-OptiQ-4bit"
        / "snapshots"
    )
    if not snaps.is_dir():
        raise SystemExit("no tier-0 snapshot found; pass --model")
    return next(snaps.iterdir())


def build_null() -> Any:
    """E. A processor that does nothing at all, not even read `tokens`.

    Isolates what merely *having* a `logits_processors` entry costs, which the
    rung-0 decomposition cannot see: it measures host work per token and assumes
    the rest of the step is unchanged by the presence of a processor.
    """

    def processor(tokens: Any, logits: Any) -> Any:
        return logits

    return processor


def build_null_sync() -> Any:
    """F. E plus the `tokens.tolist()` every real processor must do.

    `constrained-decoding.md` §3.1 withdrew the sync as a cost on the strength of a
    synthetic loop (13.24 vs 13.12 ms/token). That loop was ~4 GB and its own
    generation was not pipelined the way `stream_generate` pipelines a real one,
    so it cannot price a sync whose cost is the pipelining it prevents.
    """

    def processor(tokens: Any, logits: Any) -> Any:
        tokens.tolist()
        return logits

    return processor


def build_current(token_filter: Any, mx: Any) -> Any:
    """Verbatim copy of `mlx_tier0.build_logits_processor`'s inner function."""

    def processor(tokens: Any, logits: Any) -> Any:
        allowed = token_filter(tokens.tolist())
        if not allowed:
            return logits
        vocab = logits.shape[-1]
        ids = [int(t) for t in allowed if 0 <= int(t) < vocab]
        if not ids:
            return logits
        mask = mx.full((vocab,), -float("inf"), dtype=logits.dtype)
        mask[mx.array(ids, dtype=mx.int32)] = 0.0
        return logits + mask

    return processor


def build_no_conv(token_filter: Any, mx: Any, safe_width: bool) -> Any:
    """C. The bounds guard stays; it stops being a per-token loop."""

    def processor(tokens: Any, logits: Any) -> Any:
        allowed = token_filter(tokens.tolist())
        if not allowed:
            return logits
        vocab = logits.shape[-1]
        ids = allowed if safe_width else [t for t in allowed if 0 <= t < vocab]
        if not ids:
            return logits
        mask = mx.full((vocab,), -float("inf"), dtype=logits.dtype)
        mask[mx.array(ids, dtype=mx.int32)] = 0.0
        return logits + mask

    return processor


def build_cached(
    token_filter: Any, mx: Any, safe_width: bool, stats: dict[str, float]
) -> Any:
    """D. Keyed on list *content*: `id()` never repeats under lm-format-enforcer >= 0.11
    (measured 0/42), and a key that can collide is a silently wrong constraint."""
    slot: dict[str, Any] = {"src": None, "mask": None}

    def processor(tokens: Any, logits: Any) -> Any:
        allowed = token_filter(tokens.tolist())
        if not allowed:
            return logits
        if slot["src"] is None or slot["src"] != allowed:
            stats["miss"] = stats.get("miss", 0) + 1
            vocab = logits.shape[-1]
            ids = allowed if safe_width else [t for t in allowed if 0 <= t < vocab]
            if not ids:
                return logits
            mask = mx.full((vocab,), -float("inf"), dtype=logits.dtype)
            mask[mx.array(ids, dtype=mx.int32)] = 0.0
            slot["src"], slot["mask"] = allowed, mask
        else:
            stats["hit"] = stats.get("hit", 0) + 1
        return logits + slot["mask"]

    return processor


def build_timed(
    token_filter: Any, mx: Any, safe_width: bool, stats: dict[str, float]
) -> Any:
    """G. D with its own host side timed, to attribute what A-vs-D leaves over.

    MLX is lazy: `mx.full`, the scatter and `logits + mask` all return without
    computing, so this stopwatch covers Python and dispatch and *excludes* the GPU
    work it queues. That is the point. Host work and queued GPU work are the two
    candidates for the 22.15 ms/token residual, and they land on opposite sides of
    this boundary — if the timings here come to ~4 ms/token, the rest is GPU.
    """
    slot: dict[str, Any] = {"src": None, "mask": None}
    per_call: list[tuple[float, int]] = []
    stats["_per_call"] = per_call  # type: ignore[assignment]  # carried out, not summed

    def processor(tokens: Any, logits: Any) -> Any:
        t0 = time.perf_counter()
        seq = tokens.tolist()
        t1 = time.perf_counter()
        allowed = token_filter(seq)
        t2 = time.perf_counter()
        out = logits
        if allowed:
            if slot["src"] is None or slot["src"] != allowed:
                vocab = logits.shape[-1]
                ids = allowed if safe_width else [t for t in allowed if 0 <= t < vocab]
                if ids:
                    mask = mx.full((vocab,), -float("inf"), dtype=logits.dtype)
                    mask[mx.array(ids, dtype=mx.int32)] = 0.0
                    slot["src"], slot["mask"] = allowed, mask
            if slot["mask"] is not None:
                out = logits + slot["mask"]
        t3 = time.perf_counter()
        stats["tolist_s"] = stats.get("tolist_s", 0.0) + (t1 - t0)
        stats["filter_s"] = stats.get("filter_s", 0.0) + (t2 - t1)
        stats["mask_s"] = stats.get("mask_s", 0.0) + (t3 - t2)
        stats["calls"] = stats.get("calls", 0.0) + 1
        stats["seq_len"] = float(len(seq))
        per_call.append(((t2 - t1) * 1000, len(allowed) if allowed else 0))
        return out

    return processor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--runs", type=int, default=1, help="start at 1; ladder to 8")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    model_path = args.model or default_model()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    from orbit.gateway.toolcall.constrain import Constrainer, tool_call_schema
    from orbit.types import ToolDef

    print(f"pre-load : headroom {headroom_gb():.1f} GB")
    t0 = time.perf_counter()
    model, tokenizer = load(str(model_path))
    resident = mx.get_active_memory() / 2**30
    print(f"loaded   : {time.perf_counter() - t0:.1f} s, active {resident:.2f} GiB")
    if resident > WORKING_SET_CEILING_GIB:
        raise SystemExit(
            f"ABORT: {resident:.2f} GiB exceeds the Metal working-set ceiling"
        )

    inner = tokenizer._tokenizer
    vocab_n = len(inner)
    constrainer = Constrainer()
    t0 = time.perf_counter()
    vocabulary = constrainer.vocabulary(tokenizer)
    print(f"lmfe vocab: {time.perf_counter() - t0:.2f} s")

    tools = [
        ToolDef(
            name=t["name"], description=t["description"], parameters=t["parameters"]
        )
        for t in TOOLS_JSON
    ]
    schema = tool_call_schema(tools)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        tools=[{"type": "function", "function": t} for t in TOOLS_JSON],
    )
    print(f"prompt   : {len(inner.encode(text, add_special_tokens=False))} tokens")

    print("warm-up (the first generation pays ~9 s of Metal kernel compilation) ...")
    t0 = time.perf_counter()
    for _ in stream_generate(model, tokenizer, text, max_tokens=8):
        pass
    print(f"warm-up  : {time.perf_counter() - t0:.1f} s")

    width: list[int] = []

    def probe(_tokens: Any, logits: Any) -> Any:
        width.append(logits.shape[-1])
        return logits

    for _ in stream_generate(
        model, tokenizer, text, max_tokens=1, logits_processors=[probe]
    ):
        pass
    safe_width = bool(width) and width[0] >= vocab_n
    print(
        f"logit width {width[0]} vs tokenizer vocab {vocab_n} -> bounds filter "
        f"{'provably unnecessary' if safe_width else 'REQUIRED'}\n"
    )

    variants = [
        "A unconstrained",
        "E null processor",
        "F null + tolist()",
        "B current",
        "C no int() conv",
        "D C + mask cache",
        "G D + host stopwatch",
    ]
    results: dict[str, dict[str, Any]] = {}
    for label in variants:
        rows = []
        stats: dict[str, float] = {}
        for _ in range(args.runs):
            token_filter = constrainer.token_filter(schema, vocabulary)
            procs = {
                "A": None,
                "E": [build_null()],
                "F": [build_null_sync()],
                "B": [build_current(token_filter, mx)],
                "C": [build_no_conv(token_filter, mx, safe_width)],
                "D": [build_cached(token_filter, mx, safe_width, stats)],
                "G": [build_timed(token_filter, mx, safe_width, stats)],
            }[label[0]]

            pi = _vm_stat("Pageins")
            t0 = time.perf_counter()
            last, ntok, out = None, 0, []
            for r in stream_generate(
                model,
                tokenizer,
                text,
                max_tokens=args.max_tokens,
                logits_processors=procs,
            ):
                last, ntok = r, ntok + 1
                out.append(r.text)
            wall = time.perf_counter() - t0
            assert last is not None, "stream_generate yielded nothing"
            rows.append(
                {
                    "wall_s": wall,
                    "gen_tokens": ntok,
                    "decode_tps": last.generation_tps,
                    "ms_per_token": wall / max(ntok, 1) * 1000,
                    "pagein_delta": _vm_stat("Pageins") - pi,
                    "text": "".join(out),
                }
            )

        tps = statistics.median([float(r["decode_tps"]) for r in rows])
        ms = statistics.median([float(r["ms_per_token"]) for r in rows])
        pi_d = max(int(r["pagein_delta"]) for r in rows)
        results[label] = {"decode_tps": tps, "ms_per_token": ms, "rows": rows}
        extra = ""
        if "calls" in stats:
            calls = stats["calls"]
            per_call: Any = stats.pop("_per_call", [])
            results[label]["per_call"] = per_call
            host = {
                k: stats[f"{k}_s"] / calls * 1000 for k in ("tolist", "filter", "mask")
            }
            host["total"] = sum(host.values())
            results[label]["host_ms_per_call"] = host
            results[label]["seq_len"] = stats["seq_len"]
            extra = f"  host {host['total']:.2f} ms/call over {calls:.0f}"
        elif stats:
            hit, miss = stats.get("hit", 0), stats.get("miss", 0)
            extra = f"  cache {hit:.0f}/{hit + miss:.0f} hit"
            results[label]["cache"] = {"hit": hit, "miss": miss}
        print(
            f"{label:20s} decode {tps:6.2f} tok/s   {ms:6.2f} ms/tok"
            f"   pagein +{pi_d}{extra}"
        )
        if pi_d > 100_000:
            print(
                f"  VOID: {pi_d * 16384 / 2**30:.1f} GiB re-read from disk — the "
                "weights are being evicted and these timings are fiction"
            )

    a_ms = results["A unconstrained"]["ms_per_token"]
    b_ms = results["B current"]["ms_per_token"]
    print(
        f"\n{'variant':20s} {'tok/s':>8s} {'ms/tok':>8s} {'vs A':>7s} {'recovered':>10s}"
    )
    for label in variants:
        r = results[label]
        rec = ""
        if label[0] in "CD" and b_ms != a_ms:
            rec = f"{(b_ms - r['ms_per_token']) / (b_ms - a_ms) * 100:9.0f}%"
        print(
            f"{label:20s} {r['decode_tps']:8.2f} {r['ms_per_token']:8.2f} "
            f"{a_ms and r['ms_per_token'] / a_ms:6.2f}x {rec:>10s}"
        )

    g = results.get("G D + host stopwatch", {})
    if "host_ms_per_call" in g:
        h = g["host_ms_per_call"]
        gap = g["ms_per_token"] - a_ms
        print(
            f"\nT26 attribution, per generated token (seq len {g['seq_len']:.0f}):\n"
            f"  tokens.tolist()          {h['tolist']:6.2f} ms\n"
            f"  token_filter (LMFE)      {h['filter']:6.2f} ms\n"
            f"  mask build + add         {h['mask']:6.2f} ms\n"
            f"  host total, measured     {h['total']:6.2f} ms\n"
            f"  gap over A               {gap:6.2f} ms\n"
            f"  NOT host work            {gap - h['total']:6.2f} ms  "
            f"({(gap - h['total']) / gap * 100 if gap else 0:.0f}% of the gap)"
        )
        pc = g.get("per_call", [])
        if pc:
            print("\n  LMFE per call, by allowed-set size:")
            buckets: dict[int, list[float]] = {}
            for t, n in pc:
                buckets.setdefault(n, []).append(t)
            for n in sorted(buckets, reverse=True)[:8]:
                ts = buckets[n]
                print(
                    f"    |allowed|={n:>7}  n={len(ts):>3}  "
                    f"median {statistics.median(ts):6.2f} ms  "
                    f"mean {statistics.mean(ts):6.2f} ms  max {max(ts):6.2f} ms"
                )

    print(
        f"\nfinal: active {mx.get_active_memory() / 2**30:.2f} GiB, "
        f"peak {mx.get_peak_memory() / 2**30:.2f} GiB, headroom {headroom_gb():.1f} GB"
    )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "model": str(model_path),
                    "runs": args.runs,
                    "max_tokens": args.max_tokens,
                    "logit_width": width[0],
                    "tokenizer_vocab": vocab_n,
                    "safe_width": safe_width,
                    "resident_gib": resident,
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
