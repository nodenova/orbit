"""G1 (sec 9.3) and T16 as one measurement, against real tier-0 weights.

Both ask whether two execution paths produce the same *text*, and both are really
asking a question about two numbers: how far apart the paths' logits are, and how
big the greedy top1-top2 margin they have to survive is. A gate that compares only
text answers "no" without saying whether it missed by 1e-5 or by a mile, and cannot
distinguish a broken path from a near-tied token.

So each arm records, per step, the full logit vector and the margin. Arms are run
one at a time against **one** loaded model - on MLX a CPU arm costs no extra memory,
because unified memory means the CPU and the GPU read the same buffers, so G1 does
not need the 2 x 23.0 GiB two-backend shape `eval/gates.py` gives it.

Arm B is teacher-forced to arm A's token sequence, so every step compares logits
computed from identical inputs. Without that the arms diverge after the first flip
and every later comparison measures two different prompts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

PROMPT = (
    "Explain in two sentences why a merge conflict in a lock file is usually "
    "resolved by regenerating the file rather than by editing it."
)


def headroom_gb() -> float:
    """`total - active`, the only measure that means anything here (real-weights §3)."""
    out = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, check=False
    ).stdout
    pages = {}
    for line in out.splitlines()[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            pages[k.strip()] = int(v.strip().rstrip("."))
    total = 36 * (1 << 30)
    return (total - pages.get("Pages active", 0) * 16384) / 1e9


def pageouts() -> int:
    out = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, check=False
    ).stdout
    for line in out.splitlines():
        if "Pageouts" in line:
            return int(line.split(":")[1].strip().rstrip("."))
    return 0


class Arm:
    """One execution path's trace: chosen ids, per-step logits, per-step margin."""

    def __init__(self, label: str):
        self.label = label
        self.ids: list[int] = []
        self.logits: list[np.ndarray] = []
        self.margins: list[float] = []
        self.wall_s = 0.0
        self.error: str | None = None

    def record(self, logits: mx.array) -> None:
        # float32 in MLX first: these logits are bfloat16, which has no numpy dtype,
        # so the buffer protocol refuses the conversion.
        row = logits.reshape(-1).astype(mx.float32)
        mx.eval(row)
        arr = np.array(row, copy=True, dtype=np.float32)
        self.logits.append(arr)
        part = np.partition(arr, -2)
        self.margins.append(float(part[-1] - part[-2]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "tokens": len(self.ids),
            "wall_s": round(self.wall_s, 2),
            "min_margin": round(min(self.margins), 6) if self.margins else None,
            "median_margin": round(float(np.median(self.margins)), 4)
            if self.margins
            else None,
            "error": self.error,
        }


def run_arm(
    model: Any,
    prompt_ids: mx.array,
    *,
    label: str,
    device: mx.DeviceType,
    prefill_step_size: int,
    max_tokens: int,
    forced: list[int] | None = None,
) -> Arm:
    from mlx_lm.generate import generate_step

    arm = Arm(label)
    step = {"i": 0}

    def recorder(tokens: mx.array, logits: mx.array) -> mx.array:
        arm.record(logits)
        return logits

    def sampler(logprobs: mx.array) -> mx.array:
        i = step["i"]
        step["i"] = i + 1
        if forced is not None and i < len(forced):
            return mx.array([forced[i]], dtype=mx.uint32)
        return mx.argmax(logprobs, axis=-1)

    # `mx.stream(device)` alone does NOT move the work. mlx_lm.generate binds a
    # module-level `generation_stream = mx.new_thread_local_stream(mx.default_device())`
    # at import and wraps generate_step's body in it, so a caller's stream context is
    # overridden from the inside: a CPU arm returns Metal's own logits in 0.1 s, which
    # reads as a clean G1 pass and measures nothing. The device has to be swapped
    # globally and the module's stream rebound with it - and that is why two backends
    # on different devices cannot coexist in one process (sec 9.3, HANDOFF T22).
    # `import mlx_lm.generate` binds the *function* of that name from the package
    # namespace, not the module.
    gen = importlib.import_module("mlx_lm.generate")

    previous_device = mx.default_device()
    previous_stream = gen.generation_stream
    if device != previous_device:
        mx.set_default_device(device)
        # Assigned at import, so mypy does not see it as a module attribute.
        gen.generation_stream = mx.new_thread_local_stream(device)  # type: ignore[attr-defined]

    t0 = time.perf_counter()
    try:
        for token, _ in generate_step(
            prompt_ids,
            model,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=[recorder],
            prefill_step_size=prefill_step_size,
        ):
            arm.ids.append(int(token))
    except Exception as exc:  # noqa: BLE001 - an arm with no CPU kernel for one op is
        # the finding this probe exists to report, and it must not take the arms that
        # did run down with it.
        arm.error = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        mx.set_default_device(previous_device)
        gen.generation_stream = previous_stream  # type: ignore[attr-defined]
    arm.wall_s = time.perf_counter() - t0
    return arm


def compare(a: Arm, b: Arm) -> dict[str, Any]:
    """Per-step divergence against per-step margin - the whole point of the probe."""
    n = min(len(a.logits), len(b.logits))
    if n == 0:
        return {"steps": 0, "note": "one arm produced nothing"}
    deltas = [float(np.max(np.abs(a.logits[i] - b.logits[i]))) for i in range(n)]
    at_risk = [i for i in range(n) if deltas[i] >= a.margins[i]]
    fragile = [i for i in range(n) if a.margins[i] < max(deltas)]
    # Arm B is teacher-forced, so comparing emitted ids is vacuous - they agree by
    # construction. What the gate actually wants to know is whether the argmax each
    # arm *would* have taken agrees, step by step, from identical inputs.
    flips = [
        i
        for i in range(n)
        if int(np.argmax(a.logits[i])) != int(np.argmax(b.logits[i]))
    ]
    return {
        "steps": n,
        "max_logit_delta": round(max(deltas), 6),
        "steps_with_margin_below_max_delta": fragile,
        "median_logit_delta": round(float(np.median(deltas)), 6),
        "bitwise_identical_logits": all(
            bool(np.array_equal(a.logits[i], b.logits[i])) for i in range(n)
        ),
        "min_margin": round(min(a.margins[:n]), 6),
        "steps_where_delta_exceeds_margin": at_risk,
        "steps_where_argmax_flips": flips,
        "argmax_agreement": round(1.0 - len(flips) / n, 4),
        "margin_over_delta": round(min(a.margins[:n]) / max(max(deltas), 1e-12), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="orbit.toml")
    ap.add_argument("--decode-tokens", type=int, default=8)
    ap.add_argument("--cpu-tokens", type=int, default=0, help="0 skips the CPU arm")
    ap.add_argument("--chunk-a", type=int, default=2048)
    ap.add_argument("--chunk-b", type=int, default=64)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="pad with Gate B's filler to roughly this many tokens; 0 leaves it short",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from mlx_lm import load

    from orbit.config import Config

    cfg = Config.load(args.config)
    path = cfg.tier0.container_path or cfg.tier0.model
    print(f"headroom {headroom_gb():.1f} GB, pageouts {pageouts()}", flush=True)
    print(f"loading {path}", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(path)
    print(
        f"loaded in {time.perf_counter() - t0:.1f} s, "
        f"peak {mx.get_peak_memory() / 1e9:.2f} GB, headroom {headroom_gb():.1f} GB",
        flush=True,
    )

    content = args.prompt
    if args.prompt_tokens:
        from orbit.backends.mlx_tier1 import prefill_filler

        # Gate B's filler, reused rather than reinvented: identifier-diverse code
        # shapes, which is what a chunk-arrangement comparison needs to route
        # through many experts rather than one.
        content = f"{prefill_filler(args.prompt_tokens * 3)}\n\n{args.prompt}"
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    ids = mx.array(tokenizer.encode(text, add_special_tokens=False))
    print(f"prompt {ids.size} tokens", flush=True)

    # ~9 s of Metal kernel compilation lands on whichever arm runs first (real-weights §5).
    run_arm(
        model,
        ids,
        label="warmup",
        device=mx.gpu,
        prefill_step_size=args.chunk_a,
        max_tokens=2,
    )

    report: dict[str, Any] = {
        "model": cfg.tier0.model,
        "prompt_tokens": int(ids.size),
        "decode_tokens": args.decode_tokens,
        "mlx": mx.__version__,
        "arms": [],
        "comparisons": {},
    }

    ref = run_arm(
        model,
        ids,
        label=f"gpu, prefill_step_size={args.chunk_a}",
        device=mx.gpu,
        prefill_step_size=args.chunk_a,
        max_tokens=args.decode_tokens,
    )
    report["arms"].append(ref.as_dict())
    print(f"  A {ref.as_dict()}", flush=True)
    if ref.error:
        print("reference arm failed; nothing to compare against", flush=True)
        return 1
    print(f"  A text: {tokenizer.decode(ref.ids)!r}", flush=True)

    same = run_arm(
        model,
        ids,
        label=f"gpu, prefill_step_size={args.chunk_a} (repeat)",
        device=mx.gpu,
        prefill_step_size=args.chunk_a,
        max_tokens=args.decode_tokens,
        forced=ref.ids,
    )
    report["arms"].append(same.as_dict())
    report["comparisons"]["repeat_same_config"] = compare(ref, same)
    print(f"  repeat: {report['comparisons']['repeat_same_config']}", flush=True)

    chunked = run_arm(
        model,
        ids,
        label=f"gpu, prefill_step_size={args.chunk_b}",
        device=mx.gpu,
        prefill_step_size=args.chunk_b,
        max_tokens=args.decode_tokens,
        forced=ref.ids,
    )
    report["arms"].append(chunked.as_dict())
    report["comparisons"]["prefill_chunk_arrangement"] = compare(ref, chunked)
    print(f"  chunk: {report['comparisons']['prefill_chunk_arrangement']}", flush=True)
    print(
        f"headroom {headroom_gb():.1f} GB, peak {mx.get_peak_memory() / 1e9:.2f} GB, "
        f"pageouts {pageouts()}",
        flush=True,
    )

    if args.cpu_tokens > 0:
        cpu = run_arm(
            model,
            ids,
            label="cpu",
            device=mx.cpu,
            prefill_step_size=args.chunk_a,
            max_tokens=args.cpu_tokens,
            forced=ref.ids,
        )
        report["arms"].append(cpu.as_dict())
        print(f"  C {cpu.as_dict()}", flush=True)
        if not cpu.error:
            report["comparisons"]["device_cpu_vs_gpu"] = compare(ref, cpu)
            print(
                f"  device: {report['comparisons']['device_cpu_vs_gpu']}",
                flush=True,
            )

    report["headroom_gb_end"] = round(headroom_gb(), 1)
    report["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
