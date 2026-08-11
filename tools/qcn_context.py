"""Where the Metal single-buffer ceiling bites Qwen3-Coder-Next prefill, and how to move it.

`--plan` prints the reachable context per `--prefill-step-size` with no engine, no
weights and no GPU. Without it, probes a running `optiq serve` at exact token counts.

The ceiling is not a memory limit: resident stays at 1.36 GB and headroom never moves.
`head_dim` is 256, outside the head dims MLX 0.32.0 fuses, so full-attention layers take
the unfused path and materialise a bf16 score matrix per prefill chunk. That array, not
the working set, is what `max_buffer_length` refuses — see docs/platform.md §1.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

SNAPSHOT = (
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-Next-4bit"
    "/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5"
)
MAX_BUFFER_BYTES = 22_613_000_192
NATIVE_CONTEXT = 262_144
N_HEADS = 16
# Measured, not derived: transient per score element is 2.06-2.19 B across mask kinds
# (none/causal/array/bool) and independent of them, and the 2 B reading reproduces the
# abort byte-for-byte at 16 x 8192 x 90112 x 2 == 23_622_320_128.
SCORE_BYTES = 2

FILLER = (
    "def resolve_manifest_entry(entry, registry, *, strict=True):\n"
    "    candidate = registry.lookup(entry.name, version=entry.version)\n"
    "    if candidate is None and strict:\n"
    "        raise ManifestError(f'unresolved {entry.name!r} at {entry.version!r}')\n"
    "    return candidate.materialise(entry.overrides)\n\n"
)


def chunk_allocs(n_tokens: int, step: int) -> Iterator[int]:
    """Score-matrix bytes per chunk; mlx-lm prefills n_tokens-1 in `step` slices."""
    remaining, offset = n_tokens - 1, 0
    while remaining > 0:
        lq = min(step, remaining)
        offset += lq
        yield N_HEADS * lq * offset * SCORE_BYTES
        remaining -= lq


def worst_alloc(n_tokens: int, step: int) -> int:
    return max(chunk_allocs(n_tokens, step), default=0)


def ceiling(step: int) -> int:
    n = step
    while worst_alloc(n + step, step) <= MAX_BUFFER_BYTES:
        n += step
    lo, hi = n, n + step
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if worst_alloc(mid, step) <= MAX_BUFFER_BYTES:
            lo = mid
        else:
            hi = mid
    return lo


def vm_stat_field(needle: str) -> int:
    out = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, check=False
    ).stdout
    for line in out.splitlines():
        if needle in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("."))
    return -1


def active_gib() -> float:
    return vm_stat_field("Pages active") * 16384 / 1024**3


def build_prompt(tokenizer: Any, n_tokens: int, salt: str) -> str:
    # The salt defeats the server's prompt-cache prefix reuse. Without it a later probe
    # resumes mid-context, so Lq stops being the configured step and the chunk the abort
    # lands on is no longer the one this script predicts.
    text = f"# probe {salt}\n" + FILLER * (n_tokens // 40 + 8)
    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)[
        "offset_mapping"
    ]
    if len(offsets) < n_tokens:
        sys.exit(f"filler too short: {len(offsets)} < {n_tokens}")
    return text[: offsets[n_tokens - 1][1]]


def probe(url: str, prompt: str, timeout: float) -> tuple[float, dict[str, Any]]:
    body = json.dumps(
        {"prompt": prompt, "max_tokens": 1, "temperature": 0.0, "stream": False}
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload: dict[str, Any] = json.load(r)
    return time.monotonic() - t0, payload


def print_plan() -> None:
    print(
        f"max_buffer_length {MAX_BUFFER_BYTES:,} B, native context {NATIVE_CONTEXT:,}"
    )
    print(f"worst single array = {N_HEADS} x Lq x Lk x {SCORE_BYTES} B\n")
    print(f"{'step':>6} {'max prompt':>12} {'GiB at max':>11} {'reaches native':>15}")
    for step in (8192, 4096, 2048, 1024, 512):
        c = ceiling(step)
        reach = "yes" if c >= NATIVE_CONTEXT else "no"
        print(f"{step:>6} {c:>12,} {worst_alloc(c, step) / 1024**3:>11.2f} {reach:>15}")
    print(
        f"\nlargest step reaching {NATIVE_CONTEXT:,}: {MAX_BUFFER_BYTES // (N_HEADS * SCORE_BYTES * (NATIVE_CONTEXT - 1)):,}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8081/v1/completions")
    ap.add_argument("--step", type=int, help="the engine's --prefill-step-size")
    ap.add_argument("--tokens", type=int, nargs="+")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    if args.plan:
        print_plan()
        return 0
    if args.step is None or not args.tokens:
        ap.error("--step and --tokens are required unless --plan")

    import os

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.path.expanduser(SNAPSHOT), local_files_only=True
    )

    print(f"step {args.step} -> predicted ceiling {ceiling(args.step):,} tokens")
    print(f"active {active_gib():.2f} GiB, pageouts {vm_stat_field('Pageouts')}\n")
    print(
        f"{'asked':>9} {'engine saw':>11} {'wall s':>9} {'prefill t/s':>12} "
        f"{'GiB':>7} {'vs limit':>9} {'result':>7}"
    )

    for n in args.tokens:
        alloc = worst_alloc(n, args.step)
        shape = f"{alloc / 1024**3:>7.2f} {alloc / MAX_BUFFER_BYTES:>9.2f}"
        try:
            wall, payload = probe(
                args.endpoint, build_prompt(tok, n, f"{args.step}-{n}"), args.timeout
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"{n:>9,} {'-':>11} {'-':>9} {'-':>12} {shape} {'DEAD':>7}  {exc}")
            print("\nengine gone: read its log for the metal::malloc byte count")
            return 1
        saw = payload.get("usage", {}).get("prompt_tokens", -1)
        print(f"{n:>9,} {saw:>11,} {wall:>9.1f} {saw / wall:>12.1f} {shape} {'ok':>7}")

    print(f"\nactive {active_gib():.2f} GiB, pageouts {vm_stat_field('Pageouts')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
