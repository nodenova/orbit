"""Host health gate — fp16 bandwidth, GEMM and quantised matvec. Loads no weights.

Run this before trusting any throughput number measured on this machine, and again
after any large model load. `HANDOFF.md` T18 is why: this host has been observed at
~9% of its recorded bandwidth with compute at ~50%, after a tier-0 load and with
nothing resident. Healthy figures for the baseline M4 Max are in `BASELINE.md` §2.1;
the pass/stop thresholds are `PROCESSES.md` §3.1.

This was the first of the benchmarks to be committed rather than left in the
gitignored `specs/bench/`, because a gate deciding whether a measurement may be
believed has to survive a fresh clone. The rest followed for the same reason and
that directory is now gone; `BASELINE.md` §8 lists them.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import mlx.core as mx

ELEMENTS = 256 * 1024 * 1024
GEMM_SIZES = (2048, 4096, 8192)
QUANT_BITS = (4, 2)
QUANT_DIM = 8192
QUANT_GROUP = 64


def median_s(fn: Callable[[], Any], warm: int = 3, iters: int = 10) -> float:
    """Median wall-clock of `fn`, each call forced to completion with `mx.eval`."""
    for _ in range(warm):
        mx.eval(fn())
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def bandwidth() -> None:
    a = mx.random.normal((ELEMENTS,), dtype=mx.float16)
    mx.eval(a)
    rw = median_s(lambda: a + 1.0)
    print(f"bandwidth (r+w, fp16 elementwise): {2 * a.nbytes / rw / 1e9:.0f} GB/s")
    read = median_s(lambda: mx.sum(a))
    print(f"bandwidth (read-only reduction):   {a.nbytes / read / 1e9:.0f} GB/s")


def gemm(m: int) -> None:
    x = mx.random.normal((m, m), dtype=mx.float16)
    y = mx.random.normal((m, m), dtype=mx.float16)
    mx.eval(x, y)
    took = median_s(lambda: x @ y, warm=2, iters=6)
    print(f"GEMM fp16 {m}^3: {2 * m**3 / took / 1e12:.2f} TFLOP/s")


def qmatvec(bits: int) -> None:
    w = mx.random.normal((QUANT_DIM, QUANT_DIM), dtype=mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=QUANT_GROUP, bits=bits)
    v = mx.random.normal((1, QUANT_DIM), dtype=mx.float16)
    mx.eval(wq, scales, biases, v)
    took = median_s(
        lambda: mx.quantized_matmul(
            v, wq, scales, biases, group_size=QUANT_GROUP, bits=bits
        ),
        iters=20,
    )
    read = QUANT_DIM * QUANT_DIM * bits / 8
    print(
        f"qmatvec {bits}-bit {QUANT_DIM}x{QUANT_DIM}: "
        f"{took * 1e6:.0f} us -> {read / took / 1e9:.0f} GB/s effective weight read"
    )


def main() -> None:
    bandwidth()
    for m in GEMM_SIZES:
        gemm(m)
    for bits in QUANT_BITS:
        qmatvec(bits)


if __name__ == "__main__":
    main()
