"""SSD read at Qwen3-Coder-Next's exact expert-streaming access pattern.

`ds4_streambench.py` with the strides retargeted. Everything structural is the same --
`StreamingQuantizedSwitchLinear.__call__` still does three sequential reads per
projection (weight, scales, biases), each a `ThreadPoolExecutor.map` over the k unique
active experts, so decode runs at queue depth k and there is a barrier between each.

What differs is the size of every read, and it differs in the direction that costs:

  * routed experts are stacked one tensor per (layer, projection) -- 256 MiB holding all
    512 experts -- so a per-expert slice is exactly **512 KiB**, a quarter of DeepSeek's
    2 MiB row. Scales and biases are **32 KiB** each against 128 KiB.
  * `platform.md` §2.1: 128 KiB at qd 6 runs at 2.50 GB/s against 6.04 for 2 MiB, so
    smaller blocks cost more than their share. This model reads *nothing* at 2 MiB. That
    is the one derived number in the Gate B projection worth measuring before trusting.
  * meta is 4.84 GB by the header sum (`qcn_headers.py`), under a 6 GB
    `OPTIQ_STREAM_SCALES_BUDGET_GB` -- so unlike DeepSeek's 9.3 GB it can be made
    resident cheaply, which is what arm 4 prices.

Per decoded token at top-10 across 48 layers that is 432 sequential barriers at queue
depth 10, and 0.849 GB. Per prefill chunk the router reaches ~all 512 experts, so the
same pattern runs at k=512 and reads the whole 43.49 GB routed set once.

Corpus and method unchanged: local quants tiled so no byte is requested twice,
`F_NOCACHE` on every fd, `iostat` bracketed. **Read the per-window `dev/req` column, not
the average** -- windows below ~1.0 were served partly from inactive pages and are not
storage measurements.
"""

import fcntl
import glob
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
WEIGHT = 512 * 1024
META = 32 * 1024
SLOT = WEIGHT
PER_SLOT = SLOT // META

CORPORA = (
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-122B-A10B-OptiQ-2bit/blobs/*",
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-OptiQ-4bit/blobs/*",
    "~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-0731-OptiQ-2bit/blobs/*",
)

LAYERS = 48
PROJECTIONS = 3
MODULES = LAYERS * PROJECTIONS
TOPK = 10
N_EXPERTS = 512


def device_mb() -> float:
    out = subprocess.run(
        ["iostat", "-Id", "disk0"], capture_output=True, text=True, check=False
    ).stdout
    return float(out.strip().splitlines()[-1].split()[2])


class Pool:
    """Cold 512 KiB slots. A weight read consumes a whole slot; meta reads are carved
    16-to-a-slot, so mixing sizes neither wastes the corpus nor re-reads a byte."""

    def __init__(self, seed: int = 1):
        jobs = []
        for pattern in CORPORA:
            for path in glob.glob(os.path.expanduser(pattern)):
                if os.path.getsize(path) < 512 * 1024 * 1024:
                    continue
                fd = os.open(path, os.O_RDONLY)
                fcntl.fcntl(fd, F_NOCACHE, 1)
                jobs += [
                    (fd, off) for off in range(0, os.fstat(fd).st_size - SLOT, SLOT)
                ]
        random.seed(seed)
        random.shuffle(jobs)
        self._slots = jobs
        self._i = 0
        self._passes = 0
        self._meta: list[tuple[int, int]] = []

    def weight(self, n: int) -> list[tuple[int, int]]:
        if self._i + n > len(self._slots):
            self._i = 0
            self._passes += 1
        out = self._slots[self._i : self._i + n]
        self._i += n
        return out

    def meta(self, n: int) -> list[tuple[int, int]]:
        while len(self._meta) < n:
            fd, off = self.weight(1)[0]
            self._meta += [(fd, off + j * META) for j in range(PER_SLOT)]
        out, self._meta = self._meta[:n], self._meta[n:]
        return out

    @property
    def consumed_gb(self) -> float:
        return (self._passes * len(self._slots) + self._i) * SLOT / 1e9


_pool = ThreadPoolExecutor(max_workers=24, thread_name_prefix="qcn-bench")


def batch(jobs: list[tuple[int, int]], size: int) -> int:
    list(_pool.map(lambda j: os.pread(j[0], size, j[1]), jobs))
    return len(jobs) * size


def projection(p: Pool, k: int, meta_resident: bool = False) -> int:
    """One StreamingQuantizedSwitchLinear call: weight, then scales, then biases.

    `meta_resident` is the OPTIQ_STREAM_SCALES_BUDGET_GB arm -- the 32 KiB reads never
    reach the SSD because the scales and biases are already in memory.
    """
    n = batch(p.weight(k), WEIGHT)
    if not meta_resident:
        n += batch(p.meta(k), META) + batch(p.meta(k), META)
    return n


def window(label: str, nbytes: int, el: float, d0: float, derived: str) -> None:
    gb = nbytes / 1e9
    print(
        f"  {label:>9s} {el:>7.2f} {gb:>8.2f} {gb / el:>7.2f} "
        f"{(device_mb() - d0) / 1e3 / gb:>8.2f}   {derived}"
    )


def decode_pattern(
    p: Pool, tokens: int, per_window: int, meta_resident: bool = False
) -> None:
    print(f"  {'tokens':>9s} {'s':>7s} {'GB':>8s} {'GB/s':>7s} {'dev/req':>8s}   tok/s")
    for i in range(0, tokens, per_window):
        d0, t0 = device_mb(), time.time()
        n = sum(projection(p, TOPK, meta_resident) for _ in range(per_window * MODULES))
        el = time.time() - t0
        window(f"{i}-{i + per_window}", n, el, d0, f"{per_window / el:.3f}")


def prefill_pattern(
    p: Pool, modules: int, per_window: int, meta_resident: bool = False
) -> None:
    print(
        f"  {'modules':>9s} {'s':>7s} {'GB':>8s} {'GB/s':>7s} {'dev/req':>8s}   "
        f"full 43.5 GB sweep"
    )
    for i in range(0, modules, per_window):
        d0, t0 = device_mb(), time.time()
        n = sum(projection(p, N_EXPERTS, meta_resident) for _ in range(per_window))
        el = time.time() - t0
        window(
            f"{i}-{i + per_window}",
            n,
            el,
            d0,
            f"{el / per_window * MODULES:.1f} s/chunk",
        )


def isolated(p: Pool, label: str, size: int, qd: int, target_gb: float) -> None:
    n = max(1, int(target_gb * 1e9 // size))
    jobs = p.weight(n) if size == WEIGHT else p.meta(n)
    d0, t0 = device_mb(), time.time()
    for i in range(0, n, qd):
        batch(jobs[i : i + qd], size)
    el, gb = time.time() - t0, n * size / 1e9
    print(
        f"  {label:<16s} qd {qd:>3d}  {gb:6.2f} GB  {gb / el:6.2f} GB/s  "
        f"dev/req {(device_mb() - d0) / 1e3 / gb:5.2f}"
    )


if __name__ == "__main__":
    p = Pool()
    print(f"corpus {len(p._slots) * SLOT / 1e9:.1f} GB, F_NOCACHE, each byte once\n")

    print("1. decode -- one token = 432 sequential barriers at qd 10, 0.849 GB")
    decode_pattern(p, tokens=48, per_window=8)

    print("\n2. prefill -- the full-expert sweep, k = 512, pool depth 24")
    prefill_pattern(p, modules=144, per_window=24)

    print("\n3. isolated -- the two strides this model reads")
    for size, label in ((WEIGHT, "weight 512 KiB"), (META, "meta    32 KiB")):
        for qd in (10, 24):
            isolated(p, label, size, qd, 2.0)

    print("\n4. meta resident (OPTIQ_STREAM_SCALES_BUDGET_GB=6) -- weights only")
    decode_pattern(p, tokens=16, per_window=8, meta_resident=True)
    prefill_pattern(p, modules=72, per_window=24, meta_resident=True)

    print(f"\nconsumed {p.consumed_gb:.1f} GB")
