"""SSD read at DeepSeek-V4-Flash's exact expert-streaming access pattern.

`ssdbench_verified.py` measures block sizes and queue depths in isolation. This one
replays what `optiq/runtime/moe_stream.py` actually issues for this model, read off the
quant's safetensors headers:

  * `StreamingQuantizedSwitchLinear.__call__` does three *sequential* reads per
    projection -- weight, then scales, then biases -- each a `ThreadPoolExecutor.map`
    over the k unique active experts. So the queue depth is k, not the pool's 24, and
    there is a barrier between each.
  * DeepSeek-V4's scales/biases total 8.66 GB by the header sum below, which optiq
    reports as 9.3 GB because it counts the shared expert's meta too. Either way it is
    over the `max(2 GB, 10 % RAM)` budget the engine prints as 3.9 GB, so `stream_meta`
    is on and the meta rides the same `os.pread`.
    **Corrected 2026-08-09:** an earlier version of this note said "the 122B proxy kept
    its ~1 GB of meta resident and never paid this", and used that to argue the meta
    path was untested. Serving the 122B prints `expert scales/biases 7.2 GB vs budget
    3.9 GB -> STREAM`, so the proxy streams its meta as well and always did. The claim
    was never measured; it was inferred from the resident total.
  * Strides from the headers: weight rows are exactly 2 MiB (gate/up [2048, 256] U32,
    down [4096, 128] U32); scales and biases are exactly 128 KiB. Nine reads per expert
    per layer, of which six are the small ones.

Per decoded token at top-6 across 43 layers that is 387 sequential barriers at queue
depth 6, and 1.826 GB. Per prefill chunk the router reaches ~all 256 experts, so the
same pattern runs at k=256 and reads the whole 77.91 GB routed set once.

Corpus is the two local quants (71.5 GB, larger than RAM) tiled so no byte is requested
twice, `F_NOCACHE` on every fd. **Read the per-window `dev/req` column, not the average**
-- this host holds ~22 GB of the corpus in inactive pages, so early windows are served
partly from memory and read fast for a reason that has nothing to do with the SSD. Only
windows at dev/req >= ~1.0 are storage measurements. Expert streaming is a sustained
workload -- 1.8 GB per decoded token, 78 GB per prefill chunk -- so the sustained figure
is the one to plan against.
"""

import fcntl
import glob
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
SLOT = 2 * 1024 * 1024
WEIGHT = 2 * 1024 * 1024
META = 128 * 1024
PER_SLOT = SLOT // META

CORPORA = (
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-122B-A10B-OptiQ-2bit/blobs/*",
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-OptiQ-4bit/blobs/*",
)

LAYERS = 43
PROJECTIONS = 3
MODULES = LAYERS * PROJECTIONS
TOPK = 6
N_EXPERTS = 256
TOKEN_GB = 1.826


def device_mb() -> float:
    out = subprocess.run(
        ["iostat", "-Id", "disk0"], capture_output=True, text=True, check=False
    ).stdout
    return float(out.strip().splitlines()[-1].split()[2])


class Pool:
    """Cold 2 MiB slots. A weight read consumes a whole slot; meta reads are carved
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
        # Wraps rather than stopping. The corpus (71.5 GB) is larger than RAM but not
        # larger than this benchmark needs: `dev/req` only settles at ~1.0 once enough
        # volume has moved that the ~22 GB of the corpus already in inactive pages stops
        # dominating. F_NOCACHE keeps a second pass from being a cache read.
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


_pool = ThreadPoolExecutor(max_workers=24, thread_name_prefix="ds4-bench")


def batch(jobs: list[tuple[int, int]], size: int) -> int:
    """One `_ShardWeightReader.read`: size-byte preads over k experts, concurrent, and
    the caller blocks until all of them land."""
    list(_pool.map(lambda j: os.pread(j[0], size, j[1]), jobs))
    return len(jobs) * size


def projection(p: Pool, k: int) -> int:
    """One StreamingQuantizedSwitchLinear call: weight, then scales, then biases."""
    return batch(p.weight(k), WEIGHT) + batch(p.meta(k), META) + batch(p.meta(k), META)


def window(label: str, nbytes: int, el: float, d0: float, derived: str) -> None:
    gb = nbytes / 1e9
    print(
        f"  {label:>9s} {el:>7.2f} {gb:>8.2f} {gb / el:>7.2f} "
        f"{(device_mb() - d0) / 1e3 / gb:>8.2f}   {derived}"
    )


def decode_pattern(p: Pool, tokens: int, per_window: int) -> None:
    print(f"  {'tokens':>9s} {'s':>7s} {'GB':>8s} {'GB/s':>7s} {'dev/req':>8s}   tok/s")
    for i in range(0, tokens, per_window):
        d0, t0 = device_mb(), time.time()
        n = sum(projection(p, TOPK) for _ in range(per_window * MODULES))
        el = time.time() - t0
        window(f"{i}-{i + per_window}", n, el, d0, f"{per_window / el:.3f}")


def prefill_pattern(p: Pool, modules: int, per_window: int) -> None:
    print(
        f"  {'modules':>9s} {'s':>7s} {'GB':>8s} {'GB/s':>7s} {'dev/req':>8s}   "
        f"full 78 GB sweep"
    )
    for i in range(0, modules, per_window):
        d0, t0 = device_mb(), time.time()
        n = sum(projection(p, N_EXPERTS) for _ in range(per_window))
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

    print("1. decode -- one token = 387 sequential barriers at qd 6, 1.826 GB")
    decode_pattern(p, tokens=64, per_window=8)

    print("\n2. prefill -- the full-expert sweep, k = 256, pool depth 24")
    prefill_pattern(p, modules=129, per_window=16)

    print("\n3. isolated -- the two strides this model reads")
    for size, label in ((WEIGHT, "weight  2 MiB"), (META, "meta  128 KiB")):
        for qd in (6, 24):
            isolated(p, label, size, qd, 2.0)

    print(f"\nconsumed {p.consumed_gb:.1f} GB")
