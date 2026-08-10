"""SSD random read, checked against the block device's own counters.

This exists because the benchmark it replaces was wrong in a way that read as a finding.
That one read the 23 GiB tier-0 blobs on a 36 GiB machine with `F_NOCACHE` set, which
stops *that fd* from populating the unified buffer cache but does not stop a read being
served from pages something else already cached, and does not stop the NVMe controller's
readahead. It reported 50.6 GB/s at 8 MB blocks -- which no consumer NVMe does, and which
is how `BASELINE.md` §2.1 came to say that blocks above 2 MB "get worse", a claim that was
an artifact of the same effect at the other end. It has since been deleted; this is the
only SSD benchmark in the tree.

Two changes make the number checkable:

  1. Read the 43.6 GiB tier-1 blobs -- larger than RAM, so the cache cannot hold the
     working set -- and tile them so no byte is requested twice.
  2. Bracket every window with `iostat -Id disk0` and print
     device_bytes / requested_bytes. ~1.0 means the disk really moved it; well below 1
     means memory served it and the throughput figure is not a storage measurement;
     above 1 means readahead is fetching bytes nobody asked for.

Reporting per-window rather than one average is what separates burst from sustained,
and expert streaming is a sustained workload: it reads 1-2 GB per decoded token,
i.e. hundreds of GB per minute of generation.

Measured here (M4 Max, 36 GiB, 1 TB, macOS 26.5.2), 2 MiB blocks at qd 24:
    0-4 GB    3.53 GB/s  (cold)
    4-28 GB   6.7-6.9    <- the burst figure BASELINE.md quotes as 6.93
    28-44 GB  3.6-4.8    <- decay
    ~375 GB read continuously: 2.81 GB/s, device/requested 1.34

Plan streaming against the sustained number, not the burst one.
"""

import fcntl
import glob
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
CORPUS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-122B-A10B-OptiQ-2bit/blobs/*"
)
WINDOW_BYTES = 4_000_000_000


def device_mb() -> float:
    """Cumulative MB transferred by disk0 since boot, from `iostat -Id`."""
    out = subprocess.run(
        ["iostat", "-Id", "disk0"], capture_output=True, text=True, check=False
    ).stdout
    return float(out.strip().splitlines()[-1].split()[2])


def tile(blk: int, seed: int = 1) -> list[tuple[int, int]]:
    """Every offset in the corpus exactly once, shuffled. Open fds are never closed;
    the process is expected to be short-lived."""
    jobs = []
    for path in glob.glob(CORPUS):
        if os.path.getsize(path) < 512 * 1024 * 1024:
            continue
        fd = os.open(path, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        jobs += [(fd, off) for off in range(0, os.fstat(fd).st_size - blk, blk)]
    random.seed(seed)
    random.shuffle(jobs)
    return jobs


def sweep(blk: int, qd: int) -> None:
    jobs = tile(blk)
    per_window = max(1, WINDOW_BYTES // blk)
    print(
        f"\n{blk // 1024 // 1024 or blk / 1024 / 1024} MiB blocks, qd {qd}, "
        f"{len(jobs) * blk / 1e9:.1f} GB requested, each byte once"
    )
    print(f"{'window':>16} {'GB/s':>8} {'dev/req':>9}")
    for i in range(0, len(jobs) - per_window, per_window):
        chunk = jobs[i : i + per_window]
        before = device_mb()
        start = time.time()
        with ThreadPoolExecutor(max_workers=qd) as pool:
            list(pool.map(lambda j: os.pread(j[0], blk, j[1]), chunk))
        elapsed = time.time() - start
        requested = len(chunk) * blk / 1e9
        print(
            f"{i * blk / 1e9:>7.0f}-{(i + per_window) * blk / 1e9:>5.0f} GB "
            f"{requested / elapsed:>8.2f} {(device_mb() - before) / 1e3 / requested:>9.2f}"
        )


if __name__ == "__main__":
    for block_mb, queue_depth in ((0.5, 24), (2, 24), (2, 1)):
        sweep(int(block_mb * 1024 * 1024), queue_depth)
