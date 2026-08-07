"""Latency suite and measurement discipline (spec sec 10.4, 10.5).

TTFT and tok/s at context frontiers 2k / 4k / 8k / 16k / 32k, cold and warm, per
tier. **Instantaneous throughput at each frontier, not a whole-run average** — an
average over a run that started at 2k and ended at 32k reports a number that was
true at no point during it.

Sec 10.5 is the part that is easy to skip and expensive to skip. Every published
datapoint records chip bin, RAM, **SSD capacity**, engine commit, container hash,
adapter hash, exact command, cache state, throughput, TTFT, expert hit rate, bytes
read, and the quality check. One variable per run. `Environment.detect()` fills in
what it can and leaves the rest explicitly unknown rather than guessing — a
measurement with a guessed SSD capacity is worse than one that says it does not know,
because the whole point of sec 2.3 is that capacity is a performance spec.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..backends.base import Backend
from ..types import GenRequest, Message, Role, Sampling

FRONTIERS: tuple[int, ...] = (2_000, 4_000, 8_000, 16_000, 32_000)

# Latency contract (sec 7.3), asserted by `check_contract`.
CONTRACT = {
    "chat": {"ttft_s": 2.0, "tok_per_s": 40.0},
    "read_only": {"ttft_s": 3.0},
    "code_change": {"total_s": 30.0},
    "failed_test": {"total_s": 90.0},
}


@dataclass
class Environment:
    """Everything sec 10.5 requires recorded alongside a number."""

    chip: str = "unknown"
    cpu_cores: int = 0
    gpu_cores: int = 0
    ram_gb: float = 0.0
    ssd_capacity_gb: float = 0.0
    memory_bandwidth_gb_s: float = 0.0
    os_version: str = ""
    engine_commit: str = ""
    python: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def detect(cls) -> Environment:
        from ..attest.receipt import engine_commit

        env = cls(os_version=platform.platform(), python=platform.python_version())
        env.engine_commit = engine_commit()
        if platform.system() != "Darwin":
            # Off-target. Say so rather than reporting a Linux box's numbers as if
            # they characterised an M4 Max.
            env.chip = f"non-darwin ({platform.machine()})"
            return env
        env.chip = _sysctl("machdep.cpu.brand_string") or "unknown"
        env.cpu_cores = int(_sysctl("hw.ncpu") or 0)
        mem = _sysctl("hw.memsize")
        env.ram_gb = round(int(mem) / (1 << 30), 1) if mem else 0.0
        env.ssd_capacity_gb = _disk_capacity_gb()
        # Bandwidth is not queryable; it is a bin fact (546 vs 410 GB/s on M4 Max,
        # sec 2.3) that the operator must supply. Left at 0 = unknown.
        return env


def _sysctl(key: str) -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _disk_capacity_gb() -> float:
    """SSD capacity — a performance spec, not a storage detail (sec 2.3).

    Apple SSD read bandwidth scales with NAND package count, so 512 GB vs 1 TB is
    roughly 2x on the tier-1 path. A latency report that omits it is unreproducible.
    """
    try:
        import shutil as _shutil

        return round(_shutil.disk_usage("/").total / (1000**3), 0)
    except OSError:
        return 0.0


@dataclass
class LatencySample:
    frontier_tokens: int
    tier: int
    cold: bool
    ttft_s: float
    total_s: float
    output_tokens: int
    # Instantaneous decode rate, excluding prefill (sec 10.4).
    decode_tok_per_s: float
    prefill_tok_per_s: float
    adapter: str | None = None
    expert_hit_rate: float | None = None
    bytes_read: int | None = None
    quality_check: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LatencyReport:
    environment: dict[str, Any] = field(default_factory=dict)
    container_hash: str | None = None
    adapter_hash: str | None = None
    command: str = ""
    samples: list[LatencySample] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "container_hash": self.container_hash,
            "adapter_hash": self.adapter_hash,
            "command": self.command,
            "samples": [s.as_dict() for s in self.samples],
            "contract": check_contract(self.samples),
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    def table(self) -> str:
        head = f"{'ctx':>7} {'tier':>4} {'cache':>6} {'TTFT s':>8} {'prefill t/s':>12} {'decode t/s':>11}"
        lines = [head, "-" * len(head)]
        for s in self.samples:
            lines.append(
                f"{s.frontier_tokens:>7} {s.tier:>4} {'cold' if s.cold else 'warm':>6} "
                f"{s.ttft_s:>8.2f} {s.prefill_tok_per_s:>12.1f} {s.decode_tok_per_s:>11.1f}"
            )
        return "\n".join(lines)


def _filler(approx_tokens: int) -> str:
    unit = "def handle(request):\n    return process(request)\n\n"
    return unit * max(1, approx_tokens // 12)


async def measure(
    backend: Backend,
    *,
    frontiers: Sequence[int] = FRONTIERS,
    adapter: str | None = None,
    max_tokens: int = 64,
    cold: bool = True,
) -> list[LatencySample]:
    """One sample per frontier.

    TTFT is measured to the first non-empty delta, so a backend that emits an empty
    priming chunk does not report an artificially good number.
    """
    samples: list[LatencySample] = []
    for frontier in frontiers:
        req = GenRequest(
            messages=[Message(role=Role.USER, content=_filler(frontier))],
            adapter=adapter,
            sampling=Sampling(temperature=0.0, seed=0, max_tokens=max_tokens),
        )
        input_tokens = backend.count_tokens(backend.render(req))

        t0 = time.perf_counter()
        ttft = 0.0
        out_tokens = 0
        first_seen = False
        async for delta in backend.stream(req):
            if delta.text and not first_seen:
                ttft = time.perf_counter() - t0
                first_seen = True
            if delta.done and delta.result is not None:
                out_tokens = delta.result.usage.output_tokens
        total = time.perf_counter() - t0

        decode_s = max(1e-6, total - ttft)
        samples.append(
            LatencySample(
                frontier_tokens=frontier,
                tier=backend.tier,
                cold=cold,
                ttft_s=round(ttft, 4),
                total_s=round(total, 4),
                output_tokens=out_tokens,
                decode_tok_per_s=round(out_tokens / decode_s, 1),
                prefill_tok_per_s=round(input_tokens / max(1e-6, ttft), 1),
                adapter=adapter,
            )
        )
    return samples


def check_contract(samples: Sequence[LatencySample]) -> dict[str, Any]:
    """Sec 7.3 latency contract, per tier-0 chat expectations."""
    tier0 = [s for s in samples if s.tier == 0 and not s.cold]
    if not tier0:
        return {"checked": False, "reason": "no warm tier-0 samples"}
    worst_ttft = max(s.ttft_s for s in tier0)
    worst_decode = min(s.decode_tok_per_s for s in tier0)
    return {
        "checked": True,
        "chat_ttft_s": {
            "worst": round(worst_ttft, 2),
            "budget": CONTRACT["chat"]["ttft_s"],
            "pass": worst_ttft <= CONTRACT["chat"]["ttft_s"],
        },
        "chat_tok_per_s": {
            "worst": round(worst_decode, 1),
            "budget": CONTRACT["chat"]["tok_per_s"],
            "pass": worst_decode >= CONTRACT["chat"]["tok_per_s"],
        },
    }


def m0_gate_a(samples: Sequence[LatencySample], toolcall_failure_rate: float) -> dict[str, Any]:
    """M0 Gate A (sec 11): tier-0 warm TTFT <5 s at >=30 tok/s, tool-call failures <5%.

    The kill condition attaches here: if Gate A fails badly the interactive premise
    is wrong and the product pivots to async review-assist. That is a decision worth
    reaching in three days, which is the only reason this function is this simple.
    """
    warm = [s for s in samples if s.tier == 0 and not s.cold]
    if not warm:
        return {"pass": False, "reason": "no warm tier-0 samples"}
    worst_ttft = max(s.ttft_s for s in warm)
    worst_decode = min(s.decode_tok_per_s for s in warm)
    ttft_ok = worst_ttft < 5.0
    decode_ok = worst_decode >= 30.0
    tools_ok = toolcall_failure_rate < 0.05
    return {
        "pass": ttft_ok and decode_ok and tools_ok,
        "ttft_s": {"worst": round(worst_ttft, 2), "budget": 5.0, "pass": ttft_ok},
        "decode_tok_per_s": {"worst": round(worst_decode, 1), "budget": 30.0, "pass": decode_ok},
        "toolcall_failure_rate": {
            "value": round(toolcall_failure_rate, 4),
            "budget": 0.05,
            "pass": tools_ok,
        },
        "kill_condition": (
            "Gate A failed. The interactive premise may be wrong — consider pivoting "
            "to async review-assist before building further (sec 11)."
            if not (ttft_ok and decode_ok and tools_ok)
            else ""
        ),
    }
