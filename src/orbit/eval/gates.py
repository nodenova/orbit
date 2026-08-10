"""Blocking gates (spec sec 4.2, 9.3, 10.2).

Four gates that must pass, and each one exists because failing it silently would
invalidate a claim the product makes:

* **Tool-call validity (sec 10.2)** — >=99% well-formed over >=100 runs. A fluent
  model that cannot call tools is worth nothing in an agent loop, and 2-bit
  specifically fails here while looking fine on prose. Every change to compaction,
  quantization, adapter or sampling re-runs it.
* **Adapter isolation (sec 4.2)** — with N adapters mounted, greedy output under
  adapter *i* must be byte-identical to greedy output with only adapter *i*
  mounted. If it isn't, multi-tenancy is a lie and every receipt naming an adapter
  is wrong.
* **G1 backend equivalence (sec 9.3)** — greedy output byte-identical between the
  CPU reference path and the Metal path.
* **G2 placement invariance (sec 9.3)** — tier-1 greedy output byte-identical with
  the expert cache at 0 and at max. **If this fails, placement is silently changing
  the model, which invalidates every other claim** — including the determinism
  claim the regulated buyer is paying for.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbit.backends.base import Backend
from orbit.types import GenRequest, Message, Role, Sampling, ToolDef


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.name,
            "pass": self.passed,
            "reason": self.reason,
            **self.detail,
        }


# --- sec 10.2 tool-call validity --------------------------------------------

TOOLCALL_THRESHOLD = 0.99
TOOLCALL_MIN_RUNS = 100

# A fixed multi-step harness scenario (sec 10.2). Fixed on purpose: the gate is a
# regression detector, and a scenario that drifts between runs measures nothing.
SCENARIO_TOOLS: tuple[ToolDef, ...] = (
    ToolDef(
        name="read_file",
        description="Read a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    ),
    ToolDef(
        name="grep",
        description="Search file contents.",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}},
            "required": ["pattern"],
        },
    ),
    ToolDef(
        name="edit_file",
        description="Replace an exact string in a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    ),
    ToolDef(
        name="run_bash",
        description="Run a shell command.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    ),
)

SCENARIO_STEPS: tuple[str, ...] = (
    "Find every file that still imports the old `retry_helper` module.",
    "Read the first of those files.",
    "Replace the import with the new `orbit.retry` module.",
    "Run the test suite to check nothing broke.",
)


@dataclass
class ToolCallGateReport:
    runs: int = 0
    wellformed: int = 0
    repaired: int = 0
    failed: int = 0
    by_outcome: dict[str, int] = field(default_factory=dict)
    rejected_names: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.wellformed / self.runs if self.runs else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "wellformed": self.wellformed,
            "repaired": self.repaired,
            "failed": self.failed,
            "rate": round(self.rate, 4),
            "threshold": TOOLCALL_THRESHOLD,
            "by_outcome": dict(sorted(self.by_outcome.items())),
            "rejected_names": self.rejected_names[:20],
        }


# (GenRequest) -> the pipeline's tool-call outcome dict, as produced by
# `Pipeline._settle_tool_calls`.
TurnRunner = Callable[[GenRequest], Awaitable[dict[str, Any]]]


async def toolcall_gate(
    run_turn: TurnRunner, *, runs: int = TOOLCALL_MIN_RUNS
) -> GateResult:
    """Sec 10.2. Runs the fixed scenario `runs` times and tallies outcomes."""
    report = ToolCallGateReport()
    for i in range(runs):
        step = SCENARIO_STEPS[i % len(SCENARIO_STEPS)]
        req = GenRequest(
            messages=[Message(role=Role.USER, content=step)],
            tools=SCENARIO_TOOLS,
            # Vary the seed so the gate samples the model's behaviour rather than
            # measuring one generation a hundred times.
            sampling=Sampling(temperature=0.2, seed=i, max_tokens=256),
        )
        info = await run_turn(req)
        outcome = str(info.get("outcome", "unknown"))
        report.runs += 1
        report.by_outcome[outcome] = report.by_outcome.get(outcome, 0) + 1
        if outcome == "failed":
            report.failed += 1
        else:
            report.wellformed += 1
        if outcome.endswith("repaired") or outcome == "repaired":
            report.repaired += 1
        report.rejected_names.extend(info.get("rejected", []) or [])

    passed = report.runs >= TOOLCALL_MIN_RUNS and report.rate >= TOOLCALL_THRESHOLD
    return GateResult(
        name="toolcall_validity",
        passed=passed,
        detail=report.as_dict(),
        reason=(
            "ok"
            if passed
            else (
                f"{report.rate:.1%} well-formed over {report.runs} runs "
                f"(need >={TOOLCALL_THRESHOLD:.0%} over >={TOOLCALL_MIN_RUNS}). "
                "This gate is blocking: a model that cannot call tools is worth "
                "nothing in an agent loop (sec 10.2)."
            )
        ),
    )


# --- sec 4.2 adapter isolation ----------------------------------------------

# Backend factory: (adapters_to_mount) -> Backend. Building a backend with a subset
# mounted is the whole point of the test, so it cannot be done against one instance.
BackendFactory = Callable[[Sequence[str]], Backend]


async def _release(backend: Backend) -> None:
    """Hand back a comparison arm's weights before the next arm is built.

    Every gate below compares a backend built one way against a backend built
    another, and on the baseline platform one tier 0 is 23.0 GiB against Metal's
    28.08 GiB ceiling. Two live is not a slow gate, it is a wedged machine — which
    is why these arms are recorded and compared in sequence rather than gathered.
    `unload()` is the rung-2 `Occupant` seam and is what actually returns the
    tensors; `close()` alone does not, and a backend with neither is a no-op here.
    """
    unload = getattr(backend, "unload", None)
    if callable(unload):
        await unload()
    else:
        await backend.close()


_ISOLATION_PROMPTS = (
    "Fix the off-by-one in the pagination helper.",
    "Add a timeout parameter to the client constructor.",
    "Explain how the retry backoff is computed.",
)


async def adapter_isolation_gate(
    factory: BackendFactory,
    adapters: Sequence[str],
    *,
    prompts: Sequence[str] = _ISOLATION_PROMPTS,
) -> GateResult:
    """Sec 4.2, blocking.

    With N adapters mounted, greedy output under adapter *i* must be byte-identical
    to greedy output with only adapter *i* mounted. A failure means adapter deltas
    are leaking across requests — which under concurrency is a silent wrong-answer
    bug, and which makes every receipt naming an adapter false.

    The N-mounted arm is run to completion and *recorded*, then released, before any
    solo arm is built. It used to hold both live across one `asyncio.gather`, which
    on this platform is 2 × 23.0 GiB against a 28.08 GiB ceiling — the gate could
    not run here at all, and `operations.md` §4 called it the likeliest way to wedge
    the machine. Recording costs one dict of ≤128-token strings per comparison.
    """
    if not adapters:
        return GateResult(
            name="adapter_isolation",
            passed=True,
            reason="no adapters mounted; nothing to isolate",
        )

    requests = {
        (name, prompt): GenRequest(
            messages=[Message(role=Role.USER, content=prompt)],
            adapter=name,
            sampling=Sampling(temperature=0.0, top_p=1.0, seed=0, max_tokens=128),
        )
        for name in adapters
        for prompt in prompts
    }

    all_mounted = factory(list(adapters))
    try:
        recorded = {
            key: (await all_mounted.generate(req)).text for key, req in requests.items()
        }
    finally:
        await _release(all_mounted)

    mismatches: list[dict[str, str]] = []
    checked = 0

    for name in adapters:
        solo = factory([name])
        try:
            for prompt in prompts:
                solo_out = await solo.generate(requests[(name, prompt)])
                checked += 1
                if recorded[(name, prompt)] != solo_out.text:
                    mismatches.append(
                        {
                            "adapter": name,
                            "prompt": prompt[:60],
                            "n_mounted": recorded[(name, prompt)][:120],
                            "solo": solo_out.text[:120],
                        }
                    )
        finally:
            await _release(solo)

    passed = not mismatches
    return GateResult(
        name="adapter_isolation",
        passed=passed,
        detail={
            "adapters": list(adapters),
            "comparisons": checked,
            "mismatches": mismatches[:10],
        },
        reason=(
            "ok"
            if passed
            else (
                f"{len(mismatches)}/{checked} comparisons differ. Adapter deltas are "
                "leaking between mounts — multi-tenancy is unsafe and receipts "
                "naming an adapter are wrong (sec 4.2)."
            )
        ),
    )


# --- sec 9.3 determinism ----------------------------------------------------


async def g1_backend_equivalence(
    reference: Backend,
    accelerated: Backend,
    *,
    prompts: Sequence[str] = _ISOLATION_PROMPTS,
) -> GateResult:
    """G1: greedy output byte-identical between the CPU reference and Metal paths.

    Measured against real weights 2026-08-10, and both halves of this signature are
    wrong for MLX — `platform.md` §2.4 has the numbers:

    * The arms cannot be concurrent, and on one process they cannot coexist at all.
      `mlx_lm.generate` binds a module-level `generation_stream` to the default
      device **at import**, and wraps generate_step's body in it, so a caller's
      `mx.stream(mx.cpu)` is overridden from the inside: the CPU arm silently
      returns Metal's logits. Swapping devices means mutating that global, which is
      backend-global state of exactly the kind §6 forbids under concurrency.
    * Byte-identity is not the platform's to give. 30 of this model's 40 layers are
      `linear_attention`, and `mlx_lm/models/gated_delta.py` dispatches on
      `mx.default_device() != mx.gpu` to a *different algorithm* — so a red here is
      not a reduction order anyone can pin, and CPU-vs-Metal measured 4.375 logits
      of divergence against a 1.625 greedy margin on the first step.

    So this runs its arms in sequence, and a real G1 on this platform is two
    processes compared by recorded output rather than two backends in one (T33).
    """
    mismatches: list[dict[str, str]] = []
    for prompt in prompts:
        req = GenRequest(
            messages=[Message(role=Role.USER, content=prompt)],
            sampling=Sampling(temperature=0.0, top_p=1.0, seed=0, max_tokens=128),
        )
        ref = await reference.generate(req)
        acc = await accelerated.generate(req)
        if ref.text != acc.text:
            mismatches.append(
                {
                    "prompt": prompt[:60],
                    "reference": ref.text[:120],
                    "accelerated": acc.text[:120],
                }
            )
    passed = not mismatches
    return GateResult(
        name="g1_backend_equivalence",
        passed=passed,
        detail={"prompts": len(prompts), "mismatches": mismatches},
        reason=(
            "ok"
            if passed
            else "CPU and Metal paths compute different functions; pin the reduction order (sec 9.3)."
        ),
    )


async def g2_placement_invariance(
    factory: Callable[[int], Backend],
    *,
    cache_bytes_low: int = 0,
    cache_bytes_high: int = 18 * (1 << 30),
    prompts: Sequence[str] = _ISOLATION_PROMPTS,
) -> GateResult:
    """G2: tier-1 greedy output identical with the expert cache at 0 and at max.

    The most important gate in the product. Tier 1 streams experts from NVMe and
    caches some in memory; whether a given expert was served from RAM or disk must
    not change the answer. If it does, placement is silently changing the model, and
    the determinism claim — the thing the regulated buyer is actually buying — is
    false.

    **Which is why it must not report a pass it did not measure.** The two arms here
    differ only in `tier1.expert_cache_bytes`, and `expert_cache_provenance` already
    records that this value reaches no engine: nothing sends it, and
    `--stream-experts-cache` is a per-projection count that mlx-optiq 0.4.18 drops
    before its shard reader. So on this deployment both arms are the same engine at
    the same placement, and a byte comparison between them is guaranteed green —
    a vacuous pass on the one gate whose failure invalidates every other claim.
    It therefore reports **not measured** until that function says the value lands,
    which is the single place the fact is recorded (T34).
    """
    from orbit.backends.mlx_tier1 import expert_cache_provenance

    if not expert_cache_provenance(cache_bytes_high)["reached_engine"]:
        return GateResult(
            name="g2_placement_invariance",
            passed=False,
            detail={
                "measured": False,
                "cache_bytes_low": cache_bytes_low,
                "cache_bytes_high": cache_bytes_high,
            },
            reason=(
                "not measured: expert_cache_bytes reaches no engine, so both arms "
                "run at the same placement and a comparison between them would pass "
                "without testing anything. Vary placement on the engine's own command "
                "line and compare recorded output across two runs (sec 9.3, T34)."
            ),
        )

    cold = factory(cache_bytes_low)
    mismatches: list[dict[str, str]] = []
    requests = {
        prompt: GenRequest(
            messages=[Message(role=Role.USER, content=prompt)],
            sampling=Sampling(temperature=0.0, top_p=1.0, seed=0, max_tokens=128),
        )
        for prompt in prompts
    }
    try:
        recorded = {p: (await cold.generate(req)).text for p, req in requests.items()}
    finally:
        await _release(cold)

    warm = factory(cache_bytes_high)
    try:
        for prompt in prompts:
            b = await warm.generate(requests[prompt])
            if recorded[prompt] != b.text:
                mismatches.append(
                    {
                        "prompt": prompt[:60],
                        "cache_0": recorded[prompt][:120],
                        "cache_max": b.text[:120],
                    }
                )
    finally:
        await _release(warm)

    passed = not mismatches
    return GateResult(
        name="g2_placement_invariance",
        passed=passed,
        detail={
            "measured": True,
            "cache_bytes_low": cache_bytes_low,
            "cache_bytes_high": cache_bytes_high,
            "mismatches": mismatches,
        },
        reason=(
            "ok"
            if passed
            else (
                "Expert placement changes the output. Every determinism claim in the "
                "receipt is invalid until this passes (sec 9.3)."
            )
        ),
    )


def write_report(results: Sequence[GateResult], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "all_pass": all(r.passed for r in results),
        "gates": [r.as_dict() for r in results],
    }
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p
