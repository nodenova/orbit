"""Fallback ladder rung 2 — the 80B swapped into residency (spec sec 5.5).

    2. Tier 1 = an 80B swapped into residency, evicting tier 0 (~10 s each way)

The rung for a machine that has a second model worth verifying with but cannot hold
it alongside tier 0. Unified memory is the whole constraint: no Apple Silicon box in
this class fits a 4-bit 35B and a 4-bit 80B at once, so admitting one *means* evicting
the other. Every design decision below follows from that one sentence.

On the baseline host (36 GB, 28.08 GiB working set — `docs/BASELINE.md`) the second
occupant does not exist and there is no room for it: tier 0 alone is 23.0 GiB. The
policy below is built and tested against the mock, which is where the concurrency bugs
are; what it lacks is a resident verifier to swap in.

Three consequences, each of which is a way this rung goes silently wrong if it is
left implicit:

* **Residency is exclusive, so tier 0 cannot serve while the verifier is resident.**
  Not "should not" — the weights are gone. A tier-0 request arriving mid-verdict has
  to wait for the swap back; the only alternative is generating from a model that is
  not in memory. `ResidencySwitch` is that mutual exclusion and `SwapGuard` is what
  puts tier-0 requests behind it. Without the guard this rung is a correctness bug
  wearing a latency costume, and `Pipeline` installs the guard for that reason.

* **The swap costs what it costs, and is not amortised by pretending.** ~10 s each
  way against an 18 s rerank: a verified turn costs roughly three times an unverified
  one, which is why this is rung 2 and not the design target. So the switch *measures*
  every transition, and the backend declines once the measured round trip exceeds the
  configured budget — declining on a measurement, never on a guess, and never before
  it has one. A declined call degrades to a failed `Verdict` (sec 5.5), which is the
  ladder doing its job rather than the turn failing.

* **Swapping back is lazy: whoever is resident stays resident until someone else needs
  the memory.** The eager alternative — restore tier 0 as soon as the verdict is in —
  adds ~10 s to the tail of every verified turn for a model that turn's answer does
  not need, and throws the work away entirely when the next event is a second verifier
  call. Lazily, a rerank-then-review pair pays for one swap instead of two, and the
  swap back lands on the request that actually wants tier 0.

The occupants are the part that needs the hardware. `Occupant` is deliberately two
async methods and nothing else, so the policy above the line is exercisable against
the mock — the policy is where the concurrency bugs live, and they are the ones a
test can catch.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from orbit.backends.base import Backend, BackendUnavailable, Delta, ToolCallRenderer
from orbit.types import GenRequest, GenResult

RUNG = "resident_swap"

TIER0 = "tier0"
TIER1 = "tier1"


def _other(side: str) -> str:
    return TIER1 if side == TIER0 else TIER0


class SwapBudgetExceeded(BackendUnavailable):
    """The measured swap round trip costs more than the rung is allowed to spend."""


class Occupant(Protocol):
    """Something that can be moved in and out of unified memory.

    Two methods, both required to be *actual* memory movements. An implementation
    that no-ops `unload` turns this rung into a claim that a swap happened when it
    did not — the attestation would name a rung whose defining cost was never paid,
    and on a real machine the second model would simply fail to allocate.

    `load` must leave the occupant able to serve immediately and must restore it to
    the same identity it had before: same container hash, same mounted adapters,
    same adapter hashes. A tier 0 that comes back from a swap with a different
    adapter set produces a receipt that names what did not run.
    """

    name: str

    async def load(self) -> None: ...

    async def unload(self) -> None: ...


def _check_occupant(side: str, occupant: Any) -> None:
    """Reject an occupant that cannot actually leave memory.

    `Occupant` is a static protocol and buys nothing at runtime, so the check happens
    here, at construction, where the error can say what is missing and why it matters.
    Deliberately *not* a default pair of no-op hooks on `Backend`: a backend that
    inherited those would satisfy the protocol while swapping nothing, and this rung's
    whole cost model would quietly become a fiction.
    """
    for hook in ("load", "unload"):
        if not callable(getattr(occupant, hook, None)):
            raise ValueError(
                f"{side} occupant {type(occupant).__name__} has no {hook}(). Rung 2 "
                "evicts one model to admit the other, so both have to be able to "
                "leave memory and come back."
            )


@dataclass
class SwapStats:
    """What the swaps actually cost. Reported, because this rung's failure mode is
    thrash and thrash is invisible unless it is counted."""

    swaps: int = 0
    swap_seconds: float = 0.0
    # Most recent measured transition in each direction. The round trip is their
    # sum, and it is 0 until both legs have been observed — which is what makes the
    # budget guard measurement-driven rather than a guess.
    last_s: dict[str, float] = field(default_factory=lambda: {TIER0: 0.0, TIER1: 0.0})
    # Holds that had to wait for someone else's residency.
    waits: int = 0
    # Calls refused because the measured round trip is over budget.
    declined: int = 0
    failed: int = 0

    @property
    def round_trip_s(self) -> float:
        return self.last_s[TIER0] + self.last_s[TIER1]

    def record(self, side: str, seconds: float) -> None:
        self.swaps += 1
        self.swap_seconds += seconds
        self.last_s[side] = seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "swaps": self.swaps,
            "swap_seconds": round(self.swap_seconds, 3),
            "round_trip_s": round(self.round_trip_s, 3),
            "waits": self.waits,
            "declined": self.declined,
            "failed": self.failed,
        }


class ResidencySwitch:
    """Owns which of two models is in unified memory. Exactly one, ever.

    Fairness is deliberate and costs something. Once a side is waiting to swap in,
    new arrivals for the *resident* side queue behind it instead of joining the
    batch in front of them. Without that rule a steady stream of tier-0 turns starves
    the verifier indefinitely — merge quality quietly switches itself off under load,
    which is the one failure this product cannot have. With it, a verdict stalls
    every concurrent tier-0 request for the length of one swap. That stall is the
    honest price of this rung, and it is most of why it is rung 2.
    """

    def __init__(self, tier0: Occupant, tier1: Occupant, *, resident: str = TIER0):
        _check_occupant(TIER0, tier0)
        _check_occupant(TIER1, tier1)
        self._occupants = {TIER0: tier0, TIER1: tier1}
        # Which side is in memory, or None when a swap failed halfway and we do not
        # know. None is the honest state: reporting the old occupant as resident
        # after a failed unload is exactly the lie this class exists to prevent.
        self._resident: str | None = resident
        self._cond = asyncio.Condition()
        self._in_flight = {TIER0: 0, TIER1: 0}
        self._pending = {TIER0: 0, TIER1: 0}
        self.stats = SwapStats()

    @property
    def resident(self) -> str | None:
        return self._resident

    def _may_enter(self, side: str) -> bool:
        if self._resident == side:
            # Yield to a swap that is already queued (see the fairness note).
            return self._pending[_other(side)] == 0
        # Swapping in means evicting the other side, which cannot happen while
        # anyone is mid-generation on it.
        return self._in_flight[_other(side)] == 0

    @asynccontextmanager
    async def hold(self, side: str) -> AsyncIterator[None]:
        """Hold `side` resident for the duration of the block."""
        async with self._cond:
            waited = False
            self._pending[side] += 1
            try:
                while not self._may_enter(side):
                    waited = True
                    await self._cond.wait()
            finally:
                self._pending[side] -= 1
            if waited:
                self.stats.waits += 1
            if self._resident != side:
                await self._swap_to(side)
            self._in_flight[side] += 1
        try:
            yield
        finally:
            async with self._cond:
                self._in_flight[side] -= 1
                self._cond.notify_all()

    async def _swap_to(self, side: str) -> None:
        """Evict whoever is resident, admit `side`. Runs under the condition lock,
        so nothing else touches memory while it is in progress."""
        outgoing = self._resident
        t0 = time.perf_counter()
        try:
            if outgoing is not None:
                await self._occupants[outgoing].unload()
            # From here until `load` returns, nothing is resident. Recorded as such
            # so a failure below leaves the switch saying "nothing is in memory"
            # rather than naming a model that has been freed.
            self._resident = None
            await self._occupants[side].load()
        except Exception:
            self.stats.failed += 1
            self._cond.notify_all()
            raise
        self._resident = side
        self.stats.record(side, time.perf_counter() - t0)

    def stats_dict(self) -> dict[str, Any]:
        return {"rung": RUNG, "resident": self._resident, **self.stats.as_dict()}


class SwapGuard(Backend):
    """Tier 0, behind the residency switch.

    Pure delegation apart from `generate`/`stream`, which wait for tier 0 to be the
    model that is actually in memory. Everything else — hashes, rendering, KV state,
    token counting — is the wrapped backend's answer verbatim, because this wrapper
    changes *when* tier 0 runs and nothing about *what* it is.
    """

    tier = 0

    def __init__(self, tier0: Backend, switch: ResidencySwitch):
        self._tier0 = tier0
        self._switch = switch
        self.name = tier0.name

    @property
    def wrapped(self) -> Backend:
        return self._tier0

    async def generate(self, req: GenRequest) -> GenResult:
        async with self._switch.hold(TIER0):
            return await self._tier0.generate(req)

    async def stream(self, req: GenRequest) -> AsyncIterator[Delta]:
        # The hold spans the whole stream, not just its first token: a swap
        # mid-decode would evict the weights the decode is running on.
        async with self._switch.hold(TIER0):
            async for delta in self._tier0.stream(req):
                yield delta

    # --- passthrough --------------------------------------------------------

    def render(self, req: GenRequest, render_tool_call: ToolCallRenderer = None) -> str:
        return self._tier0.render(req, render_tool_call)

    def renders_canonically(self) -> bool:
        return self._tier0.renders_canonically()

    def count_tokens(self, text: str) -> int:
        return self._tier0.count_tokens(text)

    def container_hash(self) -> str | None:
        return self._tier0.container_hash()

    def adapter_hash(self, adapter: str | None) -> str | None:
        return self._tier0.adapter_hash(adapter)

    def profile_hash(self, adapter: str | None) -> str | None:
        return self._tier0.profile_hash(adapter)

    def mounted_adapters(self) -> tuple[str, ...]:
        return self._tier0.mounted_adapters()

    def supports_state(self) -> bool:
        return self._tier0.supports_state()

    def state_key(self, adapter: str | None) -> str:
        return self._tier0.state_key(adapter)

    def accepts_state(self, state: Any, adapter: str | None) -> bool:
        return self._tier0.accepts_state(state, adapter)

    def export_state(
        self, req: GenRequest, rendered_prefix: str, result: GenResult
    ) -> Any:
        return self._tier0.export_state(req, rendered_prefix, result)

    async def close(self) -> None:
        await self._tier0.close()


class ResidentSwapBackend(Backend):
    """The 80B, serving the tier-1 role from residency it has to be swapped into."""

    name = "resident-swap-tier1"
    tier = 1

    def __init__(
        self, verifier: Backend, switch: ResidencySwitch, *, budget_s: float = 0.0
    ):
        self._verifier = verifier
        self._switch = switch
        # Round-trip ceiling in seconds; 0 disables the guard. Off by default
        # because a ceiling that fires on a machine nobody has measured yet would
        # disable verification for a cost that was never observed.
        self.budget_s = budget_s

    @property
    def switch(self) -> ResidencySwitch:
        return self._switch

    def guard_tier0(self, tier0: Backend) -> Backend:
        """Wrap tier 0 so its requests wait for it to be resident.

        Not optional. An unguarded tier 0 will happily be asked to generate while
        its weights are out of memory.
        """
        return SwapGuard(tier0, self._switch)

    async def generate(self, req: GenRequest) -> GenResult:
        measured = self._switch.stats.round_trip_s
        if self.budget_s > 0 and measured > self.budget_s:
            self._switch.stats.declined += 1
            raise SwapBudgetExceeded(
                f"rung 2 swap round trip measured at {measured:.1f}s, over the "
                f"{self.budget_s:.1f}s budget; declining rather than spending it "
                "(sec 5.5 — the call degrades to a failed verdict)"
            )
        async with self._switch.hold(TIER1):
            return await self._verifier.generate(req)

    # --- attestation --------------------------------------------------------
    #
    # The container is the 80B's: it is the model that produced the verdict. No
    # adapter, ever — the repo adapter belongs to tier 0 and does not follow the
    # verifier into residency, which is the same reason rung 3 strips it.

    def container_hash(self) -> str | None:
        return self._verifier.container_hash()

    def adapter_hash(self, adapter: str | None) -> str | None:
        return None

    def profile_hash(self, adapter: str | None) -> str | None:
        return None

    def mounted_adapters(self) -> tuple[str, ...]:
        return ()

    def render(self, req: GenRequest, render_tool_call: ToolCallRenderer = None) -> str:
        return self._verifier.render(req, render_tool_call)

    def renders_canonically(self) -> bool:
        return self._verifier.renders_canonically()

    def count_tokens(self, text: str) -> int:
        return self._verifier.count_tokens(text)

    def supports_state(self) -> bool:
        # A KV cache does not survive its own model being evicted, and this model is
        # evicted between calls by construction.
        return False

    async def close(self) -> None:
        await self._verifier.close()

    def stats(self) -> dict[str, Any]:
        return self._switch.stats_dict() | {"budget_s": self.budget_s}
