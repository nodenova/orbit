"""Escalation policy — the cascade (spec sec 7.2, 7.3).

Rule-based on purpose. Two triggers, both measurable:

**T1 — best-of-N rerank on `code_change` turns.** Tier 0 generates N candidates at
t=0.6 with the adapter mounted; tier 1 reranks. ~3 x 1.5 s generation + ~18 s rerank
~= 23 s. Off for `chat` and `read_only`, where it would buy nothing and cost
everything.

**T2 — failure escalation.** Patch applied, tests fail -> tier 1 `review` with the
failure output attached -> tier 0 regenerates with the critique in context. Bounded
to one escalation per turn so the worst case stays bounded.

**The pressure valve (sec 7.3)** is the part that is easy to leave out and
shouldn't be. Past ~45 s a `code_change` turn stops feeling interactive, so the
router degrades to N=1 with no rerank — automatically, not as a setting somebody has
to find. It is deliberately sticky within a session: flapping between 23 s and 3 s
turns is worse for a user than consistently choosing one.

Candidate seeds are derived, not random: candidate *i* of a request always uses
seed `base_seed + i`, so a receipt naming `candidate_selected: 1` identifies an
exactly reproducible generation.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from tandem.backends.base import Backend
from tandem.config import RouterConfig
from tandem.router.classify import Classification, classify
from tandem.tier1.verifier import Candidate, Tier1Verifier, Verdict
from tandem.types import GenRequest, GenResult, Message, Role, TurnClass


@dataclass
class CascadeInfo:
    """What the router did, for the receipt and the audit log."""

    turn: TurnClass = TurnClass.CHAT
    classification: dict[str, Any] = field(default_factory=dict)
    candidates_generated: int = 1
    candidate_selected: int = 0
    tier1_invoked: bool = False
    tier1_call: str | None = None
    tier1_verdicts: list[dict[str, Any]] = field(default_factory=list)
    escalated: bool = False
    degraded: bool = False
    degrade_reason: str = ""
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn.value,
            "classification": self.classification,
            "candidates_generated": self.candidates_generated,
            "candidate_selected": self.candidate_selected,
            "tier1_invoked": self.tier1_invoked,
            "tier1_call": self.tier1_call,
            "tier1_verdicts": self.tier1_verdicts,
            "escalated": self.escalated,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# A caller-supplied hook that applies a patch and runs the repo's tests. Returns
# (passed, output). None means the host cannot test, and T2 stays dormant — which is
# the honest behaviour: escalating on a failure we never observed would be theatre.
TestRunner = Callable[[str], Awaitable[tuple[bool, str]]]


class Cascade:
    """Turn class -> tier plan."""

    def __init__(
        self,
        tier0: Backend,
        verifier: Tier1Verifier,
        cfg: RouterConfig | None = None,
        *,
        test_runner: TestRunner | None = None,
    ):
        self.tier0 = tier0
        self.verifier = verifier
        self.cfg = cfg or RouterConfig()
        self.test_runner = test_runner
        # Sticky pressure-valve state (sec 7.3).
        self._degraded = False
        self._degrade_reason = ""
        self.turn_latencies: list[float] = []

    @property
    def degraded(self) -> bool:
        return self._degraded

    def reset_pressure_valve(self) -> None:
        self._degraded = False
        self._degrade_reason = ""

    async def produce(self, req: GenRequest) -> tuple[GenResult, CascadeInfo]:
        t0 = time.perf_counter()
        cls: Classification = classify(req)
        info = CascadeInfo(turn=cls.turn, classification=cls.as_dict())

        if cls.turn in (TurnClass.CHAT, TurnClass.READ_ONLY):
            result = await self.tier0.generate(req)
            info.elapsed_s = time.perf_counter() - t0
            self._record(info)
            return result, info

        if cls.turn is TurnClass.PLAN:
            result, info = await self._plan_turn(req, info)
            info.elapsed_s = time.perf_counter() - t0
            self._record(info)
            return result, info

        result, info = await self._code_change_turn(req, info, t0)
        info.elapsed_s = time.perf_counter() - t0
        self._record(info)
        return result, info

    # --- turn shapes --------------------------------------------------------

    async def _plan_turn(
        self, req: GenRequest, info: CascadeInfo
    ) -> tuple[GenResult, CascadeInfo]:
        result = await self.tier0.generate(req)
        if not self._tier1_usable():
            return result, info
        verdict = await self.verifier.plan_critique(
            result.text, _context_of(req), seed=req.sampling.seed
        )
        info.tier1_invoked = True
        info.tier1_call = "plan_critique"
        info.tier1_verdicts.append(verdict.as_dict())
        if verdict.ok:
            result.text = _append_critique(result.text, verdict)
        return result, info

    async def _code_change_turn(
        self, req: GenRequest, info: CascadeInfo, t0: float
    ) -> tuple[GenResult, CascadeInfo]:
        n = self._candidate_count()
        info.degraded = self._degraded
        info.degrade_reason = self._degrade_reason

        candidates = await self._generate_candidates(req, n)
        info.candidates_generated = len(candidates)

        chosen_idx = 0
        if len(candidates) > 1 and self.cfg.rerank_enabled and self._tier1_usable():
            verdict = await self.verifier.rerank(
                [Candidate(index=i, text=r.text) for i, r in enumerate(candidates)],
                _context_of(req),
                seed=req.sampling.seed,
            )
            info.tier1_invoked = True
            info.tier1_call = "rerank"
            info.tier1_verdicts.append(verdict.as_dict())
            if verdict.ok:
                chosen_idx = int(verdict.data["choice"])
            # A failed rerank falls through to candidate 0 — the first sample at the
            # configured temperature, which is what a no-tier-1 install would have
            # produced anyway (sec 5.5, rung 3).

        info.candidate_selected = chosen_idx
        result = candidates[chosen_idx]

        result, info = await self._maybe_escalate(req, result, info, t0)
        return result, info

    async def _generate_candidates(self, req: GenRequest, n: int) -> list[GenResult]:
        if n <= 1:
            return [await self.tier0.generate(req)]
        base_seed = req.sampling.seed
        reqs = [
            req.with_(
                sampling=type(req.sampling)(
                    temperature=self.cfg.candidate_temperature,
                    top_p=req.sampling.top_p,
                    # Derived, not random: candidate i is reproducible from the
                    # receipt (sec 9.1) without recording N separate seeds.
                    seed=base_seed + i,
                    max_tokens=req.sampling.max_tokens,
                    stop=req.sampling.stop,
                )
            )
            for i in range(n)
        ]
        # Tier 0 is one resident model; these serialise on the GPU regardless. Using
        # gather keeps the code honest about intent and lets a future batched
        # backend actually overlap them.
        return list(await asyncio.gather(*(self.tier0.generate(r) for r in reqs)))

    async def _maybe_escalate(
        self, req: GenRequest, result: GenResult, info: CascadeInfo, t0: float
    ) -> tuple[GenResult, CascadeInfo]:
        """T2: tests fail -> tier-1 review -> tier-0 regenerate with the critique."""
        if self.test_runner is None or not self._tier1_usable():
            return result, info
        if self.cfg.max_escalations_per_turn < 1:
            return result, info

        passed, output = await self.test_runner(result.text)
        if passed:
            return result, info

        verdict = await self.verifier.review(
            result.text, _context_of(req), failure_output=output, seed=req.sampling.seed
        )
        info.tier1_invoked = True
        info.tier1_call = "review"
        info.tier1_verdicts.append(verdict.as_dict())
        if not verdict.ok:
            return result, info

        info.escalated = True
        critique = _format_issues(verdict)
        regen_req = req.with_(
            messages=[
                *req.messages,
                Message(role=Role.ASSISTANT, content=result.text),
                Message(
                    role=Role.USER,
                    content="The repository's tests fail on that patch.\n\n"
                    f"Test output:\n```\n{output.strip()[:4000]}\n```\n\n"
                    f"Review:\n{critique}\n\nProduce a corrected patch.",
                ),
            ]
        )
        regenerated = await self.tier0.generate(regen_req)
        regenerated.receipt = result.receipt
        return regenerated, info

    # --- policy -------------------------------------------------------------

    def _candidate_count(self) -> int:
        if self._degraded:
            return 1
        return max(1, self.cfg.candidates)

    def _tier1_usable(self) -> bool:
        return self.verifier.available and not self._degraded

    def _record(self, info: CascadeInfo) -> None:
        """Pressure valve (sec 7.3).

        Only `code_change` latency counts: a slow `chat` turn is a slow model, not a
        cascade that needs disabling, and folding it in would degrade the gate for
        the wrong reason.
        """
        if info.turn is not TurnClass.CODE_CHANGE:
            return
        self.turn_latencies.append(info.elapsed_s)
        if not self._degraded and info.elapsed_s > self.cfg.degrade_after_s:
            self._degraded = True
            self._degrade_reason = (
                f"code_change turn took {info.elapsed_s:.1f}s "
                f"(> {self.cfg.degrade_after_s:.0f}s); degraded to N=1, no rerank"
            )

    def stats(self) -> dict[str, Any]:
        lat = sorted(self.turn_latencies)
        return {
            "code_change_turns": len(lat),
            "p50_s": round(lat[len(lat) // 2], 2) if lat else None,
            "p95_s": round(lat[int(len(lat) * 0.95)], 2) if lat else None,
            "degraded": self._degraded,
            "degrade_reason": self._degrade_reason,
            "tier1": self.verifier.stats(),
        }


# --- helpers ----------------------------------------------------------------


def _context_of(req: GenRequest, limit: int = 8_000) -> str:
    """Recent conversation as flat context for a tier-1 call.

    Taken from the end: tier 1 gets ~8k of context by budget (sec 5.3), and the
    turn's actual task is at the end, not the beginning.
    """
    parts: list[str] = []
    for msg in reversed(req.messages):
        if not msg.content:
            continue
        parts.append(f"[{msg.role.value}] {msg.content}")
        if sum(len(p) for p in parts) > limit:
            break
    return "\n\n".join(reversed(parts))[-limit:]


def _format_issues(verdict: Verdict) -> str:
    issues = verdict.data.get("issues", [])
    if not issues:
        return f"verdict: {verdict.data.get('verdict', 'revise')}"
    lines = [f"verdict: {verdict.data.get('verdict', 'revise')}"]
    for issue in issues:
        where = issue.get("where", "")
        lines.append(
            f"- [{issue.get('severity', 'minor')}] {where} {issue.get('what', '')}".rstrip()
        )
    return "\n".join(lines)


def _append_critique(plan_text: str, verdict: Verdict) -> str:
    risks = verdict.data.get("risks", [])
    missing = verdict.data.get("missing", [])
    if not risks and not missing:
        return plan_text
    out = [plan_text, "\n\n---\n**Verifier critique**"]
    if risks:
        out.append("\n\nRisks:")
        out.extend(f"\n- {r}" for r in risks)
    if missing:
        out.append("\n\nMissing:")
        out.extend(f"\n- {m}" for m in missing)
    return "".join(out)
