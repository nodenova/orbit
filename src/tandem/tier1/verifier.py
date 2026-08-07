"""Tier-1 verifier API (spec sec 5.1).

Three calls, and only three. There is deliberately no `generate` here: tier 1 never
writes a patch. The asymmetry that justifies the whole architecture — ~1,100 tok/s
prefill against ~11 tok/s decode — means the same model that reranks five candidates
in 18 s would take six minutes to write one (sec 5.3). The interface is the place to
make that structural, so a future caller cannot accidentally reintroduce the six
minutes.

Every call degrades rather than fails. If tier 1 is unreachable, slow, or returns a
judgement we cannot parse, the caller gets a `Verdict` that says so and the router
falls back to tier 0 alone (sec 5.5, rung 3). A merge-quality gate that takes the
whole session down when the verifier hiccups is worse than no gate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import Backend
from ..types import GenRequest, Message, Role, Sampling
from .schemas import PLAN_CRITIQUE, REVIEW, rerank_schema


@dataclass
class Verdict:
    """Outcome of one tier-1 call."""

    call: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "call": self.call,
            "ok": self.ok,
            "data": self.data,
            "latency_s": round(self.latency_s, 2),
            "error": self.error,
        }


@dataclass
class Candidate:
    """One tier-0 candidate patch put up for reranking."""

    index: int
    text: str
    # Filled in by the router when a candidate has already been applied and tested.
    tests_passed: bool | None = None


class Tier1Verifier:
    """Reads and judges. Never generates."""

    def __init__(self, backend: Backend | None, *, timeout_s: float = 180.0):
        self.backend = backend
        self.timeout_s = timeout_s
        self.calls: list[Verdict] = []

    @property
    def available(self) -> bool:
        return self.backend is not None

    async def rerank(
        self, candidates: list[Candidate], context: str, *, seed: int = 0
    ) -> Verdict:
        """Pick the best of N candidates. ~18 s for 5x2k candidates + 8k context."""
        if len(candidates) <= 1:
            return Verdict(
                call="rerank",
                ok=True,
                data={"choice": 0, "reason": "single candidate; no rerank needed"},
            )
        prompt = _rerank_prompt(candidates, context)
        verdict = await self._call("rerank", rerank_schema(len(candidates)), prompt, seed=seed)
        if verdict.ok:
            choice = verdict.data.get("choice")
            # A verifier that returns an out-of-range index has not made a choice we
            # can act on; treat it as a failed call rather than clamping silently
            # into a selection nobody made.
            if not isinstance(choice, int) or not (0 <= choice < len(candidates)):
                return Verdict(
                    call="rerank",
                    ok=False,
                    latency_s=verdict.latency_s,
                    error=f"choice {choice!r} out of range for {len(candidates)} candidates",
                )
        return verdict

    async def review(
        self, patch: str, context: str, *, conventions: str = "", failure_output: str = "", seed: int = 0
    ) -> Verdict:
        """Judge one patch. ~25 s at 12k input."""
        prompt = _review_prompt(patch, context, conventions, failure_output)
        return await self._call("review", REVIEW, prompt, seed=seed)

    async def plan_critique(self, plan: str, repo_map: str, *, seed: int = 0) -> Verdict:
        prompt = _plan_prompt(plan, repo_map)
        return await self._call("plan_critique", PLAN_CRITIQUE, prompt, seed=seed)

    async def _call(self, name: str, schema: dict[str, Any], prompt: str, *, seed: int) -> Verdict:
        if self.backend is None:
            return Verdict(call=name, ok=False, error="tier 1 not enabled")
        req = GenRequest(
            system=_SYSTEM,
            messages=[Message(role=Role.USER, content=prompt)],
            # Greedy: a judgement is not a sample, and two runs of the same rerank
            # must agree for the receipt's determinism claim to mean anything.
            sampling=Sampling(temperature=0.0, top_p=1.0, seed=seed, max_tokens=512),
            json_schema=schema,
            request_id=f"tier1-{name}",
        )
        t0 = time.perf_counter()
        try:
            result = await self.backend.generate(req)
        except Exception as exc:  # noqa: BLE001 - degrade, never take the turn down
            verdict = Verdict(
                call=name, ok=False, latency_s=time.perf_counter() - t0, error=str(exc)
            )
            self.calls.append(verdict)
            return verdict
        latency = time.perf_counter() - t0

        try:
            data = json.loads(result.text)
            if not isinstance(data, dict):
                raise ValueError("verdict was not a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            verdict = Verdict(
                call=name,
                ok=False,
                latency_s=latency,
                error=f"unparseable verdict ({exc}); 2-bit JSON failure mode, sec 5.2",
            )
            self.calls.append(verdict)
            return verdict

        missing = [k for k in schema.get("required", []) if k not in data]
        if missing:
            verdict = Verdict(
                call=name, ok=False, latency_s=latency, error=f"verdict missing {missing}"
            )
            self.calls.append(verdict)
            return verdict

        verdict = Verdict(call=name, ok=True, data=data, latency_s=latency)
        self.calls.append(verdict)
        return verdict

    def stats(self) -> dict[str, Any]:
        if not self.calls:
            return {"calls": 0}
        ok = sum(1 for c in self.calls if c.ok)
        return {
            "calls": len(self.calls),
            "ok": ok,
            "failure_rate": round(1 - ok / len(self.calls), 3),
            "mean_latency_s": round(sum(c.latency_s for c in self.calls) / len(self.calls), 2),
            "by_call": {
                name: sum(1 for c in self.calls if c.call == name)
                for name in {c.call for c in self.calls}
            },
        }


_SYSTEM = (
    "You are a code reviewer for a specific repository. You read diffs and judge "
    "them. You never write code and never propose a patch. Answer only with the "
    "JSON object the schema requires."
)


def _rerank_prompt(candidates: list[Candidate], context: str) -> str:
    parts = ["# Context\n", context.strip(), "\n\n# Candidate patches\n"]
    for c in candidates:
        parts.append(f"\n## Candidate {c.index}\n")
        if c.tests_passed is not None:
            parts.append(f"(repository tests: {'pass' if c.tests_passed else 'fail'})\n")
        parts.append("```diff\n")
        parts.append(c.text.strip())
        parts.append("\n```\n")
    parts.append(
        "\n# Task\nChoose the candidate most likely to be merged by this "
        "repository's maintainers. Weigh convention conformance and blast radius, "
        "not just whether it works. Return the candidate index.\n"
    )
    return "".join(parts)


def _review_prompt(patch: str, context: str, conventions: str, failure_output: str) -> str:
    parts = ["# Context\n", context.strip(), "\n"]
    if conventions:
        parts.append("\n# Repository conventions\n" + conventions.strip() + "\n")
    if failure_output:
        # T2 escalation (sec 7.2): the failing test output is the whole point of
        # the call, so it goes above the patch where a truncating reader sees it.
        parts.append("\n# Test failure\n```\n" + failure_output.strip() + "\n```\n")
    parts.append("\n# Patch\n```diff\n" + patch.strip() + "\n```\n")
    parts.append(
        "\n# Task\nWould this be merged as-is? Return a verdict and the specific "
        "issues that would draw a review comment.\n"
    )
    return "".join(parts)


def _plan_prompt(plan: str, repo_map: str) -> str:
    return (
        "# Repository map\n"
        + repo_map.strip()
        + "\n\n# Proposed plan\n"
        + plan.strip()
        + "\n\n# Task\nList the risks this plan carries and what it is missing.\n"
    )
