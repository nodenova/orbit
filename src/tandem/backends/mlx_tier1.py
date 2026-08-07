"""Tier 1 — streamed verifier, reached over a process boundary (spec sec 5.4).

v1 runs `mlx-optiq --stream-experts` as a **separate process** and talks to it over
its OpenAI-compatible endpoint. That is not laziness about integration: mlx-optiq's
source lives in a private monorepo, so it is unforkable and unauditable, and the
only defensible way to depend on it is as a black box behind a pinned version and a
process boundary (sec 13). Phase 2 replaces it with an in-house streaming loader on
mlx-lm and this file's interface does not change.

Two invariants hold on every tier-1 call, and both now live in `tier1_call.py`
because rung 4 is a second transport and a clamp that drifts between transports is
not a clamp:

* **Tier 1 never generates a patch** (sec 5.1). `max_tokens` is clamped to the call
  type's budget. A `review` cannot quietly become a 4,000-token generation that
  takes six minutes — the clamp makes that structurally impossible, not merely
  discouraged.
* **Output is schema-constrained.** 2-bit's specific documented weakness is broken
  JSON and invented schema fields (sec 5.2); constrained decoding is what makes
  that weakness not bite. If the endpoint cannot constrain, we validate and reject
  rather than pass malformed judgements upstream.

What is left here is the part that is specific to this transport: the client, the
process boundary, and the prefill instrumentation Gate B reads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..attest.hashing import hash_artefact
from ..types import GenRequest, GenResult, Message, Role, Sampling, StopReason, Usage
from .base import Backend
from .tier1_call import Tier1Unavailable, build_payload, read_completion, validate_or_raise


@dataclass
class PrefillSample:
    """One measurement of streamed prefill throughput — M0 Gate B (sec 11)."""

    input_tokens: int
    output_tokens: int
    prefill_s: float
    total_s: float

    @property
    def prefill_tok_per_s(self) -> float:
        return self.input_tokens / self.prefill_s if self.prefill_s > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "prefill_s": round(self.prefill_s, 3),
            "total_s": round(self.total_s, 3),
            "prefill_tok_per_s": round(self.prefill_tok_per_s, 1),
        }


class OptiqTier1Backend(Backend):
    """Streamed 122B verifier behind mlx-optiq's OpenAI endpoint."""

    name = "optiq-tier1"
    tier = 1

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        container_path: str | None = None,
        timeout_s: float = 180.0,
        expert_cache_bytes: int | None = None,
        pinned_version: str = "",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.expert_cache_bytes = expert_cache_bytes
        self.pinned_version = pinned_version
        self._container_hash = hash_artefact(container_path)
        self._client = httpx.AsyncClient(timeout=timeout_s)
        # Every prefill measurement taken this run, for the M0 Gate B report.
        self.prefill_samples: list[PrefillSample] = []

    def container_hash(self) -> str | None:
        return self._container_hash

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> tuple[bool, str]:
        try:
            r = await self._client.get(f"{self.endpoint}/models", timeout=5.0)
        except httpx.HTTPError as exc:
            return False, f"tier 1 endpoint unreachable at {self.endpoint}: {exc}"
        if r.status_code != 200:
            return False, f"tier 1 endpoint returned {r.status_code}"
        return True, "ok"

    async def generate(self, req: GenRequest) -> GenResult:
        """Serve a canonical request. Always schema-constrained, always clamped."""
        payload = build_payload(req, model=self.model)

        t0 = time.perf_counter()
        try:
            resp = await self._client.post(f"{self.endpoint}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise Tier1Unavailable(f"tier 1 call failed: {exc}") from exc
        total_s = time.perf_counter() - t0
        if resp.status_code != 200:
            raise Tier1Unavailable(f"tier 1 returned {resp.status_code}: {resp.text[:200]}")

        body = resp.json()
        text, in_tok, out_tok = read_completion(body, payload, self.count_tokens)

        # Attribute time between prefill and decode using the measured decode rate
        # if the endpoint reports one; otherwise treat the whole call as prefill,
        # which understates prefill throughput rather than flattering it.
        #
        # A reported decode at least as long as the whole call is an incoherent
        # measurement — engine-side and client-side clocks disagreeing, or a
        # response served from the engine's own cache. Subtracting it anyway would
        # floor `prefill_s` at 1e-6 and hand Gate B a throughput in the billions,
        # which is the one direction a floor test must never fail in. Discard the
        # reported figure and fall back to the conservative default above.
        decode_s = _reported_decode_seconds(body)
        if decode_s >= total_s:
            decode_s = 0.0
        prefill_s = max(1e-6, total_s - decode_s)
        self.prefill_samples.append(
            PrefillSample(in_tok, out_tok, prefill_s=prefill_s, total_s=total_s)
        )

        if req.json_schema is not None:
            validate_or_raise(text, req.json_schema)

        return GenResult(
            text=text,
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
            total_s=total_s,
        )

    async def measure_prefill(self, input_tokens: int) -> PrefillSample:
        """M0 Gate B: instrument streamed prefill at a given input size (sec 11, 14.3).

        Gate B is >=200 tok/s. Below that, batch-union prefill (sec 5.3) is not
        happening inside the engine and the in-house loader becomes M-blocking
        rather than optional — which is a three-week schedule fact, so it is worth
        measuring on day three rather than discovering in month two.
        """
        # ~4 chars/token of filler; content is irrelevant, length is the variable.
        # Sized in characters rather than in repeats of the unit: the unit is 23
        # chars, so a repeat count derived from the token count undershoots the
        # target by ~4%, and a sample labelled 16k that carried 15.3k tokens is a
        # measurement of a different input size than the one Gate B is quoted at.
        unit = "def f():\n    return 1\n\n"
        target_chars = input_tokens * 4
        filler = (unit * (target_chars // len(unit) + 1))[:target_chars]
        req = GenRequest(
            messages=[Message(role=Role.USER, content=filler)],
            system="Reply with the single word: ok",
            sampling=Sampling(temperature=0.0, max_tokens=4),
        )
        await self.generate(req)
        return self.prefill_samples[-1]

    def gate_b_report(self) -> dict[str, Any]:
        if not self.prefill_samples:
            return {"pass": False, "reason": "no samples taken"}
        rates = [s.prefill_tok_per_s for s in self.prefill_samples]
        worst = min(rates)
        return {
            "pass": worst >= 200.0,
            "threshold_tok_per_s": 200.0,
            "worst_tok_per_s": round(worst, 1),
            "samples": [s.as_dict() for s in self.prefill_samples],
            "note": (
                "Below 200 tok/s the engine is not doing batch-union prefill; "
                "Phase 2 in-house loader becomes M-blocking (spec sec 5.4, 11)."
            ),
        }


def _reported_decode_seconds(body: dict[str, Any]) -> float:
    """Best-effort decode-time extraction from engine-specific timing fields."""
    for key in ("timings", "timing", "metrics"):
        block = body.get(key)
        if isinstance(block, dict):
            for field_name in ("predicted_ms", "decode_ms", "generation_ms"):
                if field_name in block:
                    return float(block[field_name]) / 1000.0
            for field_name in ("predicted_s", "decode_s"):
                if field_name in block:
                    return float(block[field_name])
    return 0.0
