"""Response-attached attestation metadata (spec sec 9.1).

The receipt is the product's answer to "which base and which adapter produced this
change?". It rides on every response and is mirrored into the audit log.

The shape is fixed by the spec; `Receipt.as_dict()` emits exactly it. Fields that do
not apply (no tier-1 call, no adapter) are still present with honest nulls — a
consumer diffing two receipts should never have to distinguish "absent" from
"unknown".
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..types import Sampling

# The floating-point reduction order the engine is pinned to. G1/G2 (sec 9.3) assert
# that both backends compute the same function; this string names which pinning was
# in force, so a receipt from before a kernel change is distinguishable from after.
REDUCTION_ORDER = "pinned-v1"


@lru_cache(maxsize=1)
def engine_commit() -> str:
    """Commit of the engine that served the request.

    Prefers an explicit build stamp so a wheel installed without a .git directory
    still attests honestly; falls back to git, then to "unknown". Never guesses.
    """
    stamped = os.environ.get("TANDEM_ENGINE_COMMIT")
    if stamped:
        return stamped.strip()
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@dataclass
class Tier0Attestation:
    container_blake3: str | None = None
    adapter_blake3: str | None = None
    profile_blake3: str | None = None
    adapter_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "container_blake3": self.container_blake3,
            "adapter_blake3": self.adapter_blake3,
            "profile_blake3": self.profile_blake3,
            "adapter_name": self.adapter_name,
        }


@dataclass
class Tier1Attestation:
    container_blake3: str | None = None
    invoked: bool = False
    call: str | None = None  # rerank | review | plan_critique
    # Which rung of the sec 5.5 ladder served this verdict. A rung-3 second opinion
    # is the resident model with its adapter unmounted — weaker than the 122B, and
    # a receipt that did not say so would let a customer read a base-model verdict
    # as a streamed-verifier one.
    rung: str | None = None
    # Expert-cache occupancy at call time. G2 (sec 9.3) requires output to be
    # invariant to this; recording it is what makes a violation detectable after
    # the fact rather than only under a deliberate test.
    expert_cache_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "container_blake3": self.container_blake3,
            "invoked": self.invoked,
            "call": self.call,
            "rung": self.rung,
            "expert_cache_bytes": self.expert_cache_bytes,
        }


@dataclass
class Receipt:
    tier0: Tier0Attestation = field(default_factory=Tier0Attestation)
    tier1: Tier1Attestation = field(default_factory=Tier1Attestation)
    compaction_template: str | None = None
    sampling: Sampling = field(default_factory=Sampling)
    candidates_generated: int = 1
    candidate_selected: int = 0
    # Populated when the router escalated on a failing test (T2, sec 7.2).
    escalated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier0": self.tier0.as_dict(),
            "tier1": self.tier1.as_dict(),
            "engine_commit": engine_commit(),
            "compaction_template": self.compaction_template,
            "seed": self.sampling.seed,
            "sampling": {
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
            },
            "reduction_order": REDUCTION_ORDER,
            "candidates_generated": self.candidates_generated,
            "candidate_selected": self.candidate_selected,
            "escalated": self.escalated,
        }
