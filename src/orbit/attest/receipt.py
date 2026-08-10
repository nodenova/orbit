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
import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from orbit.types import Sampling

# The floating-point reduction order the engine is pinned to. G1/G2 (sec 9.3) assert
# that both backends compute the same function; this string names which pinning was
# in force, so a receipt from before a kernel change is distinguishable from after.
REDUCTION_ORDER = "pinned-v1"

# A commit id and nothing else. Short shas are allowed because a build pipeline may
# stamp one, but a tag, a branch name or "dirty" is not a commit and must not be
# attested as one.
_SHA_RE = re.compile(r"\A[0-9a-fA-F]{7,64}\Z")


def _stamped_commit() -> str | None:
    """`$ORBIT_ENGINE_COMMIT`, if it is plausibly a commit id.

    Validated rather than trusted: a build that exports a tag, a branch name or an
    empty-after-strip value would otherwise land in the receipt *as the commit*, and
    the receipt is what an auditor re-derives a run from. Rejecting it falls through
    to git or to "unknown" — both honest answers, unlike a plausible-looking wrong
    one.
    """
    raw = (os.environ.get("ORBIT_ENGINE_COMMIT") or "").strip()
    if not raw or not _SHA_RE.match(raw):
        return None
    return raw.lower()


def _git_head_of_this_checkout() -> str | None:
    """HEAD of the repo this very file lives in — never of any other repo.

    The old fallback ran `git -C <__file__ parents[3]> rev-parse HEAD`. For a source
    checkout that is the repo root; for a wheel in site-packages it is some
    unrelated directory that may well be a git repository of the user's, whose HEAD
    would then be attested as the engine that served the request. The guard is that
    the candidate root must actually contain *this file* where the source layout
    puts it: true for a checkout or an editable install, false for a wheel, in which
    case no git process runs at all.
    """
    here = Path(__file__).resolve()
    root = here.parents[3]  # <root>/src/orbit/attest/receipt.py
    if (root / "src" / "orbit" / "attest" / "receipt.py").resolve() != here:
        return None
    if not (root / ".git").exists():  # a worktree's .git is a file, hence exists()
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = out.stdout.strip()
    if out.returncode != 0 or not _SHA_RE.match(head):
        return None
    return head.lower()


@lru_cache(maxsize=1)
def engine_commit() -> str:
    """Commit of the engine that served the request.

    Prefers an explicit build stamp so a wheel installed without a .git directory
    still attests honestly; falls back to git only when this file is inside that
    checkout; otherwise "unknown". Never guesses — and both halves used to (M28):
    the stamp was returned verbatim, and the fallback would attest a stranger's
    HEAD.

    "unknown" rather than None because the field is present in every receipt and a
    consumer diffing two receipts must not have to tell absent from unknown; the
    None cases are internal to the two helpers.
    """
    return _stamped_commit() or _git_head_of_this_checkout() or "unknown"


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
    # The *configured* expert-cache budget — explicitly not occupancy at call time,
    # which is what this field used to claim.
    #
    # It cannot be occupancy. The engine sits behind a process boundary (sec 5.4),
    # reports no cache state on any endpoint it serves, and as of mlx-optiq 0.4.18
    # has no expert LRU at all: `--stream-experts-cache` is accepted and discarded
    # before it reaches the shard reader, so every expert is `pread` per call and the
    # only cache is the OS page cache. See `backends.mlx_tier1.expert_cache_provenance`.
    #
    # The rename is the fix. G2 (sec 9.3) requires output to be invariant to cache
    # state, and the old name implied this field evidenced that — while holding a
    # config constant identical in every receipt, cold or hot, so a G2 violation
    # would have been exactly as invisible with it as without it. A field that reads
    # as evidence and is not is worse than no field, so the name now says "configured",
    # and G2 remains what the deliberate test proves.
    expert_cache_configured_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "container_blake3": self.container_blake3,
            "invoked": self.invoked,
            "call": self.call,
            "rung": self.rung,
            "expert_cache_configured_bytes": self.expert_cache_configured_bytes,
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
    # The id the wire response carries and the audit line records. Without it a
    # customer holding a response and an auditor holding the log share no field and
    # can correlate only by timestamp (M2).
    request_id: str = ""
    # Tip of the audit chain after this turn's record was appended. This is the
    # out-of-band anchor `verify_chain(expected_tip=...)` needs: a link that has left
    # the machine, in the customer's hands, is one the log's writer can no longer
    # quietly restate. Defaulted empty — a receipt is still valid without it, it just
    # anchors nothing (C1).
    audit_tip: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
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
            "audit_tip": self.audit_tip,
        }
