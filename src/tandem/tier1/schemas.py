"""Tier-1 call schemas (spec sec 5.1).

Every tier-1 call is constrained to one of these, which is what makes output length
bounded *by construction* rather than by asking nicely. That matters twice over:

* Tier 1 decodes at ~9-13 tok/s. An unbounded verdict is a minutes-long call.
* 2-bit's documented failure mode is broken JSON and invented schema fields
  (sec 5.2). Under a closed schema with `additionalProperties: false`, an invented
  field cannot be emitted.

`title` doubles as the call type: the tier-1 backend reads it to pick the output
budget, so a schema and its token ceiling cannot drift apart.
"""

from __future__ import annotations

from typing import Any

# ~15 output tokens (sec 5.1).
#
# The template shape. Callers should use `rerank_schema(n)` instead: with no upper
# bound on `choice`, a constrained decode can still emit an index that does not
# name a candidate, and "impossible by construction" is the entire reason tier-1
# output is schema-constrained. The verifier catches an out-of-range choice at
# runtime and degrades to candidate 0, but that is a rerank wasted — ~18 s and a
# merge-quality gate that did not happen — for a failure the schema can prevent.
RERANK: dict[str, Any] = {
    "title": "rerank",
    "type": "object",
    "properties": {
        "choice": {"type": "integer", "minimum": 0},
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["choice", "reason"],
    "additionalProperties": False,
}


def rerank_schema(n_candidates: int) -> dict[str, Any]:
    """RERANK bounded to the candidates actually on offer.

    `maximum` is what makes an unusable answer unrepresentable rather than merely
    detected. Matters most for the 2-bit verifier, whose documented failure mode is
    inventing structure (sec 5.2).
    """
    if n_candidates < 1:
        raise ValueError("rerank needs at least one candidate")
    schema = {
        **RERANK,
        "properties": {
            **RERANK["properties"],
            "choice": {"type": "integer", "minimum": 0, "maximum": n_candidates - 1},
        },
    }
    return schema


# ~150 output tokens.
REVIEW: dict[str, Any] = {
    "title": "review",
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "revise", "reject"]},
        "issues": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["blocking", "major", "minor"],
                    },
                    "where": {"type": "string", "maxLength": 120},
                    "what": {"type": "string", "maxLength": 240},
                },
                "required": ["severity", "what"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "issues"],
    "additionalProperties": False,
}

# ~200 output tokens.
PLAN_CRITIQUE: dict[str, Any] = {
    "title": "plan_critique",
    "type": "object",
    "properties": {
        "risks": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 200},
        },
        "missing": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 200},
        },
    },
    "required": ["risks", "missing"],
    "additionalProperties": False,
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "rerank": RERANK,
    "review": REVIEW,
    "plan_critique": PLAN_CRITIQUE,
}
