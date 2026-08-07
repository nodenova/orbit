"""What every tier-1 transport does to a call, in one place (spec sec 5.1, 5.2).

There is more than one transport now — mlx-optiq behind the process boundary (rung 1)
and a remote endpoint (rung 4) — and two of the things a transport does are
invariants, not details:

* **The output clamp.** `max_tokens` is bounded by the call type's budget, so a
  `review` cannot quietly become a 4,000-token generation. Tier 1 reads at ~1,100
  tok/s and writes at ~11 (sec 5.3): the model that reranks five candidates in 18 s
  takes six minutes to write one. The clamp is the promise that it never tries, and
  it is enforced in code rather than requested of the model.
* **Refusing a judgement we cannot parse.** 2-bit's documented failure mode is broken
  JSON and invented schema fields (sec 5.2). Coercing one quietly is how that failure
  mode reaches a merge decision.

A clamp that is right in one transport and drifts in another is not a clamp, so both
live here rather than being copied. That also makes them *testable*: neither transport
can run on a CI box, and until this module existed the ceiling that keeps tier 1 a
verifier had no test at all.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..types import GenRequest
from .base import BackendUnavailable

# Hard output ceilings per call type (sec 5.1). Generous against the spec's budgets
# so a schema with a long `reason` still fits, tight enough that generation is
# impossible.
CALL_BUDGETS: dict[str, int] = {
    "rerank": 128,
    "review": 512,
    "plan_critique": 640,
}
DEFAULT_BUDGET = 256


class Tier1Unavailable(BackendUnavailable):
    """Tier 1 could not answer. Always degrades to a failed `Verdict` upstream."""


def call_type_of(req: GenRequest) -> str:
    """Which of the three calls (sec 5.1) this request is.

    Read off the schema title, because the schema is the thing the caller and the
    engine both already agree on. An unrecognised title falls to `review`, the
    middle budget — an unknown call must not inherit the *largest* ceiling.
    """
    schema = req.json_schema or {}
    title = str(schema.get("title", ""))
    return title if title in CALL_BUDGETS else "review"


def clamp_max_tokens(req: GenRequest) -> int:
    """The call's output ceiling. Never above its budget, whatever was asked for."""
    budget = CALL_BUDGETS.get(call_type_of(req), DEFAULT_BUDGET)
    return min(req.sampling.max_tokens or budget, budget)


def build_payload(req: GenRequest, *, model: str) -> dict[str, Any]:
    """The OpenAI-shaped body both transports post.

    Greedy, always: a judgement is not a sample, and two runs of the same rerank
    have to agree for the receipt's determinism claim to mean anything.
    """
    messages: list[dict[str, Any]] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    for m in req.messages:
        messages.append({"role": m.role.value, "content": m.content})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": clamp_max_tokens(req),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": req.sampling.seed,
        "stream": False,
    }
    if req.json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": call_type_of(req),
                "schema": req.json_schema,
                "strict": True,
            },
        }
    return payload


def read_completion(
    body: dict[str, Any], payload: dict[str, Any], count_tokens: Callable[[str], int]
) -> tuple[str, int, int]:
    """Pull (text, input tokens, output tokens) out of a chat-completions body.

    Falls back to an estimate when the engine reports no usage block, because a
    missing count must not silently become a zero — zero input tokens would report
    an infinite prefill rate straight into the Gate B number.
    """
    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise Tier1Unavailable(f"tier 1 returned no choices: {exc}") from exc
    text = choice.get("message", {}).get("content") or ""
    usage = body.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens", 0)) or count_tokens(
        "\n".join(str(m.get("content", "")) for m in payload.get("messages", []))
    )
    out_tok = int(usage.get("completion_tokens", 0)) or count_tokens(text)
    return text, in_tok, out_tok


def validate_or_raise(text: str, schema: dict[str, Any]) -> None:
    """Reject a judgement we cannot parse rather than pass it upstream.

    A malformed verdict from a 2-bit verifier must fail loudly: coercing it would let
    the model's known failure mode (sec 5.2) reach the merge decision wearing the
    shape of an answer.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Tier1Unavailable(f"tier 1 returned non-JSON under a schema: {exc}") from exc
    if not isinstance(obj, dict):
        raise Tier1Unavailable("tier 1 returned a JSON value that is not an object")
    missing = [k for k in schema.get("required", []) if k not in obj]
    if missing:
        raise Tier1Unavailable(f"tier 1 judgement missing required fields: {missing}")
