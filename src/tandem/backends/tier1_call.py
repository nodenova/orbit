"""What every tier-1 transport does to a call, in one place (spec sec 5.1, 5.2).

There is more than one transport now — mlx-optiq behind the process boundary (rung 1)
and a remote endpoint (rung 4) — and three of the things a transport does are
invariants, not details:

* **The output clamp.** `max_tokens` is bounded by the call type's budget, so a
  `review` cannot quietly become a 4,000-token generation. Tier 1 reads at ~1,100
  tok/s and writes at ~11 (sec 5.3): the model that reranks five candidates in 18 s
  takes six minutes to write one. The clamp is the promise that it never tries, and
  it is enforced in code rather than requested of the model.
* **Refusing a judgement we cannot parse.** 2-bit's documented failure mode is broken
  JSON and invented schema fields (sec 5.2). Coercing one quietly is how that failure
  mode reaches a merge decision.
* **Tier 1 does not think.** Reasoning models put a `<think>` block in front of the
  answer, and for a verifier that breaks the clamp and the determinism claim at the
  same time. See the reasoning-control section below.

A clamp that is right in one transport and drifts in another is not a clamp, so all
three live here rather than being copied. That also makes them *testable*: neither
transport can run on a CI box, and until this module existed the ceiling that keeps
tier 1 a verifier had no test at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from tandem.backends.base import BackendUnavailable
from tandem.types import GenRequest

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


# --- reasoning control ------------------------------------------------------
#
# DeepSeek-V4-Flash ships a reasoning mode, and for a verifier it has to be off.
# Two invariants break if it is on, and both break *silently*:
#
# * **The clamp stops bounding the thing it exists to bound.** `CALL_BUDGETS` caps
#   the completion, and a `<think>` block is part of the completion. A rerank
#   clamped to 128 tokens spends them reasoning and is cut off before the JSON
#   verdict exists, so `validate_or_raise` refuses it — and every rerank degrades
#   to a failed `Verdict`. Best-of-N then stops reranking on every turn: the
#   router's documented degradation (sec 5.5), reached by accident and reported
#   as nothing at all.
# * **`temperature` stops being honoured.** DeepSeek's API documentation states
#   that thinking mode does not support `temperature`, `top_p`, `presence_penalty`
#   or `frequency_penalty`, and that setting them is not an error. `build_payload`
#   asks for greedy precisely so two runs of the same rerank agree; under thinking
#   mode that request is accepted and discarded, and the receipt's determinism
#   claim (sec 9.3) becomes false with nothing on the wire to say so.
#
# So this is not a tuning knob, and there is deliberately no value here that turns
# reasoning *on*. The only choice is which spelling of "off" the engine reads.
#
# `auto` guesses from the model name, and guessing is safe here only because the
# guess is not what makes the invariant hold: `read_completion` refuses a reasoned
# answer whatever the request asked for, so a model this fails to recognise fails
# loudly on the first call instead of quietly on every one.
REASONING_CONTROLS = ("auto", "deepseek_v4", "none")

# A model matches when every marker in any one group appears in its name. Two
# groups because the two engines name this model differently: mlx-optiq serves
# `DeepSeek-V4-Flash-0731-OptiQ-2bit-mixed`, ds4 serves it as `ds4f-q2`. Bare
# "ds4" is not a marker — it is the engine's name as well as the model's, and a
# marker that matches an engine would claim every model it serves.
_DEEPSEEK_V4_MARKERS: tuple[tuple[str, ...], ...] = (("deepseek", "v4"), ("ds4f",))


def resolve_reasoning_control(control: str, model: str) -> str:
    """Which reasoning-control dialect to speak for `model`. Never "on"."""
    if control not in REASONING_CONTROLS:
        raise ValueError(
            f"tier1.reasoning_control={control!r} is not one of "
            f"{', '.join(REASONING_CONTROLS)}"
        )
    if control != "auto":
        return control
    name = model.lower()
    matched = any(all(m in name for m in group) for group in _DEEPSEEK_V4_MARKERS)
    return "deepseek_v4" if matched else "none"


def _disable_thinking(payload: dict[str, Any]) -> None:
    """Turn reasoning off in both documented spellings.

    Two authorities document different keys and the engines disagree about which
    they read: DeepSeek's own API docs disable thinking with a top-level `thinking`
    object, while the vLLM recipe for this model does it through
    `chat_template_kwargs`. Sending both costs one ignored key on an engine that
    reads the other; sending one costs a reasoning block on an engine that reads
    the other, which is the failure this exists to prevent.

    An engine strict enough to reject the key it does not know fails the call
    outright, which is the acceptable direction — loud, on the first call, and
    fixable with `reasoning_control = "none"` plus the engine's own configuration.
    """
    payload["thinking"] = {"type": "disabled"}
    payload["chat_template_kwargs"] = {"thinking": False}


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


def build_payload(
    req: GenRequest, *, model: str, reasoning_control: str = "auto"
) -> dict[str, Any]:
    """The OpenAI-shaped body both transports post.

    Greedy, always: a judgement is not a sample, and two runs of the same rerank
    have to agree for the receipt's determinism claim to mean anything. On a
    reasoning model "greedy" is only honoured with thinking off, which is what
    `reasoning_control` arranges — see the section above.
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
    if resolve_reasoning_control(reasoning_control, model) == "deepseek_v4":
        _disable_thinking(payload)
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
    refuse_reasoned_answer(choice, body)
    text = choice.get("message", {}).get("content") or ""
    usage = body.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens", 0)) or count_tokens(
        "\n".join(str(m.get("content", "")) for m in payload.get("messages", []))
    )
    out_tok = int(usage.get("completion_tokens", 0)) or count_tokens(text)
    return text, in_tok, out_tok


def refuse_reasoned_answer(choice: dict[str, Any], body: dict[str, Any]) -> None:
    """Refuse a verdict that arrived with a reasoning trace attached.

    Unconditional, on every transport and every model — deliberately not gated on
    `reasoning_control`. The request-side flag is a guess about what the engine
    reads; this is the observation of what it actually did, and it is the half that
    makes the invariant hold. An engine that ignored the flag, a model the name
    match did not recognise, an operator who configured thinking on at the engine:
    all three land here.

    Failing is the right answer rather than an over-reaction, because a reasoned
    verdict is wrong in a way that reading it cannot reveal. `temperature` was
    silently ignored, so the judgement is a *sample* — and the receipt about to be
    written says it was greedy (sec 9.3). Degrading to a failed `Verdict` costs one
    rerank; accepting it puts an unreproducible judgement behind an attestation
    that claims otherwise.
    """
    message = choice.get("message")
    reasoning = (
        (message or {}).get("reasoning_content") if isinstance(message, dict) else None
    )
    details = (body.get("usage") or {}).get("completion_tokens_details") or {}
    try:
        reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        reasoning_tokens = 0
    if not (reasoning and str(reasoning).strip()) and reasoning_tokens <= 0:
        return
    raise Tier1Unavailable(
        "tier 1 answered with a reasoning trace "
        f"({reasoning_tokens or len(str(reasoning or ''))} reasoning tokens/chars), "
        "so thinking mode is on at the engine. A verifier must not think: the "
        "reasoning spends the sec 5.1 output clamp before the verdict exists, and "
        "thinking mode ignores `temperature`, which makes the greedy judgement the "
        "receipt attests to (sec 9.3) a sample instead. Turn thinking off at the "
        "engine, or set tier1.reasoning_control to a dialect it reads."
    )


def validate_or_raise(text: str, schema: dict[str, Any]) -> None:
    """Reject a judgement we cannot parse rather than pass it upstream.

    A malformed verdict from a 2-bit verifier must fail loudly: coercing it would let
    the model's known failure mode (sec 5.2) reach the merge decision wearing the
    shape of an answer.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Tier1Unavailable(
            f"tier 1 returned non-JSON under a schema: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise Tier1Unavailable("tier 1 returned a JSON value that is not an object")
    missing = [k for k in schema.get("required", []) if k not in obj]
    if missing:
        raise Tier1Unavailable(f"tier 1 judgement missing required fields: {missing}")
