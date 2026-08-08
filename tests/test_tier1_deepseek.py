"""Tier 1 on a reasoning model — DeepSeek-V4-Flash (spec sec 5.1, 5.2, 9.3).

DeepSeek-V4-Flash is the first tier-1 candidate that reasons by default, and that
breaks two invariants at once in a way neither of them notices:

* the `<think>` block is part of the completion, so it spends the sec 5.1 clamp
  before the JSON verdict exists — every rerank then degrades to a failed
  `Verdict` and best-of-N silently stops reranking;
* thinking mode ignores `temperature`, so the greedy judgement the receipt attests
  to (sec 9.3) is actually a sample, and nothing on the wire says so.

Both halves are tested here: the request that asks for thinking off, and — the half
that is load-bearing — the refusal of an answer that reasoned anyway. The second
does not depend on the first having worked, which is the whole point of it.

Everything runs against a `MockTransport`. The Apple Silicon is on the other side
of the socket; the payload, the refusal and the filler are not.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tandem.backends.mlx_tier1 import OptiqTier1Backend, prefill_filler
from tandem.backends.remote_tier1 import RemoteTier1Backend
from tandem.backends.tier1_call import (
    Tier1Unavailable,
    build_payload,
    resolve_reasoning_control,
)
from tandem.tier1.schemas import rerank_schema
from tandem.types import GenRequest, Message, Role, Sampling

DEEPSEEK = "mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit-mixed"
QWEN = "mlx-community/Qwen3.5-122B-A10B-OptiQ-2bit"
RERANK_BODY = '{"choice": 1, "reason": "candidate 1 matches the repo\'s error style"}'


def _rerank_req(n: int = 3) -> GenRequest:
    return GenRequest(
        messages=[Message(role=Role.USER, content="pick one")],
        system="You are a verifier.",
        json_schema=rerank_schema(n),
        sampling=Sampling(temperature=0.0, top_p=1.0, seed=7, max_tokens=4096),
    )


def _endpoint(handler, *, model: str = DEEPSEEK, **kw):
    backend = OptiqTier1Backend("http://127.0.0.1:9999/v1", model=model, **kw)
    original = backend._client
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return backend, original


def _answering(body: str = RERANK_BODY, *, message_extra: dict | None = None,
               usage: dict | None = None):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        seen.append(json.loads(request.content))
        message = {"content": body}
        message.update(message_extra or {})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message}],
                "usage": usage or {"prompt_tokens": 8000, "completion_tokens": 24},
            },
        )

    return handler, seen


# --- naming the family ------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        DEEPSEEK,
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "DeepSeek-V4-Pro",
        "ds4f-q2",  # ds4 serves the same weights under its own short name
        "ds4f-mxfp4",
    ],
)
def test_the_deepseek_v4_family_is_recognised_by_name(model):
    assert resolve_reasoning_control("auto", model) == "deepseek_v4"


@pytest.mark.parametrize(
    "model",
    [
        QWEN,
        "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
        "deepseek-ai/DeepSeek-V3.2",  # V3 does not carry the reasoning mode
        "ds4",  # the *engine*, not a model — must not claim everything it serves
    ],
)
def test_auto_claims_nothing_it_does_not_recognise(model):
    assert resolve_reasoning_control("auto", model) == "none"


def test_an_unknown_control_is_a_startup_error_not_a_failed_verdict():
    """Raised at construction, so a typo is found on `tandem doctor` rather than
    on the first reranked turn of a real session."""
    with pytest.raises(ValueError, match="reasoning_control"):
        OptiqTier1Backend(
            "http://127.0.0.1:9999/v1", model=DEEPSEEK, reasoning_control="high"
        )


def test_there_is_no_way_to_ask_for_reasoning():
    """The knob selects a dialect of "off". A value meaning "on" would be a knob
    for disabling the sec 5.1 clamp, which is not a knob this has."""
    from tandem.backends.tier1_call import REASONING_CONTROLS

    assert REASONING_CONTROLS == ("auto", "deepseek_v4", "none")


# --- the request half -------------------------------------------------------


def test_a_deepseek_payload_turns_thinking_off_in_both_dialects():
    """DeepSeek's API docs and the vLLM recipe document different keys, and the
    engines disagree about which they read. Sending one is a coin flip."""
    payload = build_payload(_rerank_req(), model=DEEPSEEK)

    assert payload["thinking"] == {"type": "disabled"}
    assert payload["chat_template_kwargs"] == {"thinking": False}


def test_a_non_reasoning_model_gets_no_extra_keys():
    """Today's default is a Qwen, and it must go out on the wire unchanged: an
    unknown key is a 400 on a strict engine, which would take tier 1 down for
    every deployment that never asked for DeepSeek."""
    payload = build_payload(_rerank_req(), model=QWEN)

    assert "thinking" not in payload
    assert "chat_template_kwargs" not in payload


def test_the_clamp_and_the_schema_still_ride_along_on_deepseek():
    """The new keys are additions, not a different payload."""
    payload = build_payload(_rerank_req(n=3), model=DEEPSEEK)

    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.0
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["choice"][
        "maximum"
    ] == 2


def test_an_explicit_dialect_overrides_a_name_the_guess_misses():
    payload = build_payload(
        _rerank_req(), model="local-verifier", reasoning_control="deepseek_v4"
    )

    assert payload["thinking"] == {"type": "disabled"}


def test_none_sends_nothing_even_for_a_deepseek():
    """The escape hatch for an engine strict enough to reject the key it does not
    know: configure non-think at the engine and stop sending it."""
    payload = build_payload(_rerank_req(), model=DEEPSEEK, reasoning_control="none")

    assert "thinking" not in payload


@pytest.mark.asyncio
async def test_the_dialect_reaches_the_actual_request_body():
    """A payload helper nobody calls is not a payload. This is the wire."""
    handler, seen = _answering()
    backend, original = _endpoint(handler)
    try:
        await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()

    assert seen[0]["thinking"] == {"type": "disabled"}
    assert seen[0]["chat_template_kwargs"] == {"thinking": False}


# --- the response half, which is the one that holds -------------------------


@pytest.mark.asyncio
async def test_a_verdict_that_reasoned_anyway_is_refused():
    """The engine ignored the flag. The verdict parses, the schema is satisfied,
    and it is still wrong: `temperature` was discarded, so this is a sample and
    the receipt about to be written claims it was greedy."""
    handler, _ = _answering(
        message_extra={"reasoning_content": "Let me compare the two diffs..."}
    )
    backend, original = _endpoint(handler)
    try:
        with pytest.raises(Tier1Unavailable, match="reasoning trace"):
            await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()


@pytest.mark.asyncio
async def test_reasoning_is_refused_from_the_usage_block_too():
    """Some engines report the reasoning in `usage` and strip the text. The
    detector cannot depend on getting the trace itself."""
    handler, _ = _answering(
        usage={
            "prompt_tokens": 8000,
            "completion_tokens": 24,
            "completion_tokens_details": {"reasoning_tokens": 900},
        }
    )
    backend, original = _endpoint(handler)
    try:
        with pytest.raises(Tier1Unavailable, match="reasoning trace"):
            await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()


@pytest.mark.asyncio
async def test_the_refusal_does_not_depend_on_the_request_having_asked():
    """Deliberately not gated on `reasoning_control`. The request-side flag is a
    guess about what the engine reads; this is what it did. A model the name match
    missed has to fail loudly on the first call, not quietly on every one."""
    handler, _ = _answering(message_extra={"reasoning_content": "hmm"})
    backend, original = _endpoint(handler, model="some-unrecognised-verifier")
    try:
        assert backend.reasoning_control == "none"
        with pytest.raises(Tier1Unavailable, match="reasoning trace"):
            await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()


@pytest.mark.asyncio
async def test_an_empty_reasoning_field_is_not_a_reasoning_trace():
    """Engines that always emit the key must not fail every call."""
    handler, _ = _answering(message_extra={"reasoning_content": ""})
    backend, original = _endpoint(handler)
    try:
        result = await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()

    assert json.loads(result.text)["choice"] == 1


@pytest.mark.asyncio
async def test_rung_4_refuses_a_reasoned_verdict_as_well():
    """A hosted DeepSeek reasons by default, and both invariants break exactly the
    same way for being someone else's engine."""

    async def transport(payload: dict) -> dict:
        assert payload["thinking"] == {"type": "disabled"}
        return {
            "choices": [
                {"message": {"content": RERANK_BODY, "reasoning_content": "step 1..."}}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    backend = RemoteTier1Backend(transport, model="deepseek-v4-flash-0731")
    with pytest.raises(Tier1Unavailable, match="reasoning trace"):
        await backend.generate(_rerank_req())


# --- Gate B's filler --------------------------------------------------------


def test_the_filler_is_exactly_the_length_asked_for():
    """Gate B is quoted at an input size. A sample labelled 16k that carried 15.3k
    is a measurement of something else."""
    for n in (1, 23, 4096, 65536):
        assert len(prefill_filler(n)) == n


def test_the_filler_is_deterministic():
    """A gate whose number moves between runs is not a gate."""
    assert prefill_filler(20000) == prefill_filler(20000)


def test_the_filler_does_not_collapse_the_expert_union():
    """The old filler was one 23-character line repeated. On a streamed MoE that
    is not a neutral choice: identical tokens route to identical experts — by
    construction on the layers that use hash routing — so the chunk's expert union
    collapses, the engine's cache serves the whole sweep from RAM, and Gate B
    reports a throughput no real prompt reaches. A floor test must not fail in the
    flattering direction.

    Measured as distinct lines and distinct whitespace-separated tokens, both far
    above what a repeated unit can produce.
    """
    text = prefill_filler(64_000)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    words = {w for w in text.replace("(", " ").replace(")", " ").split() if w}

    assert len(set(lines)) > len(lines) * 0.5
    assert len(words) > 1000


def test_the_filler_still_looks_like_source():
    """Gate B measures prefill on code, because that is what tier 1 reads."""
    text = prefill_filler(4000)

    assert "def " in text and "return " in text and "raise ValueError" in text


@pytest.mark.asyncio
async def test_gate_b_reports_the_cache_size_the_number_depends_on():
    """Streamed prefill throughput is a function of how much of the expert set is
    already resident (sec 10.5). A rate quoted without it cannot be reproduced."""
    handler, _ = _answering()
    backend, original = _endpoint(handler, expert_cache_bytes=32 << 30)
    try:
        await backend.measure_prefill(4096)
        report = backend.gate_b_report()
    finally:
        await backend.close()
        await original.aclose()

    assert report["expert_cache_bytes"] == 32 << 30
    assert report["model"] == DEEPSEEK
    assert report["threshold_tok_per_s"] == 200.0
