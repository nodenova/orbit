"""Tier 1 over the mlx-optiq process boundary (spec sec 5.4).

`backends/mlx_tier1.py` was carried on the "never executed, Apple-Silicon-only"
list next to `mlx_tier0.py`. That was wrong about this half of the pair: it
imports no MLX. The whole point of sec 5.4 is that mlx-optiq lives behind a
*process boundary*, so what is in this file is an httpx client, and an httpx
client is testable anywhere. The Apple Silicon is on the other side of the
socket, and on a CI box a `MockTransport` stands exactly where it stands.

So the parts this covers are not approximations of the real thing — they are the
real thing: the payload built for the endpoint, the clamp that keeps tier 1 a
verifier, the schema refusal, and the Gate B arithmetic that decides whether the
in-house streaming loader is optional or M-blocking. Only the numbers coming back
over the socket are synthetic, and those were never going to be real off-target.
"""

from __future__ import annotations

import httpx
import pytest

from tandem.backends.mlx_tier1 import (
    OptiqTier1Backend,
    PrefillSample,
    _reported_decode_seconds,
)
from tandem.backends.tier1_call import Tier1Unavailable
from tandem.tier1.schemas import rerank_schema
from tandem.types import GenRequest, Message, Role, Sampling

RERANK_BODY = '{"choice": 1, "reason": "candidate 1 matches the repo\'s error style"}'


def _endpoint(handler):
    """An OptiqTier1Backend whose socket is `handler`.

    The transport is swapped after construction rather than injected, because the
    constructor building its own client is part of what is under test: a backend
    that needed a transport handed to it would not be the object `build_tier1`
    actually creates.
    """

    backend = OptiqTier1Backend("http://127.0.0.1:9999/v1", model="glm-5.2-2bit")
    original = backend._client
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return backend, original


def _ok(
    body: str = RERANK_BODY, *, usage: dict | None = None, extra: dict | None = None
):
    """A handler answering every POST with `body`, recording what it was sent."""

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        seen.append(json.loads(request.content))
        payload = {
            "choices": [{"message": {"content": body}}],
            "usage": usage
            if usage is not None
            else {"prompt_tokens": 8000, "completion_tokens": 24},
        }
        payload.update(extra or {})
        return httpx.Response(200, json=payload)

    return handler, seen


def _rerank_req(n: int = 3) -> GenRequest:
    return GenRequest(
        messages=[Message(role=Role.USER, content="pick one")],
        system="You are a verifier.",
        json_schema=rerank_schema(n),
        sampling=Sampling(temperature=0.9, top_p=0.4, seed=7, max_tokens=4096),
    )


# --- the call ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_verdict_comes_back_with_the_endpoint_s_own_token_counts():
    handler, seen = _ok()
    backend, original = _endpoint(handler)
    try:
        result = await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()

    assert result.text == RERANK_BODY
    assert result.usage.input_tokens == 8000
    assert result.usage.output_tokens == 24
    assert result.total_s > 0
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_the_output_clamp_survives_the_transport():
    """Sec 5.1. The clamp is tested in `tier1_call`; this is the wire proving it.

    A clamp enforced in a helper nobody calls is not enforced. `max_tokens=4096`
    goes in and the rerank budget comes out on the actual request body.
    """
    handler, seen = _ok()
    backend, original = _endpoint(handler)
    try:
        await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()

    assert seen[0]["max_tokens"] == 128


@pytest.mark.asyncio
async def test_a_judgement_is_greedy_whatever_the_caller_asked_for():
    """Two runs of the same rerank have to agree or the receipt's determinism
    claim is decoration. The caller's temperature is not consulted."""
    handler, seen = _ok()
    backend, original = _endpoint(handler)
    try:
        await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()

    assert seen[0]["temperature"] == 0.0
    assert seen[0]["top_p"] == 1.0
    assert seen[0]["seed"] == 7
    assert seen[0]["stream"] is False


@pytest.mark.asyncio
async def test_the_schema_rides_along_as_a_strict_response_format():
    handler, seen = _ok()
    backend, original = _endpoint(handler)
    try:
        await backend.generate(_rerank_req(n=3))
    finally:
        await backend.close()
        await original.aclose()

    fmt = seen[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "rerank"
    assert fmt["json_schema"]["strict"] is True
    # rerank_schema(n) bounds the choice by construction (sec 5.2). If the bound
    # did not reach the engine, a constrained decode could still name candidate 9.
    assert fmt["json_schema"]["schema"]["properties"]["choice"]["maximum"] == 2


# --- refusing what cannot be parsed -----------------------------------------


@pytest.mark.asyncio
async def test_broken_json_under_a_schema_is_refused_not_coerced():
    """2-bit's documented failure mode (sec 5.2). Coercing it is how that failure
    reaches a merge decision wearing the shape of an answer."""
    handler, _ = _ok('{"choice": 1, "reason": "unterminated')
    backend, original = _endpoint(handler)
    try:
        with pytest.raises(Tier1Unavailable, match="non-JSON"):
            await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()


@pytest.mark.asyncio
async def test_a_verdict_missing_a_required_field_is_refused():
    handler, _ = _ok('{"reason": "I liked it"}')
    backend, original = _endpoint(handler)
    try:
        with pytest.raises(Tier1Unavailable, match="missing required fields"):
            await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()


@pytest.mark.asyncio
async def test_a_non_200_names_the_status_and_does_not_retry():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="expert cache thrashing")

    backend, original = _endpoint(handler)
    try:
        with pytest.raises(Tier1Unavailable, match="503"):
            await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()


@pytest.mark.asyncio
async def test_a_dead_socket_is_tier1_unavailable_not_a_raw_httpx_error():
    """Every tier-1 failure has to arrive as `Tier1Unavailable`, because that is
    what degrades to a failed `Verdict` instead of failing the turn."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend, original = _endpoint(handler)
    try:
        with pytest.raises(Tier1Unavailable, match="tier 1 call failed"):
            await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()


@pytest.mark.asyncio
async def test_an_empty_choices_list_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    backend, original = _endpoint(handler)
    try:
        with pytest.raises(Tier1Unavailable, match="no choices"):
            await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()


# --- health -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_reports_the_endpoint_in_its_failure_string():
    """`tandem doctor` prints this. "unreachable at <endpoint>" is actionable;
    "False" is not."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend, original = _endpoint(handler)
    try:
        ok, reason = await backend.health()
    finally:
        await backend.close()
        await original.aclose()

    assert ok is False
    assert "127.0.0.1:9999" in reason


@pytest.mark.asyncio
async def test_health_is_ok_when_the_models_endpoint_answers():
    handler, _ = _ok()
    backend, original = _endpoint(handler)
    try:
        ok, reason = await backend.health()
    finally:
        await backend.close()
        await original.aclose()

    assert (ok, reason) == (True, "ok")


# --- Gate B (sec 11, 14.3) --------------------------------------------------


@pytest.mark.asyncio
async def test_measure_prefill_actually_reaches_the_size_it_was_asked_for():
    """Gate B is quoted at 4k/8k/16k. A sample labelled 16k that carried 15.3k
    tokens is a measurement of something else, and Gate B decides a three-week
    schedule fact."""
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        sizes.append(sum(len(m["content"]) for m in body["messages"]))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 16_000, "completion_tokens": 1},
            },
        )

    backend, original = _endpoint(handler)
    try:
        await backend.measure_prefill(16_000)
    finally:
        await backend.close()
        await original.aclose()

    # ~4 chars/token is the filler's own stated ratio; it must not undershoot.
    assert sizes[0] >= 16_000 * 4


def test_gate_b_reads_the_worst_sample_not_the_average():
    """A gate that averaged would pass on one good size and one catastrophic one.

    The realistic shape of a Gate B failure is exactly that: fast at 4k where the
    expert working set fits the cache, collapsing at 16k where it does not.
    """
    backend = OptiqTier1Backend("http://127.0.0.1:9999/v1", model="m")
    backend.prefill_samples = [
        PrefillSample(16_000, 1, prefill_s=1.0, total_s=1.0),  # 16,000 tok/s
        PrefillSample(16_000, 1, prefill_s=200.0, total_s=200.0),  # 80 tok/s
    ]
    report = backend.gate_b_report()
    assert report["pass"] is False
    assert report["worst_tok_per_s"] == 80.0
    assert len(report["samples"]) == 2


def test_gate_b_relaxed_threshold_passes_but_does_not_claim_the_spec():
    """A host judged against its own floor must not report a met sec-11 Gate B.

    This host measures 164.7 tok/s of streamed prefill against sec 11's 200, so the
    relaxed floor is what lets the rung run at all and `meets_spec` is what stops the
    result being quoted as a pass. The sample below is deliberately far worse than the
    host's real rate: what is under test is the two-verdict shape, not the hardware.
    """
    backend = OptiqTier1Backend("http://127.0.0.1:9999/v1", model="m")
    backend.prefill_samples = [PrefillSample(16_000, 1, prefill_s=615.4, total_s=615.4)]

    spec = backend.gate_b_report()
    assert spec["pass"] is False
    assert spec["threshold_tok_per_s"] == 200.0
    assert spec["relaxed"] is False

    host = backend.gate_b_report(threshold_tok_per_s=20.0)
    assert host["pass"] is True
    assert host["meets_spec"] is False
    assert host["relaxed"] is True
    assert host["threshold_tok_per_s"] == 20.0
    assert host["spec_threshold_tok_per_s"] == 200.0
    # The worst rate is the same measurement in both reports; only the verdict moved.
    assert spec["worst_tok_per_s"] == host["worst_tok_per_s"] == 26.0


def test_gate_b_with_no_samples_is_a_failure_not_a_pass():
    """The "we never measured" case must not read as green."""
    backend = OptiqTier1Backend("http://127.0.0.1:9999/v1", model="m")
    report = backend.gate_b_report()
    assert report["pass"] is False
    assert "no samples" in report["reason"]


def test_prefill_rate_of_a_zero_duration_sample_is_zero_not_infinite():
    """A clock too coarse to see the call must not report an infinite rate
    straight into the Gate B number."""
    assert PrefillSample(4096, 4, prefill_s=0.0, total_s=0.0).prefill_tok_per_s == 0.0


@pytest.mark.asyncio
async def test_decode_time_is_subtracted_from_prefill_when_the_engine_reports_it():
    import time

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.05)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": RERANK_BODY}}],
                "usage": {"prompt_tokens": 8000, "completion_tokens": 24},
                "timings": {"predicted_ms": 20.0},
            },
        )

    backend, original = _endpoint(handler)
    try:
        await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()

    sample = backend.prefill_samples[-1]
    assert sample.prefill_s == pytest.approx(sample.total_s - 0.02, abs=1e-6)
    assert sample.prefill_s < sample.total_s


@pytest.mark.asyncio
async def test_a_decode_longer_than_the_whole_call_cannot_flatter_gate_b():
    """The engine's clock and ours disagreeing must not manufacture throughput.

    Subtracting a decode time that exceeds the measured total floors `prefill_s`
    at 1e-6, which turns 8,000 input tokens into eight billion tok/s and walks an
    incoherent sample straight through a floor test. Gate B is allowed to be
    pessimistic and is never allowed to be optimistic.
    """
    handler, _ = _ok(extra={"timings": {"predicted_ms": 900_000.0}})
    backend, original = _endpoint(handler)
    try:
        await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()

    sample = backend.prefill_samples[-1]
    assert sample.prefill_s == pytest.approx(sample.total_s, abs=1e-6)
    # The 1e-6 floor is what the throughput blows up through. Not reaching it is
    # the property; the absolute rate over a mock socket means nothing either way.
    assert sample.prefill_s > 1e-6


@pytest.mark.asyncio
async def test_an_engine_that_reports_no_timings_understates_prefill_throughput():
    """The whole call is attributed to prefill. That flatters nothing: it can only
    make the measured rate lower, and Gate B is a floor."""
    handler, _ = _ok()
    backend, original = _endpoint(handler)
    try:
        await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()

    sample = backend.prefill_samples[-1]
    assert sample.prefill_s == pytest.approx(sample.total_s, abs=1e-6)


def test_decode_seconds_are_read_from_whichever_field_the_engine_uses():
    assert _reported_decode_seconds({"timings": {"predicted_ms": 1500}}) == 1.5
    assert _reported_decode_seconds({"metrics": {"decode_s": 2.0}}) == 2.0
    assert _reported_decode_seconds({"timing": {"generation_ms": 250}}) == 0.25
    assert _reported_decode_seconds({}) == 0.0
    assert _reported_decode_seconds({"timings": "not a dict"}) == 0.0


# --- usage accounting -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_missing_usage_block_is_estimated_never_zero():
    """Zero input tokens would report an infinite prefill rate."""
    handler, _ = _ok(usage={})
    backend, original = _endpoint(handler)
    try:
        result = await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()

    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


@pytest.mark.asyncio
async def test_a_remote_container_is_not_attested_to():
    """`container_hash` is None with no local container path — you cannot attest
    to a model you do not hold."""
    backend = OptiqTier1Backend("http://127.0.0.1:9999/v1", model="m")
    try:
        assert backend.container_hash() is None
    finally:
        await backend.close()


def test_gate_b_measures_rung_1_without_building_tier_0(tmp_path, monkeypatch, capsys):
    """`tandem bench tier1` must not load the resident model.

    Rung 1 reaches the engine over a socket, so tier 0 is not a dependency of the
    measurement — and on the 36 GB baseline host, building one is 23.0 GiB against
    25.9 GB of measured headroom, which is Gate B failing to run rather than Gate B
    reporting red. The rungs that do serve from tier 0 carry no prefill instrument,
    so this command can only decline for them; it declines without loading.
    """
    import argparse
    import json

    import tandem.backends
    from tandem.cli import cmd_bench

    config = tmp_path / "rung3.toml"
    config.write_text(
        'backend = "mlx"\n\n[tier1]\nenabled = true\nrung = "second_opinion"\n',
        encoding="utf-8",
    )

    def refuse(_cfg):
        raise AssertionError("Gate B built tier 0")

    monkeypatch.setattr(tandem.backends, "build_tier0", refuse)

    code = cmd_bench(argparse.Namespace(config=str(config), which="tier1"))

    assert code == 1
    assert json.loads(capsys.readouterr().out)["rung"] == "second_opinion"


def test_the_reasoning_refusal_reads_the_spelling_mlx_optiq_emits():
    """The guard is only as unconditional as the narrowest key it reads.

    mlx-optiq 0.4.18 spells this `reasoning` and sends no
    `completion_tokens_details`. Reading `reasoning_content` alone passed a
    thinking model's empty `content` through as a verdict — measured against the
    live 122B on 2026-08-10, and the reason `resolve_reasoning_control`'s "fails
    loudly on the first call" claim was false for the engine this repo runs.
    """
    from tandem.backends.tier1_call import refuse_reasoned_answer

    body = {
        "choices": [
            {"message": {"role": "assistant", "reasoning": "Thinking Process:"}}
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 32},
    }
    with pytest.raises(Tier1Unavailable, match="reasoning trace"):
        refuse_reasoned_answer(body["choices"][0], body)


@pytest.mark.asyncio
async def test_a_model_auto_does_not_recognise_fails_loudly_not_emptily():
    """The whole safety argument for guessing from the model name.

    `auto` does not claim Qwen3, so the 122B goes out with thinking on and comes
    back reasoning — the live engine's actual behaviour on 2026-08-10. That must
    reach the caller as a failure, not as `text=""` that then fails schema
    validation somewhere with no cause attached.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "reasoning": "Thinking Process:"}}
                ],
                "usage": {"prompt_tokens": 17, "completion_tokens": 32},
            },
        )

    backend, original = _endpoint(handler)
    try:
        with pytest.raises(Tier1Unavailable, match="reasoning trace"):
            await backend.generate(_rerank_req())
    finally:
        await backend.close()
        await original.aclose()
