"""Wire-layer regressions: the streaming contracts and the edge bounds (spec sec 8.1).

`test_gateway.py` covers the happy path through each protocol. This file covers the
edges that were found broken: usage on an OpenAI Chat stream, a stream that ends in
an error mid-block, the Responses content-part events, the bounds a request has to
satisfy before it reaches the model, and the retention rules on the compactor's
history.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tandem.backends.base import Delta
from tandem.backends.mock import MockBackend
from tandem.config import Config
from tandem.gateway import wire
from tandem.gateway.app import create_app
from tandem.gateway.compaction import Compactor
from tandem.gateway.pipeline import Pipeline
from tandem.gateway.wire import anthropic, openai_chat, openai_responses
from tandem.types import GenRequest, GenResult, Message, Role, StopReason, Usage

CC_SYSTEM = (
    "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
    "# Harness\nReference code as file_path:line_number.\n"
    "IMPORTANT: Assist with authorized security testing.\n"
    + ("Boilerplate the model does not need on every turn. " * 300)
)


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.attest.audit_log = str(tmp_path / "audit.jsonl")
    c.cache.disk_kv_dir = str(tmp_path / "kv")
    return c


@pytest.fixture
def client(cfg):
    return _client(create_app(cfg))


def _client(app) -> TestClient:
    """A client whose Host header passes the gateway's allow-list.

    `TestClient` defaults to `http://testserver`, which the allow-list correctly
    rejects — rejecting unfamiliar names is the whole point, since that is what
    stops a DNS-rebinding page from driving a loopback-bound gateway from a
    browser. Tests address the app the way a real local client does.
    """
    return TestClient(app, base_url="http://127.0.0.1")


def _chunks(events: list[str]) -> list[dict]:
    """Parse `data:` payloads out of an OpenAI-style SSE stream, skipping [DONE]."""
    out = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: ") and line[len("data: ") :].strip() != "[DONE]":
                out.append(json.loads(line[len("data: ") :]))
    return out


def _named(events: list[str]) -> list[str]:
    return [
        line[len("event: ") :]
        for event in events
        for line in event.splitlines()
        if line.startswith("event: ")
    ]


# --- the module contract ----------------------------------------------------


def test_every_protocol_exposes_the_documented_contract():
    """`wire/__init__.py` names these; app.py drives all three through them alone."""
    for module in (anthropic, openai_chat, openai_responses):
        for name in ("to_canonical", "from_canonical", "StreamEncoder", "error", "stream_options"):
            assert hasattr(module, name), f"{module.__name__} is missing {name}"


def test_the_dead_whole_sequence_helper_is_gone():
    """`sse_events` was documented as the contract and called by nothing.

    A second encoding path next to `StreamEncoder` is a second place for the two to
    drift; the docstring now names the encoder.
    """
    for module in (anthropic, openai_chat, openai_responses):
        assert not hasattr(module, "sse_events")
    assert "StreamEncoder" in wire.__doc__
    assert "sse_events" not in wire.__doc__


# --- M6: OpenAI Chat streaming usage (sec 8.3) ------------------------------


def test_chat_stream_options_are_read_off_the_body():
    assert openai_chat.stream_options({"stream_options": {"include_usage": True}}) == {
        "include_usage": True
    }
    assert openai_chat.stream_options({"stream_options": {"include_usage": False}}) == {
        "include_usage": False
    }
    assert openai_chat.stream_options({}) == {"include_usage": False}
    # A client that sends a non-object is a client bug, not a 500.
    assert openai_chat.stream_options({"stream_options": "yes"}) == {"include_usage": False}


def test_chat_stream_emits_a_usage_chunk_when_the_client_asks():
    """Without this, sec 8.3 context scaling never reaches an OpenAI streaming client.

    The scaled count is the entire mechanism by which the harness decides to
    compact; a stream that reports nothing leaves its context meter frozen.
    """
    body = {"stream": True, "stream_options": {"include_usage": True}, "messages": []}
    enc = openai_chat.StreamEncoder(model="t", request_id="r1", **openai_chat.stream_options(body))
    events = [*enc.open(1200), *enc.delta("hi"), *enc.close(
        GenResult(text="hi", usage=Usage(input_tokens=1200, output_tokens=7, cached_input_tokens=64))
    )]
    chunks = _chunks(events)

    final = chunks[-1]
    assert final["choices"] == []
    assert final["usage"]["prompt_tokens"] == 1200
    assert final["usage"]["completion_tokens"] == 7
    assert final["usage"]["total_tokens"] == 1207
    assert final["usage"]["prompt_tokens_details"]["cached_tokens"] == 64
    # Same completion throughout, and the terminator still comes last.
    assert final["id"] == "r1"
    assert events[-1] == "data: [DONE]\n\n"
    # Per the contract, every other chunk carries the key with a null value.
    assert all(c["usage"] is None for c in chunks[:-1])


def test_chat_stream_falls_back_to_the_prologue_prompt_count():
    """A backend that reports no prompt tokens must not report zero to the harness."""
    enc = openai_chat.StreamEncoder(model="t", include_usage=True)
    events = [*enc.open(830), *enc.delta("x"), *enc.close(GenResult(text="x", usage=Usage(output_tokens=3)))]
    final = _chunks(events)[-1]
    assert final["usage"]["prompt_tokens"] == 830
    assert final["usage"]["total_tokens"] == 833


def test_chat_stream_stays_silent_about_usage_when_not_asked():
    """An empty `choices` array is a shape clients only tolerate having asked for it."""
    enc = openai_chat.StreamEncoder(model="t")
    events = [*enc.open(1200), *enc.delta("hi"), *enc.close(
        GenResult(text="hi", usage=Usage(input_tokens=1200, output_tokens=7))
    )]
    chunks = _chunks(events)
    assert all("usage" not in c for c in chunks)
    assert all(c["choices"] for c in chunks)


def test_chat_usage_chunk_survives_a_tool_call_turn():
    enc = openai_chat.StreamEncoder(model="t", include_usage=True)
    from tandem.types import ToolCall

    result = GenResult(
        tool_calls=(ToolCall(id="c1", name="read_file", arguments={"path": "a"}),),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(input_tokens=10, output_tokens=4),
    )
    chunks = _chunks([*enc.open(10), *enc.close(result)])
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["prompt_tokens"] == 10
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"


# --- Anthropic: a stream that fails mid-block -------------------------------


def test_anthropic_fail_closes_an_open_content_block():
    """An `error` on top of an unterminated block corrupts the SDK's accumulator."""
    enc = anthropic.StreamEncoder(model="t")
    events = [*enc.open(10), *enc.delta("half a sen")]
    events += enc.fail("boom")
    names = _named(events)

    assert names == ["message_start", "content_block_start", "content_block_delta",
                     "content_block_stop", "error"]
    # `message_stop` means the turn completed. This one did not.
    assert "message_stop" not in names


def test_anthropic_fail_before_any_text_emits_only_the_error():
    enc = anthropic.StreamEncoder(model="t")
    names = _named([*enc.open(10), *enc.fail("boom")])
    assert names == ["message_start", "error"]


def test_anthropic_fail_does_not_reuse_the_closed_block_index():
    enc = anthropic.StreamEncoder(model="t")
    events = [*enc.open(0), *enc.delta("text"), *enc.fail("boom")]
    stop = next(json.loads(e.split("data: ", 1)[1]) for e in events if "content_block_stop" in e)
    assert stop["index"] == 0
    assert enc._index == 1


@pytest.mark.asyncio
async def test_a_stream_that_dies_mid_turn_still_closes_its_block(cfg):
    """End to end through `app._sse`, which is where the C3 reproduction saw it."""

    class Exploding(MockBackend):
        async def stream(self, req):
            yield Delta(text="partial answer")
            raise RuntimeError("backend died")

    pipeline = Pipeline(cfg, Exploding(use_tools=False))
    app = create_app(cfg, pipeline)
    with _client(app) as client:
        r = client.post(
            "/v1/messages",
            json={"model": "t", "max_tokens": 32, "stream": True,
                  "messages": [{"role": "user", "content": "hello there"}]},
        )
    names = [line[len("event: ") :] for line in r.text.splitlines() if line.startswith("event: ")]
    assert "error" in names
    assert names.index("content_block_stop") < names.index("error")


# --- Responses: content parts -----------------------------------------------


def test_responses_text_stream_announces_and_closes_its_content_part():
    """Deltas are addressed to a content part by index; announcing the item is not enough."""
    enc = openai_responses.StreamEncoder(model="t")
    events = [*enc.open(5), *enc.delta("hel"), *enc.delta("lo"), *enc.close(
        GenResult(text="hello", usage=Usage(input_tokens=5, output_tokens=2))
    )]
    names = _named(events)

    assert names.index("response.content_part.added") < names.index("response.output_text.delta")
    assert (
        names.index("response.output_text.done")
        < names.index("response.content_part.done")
        < names.index("response.output_item.done")
    )
    # Announced once however many deltas arrive.
    assert names.count("response.content_part.added") == 1
    assert names.count("response.content_part.done") == 1

    payloads = {json.loads(e.split("data: ", 1)[1])["type"]: json.loads(e.split("data: ", 1)[1])
                for e in events}
    assert payloads["response.content_part.added"]["part"] == {
        "type": "output_text", "text": "", "annotations": []
    }
    assert payloads["response.content_part.done"]["part"]["text"] == "hello"
    assert payloads["response.content_part.done"]["content_index"] == 0


def test_responses_tool_only_stream_announces_no_content_part():
    """A turn that produces only tool calls must not claim a text part it never filled."""
    from tandem.types import ToolCall

    enc = openai_responses.StreamEncoder(model="t")
    result = GenResult(
        tool_calls=(ToolCall(id="c1", name="read_file", arguments={"path": "a"}),),
        usage=Usage(input_tokens=5),
    )
    names = _named([*enc.open(5), *enc.close(result)])
    assert "response.content_part.added" not in names
    assert "response.content_part.done" not in names


def test_responses_stream_still_terminates(client):
    r = client.post("/v1/responses", json={"model": "t", "stream": True, "input": "hi"})
    assert r.status_code == 200
    assert "response.completed" in r.text


# --- edge bounds ------------------------------------------------------------


def test_message_count_is_bounded_on_every_protocol():
    many = [{"role": "user", "content": "x"} for _ in range(wire.MAX_MESSAGES + 1)]
    with pytest.raises(ValueError, match="too many messages"):
        anthropic.to_canonical({"messages": many})
    with pytest.raises(ValueError, match="too many messages"):
        openai_chat.to_canonical({"messages": many})
    with pytest.raises(ValueError, match="too many messages"):
        openai_responses.to_canonical({"input": many})


def test_max_tokens_is_bounded_on_every_protocol():
    huge = wire.MAX_OUTPUT_TOKENS + 1
    one = [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError, match="max_tokens too large"):
        anthropic.to_canonical({"messages": one, "max_tokens": huge})
    with pytest.raises(ValueError, match="max_tokens too large"):
        openai_chat.to_canonical({"messages": one, "max_completion_tokens": huge})
    with pytest.raises(ValueError, match="max_tokens too large"):
        openai_responses.to_canonical({"input": "hi", "max_output_tokens": huge})


def test_a_zero_max_tokens_budget_is_rejected():
    """A decode budget of nothing is a client bug, not a turn to run."""
    with pytest.raises(ValueError, match="at least 1"):
        anthropic.to_canonical({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0})


def test_input_size_is_bounded(monkeypatch):
    monkeypatch.setattr(wire, "MAX_INPUT_CHARS", 64)
    body = {"messages": [{"role": "user", "content": "x" * 100}], "max_tokens": 16}
    with pytest.raises(ValueError, match="request too large"):
        anthropic.to_canonical(body)
    with pytest.raises(ValueError, match="request too large"):
        openai_chat.to_canonical(body)
    with pytest.raises(ValueError, match="request too large"):
        openai_responses.to_canonical({"input": "x" * 100})


def test_size_of_counts_tool_results_and_the_system_prompt():
    from tandem.types import ToolResult

    msgs = [
        Message(role=Role.USER, content="abcd"),
        Message(role=Role.TOOL, tool_results=(ToolResult(tool_call_id="t", content="efg"),)),
    ]
    assert wire.size_of("xy", msgs) == 2 + 4 + 3
    assert wire.size_of(None, msgs) == 7


def test_a_request_over_the_bounds_is_a_400_not_a_500(client):
    """`app.py` already turns a `ValueError` from `to_canonical` into the protocol's 400."""
    r = client.post(
        "/v1/messages",
        json={"model": "t", "max_tokens": wire.MAX_OUTPUT_TOKENS + 1,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400
    assert r.json()["type"] == "error"
    assert "max_tokens" in r.json()["error"]["message"]


def test_an_ordinary_request_is_nowhere_near_the_bounds(client):
    """The bounds are generous on purpose; a real turn must never trip them."""
    r = client.post(
        "/v1/messages",
        json={"model": "t", "max_tokens": 4096, "system": CC_SYSTEM,
              "messages": [{"role": "user", "content": "Fix the retry loop"} for _ in range(50)]},
    )
    assert r.status_code == 200


# --- M7: compactor retention (sec 8.2) --------------------------------------


def test_compactor_history_is_bounded():
    """A served gateway runs for days; 10k retained Claude Code prompts is ~1 GB."""
    compactor = Compactor(history_limit=8)
    for _ in range(300):
        compactor.apply(GenRequest(system=CC_SYSTEM))
    assert len(compactor.history) == 8
    # The window is a sample, not the population: the gate must not claim n=8.
    assert compactor.measure()["n"] == 300
    assert compactor.measure()["sampled"] == 8
    assert compactor.measure()["window"] == 8
    assert compactor.measure()["pass"] is True


def test_keep_original_false_retains_no_prompt_text_anywhere():
    """The flag gated only the outgoing request, so the privacy option did nothing."""
    compactor = Compactor(keep_original=False)
    out, result = compactor.apply(GenRequest(system=CC_SYSTEM))

    assert out.original_system is None
    assert result.original_system is None
    # `compacted_system` held the raw prompt too on every non-applied branch.
    assert result.compacted_system is None
    assert compactor.history[-1].original_system is None
    assert compactor.history[-1].compacted_system is None
    # The measurement survives; only the text is dropped.
    assert result.applied and result.original_tokens > result.compacted_tokens


def test_keep_original_false_retains_nothing_on_the_uncompacted_branches():
    compactor = Compactor(keep_original=False)
    for req, kwargs in (
        (GenRequest(system=CC_SYSTEM), {"force_off": True}),
        (GenRequest(system="You are a helpful assistant."), {}),
    ):
        _out, result = compactor.apply(req, **kwargs)
        assert not result.applied
        assert result.original_system is None and result.compacted_system is None


def test_a_withheld_diff_says_so_rather_than_looking_empty():
    """An empty diff reads as "compaction changed nothing", the opposite of the truth."""
    _out, result = Compactor(keep_original=False).apply(GenRequest(system=CC_SYSTEM))
    assert "keep_original" in result.diff()


def test_keep_original_true_still_reaches_the_diff_view():
    out, result = Compactor().apply(GenRequest(system=CC_SYSTEM))
    assert out.original_system == CC_SYSTEM
    assert result.original_system == CC_SYSTEM
    assert "--- harness-system" in result.diff()


def test_an_unrecorded_apply_leaves_the_history_alone():
    """A token-count probe is not a served turn: it must not skew the M1 gate or
    hide the last real request behind it in the sec 8.2 diff view."""
    compactor = Compactor()
    compactor.apply(GenRequest(system=CC_SYSTEM))
    before = list(compactor.history)

    _out, result = compactor.apply(GenRequest(system=CC_SYSTEM), record=False)
    assert result.applied  # the probe is still compacted, and still measured
    assert list(compactor.history) == before
    assert compactor.measure()["n"] == 1


def test_the_gate_reports_honestly_when_the_window_holds_no_compacted_turn():
    compactor = Compactor(history_limit=2)
    compactor.apply(GenRequest(system=CC_SYSTEM))
    for _ in range(2):
        compactor.apply(GenRequest(system="You are a helpful assistant."))
    m = compactor.measure()
    assert m["pass"] is False
    assert m["n"] == 1
    assert "window" in m["reason"]


@pytest.mark.asyncio
async def test_a_served_turn_still_lands_in_the_history(cfg):
    """The bound must not turn into "nothing is recorded"."""
    pipeline = Pipeline(cfg, MockBackend(use_tools=False))
    await pipeline.run(GenRequest(system=CC_SYSTEM,
                                  messages=[Message(role=Role.USER, content="hi")]))
    assert pipeline.compactor.history
    assert pipeline.compactor.history[-1].applied
