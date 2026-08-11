"""Gateway: three wire protocols, compaction, caching, context scaling (spec sec 8)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from orbit.attest.audit import verify_chain
from orbit.backends.base import Delta
from orbit.backends.mock import Fault, MockBackend
from orbit.config import Config
from orbit.gateway.app import create_app
from orbit.gateway.cache.kv_disk import DiskKVCache, KVSnapshot, align_down
from orbit.gateway.cache.prompt_cache import CacheEntry, PromptCache, chunk_digests
from orbit.gateway.compaction import Compactor, detect, one_line, strip_tool
from orbit.gateway.context_scale import ContextScaler
from orbit.gateway.pipeline import Pipeline
from orbit.types import GenRequest, Message, Role, ToolDef

CC_SYSTEM = (
    "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
    "# Harness\nReference code as file_path:line_number.\n"
    "IMPORTANT: Assist with authorized security testing.\n"
    + ("Boilerplate the model does not need on every turn. " * 300)
)

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "A very long description. " * 30},
        "mode": {"type": "string", "enum": ["read", "write"]},
        "lines": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["path"],
}


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.attest.audit_log = str(tmp_path / "audit.jsonl")
    c.cache.disk_kv_dir = str(tmp_path / "kv")
    return c


def build_client(cfg) -> TestClient:
    """A client whose Host header passes the gateway's allow-list.

    `TestClient` defaults to `http://testserver`, which the Host allow-list
    correctly rejects — the allow-list is what stops a DNS-rebinding page from
    driving the gateway from a browser, so it has to reject unfamiliar names. Tests
    address the app the way a real local client does.
    """
    return TestClient(create_app(cfg), base_url="http://127.0.0.1")


@pytest.fixture
def client(cfg):
    return build_client(cfg)


# --- wire protocols (sec 8.1) -----------------------------------------------


def test_messages_endpoint_round_trip(client):
    r = client.post(
        "/v1/messages",
        json={
            "model": "orbit",
            "max_tokens": 128,
            "system": CC_SYSTEM,
            "messages": [{"role": "user", "content": "Fix the retry loop"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["type"] == "text"
    assert body["usage"]["input_tokens"] > 0


def test_chat_completions_round_trip(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "orbit", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_responses_round_trip(client):
    r = client.post("/v1/responses", json={"model": "orbit", "input": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response"
    assert body["output"][0]["type"] == "message"
    assert body["output_text"]


def test_tool_calls_survive_each_protocol(client):
    tool_a = {"name": "read_file", "description": "Read", "input_schema": SCHEMA}

    r = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 64,
            "tools": [tool_a],
            "messages": [{"role": "user", "content": "read a file"}],
        },
    )
    blocks = r.json()["content"]
    assert any(b["type"] == "tool_use" for b in blocks)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "t",
            "messages": [{"role": "user", "content": "read a file"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": SCHEMA},
                }
            ],
        },
    )
    calls = r.json()["choices"][0]["message"]["tool_calls"]
    # OpenAI carries arguments as a JSON *string*; clients round-trip it verbatim.
    assert isinstance(calls[0]["function"]["arguments"], str)
    assert json.loads(calls[0]["function"]["arguments"])

    r = client.post(
        "/v1/responses",
        json={
            "model": "t",
            "input": "read a file",
            "tools": [{"type": "function", "name": "read_file", "parameters": SCHEMA}],
        },
    )
    assert any(item["type"] == "function_call" for item in r.json()["output"])


def test_tool_results_round_trip_from_each_protocol():
    from orbit.gateway.wire import anthropic, openai_chat, openai_responses

    a = anthropic.to_canonical(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "read_file",
                            "input": {"path": "a"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "file body",
                        }
                    ],
                },
            ]
        }
    )
    assert a.messages[0].tool_calls[0].id == "t1"
    assert a.messages[1].role is Role.TOOL
    assert a.messages[1].tool_results[0].content == "file body"

    c = openai_chat.to_canonical(
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "t1", "content": "file body"},
            ]
        }
    )
    assert c.messages[0].tool_calls[0].arguments == {"path": "a"}
    assert c.messages[1].tool_results[0].content == "file body"

    o = openai_responses.to_canonical(
        {
            "input": [
                {
                    "type": "function_call",
                    "call_id": "t1",
                    "name": "read_file",
                    "arguments": '{"path": "a"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "t1",
                    "output": "file body",
                },
            ]
        }
    )
    assert o.messages[0].tool_calls[0].name == "read_file"
    assert o.messages[1].tool_results[0].content == "file body"


def test_responses_folds_instructions_and_system_into_one_prompt():
    from orbit.gateway.wire import openai_responses

    req = openai_responses.to_canonical(
        {
            "instructions": "top-level instructions",
            "input": [
                {"role": "system", "content": "a system item"},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    assert req.system == "top-level instructions\n\na system item"
    assert len(req.messages) == 1


def test_responses_ignores_hosted_tools():
    """Hosted tools are not ours to serve; a local model cannot execute them (sec 12)."""
    from orbit.gateway.wire import openai_responses

    req = openai_responses.to_canonical(
        {
            "input": "hi",
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "ok", "parameters": {}},
            ],
        }
    )
    assert [t.name for t in req.tools] == ["ok"]


@pytest.mark.parametrize(
    "path,payload,marker",
    [
        (
            "/v1/messages",
            {
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "message_stop",
        ),
        (
            "/v1/chat/completions",
            {"stream": True, "messages": [{"role": "user", "content": "hi"}]},
            "[DONE]",
        ),
        ("/v1/responses", {"stream": True, "input": "hi"}, "response.completed"),
    ],
)
def test_streaming_emits_a_terminal_event(client, path, payload, marker):
    r = client.post(path, json={"model": "t", **payload})
    assert r.status_code == 200
    assert marker in r.text


# --- incremental streaming (sec 7.3) ----------------------------------------


def _events(text: str) -> list[str]:
    return [
        line[len("event: ") :]
        for line in text.splitlines()
        if line.startswith("event: ")
    ]


def _data(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ") and line[len("data: ") :].strip() != "[DONE]"
    ]


def test_a_chat_turn_streams_token_by_token(client):
    """The sec 7.3 TTFT budget is only served if deltas arrive before the turn ends."""
    r = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 128,
            "stream": True,
            "messages": [{"role": "user", "content": "hello there"}],
        },
    )
    events = _events(r.text)
    assert events.count("content_block_delta") > 1
    # One text block, opened once and closed once, whatever the delta count.
    assert events.count("content_block_start") == 1
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"


def test_the_streamed_text_reassembles_into_the_completion(client):
    r = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 128,
            "stream": True,
            "messages": [{"role": "user", "content": "hello there"}],
        },
    )
    streamed = "".join(
        d["delta"]["text"]
        for d in _data(r.text)
        if d.get("type") == "content_block_delta"
    )
    same = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hello there"}],
        },
    ).json()
    assert streamed == same["content"][0]["text"]


def test_message_start_carries_the_scaled_input_tokens(client):
    """A harness reads its context meter off message_start, so 0 there is a bug."""
    r = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 64,
            "stream": True,
            "system": CC_SYSTEM,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    start = next(d for d in _data(r.text) if d.get("type") == "message_start")
    assert start["message"]["usage"]["input_tokens"] > 0


def test_a_tool_bearing_turn_does_not_stream_incrementally(client):
    """The tool-call layer (sec 8.5) needs the whole reply before it can repair it."""
    r = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 128,
            "stream": True,
            "messages": [{"role": "user", "content": "fix the retry count"}],
            "tools": [{"name": "edit_file", "input_schema": SCHEMA}],
        },
    )
    assert "tool_use" in r.text
    trace = client.get("/orbit/trace/last").json()
    assert trace["stream"]["incremental"] is False
    assert "tool" in trace["stream"]["reason"]


@pytest.mark.asyncio
async def test_a_code_change_turn_runs_to_completion_before_emitting(cfg):
    """Best-of-N has nothing to emit until the verifier has chosen (sec 7.2)."""
    cfg.tier1.enabled = True
    pipeline = Pipeline(cfg, MockBackend(use_tools=False), MockBackend(tier=1))
    # No tools, so the turn is held back for the best-of-N reason and not the
    # tool-call one.
    req = GenRequest(
        messages=[Message(role=Role.USER, content="refactor the retry helper")]
    )
    deltas = [d async for d in pipeline.stream(req)]
    # Prologue, one whole-result delta, terminator.
    assert deltas[0].result is not None and not deltas[0].done
    assert sum(1 for d in deltas if d.text) == 1
    assert deltas[-1].done
    assert pipeline.traces[-1].stream["incremental"] is False
    assert pipeline.traces[-1].cascade["candidates_generated"] > 1


@pytest.mark.asyncio
async def test_a_streamed_turn_still_writes_its_receipt_and_audit_record(cfg):
    """Attestation is not a property of the batch path (sec 9.1, 9.2)."""
    pipeline = Pipeline(cfg, MockBackend(use_tools=False))
    req = GenRequest(messages=[Message(role=Role.USER, content="hello there")])
    deltas = [d async for d in pipeline.stream(req)]
    final = deltas[-1].result
    assert final.receipt["engine_commit"]
    assert pipeline.traces[-1].stream["incremental"] is True
    ok, _reason = verify_chain(cfg.attest.audit_log)
    assert ok


@pytest.mark.asyncio
async def test_streamed_ttft_is_measured_before_the_turn_ends(cfg):
    pipeline = Pipeline(cfg, MockBackend(use_tools=False, token_delay_s=0.01))
    req = GenRequest(messages=[Message(role=Role.USER, content="hello there")])
    async for _ in pipeline.stream(req):
        pass
    trace = pipeline.traces[-1]
    assert 0 < trace.ttft_s < trace.total_s


@pytest.mark.asyncio
async def test_a_backend_that_only_emits_on_the_final_delta_still_streams(cfg):
    """`Backend.stream`'s default is one delta; the pipeline must not drop its text."""

    class NoDeltas(MockBackend):
        async def stream(self, req):
            yield Delta(done=True, result=await self.generate(req))

    pipeline = Pipeline(cfg, NoDeltas(use_tools=False))
    req = GenRequest(messages=[Message(role=Role.USER, content="hello there")])
    deltas = [d async for d in pipeline.stream(req)]
    assert "".join(d.text for d in deltas) == deltas[-1].result.text


def test_malformed_body_is_a_400_not_a_500(client):
    r = client.post(
        "/v1/messages",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
    assert r.json()["type"] == "error"


def test_api_key_is_enforced_when_set(tmp_path):
    c = Config()
    c.attest.audit_log = str(tmp_path / "audit.jsonl")
    c.server.api_key = "secret"
    client = build_client(c)
    assert (
        client.post("/v1/messages", json={"messages": [], "max_tokens": 8}).status_code
        == 401
    )
    ok = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        headers={"x-api-key": "secret"},
    )
    assert ok.status_code == 200


# Every registered route, not just /v1/messages. The admin routes carried no auth
# at all: /orbit/compaction/last returned the previous request's full raw system
# prompt — for a coding agent, repository context, file paths and project
# instructions — to any unauthenticated caller, and /orbit/health enumerated the
# mounted adapter names.
@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/v1/messages"),
        ("post", "/v1/messages/count_tokens"),
        ("post", "/v1/chat/completions"),
        ("post", "/v1/responses"),
        ("get", "/v1/models"),
        ("get", "/orbit/health"),
        ("get", "/orbit/stats"),
        ("get", "/orbit/audit/verify"),
        ("get", "/orbit/compaction/last"),
        ("get", "/orbit/trace/last"),
    ],
)
def test_every_route_requires_the_api_key_when_set(tmp_path, method, path):
    c = Config()
    c.attest.audit_log = str(tmp_path / "audit.jsonl")
    c.server.api_key = "secret"
    client = build_client(c)

    unauth = client.request(method, path, json={} if method == "post" else None)
    assert unauth.status_code == 401, f"{path} served an unauthenticated caller"

    authed = client.request(
        method,
        path,
        json={} if method == "post" else None,
        headers={"x-api-key": "secret"},
    )
    assert authed.status_code != 401


def test_the_compaction_view_does_not_disclose_the_prompt_without_the_key(tmp_path):
    """The specific disclosure H6 named, pinned end to end."""
    c = Config()
    c.attest.audit_log = str(tmp_path / "audit.jsonl")
    c.server.api_key = "secret"
    client = build_client(c)
    secret_prompt = CC_SYSTEM + "\nThe repository is at /home/alice/acme-payments."
    client.post(
        "/v1/messages",
        json={
            "system": secret_prompt,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
        headers={"x-api-key": "secret"},
    )
    leaked = client.get("/orbit/compaction/last")
    assert leaked.status_code == 401
    assert "acme-payments" not in leaked.text


def test_an_unknown_adapter_is_refused_rather_than_silently_served_by_the_base(client):
    r = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        headers={"x-orbit-adapter": "no-such-adapter"},
    )
    # Serving the base model here would attest an adapter that never ran.
    assert r.status_code == 400
    assert "no-such-adapter" in r.json()["error"]["message"]


def test_a_typod_default_adapter_is_refused_for_every_request(tmp_path):
    c = Config()
    c.attest.audit_log = str(tmp_path / "audit.jsonl")
    c.tier0.default_adapter = "a1-typo"
    client = build_client(c)
    r = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    assert r.status_code == 400


@pytest.mark.parametrize(
    "body",
    [
        {"messages": ["hello"]},
        {"messages": [{"role": "user", "content": "hi"}], "metadata": "x"},
    ],
)
def test_malformed_but_plausible_bodies_are_400_in_protocol_shape(client, body):
    """A wrong-shaped body two levels down is a client error, not a bare 500."""
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 400
    assert r.json()["type"] == "error"


def test_a_malformed_openai_tool_call_is_a_400(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "a", "function": "oops"}]},
            ],
        },
    )
    assert r.status_code == 400


def test_count_tokens_rejects_a_non_json_body(client):
    r = client.post(
        "/v1/messages/count_tokens",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


def test_count_tokens_counts_the_compacted_prompt(client):
    """M1: counting the raw harness prompt over-reports by the whole multiplier.

    Claude Code drives its context meter *and* its auto-compact threshold off this
    number, so an inflated count makes the harness throw away its own history.
    """
    body = {
        "system": CC_SYSTEM,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    }
    counted = client.post("/v1/messages/count_tokens", json=body).json()["input_tokens"]

    served = client.post("/v1/messages", json=body).json()["usage"]["input_tokens"]
    # The count endpoint and the completion path must describe the same prompt.
    # Before the fix this was ~29x apart.
    assert counted <= served * 1.5
    assert counted < 2000


def test_count_tokens_honours_the_no_compact_escape_hatch(client):
    body = {
        "system": CC_SYSTEM,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    }
    compacted = client.post("/v1/messages/count_tokens", json=body).json()[
        "input_tokens"
    ]
    raw = client.post(
        "/v1/messages/count_tokens", json=body, headers={"x-orbit-no-compact": "1"}
    ).json()["input_tokens"]
    assert raw > compacted


def test_count_tokens_probes_do_not_displace_the_diff_view(client):
    """A probe is not a served turn; the sec 8.2 view must still show the request."""
    client.post(
        "/v1/messages",
        json={
            "system": CC_SYSTEM,
            "messages": [{"role": "user", "content": "served"}],
            "max_tokens": 8,
        },
    )
    for _ in range(5):
        client.post(
            "/v1/messages/count_tokens",
            json={
                "system": CC_SYSTEM,
                "messages": [{"role": "user", "content": "probe"}],
            },
        )
    assert client.get("/orbit/compaction/last").status_code == 200


def test_the_response_id_is_the_audited_request_id(client, cfg):
    """M2: an auditor could previously correlate only by timestamp."""
    r = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    response_id = r.json()["id"]
    with open(cfg.attest.audit_log, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert records, "the turn wrote no audit record"
    assert records[-1]["request_id"] == response_id


def test_a_cache_store_failure_degrades_to_a_miss_and_still_audits(
    client, cfg, monkeypatch
):
    """C3: a cache must never fail a served request or suppress its audit record."""
    from orbit.gateway.cache.prompt_cache import PromptCache

    def boom(*_a, **_k):
        raise RuntimeError("cache exploded")

    monkeypatch.setattr(PromptCache, "store", boom)
    # Long enough to reach a chunk boundary, or there is nothing to store and the
    # failing path is never entered.
    r = client.post(
        "/v1/messages",
        json={
            "messages": [{"role": "user", "content": "hi " * 4000}],
            "max_tokens": 8,
        },
    )
    assert r.status_code == 200
    assert r.json()["content"][0]["text"]

    with open(cfg.attest.audit_log, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(records) == 1, "the answered turn vanished from the sec 9.2 chain"
    trace = client.get("/orbit/trace/last").json()
    assert "store_error" in trace["cache"]


def test_prose_on_a_tool_turn_is_not_counted_as_a_wellformed_tool_call(cfg):
    """C4's shape in the live tally: 'Sure, I'll take a look!' is not a tool call."""
    # Prevention off: under a tool-call grammar the model *cannot* answer a tool
    # turn with prose, so with constrained decoding enabled this shape is
    # unrepresentable and the test would assert nothing. `script` is the existing
    # fixed-reply mechanism, and it only reaches the model path once the schema is
    # not being enforced.
    cfg.toolcall.constrain = False
    backend = MockBackend(script=["Sure, I'll take a look!"])
    pipeline = Pipeline(cfg, backend)
    client = TestClient(create_app(cfg, pipeline), base_url="http://127.0.0.1")
    client.post(
        "/v1/messages",
        json={
            "messages": [{"role": "user", "content": "read the file"}],
            "max_tokens": 32,
            "tools": [
                {"name": "read_file", "description": "read", "input_schema": SCHEMA}
            ],
        },
    )
    rate = pipeline.tool_call_rate()
    assert rate["prose"] == 1
    assert rate["rate"] != 1.0


# --- compaction (sec 8.2) ---------------------------------------------------


def test_claude_code_prompt_is_detected_and_compacted():
    compactor = Compactor()
    req = GenRequest(system=CC_SYSTEM, messages=[Message(role=Role.USER, content="hi")])
    out, result = compactor.apply(req)
    assert result.applied
    assert result.harness == "claude_code"
    assert result.template_id == "cc-2026.08@v3"
    # The spec's incumbent measured 28x; the gate is >=10x.
    assert result.multiplier >= 10.0
    assert out.system != CC_SYSTEM
    assert out.original_system == CC_SYSTEM


def test_unrecognised_harness_is_left_alone():
    """Guessing at a prompt we do not recognise is how you mis-strip a tool."""
    compactor = Compactor()
    req = GenRequest(system="You are a helpful assistant.")
    out, result = compactor.apply(req)
    assert not result.applied
    assert out.system == "You are a helpful assistant."


def test_no_compact_escape_hatch_and_diff_view(client):
    r = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 32,
            "system": CC_SYSTEM,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"x-orbit-no-compact": "1"},
    )
    assert r.status_code == 200
    last = client.get("/orbit/compaction/last").json()
    assert last["applied"] is False
    assert last["reason"] == "compaction disabled"


def test_compaction_diff_view_shows_what_changed():
    compactor = Compactor()
    _out, result = compactor.apply(
        GenRequest(system=CC_SYSTEM, messages=[Message(role=Role.USER, content="hi")])
    )
    diff = result.diff()
    assert "--- harness-system" in diff
    assert "+++ compacted-system" in diff


def test_drifted_harness_prompt_is_flagged_stale():
    """A stale fingerprint must be loud, not a silent loss of the multiplier."""
    compactor = Compactor()
    # Two of six markers: enough to identify the harness, few enough that the
    # template was clearly written against a different release.
    thin = (
        "You are Claude Code, a coding tool.\n"
        "Anthropic's official CLI has been rewritten and none of the rest of this "
        "prompt resembles the version this template was authored against."
    )
    _out, result = compactor.apply(GenRequest(system=thin))
    assert result.applied
    assert result.stale_fingerprint
    assert "re-authoring" in result.reason


def test_realistic_prompt_is_not_flagged_stale():
    _out, result = Compactor().apply(GenRequest(system=CC_SYSTEM))
    assert result.applied and not result.stale_fingerprint


def test_tool_schemas_strip_to_name_types_and_one_line():
    tool = ToolDef(
        name="read_file",
        description="Read a file. Then do more. " * 20,
        parameters=SCHEMA,
    )
    slim = strip_tool(tool)
    assert slim.name == "read_file"
    assert len(slim.description) <= 121
    props = slim.parameters["properties"]
    assert props["path"] == {"type": "string"}
    # Enums and required survive: dropping them makes the model invent values and
    # omit mandatory arguments, straight onto the sec 10.2 gate.
    assert props["mode"]["enum"] == ["read", "write"]
    assert props["lines"]["type"] == "integer[]"
    assert slim.parameters["required"] == ["path"]


def test_one_line_truncates_at_a_sentence():
    assert one_line("First sentence. Second sentence.") == "First sentence."


def test_detect_scores_the_best_template():
    tmpl, score = detect(CC_SYSTEM)
    assert tmpl is not None and tmpl.harness == "claude_code"
    assert score >= tmpl.min_markers


# --- context scaling (sec 8.3) ----------------------------------------------


def test_context_scaling_rounds_up():
    scaler = ContextScaler(assumed_window=200_000, real_window=25_000)
    assert scaler.factor == 8.0
    assert scaler.scale(1_000) == 8_000
    # Rounds up: under-reporting would let the harness sail past the point where it
    # should have compacted, which is the failure this exists to prevent.
    assert scaler.scale(1) == 8


def test_context_scaling_off_is_identity():
    assert ContextScaler(enabled=False).scale(1234) == 1234


def test_scaled_usage_reaches_the_wire(client):
    r = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi" * 500}],
        },
    )
    reported = r.json()["usage"]["input_tokens"]
    scaler = ContextScaler()
    assert reported > 0
    assert reported % 1 == 0
    # Reported usage is scaled; the audit log holds the true count.
    assert reported >= scaler.factor


def test_count_tokens_endpoint_is_scaled(client):
    r = client.post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert r.status_code == 200
    assert r.json()["input_tokens"] > 0


# --- caches (sec 8.4) -------------------------------------------------------


def test_chunk_digests_are_a_streaming_prefix_index():
    text = "a" * 3000
    marks = chunk_digests(text, 1024)
    assert [m[0] for m in marks] == [1024, 2048, 3000]
    # Each digest is of the *prefix*, so a longer text reproduces the earlier ones.
    longer = chunk_digests(text + "b" * 500, 1024)
    assert [m[1] for m in longer[:2]] == [m[1] for m in marks[:2]]


def test_chunk_digests_never_split_a_codepoint():
    text = "é" * 2000  # two bytes each
    for offset, _digest in chunk_digests(text, 1024):
        text.encode("utf-8")[:offset].decode("utf-8")  # must not raise


def test_prompt_cache_returns_longest_prefix():
    cache = PromptCache(chunk_bytes=1024)
    turn1 = "x" * 4000
    cache.store(
        align_down(turn1, 1024),
        CacheEntry(digest="", prefix_bytes=0, n_tokens=1000, size_bytes=4096),
    )
    turn2 = turn1 + "y" * 500
    hit = cache.lookup(turn2)
    assert hit is not None
    assert hit.covered_bytes == 3072
    assert hit.remaining_bytes == len(turn2.encode()) - 3072


def test_prompt_cache_evicts_under_budget():
    cache = PromptCache(budget_bytes=4096, chunk_bytes=1024)
    for i in range(4):
        cache.store(
            chr(97 + i) * 3000,
            CacheEntry(digest="", prefix_bytes=0, n_tokens=1, size_bytes=2048),
        )
    assert cache.size_bytes <= 4096
    assert cache.evictions > 0


def test_align_down_trims_to_a_chunk_boundary():
    assert len(align_down("z" * 5000, 1024).encode()) == 4096
    assert align_down("short", 1024) == ""


def test_disk_kv_round_trip(tmp_path):
    cache = DiskKVCache(tmp_path, budget_bytes=1 << 20)
    snap = KVSnapshot(
        digest="a" * 64,
        token_ids=[1, 2, 3, 65535],
        next_logits=b"\x01\x02",
        state_blob=b"\xff" * 64,
        replay={"call_1": '{"name":"read_file"}'},
        prefix_bytes=4096,
    )
    cache.put(snap)
    got = cache.get("a" * 64)
    assert got is not None
    assert got.token_ids == [1, 2, 3, 65535]
    assert got.next_logits == b"\x01\x02"
    assert got.state_blob == b"\xff" * 64
    # The replay map rides with the state it belongs to (sec 8.5.5).
    assert got.replay == {"call_1": '{"name":"read_file"}'}


def test_disk_kv_corrupt_entry_is_a_miss_not_an_error(tmp_path):
    cache = DiskKVCache(tmp_path)
    cache.put(KVSnapshot(digest="b" * 64, token_ids=[1]))
    path = cache._path("b" * 64)
    path.write_bytes(b"garbage")
    assert cache.get("b" * 64) is None


def test_disk_kv_enforces_its_budget(tmp_path):
    cache = DiskKVCache(tmp_path, budget_bytes=2048)
    for i in range(6):
        cache.put(KVSnapshot(digest=f"{i:064x}", state_blob=b"\x00" * 1024))
    assert cache.stats()["bytes"] <= 2048 + 1024


# --- pipeline ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_repairs_a_faulted_tool_call(cfg):
    # Prevention off: the repair layer is the *fallback* for when constrained
    # decoding is unavailable, so testing it with constraint enabled would test
    # nothing — the model could not emit the malformed shape in the first place.
    cfg.toolcall.constrain = False
    backend = MockBackend(fault=Fault.XML_HYBRID)
    pipeline = Pipeline(cfg, backend)
    req = GenRequest(
        messages=[Message(role=Role.USER, content="read a file")],
        tools=(ToolDef(name="read_file", parameters=SCHEMA),),
    )
    result, trace = await pipeline.run(req)
    assert result.tool_calls
    assert result.repaired
    assert trace.toolcall["outcome"] == "repaired"


@pytest.mark.asyncio
async def test_pipeline_rejects_an_invented_tool(cfg):
    # Prevention off for the same reason: under a closed name enum the model
    # cannot invent a tool, so the rejection path would never be reached.
    cfg.toolcall.constrain = False
    backend = MockBackend(fault=Fault.UNKNOWN_TOOL)
    pipeline = Pipeline(cfg, backend)
    req = GenRequest(
        messages=[Message(role=Role.USER, content="read a file")],
        tools=(ToolDef(name="read_file", parameters=SCHEMA),),
    )
    result, trace = await pipeline.run(req)
    assert not result.tool_calls
    assert trace.toolcall["rejected"]


@pytest.mark.asyncio
async def test_tool_turns_are_cooled(cfg):
    """Sec 8.5: temperature 0.2 for tool-bearing turns, caller's value otherwise."""
    backend = MockBackend()
    pipeline = Pipeline(cfg, backend)
    from orbit.types import Sampling

    await pipeline.run(
        GenRequest(
            messages=[Message(role=Role.USER, content="read")],
            tools=(ToolDef(name="read_file", parameters=SCHEMA),),
            sampling=Sampling(temperature=0.9),
        )
    )
    assert backend.calls[-1].sampling.temperature == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_second_turn_hits_the_prompt_cache(cfg):
    """The sec 8.4 claim: TTFT stops growing with conversation length."""
    cfg.compaction.enabled = False
    pipeline = Pipeline(cfg, MockBackend(use_tools=False))
    long_body = "def handler():\n    return 1\n\n" * 400

    await pipeline.run(
        GenRequest(messages=[Message(role=Role.USER, content=long_body)])
    )
    await pipeline.run(
        GenRequest(
            messages=[
                Message(role=Role.USER, content=long_body),
                Message(role=Role.ASSISTANT, content="ok"),
                Message(role=Role.USER, content="now the other one"),
            ]
        )
    )
    assert pipeline.traces[1].cache["prefix_hit"] is True
    assert pipeline.traces[1].cache["covered_fraction"] > 0.5
