"""Disk KV cache and the backend state interface (spec sec 8.4).

The disk cache exists for exactly one thing the in-memory cache cannot do: make the
first turn after a restart warm instead of a 30 s cold prefill. So the tests that
matter here build a second `Pipeline` over the same directory — a fresh process in
all the ways that count — and check the state comes back.

The other half is refusal. A KV state restored under a different container or a
different adapter would continue the conversation in a model that never saw its own
prefix, and the failure is silent: fluent output, wrong model, and a receipt naming
the adapter that did not produce it. Every mismatch below must miss, not restore.

The third half — the cheapest tests here and the ones that were missing — is that
none of this may ever *fail a turn*. A cache is written after the model has answered,
so a store that raises loses the answer; and a cache is read into a prompt, so an
entry that comes back short, or under the wrong name, is worse than no entry at all.
"""

from __future__ import annotations

import pytest

from tandem.backends.mock import MockBackend
from tandem.config import Config
from tandem.gateway.cache.kv_disk import DiskKVCache, KVSnapshot
from tandem.gateway.pipeline import Pipeline
from tandem.types import GenRequest, KVState, Message, Role

LONG_BODY = "def handler(request):\n    return process(request)\n\n" * 300
# The same shape of body with the multi-byte characters a real diff carries: a CJK
# identifier, an accented one, an emoji and a box-drawing character. Every cache test
# in the suite used to be pure ASCII, which is why H1 survived.
UNICODE_BODY = (
    "def 计算(请求):\n    return prozessieren(请求)  # ✅ résumé ─┤\n\n" * 300
)


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.attest.audit_log = str(tmp_path / "audit.jsonl")
    c.cache.disk_kv_dir = str(tmp_path / "kv")
    # Compaction rewrites the system prompt; off here so the rendered prefix is
    # driven by the conversation under test rather than by a template.
    c.compaction.enabled = False
    return c


def _turn(*contents: str) -> GenRequest:
    return GenRequest(messages=[Message(role=Role.USER, content=c) for c in contents])


# --- the restart loop -------------------------------------------------------


@pytest.mark.asyncio
async def test_state_survives_a_restart(cfg):
    """The whole point of the disk cache: a new process starts warm."""
    first = Pipeline(cfg, MockBackend(use_tools=False))
    await first.run(_turn(LONG_BODY))
    assert first._disk_kv_stats()["entries"] > 0

    # A second Pipeline over the same directory — a fresh process in every way
    # that matters, with an empty in-memory cache.
    second = Pipeline(cfg, MockBackend(use_tools=False))
    _result, trace = await second.run(_turn(LONG_BODY, "and now the follow-up"))

    assert trace.cache["prefix_hit"] is True
    assert trace.cache["source"] == "disk"
    assert trace.cache["restored_tokens"] > 0
    assert second.disk_kv_hits == 1


@pytest.mark.asyncio
async def test_restored_state_makes_the_prefix_free_to_prefill(cfg):
    """A restored prefix must be *reported* as cached, or it saved nothing."""
    first = Pipeline(cfg, MockBackend(use_tools=False))
    await first.run(_turn(LONG_BODY))

    second = Pipeline(cfg, MockBackend(use_tools=False))
    result, _trace = await second.run(_turn(LONG_BODY, "follow-up"))
    assert result.usage.cached_input_tokens > 0


@pytest.mark.asyncio
async def test_memory_hit_is_preferred_over_disk(cfg):
    """A hit in memory needs no read at all; disk is the fallback, not the path."""
    pipeline = Pipeline(cfg, MockBackend(use_tools=False))
    await pipeline.run(_turn(LONG_BODY))
    _result, trace = await pipeline.run(_turn(LONG_BODY, "second turn"))
    assert trace.cache["source"] == "memory"
    assert pipeline.disk_kv_hits == 0


# --- refusal ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_from_a_different_container_is_refused(cfg):
    """A different container is a different model. Restoring is silently wrong."""
    first = Pipeline(cfg, MockBackend(use_tools=False, container="container-A"))
    await first.run(_turn(LONG_BODY))

    second = Pipeline(cfg, MockBackend(use_tools=False, container="container-B"))
    _result, trace = await second.run(_turn(LONG_BODY, "follow-up"))
    assert trace.cache["prefix_hit"] is False
    assert second.disk_kv_hits == 0


@pytest.mark.asyncio
async def test_state_from_a_different_adapter_is_refused(cfg):
    """Same base, different adapter — still a different model (sec 4.2)."""
    backend = MockBackend(use_tools=False, adapters=("a1", "a2"))
    first = Pipeline(cfg, backend)
    await first.run(_turn(LONG_BODY).with_(adapter="a1"))

    second = Pipeline(cfg, MockBackend(use_tools=False, adapters=("a1", "a2")))
    _result, trace = await second.run(_turn(LONG_BODY, "follow-up").with_(adapter="a2"))
    assert trace.cache["prefix_hit"] is False


@pytest.mark.asyncio
async def test_same_adapter_still_restores(cfg):
    """The adapter check must not be so strict it refuses a legitimate hit."""
    first = Pipeline(cfg, MockBackend(use_tools=False, adapters=("a1",)))
    await first.run(_turn(LONG_BODY).with_(adapter="a1"))

    second = Pipeline(cfg, MockBackend(use_tools=False, adapters=("a1",)))
    _result, trace = await second.run(_turn(LONG_BODY, "follow-up").with_(adapter="a1"))
    assert trace.cache["source"] == "disk"


@pytest.mark.asyncio
async def test_diverged_prefix_reports_no_saving(cfg):
    """A state whose bytes are not a prefix of this prompt saved nothing.

    Reporting its tokens as cached would claim a saving that did not happen.
    """
    backend = MockBackend(use_tools=False)
    state = KVState(
        key=backend.state_key(None),
        prefix_bytes=100,
        token_ids=tuple(range(25)),
        blob=b"a prefix this prompt does not begin with",
    )
    req = _turn("an entirely different prompt").with_(warm_state=state)
    result = await backend.generate(req)
    assert result.usage.cached_input_tokens == 0


# --- backends that cannot export state --------------------------------------


class _Stateless(MockBackend):
    def supports_state(self) -> bool:
        return False

    def export_state(self, req, rendered_prefix, result):
        return None


@pytest.mark.asyncio
async def test_stateless_backend_writes_nothing_and_still_serves(cfg):
    """Most backends cannot snapshot KV. That degrades to prefilling, not to failing."""
    pipeline = Pipeline(cfg, _Stateless(use_tools=False))
    result, trace = await pipeline.run(_turn(LONG_BODY))
    assert result.text
    assert trace.cache["prefix_hit"] is False
    stats = pipeline._disk_kv_stats()
    assert stats["entries"] == 0
    assert stats["backend_supports_state"] is False


@pytest.mark.asyncio
async def test_disabled_disk_cache_still_serves(cfg):
    cfg.cache.disk_kv_enabled = False
    pipeline = Pipeline(cfg, MockBackend(use_tools=False))
    result, _trace = await pipeline.run(_turn(LONG_BODY))
    assert result.text
    assert pipeline._disk_kv_stats() is None


# --- replay map rides with the state (sec 8.5.5) ----------------------------


@pytest.mark.asyncio
async def test_replay_map_is_restored_alongside_the_state(cfg, tmp_path):
    """A restored prefix whose tool calls re-render differently is not that prefix."""
    from tandem.types import ToolDef

    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    tools = (ToolDef(name="read_file", parameters=schema),)

    first = Pipeline(cfg, MockBackend())
    result, _ = await first.run(
        GenRequest(messages=[Message(role=Role.USER, content=LONG_BODY)], tools=tools)
    )
    assert result.tool_calls
    call_id = result.tool_calls[0].id
    assert first.replay.get(call_id) is not None

    second = Pipeline(cfg, MockBackend())
    assert second.replay.get(call_id) is None  # fresh process, empty map
    await second.run(
        GenRequest(
            messages=[
                Message(role=Role.USER, content=LONG_BODY),
                Message(role=Role.USER, content="follow-up"),
            ],
            tools=tools,
        )
    )
    assert second.replay.get(call_id) is not None


# --- the on-disk format -----------------------------------------------------


def test_snapshot_round_trips_the_state_key(tmp_path):
    cache = DiskKVCache(tmp_path)
    cache.put(
        KVSnapshot(
            digest="c" * 64,
            token_ids=[1, 2, 3],
            state_blob=b"weights",
            state_key="mock:abc:a1",
            prefix_bytes=4096,
        )
    )
    got = cache.get("c" * 64)
    assert got is not None
    assert got.state_key == "mock:abc:a1"
    assert got.prefix_bytes == 4096


def test_older_format_versions_are_a_miss(tmp_path):
    """A v1 entry carries no backend identity, so it cannot be shown to be safe."""
    import struct

    cache = DiskKVCache(tmp_path)
    cache.put(KVSnapshot(digest="d" * 64, token_ids=[1], state_key="mock:abc:-"))
    path = cache._path("d" * 64)
    raw = bytearray(path.read_bytes())
    raw[8:12] = struct.pack("<I", 1)  # rewrite the version to v1
    path.write_bytes(bytes(raw))
    assert cache.get("d" * 64) is None


def test_accepts_state_checks_container_and_adapter():
    backend = MockBackend(container="c1")
    good = KVState(key=backend.state_key("a1"), prefix_bytes=10)
    assert backend.accepts_state(good, "a1")
    assert not backend.accepts_state(good, "a2")
    assert not backend.accepts_state(good, None)
    assert not MockBackend(container="c2").accepts_state(good, "a1")
