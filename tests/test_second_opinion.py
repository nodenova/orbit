"""Fallback ladder rung 3 — tier 0 as its own verifier (spec sec 5.5).

The rung exists so that a machine with no 122B container still has *a* verifier.
Its value rests entirely on one property: the adapter must be unmounted. An adapted
model judging its own candidates is asking whether it agrees with itself, and the
answer carries no information. Most of what follows tests that strip.
"""

from __future__ import annotations

import pytest

from tandem.backends import (
    SECOND_OPINION_RUNG,
    SecondOpinionBackend,
    build_tier0,
    build_tier1,
)
from tandem.backends.mock import MockBackend
from tandem.config import Config
from tandem.gateway.pipeline import Pipeline
from tandem.router.cascade import Cascade
from tandem.tier1.verifier import Candidate, Tier1Verifier
from tandem.types import GenRequest, Message, Role, ToolDef

EDIT = ToolDef(
    name="edit_file",
    parameters={"type": "object", "properties": {"p": {"type": "string"}}},
)


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.attest.audit_log = str(tmp_path / "audit.jsonl")
    c.cache.disk_kv_dir = str(tmp_path / "kv")
    c.tier1.enabled = True
    c.tier1.rung = SECOND_OPINION_RUNG
    return c


# --- the strip --------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_adapter_is_unmounted_for_every_verdict():
    """The whole mechanism. Without this the model is agreeing with itself."""
    tier0 = MockBackend(use_tools=False, adapters=("a1-myrepo",))
    verifier = SecondOpinionBackend(tier0)
    await verifier.generate(
        GenRequest(
            messages=[Message(role=Role.USER, content="judge this")],
            adapter="a1-myrepo",
        )
    )
    assert tier0.calls[-1].adapter is None


@pytest.mark.asyncio
async def test_the_strip_does_not_mutate_the_caller_s_request():
    """The same request object is often the one tier 0 is about to serve *with* its
    adapter; mutating it here would unmount the adapter for the generation too."""
    tier0 = MockBackend(use_tools=False, adapters=("a1",))
    verifier = SecondOpinionBackend(tier0)
    req = GenRequest(messages=[Message(role=Role.USER, content="x")], adapter="a1")
    await verifier.generate(req)
    assert req.adapter == "a1"


@pytest.mark.asyncio
async def test_base_and_adapted_output_actually_differ():
    """If the mock ignored the adapter, every test above would pass vacuously."""
    tier0 = MockBackend(use_tools=False, adapters=("a1",))
    req = GenRequest(
        messages=[Message(role=Role.USER, content="same prompt")], adapter="a1"
    )
    adapted = await tier0.generate(req)
    base = await SecondOpinionBackend(tier0).generate(req)
    assert adapted.text != base.text


def test_the_verifier_never_attests_to_an_adapter():
    """This rung's claim is that no adapter was mounted; reporting one inverts it."""
    tier0 = MockBackend(adapters=("a1",))
    verifier = SecondOpinionBackend(tier0)
    assert verifier.adapter_hash("a1") is None
    assert verifier.profile_hash("a1") is None
    assert verifier.mounted_adapters() == ()
    # The container is tier 0's, because that is what actually ran.
    assert verifier.container_hash() == tier0.container_hash()


@pytest.mark.asyncio
async def test_closing_the_verifier_does_not_close_tier_0():
    """Tier 0 is still serving generation; closing it here would take it down."""
    closed = {"n": 0}

    class Tracking(MockBackend):
        async def close(self):
            closed["n"] += 1

    tier0 = Tracking()
    await SecondOpinionBackend(tier0).close()
    assert closed["n"] == 0


# --- construction -----------------------------------------------------------


def test_build_tier1_returns_the_second_opinion_backend(cfg):
    tier0 = build_tier0(cfg)
    tier1 = build_tier1(cfg, tier0)
    assert isinstance(tier1, SecondOpinionBackend)
    assert tier1.tier == 1


def test_rung_3_without_a_tier0_backend_is_an_error(cfg):
    """It serves the verifier from tier 0; there is nothing to serve it from."""
    with pytest.raises(ValueError, match="needs the tier-0 backend"):
        build_tier1(cfg, None)


# Rung selection as a whole — every rung by name, and the rejection of one that does
# not exist — lives in `test_fallback_rungs.py` now that all four are implemented.
# `resident_swap` used to be this file's example of an unknown rung.


def test_disabled_tier1_ignores_the_rung(cfg):
    cfg.tier1.enabled = False
    assert build_tier1(cfg, build_tier0(cfg)) is None


# --- it actually verifies ---------------------------------------------------


@pytest.mark.asyncio
async def test_rung_3_serves_a_real_rerank(cfg):
    tier0 = build_tier0(cfg)
    verifier = Tier1Verifier(build_tier1(cfg, tier0))
    verdict = await verifier.rerank(
        [Candidate(0, "patch a"), Candidate(1, "patch b")], "some context"
    )
    assert verdict.ok
    assert verdict.data["choice"] in (0, 1)


@pytest.mark.asyncio
async def test_the_cascade_reranks_through_rung_3(cfg):
    tier0 = build_tier0(cfg)
    cascade = Cascade(tier0, Tier1Verifier(build_tier1(cfg, tier0)), cfg.router)
    _result, info = await cascade.produce(
        GenRequest(
            messages=[Message(role=Role.USER, content="Fix the retry loop")],
            tools=(EDIT,),
            adapter="a1",
        )
    )
    assert info.tier1_invoked
    assert info.tier1_call == "rerank"


@pytest.mark.asyncio
async def test_the_receipt_names_the_rung(cfg):
    """A base-model verdict must not read as a streamed-verifier one."""
    tier0 = build_tier0(cfg)
    pipeline = Pipeline(cfg, tier0, build_tier1(cfg, tier0))
    result, _trace = await pipeline.run(
        GenRequest(
            messages=[Message(role=Role.USER, content="Fix the retry loop")],
            tools=(EDIT,),
        )
    )
    assert result.receipt["tier1"]["rung"] == SECOND_OPINION_RUNG


@pytest.mark.asyncio
async def test_no_tier1_leaves_the_rung_unset(tmp_path):
    cfg = Config()
    cfg.attest.audit_log = str(tmp_path / "audit.jsonl")
    cfg.cache.disk_kv_dir = str(tmp_path / "kv")
    pipeline = Pipeline(cfg, build_tier0(cfg), None)
    result, _trace = await pipeline.run(
        GenRequest(messages=[Message(role=Role.USER, content="hello")])
    )
    assert result.receipt["tier1"]["rung"] is None
    assert result.receipt["tier1"]["invoked"] is False
