"""Router and tier-1 verifier (spec sec 5, 7)."""

from __future__ import annotations

import pytest

from tandem.backends.mock import MockBackend
from tandem.config import RouterConfig
from tandem.router.cascade import Cascade
from tandem.router.classify import classify, previous_turn_produced_diff
from tandem.tier1.schemas import RERANK, REVIEW
from tandem.tier1.verifier import Candidate, Tier1Verifier
from tandem.types import GenRequest, GenResult, Message, Role, ToolDef, ToolResult, TurnClass

EDIT = ToolDef(name="edit_file", parameters={"type": "object", "properties": {"path": {"type": "string"}}})
READ = ToolDef(name="read_file", parameters={"type": "object", "properties": {"path": {"type": "string"}}})
BASH = ToolDef(name="run_bash", parameters={"type": "object", "properties": {"command": {"type": "string"}}})

DIFF = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"


# --- classification (sec 7.1) -----------------------------------------------


@pytest.mark.parametrize(
    "text,tools,expected",
    [
        ("Fix the retry loop in utils.py", (EDIT, BASH), TurnClass.CODE_CHANGE),
        ("Implement pagination for the list endpoint", (EDIT,), TurnClass.CODE_CHANGE),
        ("What does the parse_args function do?", (READ,), TurnClass.READ_ONLY),
        ("Explain how the retry backoff works", (READ,), TurnClass.READ_ONLY),
        ("How would we approach migrating to the new API?", (READ, EDIT), TurnClass.PLAN),
        ("Thanks, that's helpful", (), TurnClass.CHAT),
    ],
)
def test_turn_classification(text, tools, expected):
    req = GenRequest(messages=[Message(role=Role.USER, content=text)], tools=tools)
    assert classify(req).turn is expected


def test_plan_directive_yields_to_an_explicit_code_directive():
    """A harness ships the same tools every turn; the directive is the real signal."""
    req = GenRequest(
        messages=[Message(role=Role.USER, content="Plan and then implement the fix")],
        tools=(EDIT,),
    )
    assert classify(req).turn is TurnClass.CODE_CHANGE


def test_continuation_after_a_diff_is_still_a_code_change_turn():
    """Sec 7.1: whether the previous turn produced a diff is a classification signal."""
    req = GenRequest(
        messages=[
            Message(role=Role.USER, content="do the thing"),
            Message(role=Role.ASSISTANT, content=DIFF),
            Message(role=Role.USER, content="now the other one"),
        ],
        tools=(EDIT, BASH),
    )
    assert classify(req).turn is TurnClass.CODE_CHANGE


def test_diff_detection_reads_tool_results_too():
    msgs = [Message(role=Role.TOOL, tool_results=(ToolResult(tool_call_id="t", content=DIFF),))]
    assert previous_turn_produced_diff(msgs)
    assert not previous_turn_produced_diff([Message(role=Role.ASSISTANT, content="no diff here")])


# --- tier-1 verifier (sec 5.1) ----------------------------------------------


@pytest.mark.asyncio
async def test_rerank_returns_a_schema_valid_choice():
    verifier = Tier1Verifier(MockBackend(tier=1))
    verdict = await verifier.rerank(
        [Candidate(0, "patch a"), Candidate(1, "patch b"), Candidate(2, "patch c")], "context"
    )
    assert verdict.ok
    assert 0 <= verdict.data["choice"] < 3


@pytest.mark.asyncio
async def test_rerank_short_circuits_on_a_single_candidate():
    """No tier-1 call is worth 18 s when there is nothing to choose between."""
    backend = MockBackend(tier=1)
    verifier = Tier1Verifier(backend)
    verdict = await verifier.rerank([Candidate(0, "only")], "context")
    assert verdict.ok and verdict.data["choice"] == 0
    assert not backend.calls


@pytest.mark.asyncio
async def test_out_of_range_choice_is_a_failed_call_not_a_silent_clamp():
    def responder(_req):
        return GenResult(text='{"choice": 99, "reason": "nonsense"}')

    verifier = Tier1Verifier(MockBackend(tier=1, responder=responder))
    verdict = await verifier.rerank([Candidate(0, "a"), Candidate(1, "b")], "ctx")
    assert not verdict.ok
    assert "out of range" in verdict.error


@pytest.mark.asyncio
async def test_unparseable_verdict_degrades_rather_than_raises():
    """2-bit's documented failure mode must not take the turn down (sec 5.2)."""
    def responder(_req):
        return GenResult(text="I think candidate two is best, actually.")

    verifier = Tier1Verifier(MockBackend(tier=1, responder=responder))
    verdict = await verifier.rerank([Candidate(0, "a"), Candidate(1, "b")], "ctx")
    assert not verdict.ok
    assert "unparseable" in verdict.error


@pytest.mark.asyncio
async def test_tier1_unavailable_degrades_cleanly():
    verifier = Tier1Verifier(None)
    assert not verifier.available
    verdict = await verifier.review("patch", "ctx")
    assert not verdict.ok and "not enabled" in verdict.error


@pytest.mark.asyncio
async def test_a_raising_backend_does_not_propagate():
    class Boom(MockBackend):
        async def generate(self, req):
            raise RuntimeError("NVMe fell over")

    verifier = Tier1Verifier(Boom(tier=1))
    verdict = await verifier.review("patch", "ctx")
    assert not verdict.ok and "NVMe fell over" in verdict.error


def test_rerank_schema_bounds_the_choice_to_the_candidates_on_offer():
    """Sec 5.1: bounded *by construction*, not merely validated afterwards.

    Without `maximum`, a constrained decode can still name a candidate that does
    not exist — and the cost of catching that at runtime is a wasted ~18 s rerank
    and a merge-quality gate that did not happen.
    """
    from tandem.tier1.schemas import rerank_schema

    schema = rerank_schema(3)
    assert schema["properties"]["choice"] == {"type": "integer", "minimum": 0, "maximum": 2}
    assert schema["title"] == "rerank"  # the tier-1 budget lookup keys on this
    assert rerank_schema(1)["properties"]["choice"]["maximum"] == 0
    with pytest.raises(ValueError):
        rerank_schema(0)


@pytest.mark.asyncio
async def test_rerank_never_picks_a_candidate_that_does_not_exist():
    """Over many seeds, with the mock honouring the schema bounds as a real
    constrained decoder does."""
    verifier = Tier1Verifier(MockBackend(tier=1))
    for seed in range(40):
        verdict = await verifier.rerank(
            [Candidate(0, "a"), Candidate(1, "b"), Candidate(2, "c")], "ctx", seed=seed
        )
        assert verdict.ok, verdict.error
        assert 0 <= verdict.data["choice"] <= 2


def test_schemas_are_closed_and_bounded():
    """Output length bounded by construction, invented fields impossible (sec 5.1)."""
    for schema in (RERANK, REVIEW):
        assert schema["additionalProperties"] is False
        assert schema["required"]
    assert RERANK["properties"]["reason"]["maxLength"] == 200
    assert REVIEW["properties"]["issues"]["maxItems"] == 6


# --- cascade (sec 7.2, 7.3) -------------------------------------------------


@pytest.mark.asyncio
async def test_chat_turns_never_touch_tier_1():
    tier1 = MockBackend(tier=1)
    cascade = Cascade(MockBackend(use_tools=False), Tier1Verifier(tier1))
    _result, info = await cascade.produce(
        GenRequest(messages=[Message(role=Role.USER, content="thanks!")])
    )
    assert info.turn is TurnClass.CHAT
    assert not info.tier1_invoked
    assert not tier1.calls


@pytest.mark.asyncio
async def test_code_change_turn_generates_n_candidates_and_reranks():
    tier0 = MockBackend(use_tools=False)
    cascade = Cascade(tier0, Tier1Verifier(MockBackend(tier=1)), RouterConfig(candidates=3))
    _result, info = await cascade.produce(
        GenRequest(messages=[Message(role=Role.USER, content="Fix the retry loop")], tools=(EDIT,))
    )
    assert info.candidates_generated == 3
    assert info.tier1_invoked and info.tier1_call == "rerank"
    assert 0 <= info.candidate_selected < 3


@pytest.mark.asyncio
async def test_candidate_seeds_are_derived_so_the_receipt_reproduces_them():
    tier0 = MockBackend(use_tools=False)
    cascade = Cascade(tier0, Tier1Verifier(None), RouterConfig(candidates=3, rerank_enabled=False))
    req = GenRequest(messages=[Message(role=Role.USER, content="Fix it")], tools=(EDIT,))
    await cascade.produce(req)
    seeds = [c.sampling.seed for c in tier0.calls]
    assert seeds == [0, 1, 2]
    # Distinct seeds must actually produce distinct candidates, or best-of-N is
    # three copies of one sample.
    assert len({c.sampling.seed for c in tier0.calls}) == 3


@pytest.mark.asyncio
async def test_n_equals_one_disables_reranking():
    tier1 = MockBackend(tier=1)
    cascade = Cascade(MockBackend(use_tools=False), Tier1Verifier(tier1), RouterConfig(candidates=1))
    _result, info = await cascade.produce(
        GenRequest(messages=[Message(role=Role.USER, content="Fix it")], tools=(EDIT,))
    )
    assert info.candidates_generated == 1
    assert not tier1.calls


@pytest.mark.asyncio
async def test_failed_rerank_falls_back_to_candidate_zero():
    def responder(_req):
        return GenResult(text="not json")

    cascade = Cascade(
        MockBackend(use_tools=False),
        Tier1Verifier(MockBackend(tier=1, responder=responder)),
        RouterConfig(candidates=3),
    )
    _result, info = await cascade.produce(
        GenRequest(messages=[Message(role=Role.USER, content="Fix it")], tools=(EDIT,))
    )
    assert info.candidate_selected == 0
    assert info.tier1_verdicts and not info.tier1_verdicts[0]["ok"]


@pytest.mark.asyncio
async def test_failure_escalation_regenerates_with_the_critique():
    """T2 (sec 7.2): tests fail -> tier-1 review -> tier-0 regenerates."""
    calls = {"n": 0}

    async def failing_tests(_patch):
        calls["n"] += 1
        return False, "FAILED tests/test_retry.py::test_backoff - AssertionError"

    tier0 = MockBackend(use_tools=False)
    cascade = Cascade(
        tier0,
        Tier1Verifier(MockBackend(tier=1)),
        RouterConfig(candidates=1),
        test_runner=failing_tests,
    )
    _result, info = await cascade.produce(
        GenRequest(messages=[Message(role=Role.USER, content="Fix the retry loop")], tools=(EDIT,))
    )
    assert calls["n"] == 1
    assert info.escalated
    assert info.tier1_call == "review"
    # The regeneration must actually carry the failure output into context.
    last_prompt = tier0.calls[-1].messages[-1].content
    assert "AssertionError" in last_prompt


@pytest.mark.asyncio
async def test_escalation_is_bounded_to_one_per_turn():
    async def always_fails(_patch):
        return False, "still failing"

    tier0 = MockBackend(use_tools=False)
    cascade = Cascade(
        tier0, Tier1Verifier(MockBackend(tier=1)),
        RouterConfig(candidates=1, max_escalations_per_turn=1),
        test_runner=always_fails,
    )
    _result, info = await cascade.produce(
        GenRequest(messages=[Message(role=Role.USER, content="Fix it")], tools=(EDIT,))
    )
    # One original generation plus exactly one regeneration.
    assert len(tier0.calls) == 2
    assert info.escalated


@pytest.mark.asyncio
async def test_escalation_stays_dormant_without_a_test_runner():
    """Escalating on a failure we never observed would be theatre."""
    cascade = Cascade(MockBackend(use_tools=False), Tier1Verifier(MockBackend(tier=1)),
                      RouterConfig(candidates=1))
    _result, info = await cascade.produce(
        GenRequest(messages=[Message(role=Role.USER, content="Fix it")], tools=(EDIT,))
    )
    assert not info.escalated


# --- T2 on the served path (sec 7.2) ----------------------------------------


def _served(tmp_path, **eval_kw):
    from tandem.config import Config
    from tandem.gateway.pipeline import Pipeline

    cfg = Config()
    cfg.attest.audit_log = str(tmp_path / "audit.jsonl")
    cfg.cache.disk_kv_enabled = False
    cfg.tier1.enabled = True
    for key, value in eval_kw.items():
        setattr(cfg.eval, key, value)
    return Pipeline(cfg, MockBackend(use_tools=False), MockBackend(tier=1))


def test_the_served_path_has_no_test_runner_by_default(tmp_path):
    """Running the repo's suite on every turn is opt-in, not a default."""
    pipeline = _served(tmp_path)
    assert pipeline.cascade.test_runner is None
    assert pipeline.stats()["escalation"]["enabled"] is False


def test_opting_in_without_a_test_command_stays_dormant(tmp_path):
    """A runner that answers "passed" to everything would suppress T2 silently."""
    pipeline = _served(tmp_path, escalate_on_test_failure=True, linters=[["ruff", "check"]])
    assert pipeline.cascade.test_runner is None


def test_opting_in_makes_t2_escalation_reachable(tmp_path):
    """Item 3: the path exists and is tested, but nothing could ever trigger it."""
    pipeline = _served(
        tmp_path, escalate_on_test_failure=True, test_command=["pytest", "-q"], repo=str(tmp_path)
    )
    assert pipeline.cascade.test_runner is not None
    assert pipeline.stats()["escalation"]["test_command"] == "pytest -q"


@pytest.mark.asyncio
async def test_pressure_valve_degrades_to_n_equals_one():
    """Sec 7.3: past the budget, degrade automatically rather than as a setting."""
    tier0 = MockBackend(use_tools=False, token_delay_s=0.002)
    cascade = Cascade(
        tier0,
        Tier1Verifier(MockBackend(tier=1)),
        RouterConfig(candidates=3, degrade_after_s=0.0),
    )
    req = GenRequest(messages=[Message(role=Role.USER, content="Fix it")], tools=(EDIT,))

    first, info1 = await cascade.produce(req)
    assert info1.candidates_generated == 3
    assert cascade.degraded

    _second, info2 = await cascade.produce(req)
    assert info2.candidates_generated == 1
    assert info2.degraded
    assert "degraded to N=1" in cascade.stats()["degrade_reason"]


@pytest.mark.asyncio
async def test_pressure_valve_ignores_chat_latency():
    """A slow chat turn is a slow model, not a cascade that needs disabling."""
    cascade = Cascade(
        MockBackend(use_tools=False),
        Tier1Verifier(MockBackend(tier=1)),
        RouterConfig(candidates=3, degrade_after_s=0.0),
    )
    await cascade.produce(GenRequest(messages=[Message(role=Role.USER, content="thanks")]))
    assert not cascade.degraded
