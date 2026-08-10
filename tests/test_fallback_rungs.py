"""Fallback ladder rungs 2 and 4 (spec sec 5.5).

Rung 3 has its own file. These two are the ones whose failure modes are not "the
verdict is worse" but "the system did something nobody asked for":

* **Rung 2** evicts tier 0 to admit the verifier. If the mutual exclusion is wrong,
  tier 0 is asked to generate from weights that are not in memory — and on the mock,
  which has no memory, that failure is invisible unless the residency policy itself is
  under test. So most of what follows tests the policy, not the models.
* **Rung 4** sends the repository's code to a third party. Every test here is about
  the gates in front of it: it is never reached by falling back, it needs the consent
  sentence written out, and it makes `orbit doctor` stop claiming an offline posture.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from orbit.backends import (
    REMOTE_RUNG,
    RESIDENT_SWAP_RUNG,
    SECOND_OPINION_RUNG,
    STREAMED_RUNG,
    RemoteConsentMissing,
    RemoteTier1Backend,
    ResidencySwitch,
    ResidentSwapBackend,
    SwapGuard,
    build_tier0,
    build_tier1,
)
from orbit.backends.base import Backend
from orbit.backends.mock import MockBackend
from orbit.backends.remote_tier1 import CONSENT
from orbit.backends.resident_swap import TIER0, TIER1, SwapBudgetExceeded
from orbit.backends.tier1_call import (
    CALL_BUDGETS,
    Tier1Unavailable,
    build_payload,
    clamp_max_tokens,
    validate_or_raise,
)
from orbit.config import Config
from orbit.gateway.pipeline import Pipeline
from orbit.offline import OfflineReport, verify
from orbit.tier1.schemas import REVIEW, rerank_schema
from orbit.tier1.verifier import Candidate, Tier1Verifier
from orbit.types import GenRequest, Message, Role, Sampling, ToolDef

EDIT = ToolDef(
    name="edit_file",
    parameters={"type": "object", "properties": {"p": {"type": "string"}}},
)


def _cfg(tmp_path, rung: str) -> Config:
    c = Config()
    c.attest.audit_log = str(tmp_path / "audit.jsonl")
    c.cache.disk_kv_dir = str(tmp_path / "kv")
    c.tier1.enabled = True
    c.tier1.rung = rung
    return c


# --- occupants --------------------------------------------------------------


class Occ:
    """A model that can leave memory and come back, and remembers whether it has."""

    def __init__(
        self, name: str, *, resident: bool = False, fail_on_load: bool = False
    ):
        self.name = name
        self.resident = resident
        self.fail_on_load = fail_on_load
        self.events: list[str] = []

    async def load(self) -> None:
        self.events.append("load")
        if self.fail_on_load:
            raise RuntimeError("out of memory")
        self.resident = True

    async def unload(self) -> None:
        self.events.append("unload")
        self.resident = False


def _switch(**kw) -> tuple[ResidencySwitch, Occ, Occ]:
    a = Occ("tier0", resident=True)
    b = Occ("tier1-80b")
    return ResidencySwitch(a, b, **kw), a, b


# --- rung 2: residency is exclusive -----------------------------------------


@pytest.mark.asyncio
async def test_admitting_one_model_evicts_the_other():
    """The rung in one line: unified memory does not fit both."""
    switch, tier0, tier1 = _switch()
    async with switch.hold(TIER1):
        assert switch.resident == TIER1
        assert tier0.resident is False
        assert tier1.resident is True
    assert tier0.events == ["unload"]
    assert tier1.events == ["load"]


@pytest.mark.asyncio
async def test_a_tier0_request_waits_for_the_verifier_to_leave_memory():
    """Without this, tier 0 is asked to generate from weights that are not there."""
    switch, _t0, _t1 = _switch()
    order: list[str] = []

    async def verify_call():
        async with switch.hold(TIER1):
            order.append("verify:start")
            await asyncio.sleep(0.02)
            order.append("verify:end")

    async def generate_call():
        await asyncio.sleep(0.005)  # arrive mid-verdict
        async with switch.hold(TIER0):
            order.append("generate")

    await asyncio.gather(verify_call(), generate_call())
    assert order == ["verify:start", "verify:end", "generate"]
    assert switch.resident == TIER0
    assert switch.stats.waits == 1


@pytest.mark.asyncio
async def test_the_swap_back_is_lazy():
    """Restoring tier 0 eagerly adds ~10 s to the tail of every verified turn for a
    model that turn's answer does not need — and throws it away when the next event
    is a second verifier call."""
    switch, tier0, tier1 = _switch()
    async with switch.hold(TIER1):
        pass
    assert switch.resident == TIER1
    # A rerank followed by a review pays for one swap, not two.
    async with switch.hold(TIER1):
        pass
    assert tier0.events == ["unload"]
    assert tier1.events == ["load"]
    assert switch.stats.swaps == 1


@pytest.mark.asyncio
async def test_concurrent_tier0_requests_share_one_residency():
    switch, _t0, _t1 = _switch()

    async def gen():
        async with switch.hold(TIER0):
            await asyncio.sleep(0.01)

    await asyncio.gather(*(gen() for _ in range(4)))
    assert switch.stats.swaps == 0  # tier 0 was already resident


@pytest.mark.asyncio
async def test_a_steady_stream_of_tier0_turns_does_not_starve_the_verifier():
    """Merge quality switching itself off under load is the one failure this product
    cannot have, so a pending swap makes new arrivals for the resident side queue
    behind it. The stall is the honest price of the rung."""
    switch, _t0, _t1 = _switch()
    served = {"tier0": 0, "verified": False}
    stop = asyncio.Event()

    async def tier0_load():
        while not stop.is_set():
            async with switch.hold(TIER0):
                served["tier0"] += 1
                await asyncio.sleep(0.001)

    async def verdict():
        await asyncio.sleep(0.02)
        async with switch.hold(TIER1):
            served["verified"] = True
        stop.set()

    load = [asyncio.create_task(tier0_load()) for _ in range(3)]
    await asyncio.wait_for(verdict(), timeout=5)
    for t in load:
        t.cancel()
    await asyncio.gather(*load, return_exceptions=True)

    assert served["verified"] is True
    assert served["tier0"] > 0  # the load was real, not a starved-out no-op


@pytest.mark.asyncio
async def test_a_failed_swap_leaves_nothing_claimed_as_resident():
    """Reporting the evicted model as resident is the exact lie the switch exists to
    prevent: the next request would generate from weights that have been freed."""
    tier0, tier1 = Occ("tier0", resident=True), Occ("tier1", fail_on_load=True)
    switch = ResidencySwitch(tier0, tier1)
    with pytest.raises(RuntimeError, match="out of memory"):
        async with switch.hold(TIER1):
            pass
    assert switch.resident is None
    assert switch.stats.failed == 1
    # And the switch is not wedged: tier 0 can be admitted again.
    async with switch.hold(TIER0):
        assert switch.resident == TIER0


@pytest.mark.asyncio
async def test_a_failed_swap_does_not_leave_waiters_hanging():
    tier0, tier1 = Occ("tier0", resident=True), Occ("tier1", fail_on_load=True)
    switch = ResidencySwitch(tier0, tier1)

    async def failing():
        with pytest.raises(RuntimeError):
            async with switch.hold(TIER1):
                pass

    async def waiter():
        await asyncio.sleep(0.005)
        async with switch.hold(TIER0):
            return True

    _f, ok = await asyncio.wait_for(asyncio.gather(failing(), waiter()), timeout=5)
    assert ok is True


def test_an_occupant_that_cannot_leave_memory_is_rejected():
    """A no-op unload would make this rung's whole cost model a fiction."""
    with pytest.raises(ValueError, match="has no load"):
        ResidencySwitch(Occ("tier0"), object())


# --- rung 2: the cost is measured, not assumed ------------------------------


@pytest.mark.asyncio
async def test_the_budget_guard_does_not_fire_before_it_has_a_measurement():
    """Declining on a guess would disable verification for a cost nobody observed."""
    switch, _t0, _t1 = _switch()
    backend = ResidentSwapBackend(MockBackend(tier=1), switch, budget_s=0.001)
    result = await backend.generate(
        GenRequest(messages=[Message(role=Role.USER, content="x")])
    )
    assert result.text
    assert switch.stats.declined == 0


@pytest.mark.asyncio
async def test_an_over_budget_swap_declines_and_degrades_to_a_failed_verdict():
    switch, _t0, _t1 = _switch()
    backend = ResidentSwapBackend(MockBackend(tier=1), switch, budget_s=1.0)
    # A round trip both ways, measured as expensive.
    switch.stats.record(TIER1, 12.0)
    switch.stats.record(TIER0, 11.0)

    with pytest.raises(SwapBudgetExceeded):
        await backend.generate(
            GenRequest(messages=[Message(role=Role.USER, content="x")])
        )

    # The ladder's contract: the turn survives it.
    verdict = await Tier1Verifier(backend).rerank(
        [Candidate(0, "a"), Candidate(1, "b")], "context"
    )
    assert verdict.ok is False
    assert "budget" in verdict.error
    assert switch.stats.declined == 2


@pytest.mark.asyncio
async def test_a_zero_budget_never_declines():
    switch, _t0, _t1 = _switch()
    backend = ResidentSwapBackend(MockBackend(tier=1), switch, budget_s=0.0)
    switch.stats.record(TIER1, 999.0)
    switch.stats.record(TIER0, 999.0)
    assert await backend.generate(
        GenRequest(messages=[Message(role=Role.USER, content="x")])
    )


@pytest.mark.asyncio
async def test_swap_stats_expose_the_thrash():
    """This rung's failure mode is alternating swaps, and thrash is invisible unless
    it is counted."""
    switch, _t0, _t1 = _switch()
    for _ in range(3):
        async with switch.hold(TIER1):
            pass
        async with switch.hold(TIER0):
            pass
    stats = ResidentSwapBackend(MockBackend(tier=1), switch).stats()
    assert stats["rung"] == RESIDENT_SWAP_RUNG
    assert stats["swaps"] == 6
    assert stats["resident"] == TIER0
    # Both legs measured, so the budget guard has a round trip to read. The mock's
    # swap is instant, so this is the timer working rather than a plausible cost.
    assert switch.stats.last_s[TIER0] > 0 and switch.stats.last_s[TIER1] > 0


# --- rung 2: attestation ----------------------------------------------------


@pytest.mark.asyncio
async def test_the_verdict_attests_to_the_80b_and_to_no_adapter(tmp_path):
    cfg = _cfg(tmp_path, RESIDENT_SWAP_RUNG)
    tier0 = build_tier0(cfg)
    tier1 = build_tier1(cfg, tier0)
    assert isinstance(tier1, ResidentSwapBackend)
    # The 80B is what ran, so it is what the receipt names — and it is a different
    # container from tier 0's, or the rung would be indistinguishable from rung 3.
    assert tier1.container_hash() != tier0.container_hash()
    assert tier1.adapter_hash("a1") is None
    assert tier1.mounted_adapters() == ()
    assert tier1.supports_state() is False


@pytest.mark.asyncio
async def test_the_receipt_names_rung_2(tmp_path):
    cfg = _cfg(tmp_path, RESIDENT_SWAP_RUNG)
    tier0 = build_tier0(cfg)
    pipeline = Pipeline(cfg, tier0, build_tier1(cfg, tier0))
    result, _trace = await pipeline.run(
        GenRequest(
            messages=[Message(role=Role.USER, content="Fix the retry loop")],
            tools=(EDIT,),
        )
    )
    assert result.receipt["tier1"]["rung"] == RESIDENT_SWAP_RUNG


# --- rung 2: the guard on tier 0 --------------------------------------------


@pytest.mark.asyncio
async def test_the_pipeline_puts_tier_0_behind_the_switch(tmp_path):
    """The guard is not optional: an unguarded tier 0 will happily be asked to
    generate while its weights are out of memory."""
    cfg = _cfg(tmp_path, RESIDENT_SWAP_RUNG)
    tier0 = build_tier0(cfg)
    pipeline = Pipeline(cfg, tier0, build_tier1(cfg, tier0))
    assert isinstance(pipeline.tier0, SwapGuard)
    assert pipeline.tier0.wrapped is tier0
    assert pipeline.cascade.tier0 is pipeline.tier0


@pytest.mark.asyncio
async def test_the_guard_holds_residency_for_the_whole_stream(tmp_path):
    """A swap mid-decode would evict the weights the decode is running on."""
    switch, _t0, _t1 = _switch()
    guard = SwapGuard(MockBackend(use_tools=False), switch)
    seen: list[str] = []
    async for delta in guard.stream(
        GenRequest(messages=[Message(role=Role.USER, content="hi")])
    ):
        seen.append(switch.resident or "none")
    assert seen and set(seen) == {TIER0}


def test_the_guard_changes_when_tier_0_runs_and_nothing_about_what_it_is():
    switch, _t0, _t1 = _switch()
    tier0 = MockBackend(adapters=("a1",))
    guard = SwapGuard(tier0, switch)
    assert guard.name == tier0.name
    assert guard.container_hash() == tier0.container_hash()
    assert guard.adapter_hash("a1") == tier0.adapter_hash("a1")
    assert guard.profile_hash("a1") == tier0.profile_hash("a1")
    assert guard.mounted_adapters() == tier0.mounted_adapters()
    assert guard.state_key("a1") == tier0.state_key("a1")
    assert guard.supports_state() == tier0.supports_state()
    assert guard.count_tokens("abcdefgh") == tier0.count_tokens("abcdefgh")


def test_a_delegating_render_is_not_mistaken_for_a_chat_template(tmp_path):
    """The trap: `SwapGuard.render` overrides the method while changing none of the
    bytes. A type check reads that as a real chat template and silently drops
    replay-aware rendering (sec 8.5.5) — a cache-key bug whose only symptom is a
    lower hit rate."""
    switch, _t0, _t1 = _switch()
    tier0 = MockBackend()
    assert SwapGuard(tier0, switch).renders_canonically() is tier0.renders_canonically()

    cfg = _cfg(tmp_path, RESIDENT_SWAP_RUNG)
    pipeline = Pipeline(cfg, build_tier0(cfg), None)
    req = GenRequest(messages=[Message(role=Role.USER, content="hi")])
    plain = pipeline.render(req)
    pipeline.tier0 = SwapGuard(pipeline.tier0, switch)
    assert pipeline.render(req) == plain


# --- rung 2: construction ---------------------------------------------------


def test_rung_2_without_a_tier0_backend_is_an_error(tmp_path):
    cfg = _cfg(tmp_path, RESIDENT_SWAP_RUNG)
    with pytest.raises(ValueError, match="needs the tier-0 backend"):
        build_tier1(cfg, None)


def test_rung_2_says_what_is_missing_on_a_real_backend(tmp_path):
    """The residency policy is built; the MLX occupants are not. Saying so beats
    building a swap with nothing to swap in."""
    from orbit.backends.base import BackendUnavailable

    cfg = _cfg(tmp_path, RESIDENT_SWAP_RUNG)
    cfg.backend = "mlx"
    with pytest.raises(BackendUnavailable, match="rung 2"):
        build_tier1(cfg, MockBackend())


# --- rung 4: the gates in front of it ---------------------------------------


def test_the_ladder_never_falls_to_the_remote_rung(tmp_path):
    """Nothing degrades into rung 4. Every other rung, asked for, returns itself."""
    for rung in (STREAMED_RUNG, SECOND_OPINION_RUNG, RESIDENT_SWAP_RUNG):
        cfg = _cfg(tmp_path, rung)
        built = build_tier1(cfg, build_tier0(cfg))
        assert not isinstance(built, RemoteTier1Backend)


def test_naming_the_rung_is_not_enough(tmp_path):
    cfg = _cfg(tmp_path, REMOTE_RUNG)
    cfg.tier1.remote_endpoint = "https://api.example.com/v1"
    with pytest.raises(RemoteConsentMissing, match="explicit choice"):
        build_tier1(cfg, build_tier0(cfg))


def test_a_nearly_right_consent_string_is_not_consent(tmp_path):
    """Fuzzy matching here would be a fuzzy consent gate. The entire point of the
    sentence is that it cannot be arrived at inattentively."""
    cfg = _cfg(tmp_path, REMOTE_RUNG)
    cfg.tier1.remote_endpoint = "https://api.example.com/v1"
    cfg.tier1.remote_consent = "tier 1 may leave this machine"
    with pytest.raises(RemoteConsentMissing):
        build_tier1(cfg, build_tier0(cfg))


def test_consent_tolerates_case_and_surrounding_whitespace(tmp_path):
    cfg = _cfg(tmp_path, REMOTE_RUNG)
    cfg.tier1.remote_endpoint = "https://api.example.com/v1"
    cfg.tier1.remote_consent = f"  {CONSENT.upper()}  "
    cfg.tier1.remote_api_key_env = ""
    backend = build_tier1(cfg, build_tier0(cfg))
    assert isinstance(backend, RemoteTier1Backend)


def test_the_endpoint_label_is_the_host_and_nothing_else(tmp_path):
    """It ends up in stats() and in `orbit doctor`; a path or a query string is how
    a token ends up in a support bundle."""
    cfg = _cfg(tmp_path, REMOTE_RUNG)
    cfg.tier1.remote_endpoint = "https://api.example.com/v1/secret?key=abc123"
    cfg.tier1.remote_consent = CONSENT
    cfg.tier1.remote_api_key_env = ""
    backend = build_tier1(cfg, build_tier0(cfg))
    assert backend.stats()["endpoint"] == "api.example.com"
    assert "abc123" not in json.dumps(backend.stats())


def test_the_key_is_read_from_the_environment_not_the_config(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, REMOTE_RUNG)
    cfg.tier1.remote_endpoint = "https://api.example.com/v1"
    cfg.tier1.remote_consent = CONSENT
    monkeypatch.delenv("ORBIT_REMOTE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ORBIT_REMOTE_API_KEY is not set"):
        build_tier1(cfg, build_tier0(cfg))
    monkeypatch.setenv("ORBIT_REMOTE_API_KEY", "sk-test")
    assert isinstance(build_tier1(cfg, build_tier0(cfg)), RemoteTier1Backend)


def test_a_missing_transport_file_names_the_one_that_ships(tmp_path):
    cfg = _cfg(tmp_path, REMOTE_RUNG)
    cfg.tier1.remote_endpoint = "https://api.example.com/v1"
    cfg.tier1.remote_consent = CONSENT
    cfg.tier1.remote_transport = str(tmp_path / "nope.py")
    with pytest.raises(ValueError, match=r"tools/remote_tier1\.py"):
        build_tier1(cfg, build_tier0(cfg))


def test_the_shipped_transport_is_outside_the_package():
    """Same reason as the A2 exporter: the offline claim has to be structural."""
    from pathlib import Path

    import orbit

    path = Path(Config().tier1.remote_transport)
    assert path.is_file()
    assert Path(orbit.__file__).resolve().parent not in path.resolve().parents


# --- rung 4: what it does and does not attest to ----------------------------


def _canned(body: dict) -> RemoteTier1Backend:
    async def transport(_payload):
        return body

    return RemoteTier1Backend(
        transport, model="remote-m", endpoint_label="api.example.com"
    )


@pytest.mark.asyncio
async def test_a_remote_verdict_has_no_container_attestation():
    """None by construction, not by omission: you cannot attest to a model you do
    not hold. The rung in the receipt is what says the null is a property."""
    backend = _canned(
        {"choices": [{"message": {"content": '{"choice":1,"reason":"r"}'}}]}
    )
    assert backend.container_hash() is None
    assert backend.adapter_hash("a1") is None
    assert backend.mounted_adapters() == ()
    assert backend.supports_state() is False


@pytest.mark.asyncio
async def test_a_remote_verdict_is_validated_on_this_side_of_the_boundary():
    """'The endpoint supports constrained decoding' is a claim about a machine we do
    not control and cannot check (sec 5.2)."""
    backend = _canned(
        {"choices": [{"message": {"content": '{"reason":"no choice field"}'}}]}
    )
    with pytest.raises(Tier1Unavailable, match="missing required fields"):
        await backend.generate(
            GenRequest(
                messages=[Message(role=Role.USER, content="x")],
                json_schema=rerank_schema(2),
            )
        )


@pytest.mark.asyncio
async def test_a_transport_failure_degrades_to_a_failed_verdict():
    async def transport(_payload):
        raise ConnectionError("no route to host")

    backend = RemoteTier1Backend(transport, model="m")
    verdict = await Tier1Verifier(backend).review("diff", "context")
    assert verdict.ok is False
    assert "no route to host" in verdict.error


@pytest.mark.asyncio
async def test_a_transport_that_returns_nonsense_is_rejected():
    async def transport(_payload):
        return "not a body"

    with pytest.raises(Tier1Unavailable, match="JSON object"):
        await RemoteTier1Backend(transport, model="m").generate(
            GenRequest(messages=[Message(role=Role.USER, content="x")])
        )


def test_the_backend_holds_no_transport_of_its_own():
    with pytest.raises(ValueError, match=r"tools/remote_tier1\.py"):
        RemoteTier1Backend(None, model="m")


@pytest.mark.asyncio
async def test_the_remote_rung_serves_a_real_rerank():
    backend = _canned(
        {
            "choices": [{"message": {"content": '{"choice":1,"reason":"cleaner"}'}}],
            "usage": {"prompt_tokens": 8000, "completion_tokens": 12},
        }
    )
    verdict = await Tier1Verifier(backend).rerank(
        [Candidate(0, "a"), Candidate(1, "b")], "context"
    )
    assert verdict.ok
    assert verdict.data["choice"] == 1


# --- rung 4: the offline claim ----------------------------------------------


def test_a_configured_remote_rung_fails_the_offline_posture():
    """A configuration fact, not an observation: lsof only sees a call that already
    happened, and the posture is wrong from the moment the rung is armed."""
    clean = OfflineReport(loopback_only=True, env_ok=True)
    assert clean.ok is True
    armed = OfflineReport(loopback_only=True, env_ok=True, remote_tier1=True)
    assert armed.ok is False
    assert armed.as_dict()["remote_tier1"] is True


def test_verify_reports_why_the_posture_fails():
    report = verify(remote_tier1=True)
    assert report.ok is False
    assert "rung 4" in report.note


def test_doctor_reports_the_rung_it_would_use(tmp_path, capsys):
    from orbit.cli import main

    cfg_path = tmp_path / "orbit.toml"
    cfg_path.write_text(
        "backend = 'mock'\n\n[tier1]\nenabled = true\nrung = 'remote'\n"
        f"remote_endpoint = 'https://api.example.com/v1'\nremote_consent = '{CONSENT}'\n"
        "remote_api_key_env = ''\n",
        encoding="utf-8",
    )
    main(["--config", str(cfg_path), "doctor"])
    out = json.loads(capsys.readouterr().out)
    assert out["offline"]["ok"] is False
    assert out["offline"]["remote_tier1"] is True
    assert out["tier1"]["rung"] == REMOTE_RUNG
    assert out["tier1"]["container_hash"] is None
    assert "leave this machine" in out["tier1"]["note"]


# --- the clamp, now that it has two transports ------------------------------


def test_every_call_type_is_clamped_to_its_own_budget():
    """The promise that tier 1 stays a verifier (sec 5.1). Until both transports
    shared this code it lived only in a file that has never been executed."""
    for call, budget in CALL_BUDGETS.items():
        req = GenRequest(
            messages=[Message(role=Role.USER, content="x")],
            json_schema={"title": call, "type": "object"},
            sampling=Sampling(max_tokens=4096),
        )
        assert clamp_max_tokens(req) == budget


def test_an_unrecognised_call_does_not_inherit_the_largest_ceiling():
    req = GenRequest(
        messages=[Message(role=Role.USER, content="x")],
        json_schema={"title": "write_the_patch", "type": "object"},
        sampling=Sampling(max_tokens=4096),
    )
    assert clamp_max_tokens(req) == CALL_BUDGETS["review"]


def test_a_smaller_request_is_not_inflated_to_the_budget():
    req = GenRequest(
        messages=[Message(role=Role.USER, content="x")],
        json_schema=rerank_schema(3),
        sampling=Sampling(max_tokens=16),
    )
    assert clamp_max_tokens(req) == 16


def test_the_payload_is_greedy_and_schema_constrained():
    """A judgement is not a sample: two runs of the same rerank must agree for the
    receipt's determinism claim to mean anything."""
    payload = build_payload(
        GenRequest(
            system="judge",
            messages=[Message(role=Role.USER, content="candidates")],
            json_schema=REVIEW,
            sampling=Sampling(temperature=0.9, top_p=0.8, seed=7),
        ),
        model="m",
    )
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["seed"] == 7
    assert payload["stream"] is False
    assert payload["response_format"]["json_schema"]["schema"] is REVIEW
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "text,match",
    [
        ("{not json", "non-JSON"),
        ('["choice"]', "not an object"),
        ('{"reason":"x"}', "missing required fields"),
    ],
)
def test_an_unparseable_judgement_is_refused_rather_than_coerced(text, match):
    with pytest.raises(Tier1Unavailable, match=match):
        validate_or_raise(text, rerank_schema(2))


# --- the ladder as a whole --------------------------------------------------


def test_every_rung_is_reachable_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBIT_REMOTE_API_KEY", "sk-test")
    built: dict[str, Backend] = {}
    for rung in (STREAMED_RUNG, RESIDENT_SWAP_RUNG, SECOND_OPINION_RUNG, REMOTE_RUNG):
        cfg = _cfg(tmp_path, rung)
        cfg.tier1.remote_endpoint = "https://api.example.com/v1"
        cfg.tier1.remote_consent = CONSENT
        backend = build_tier1(cfg, build_tier0(cfg))
        assert backend is not None and backend.tier == 1
        built[rung] = backend
    assert len({type(b) for b in built.values()}) == 4


def test_an_unknown_rung_lists_the_ones_that_exist(tmp_path):
    cfg = _cfg(tmp_path, "rung_five")
    with pytest.raises(ValueError, match=r"unknown tier1\.rung"):
        build_tier1(cfg, build_tier0(cfg))
