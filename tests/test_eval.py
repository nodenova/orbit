"""Evaluation and blocking gates (spec sec 4.2, 9.3, 10)."""

from __future__ import annotations

import platform

import pytest

from tandem.backends.mock import Fault, MockBackend
from tandem.config import Config
from tandem.eval.gates import (
    adapter_isolation_gate,
    g1_backend_equivalence,
    g2_placement_invariance,
    toolcall_gate,
)
from tandem.eval.latency import Environment, LatencySample, m0_gate_a, measure
from tandem.eval.merge_eval import (
    Arm,
    ArmSummary,
    EvalCase,
    blast_radius,
    changed_line_count,
    compare_arms,
    diff_proximity,
    run,
    scored_review_proxy,
    summarise,
    tier1_review_proxy,
    touched_files,
)
from tandem.eval.worktree import PatchOutcome
from tandem.gateway.pipeline import Pipeline
from tandem.tier1.verifier import Tier1Verifier
from tandem.types import GenRequest, ToolDef

REF_DIFF = (
    "diff --git a/src/app.py b/src/app.py\n"
    "--- a/src/app.py\n+++ b/src/app.py\n"
    "@@ -1,3 +1,3 @@\n-    retries = 3\n+    retries = 5\n"
)
SAME_EDIT_DIFFERENT_OFFSET = (
    "diff --git a/src/app.py b/src/app.py\n"
    "--- a/src/app.py\n+++ b/src/app.py\n"
    "@@ -90,3 +90,3 @@\n-    retries = 3\n+    retries = 5\n"
)
BROADER_DIFF = REF_DIFF + (
    "diff --git a/src/other.py b/src/other.py\n"
    "--- a/src/other.py\n+++ b/src/other.py\n"
    "@@ -1,2 +1,2 @@\n-x = 1\n+x = 2\n"
)


# --- merge-eval metrics (sec 10.1) ------------------------------------------


def test_touched_files_and_line_counts():
    assert touched_files(BROADER_DIFF) == {"src/app.py", "src/other.py"}
    assert changed_line_count(REF_DIFF) == 2


def test_diff_proximity_is_offset_insensitive():
    """The same edit at a different line is the same patch."""
    assert diff_proximity(SAME_EDIT_DIFFERENT_OFFSET, REF_DIFF) == 1.0
    assert diff_proximity(REF_DIFF, REF_DIFF) == 1.0


def test_diff_proximity_falls_with_divergence():
    other = REF_DIFF.replace("retries = 5", "retries = 99")
    assert 0.0 < diff_proximity(other, REF_DIFF) < 1.0
    assert diff_proximity("", REF_DIFF) == 0.0


def test_blast_radius_is_a_ratio_to_the_merged_diff():
    files, lines = blast_radius(REF_DIFF, REF_DIFF)
    assert files == 1.0 and lines == 1.0
    files, lines = blast_radius(BROADER_DIFF, REF_DIFF)
    assert files == 2.0 and lines == 2.0


def test_blast_accuracy_penalises_both_directions():
    """A patch touching twice as much and one touching half must not cancel."""
    from tandem.eval.merge_eval import _blast_accuracy

    exact = ArmSummary(arm="x", mean_blast_files=1.0, mean_blast_lines=1.0)
    broad = ArmSummary(arm="y", mean_blast_files=2.0, mean_blast_lines=2.0)
    narrow = ArmSummary(arm="z", mean_blast_files=0.5, mean_blast_lines=0.5)
    assert _blast_accuracy(exact) > _blast_accuracy(broad)
    assert _blast_accuracy(exact) > _blast_accuracy(narrow)


def test_m3_gate_needs_three_of_five_metrics():
    base = ArmSummary(
        arm="base", n=10, test_pass_rate=0.4, convention_rate=0.5,
        mean_proximity=0.30, mean_blast_files=2.0, mean_blast_lines=2.0,
        mean_review_proxy=0.6,
    )
    better = ArmSummary(
        arm="+A1", n=10, test_pass_rate=0.6, convention_rate=0.7,
        mean_proximity=0.55, mean_blast_files=1.1, mean_blast_lines=1.1,
        mean_review_proxy=0.3,
    )
    result = compare_arms(base, better)
    assert result["pass"]
    assert len(result["wins"]) >= 3

    worse = ArmSummary(
        arm="+A1", n=10, test_pass_rate=0.2, convention_rate=0.3,
        mean_proximity=0.10, mean_blast_files=4.0, mean_blast_lines=4.0,
        mean_review_proxy=0.9,
    )
    assert not compare_arms(base, worse)["pass"]


def test_m3_gate_says_so_when_metrics_are_unmeasured():
    """A pass on two measurable metrics is not the M3 gate."""
    base = ArmSummary(arm="base", n=5, mean_proximity=0.2, mean_blast_files=2.0,
                      mean_blast_lines=2.0)
    cand = ArmSummary(arm="+A1", n=5, mean_proximity=0.5, mean_blast_files=1.0,
                      mean_blast_lines=1.0)
    result = compare_arms(base, cand)
    assert not result["pass"]
    assert "test_pass_rate" in result["unmeasured"]
    assert "Wire up the test hook" in result["note"]


@pytest.mark.asyncio
async def test_merge_eval_runs_four_bars():
    cases = [EvalCase(sha=f"s{i}", prompt=f"task {i}", reference_diff=REF_DIFF) for i in range(4)]

    async def gen_base(_req):
        return ""

    async def gen_a1(_req):
        return REF_DIFF

    report = await run(
        cases,
        [Arm(name="tier0 base", generate=gen_base), Arm(name="tier0 + A1", generate=gen_a1)],
    )
    assert report.n_cases == 4
    assert [a.arm for a in report.arms] == ["tier0 base", "tier0 + A1"]
    assert report.arms[1].mean_proximity == 1.0
    assert report.arms[0].mean_proximity == 0.0
    assert "tier0 base" in report.table()


@pytest.mark.asyncio
async def test_a_failing_case_does_not_void_the_run():
    calls = {"n": 0}

    async def flaky(_req):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("backend hiccup")
        return REF_DIFF

    cases = [EvalCase(sha=f"s{i}", prompt="t", reference_diff=REF_DIFF) for i in range(3)]
    report = await run(cases, [Arm(name="arm", generate=flaky)])
    assert report.arms[0].n == 2
    assert report.arms[0].errors == 1


def test_summarise_handles_an_empty_arm():
    assert summarise("empty", []).n == 0


@pytest.mark.asyncio
async def test_the_patch_is_scored_not_the_reply_it_arrived_in():
    """Prose around a fenced diff is not part of the patch (sec 10.1)."""

    async def chatty(_req):
        return f"I kept the existing helper.\n\n```diff\n{REF_DIFF}```\n\nLet me know."

    report = await run(
        [EvalCase(sha="s0", prompt="t", reference_diff=REF_DIFF)],
        [Arm(name="arm", generate=chatty)],
    )
    assert report.arms[0].mean_proximity == 1.0


@pytest.mark.asyncio
async def test_the_worktree_metrics_reach_the_arm_summary(tmp_path):
    """Two of the five M3 metrics only exist once a runner is handed in."""

    class StubRunner:
        repo = "/repo"
        measures_tests = True
        measures_lint = True
        seen: list[str] = []

        async def evaluate(self, diff, *, base_rev=""):
            self.seen.append(base_rev)
            return PatchOutcome(applied=True, lint_clean=True, tests_passed=True)

    runner = StubRunner()
    cases = [EvalCase(sha="abc123", prompt="t", reference_diff=REF_DIFF)]
    report = await run(cases, [Arm(name="arm", generate=_returns(REF_DIFF))], runner=runner)

    assert report.arms[0].test_pass_rate == 1.0
    assert report.arms[0].convention_rate == 1.0
    assert report.arms[0].apply_rate == 1.0
    assert report.measured == {"tests": True, "lint": True, "review_proxy": False}
    # Applied against the change's own parent, not against whatever HEAD happens
    # to be — the tree the merged change was actually written against.
    assert runner.seen == ["abc123^1"]


@pytest.mark.asyncio
async def test_all_five_metrics_measured_makes_the_gate_answerable():
    """The whole point of item 2: `compare_arms` can return a verdict at all."""

    class AlwaysGood:
        repo = "/repo"
        measures_tests = True
        measures_lint = True

        async def evaluate(self, diff, *, base_rev=""):
            good = bool(diff.strip())
            return PatchOutcome(applied=good, lint_clean=good, tests_passed=good)

    async def scored(case, diff):
        return 0.1 if diff.strip() else 0.9

    cases = [EvalCase(sha=f"s{i}", prompt="t", reference_diff=REF_DIFF) for i in range(3)]
    report = await run(
        cases,
        [
            Arm(name="tier0 base", generate=_returns("")),
            Arm(name="tier0 + A1", generate=_returns(REF_DIFF)),
        ],
        runner=AlwaysGood(),
        review_proxy=scored,
    )
    comparison = report.comparisons[0]
    assert comparison["unmeasured"] == []
    assert comparison["measured_metrics"] == 5
    assert comparison["pass"]


@pytest.mark.asyncio
async def test_a_review_proxy_may_leave_a_case_unscored():
    """A human pass that covered half the cases has measured half of them."""
    proxy = scored_review_proxy({"s0": 0.2})
    cases = [EvalCase(sha="s0", prompt="t", reference_diff=REF_DIFF),
             EvalCase(sha="s1", prompt="t", reference_diff=REF_DIFF)]
    report = await run(cases, [Arm(name="arm", generate=_returns(REF_DIFF))], review_proxy=proxy)
    assert report.arms[0].mean_review_proxy == 0.2


@pytest.mark.asyncio
async def test_tier1_review_proxy_scores_a_verdict():
    verifier = Tier1Verifier(MockBackend(tier=1, use_tools=False))
    proxy = tier1_review_proxy(verifier)
    score = await proxy(EvalCase(sha="s0", prompt="t", reference_diff=REF_DIFF), REF_DIFF)
    assert 0.0 <= score <= 1.0
    # No patch at all reliably draws a comment, and costs no verifier call.
    assert await proxy(EvalCase(sha="s1", prompt="t", reference_diff=REF_DIFF), "") == 1.0


@pytest.mark.asyncio
async def test_an_unusable_tier1_verdict_is_unmeasured_not_neutral():
    """A verifier declining to answer is not evidence the patch is middling."""
    proxy = tier1_review_proxy(Tier1Verifier(None))
    assert await proxy(EvalCase(sha="s0", prompt="t", reference_diff=REF_DIFF), REF_DIFF) is None


def _returns(text: str):
    async def gen(_req):
        return text

    return gen


# --- tool-call gate (sec 10.2) ----------------------------------------------


@pytest.mark.asyncio
async def test_toolcall_gate_passes_on_wellformed_output(tmp_path):
    cfg = Config()
    cfg.attest.audit_log = str(tmp_path / "audit.jsonl")
    pipeline = Pipeline(cfg, MockBackend())

    async def run_turn(req: GenRequest):
        prepared = pipeline._prepare_sampling(req)
        result, _ = await pipeline.cascade.produce(prepared)
        _r, info = await pipeline._settle_tool_calls(prepared, result)
        return info

    result = await toolcall_gate(run_turn, runs=100)
    assert result.passed
    assert result.detail["rate"] >= 0.99
    assert result.detail["runs"] == 100


@pytest.mark.asyncio
async def test_toolcall_gate_fails_below_the_threshold(tmp_path):
    cfg = Config()
    cfg.attest.audit_log = str(tmp_path / "audit.jsonl")
    # All three defences off, so the gate sees the raw model. With constrained
    # decoding on, a malformed call is unrepresentable and the gate could not fail.
    cfg.toolcall.constrain = False
    cfg.toolcall.repair = False
    cfg.toolcall.max_retries = 0
    pipeline = Pipeline(cfg, MockBackend(fault=Fault.TRUNCATED_JSON))

    async def run_turn(req: GenRequest):
        prepared = pipeline._prepare_sampling(req)
        result, _ = await pipeline.cascade.produce(prepared)
        _r, info = await pipeline._settle_tool_calls(prepared, result)
        return info

    result = await toolcall_gate(run_turn, runs=100)
    assert not result.passed
    assert "blocking" in result.reason


@pytest.mark.asyncio
async def test_toolcall_gate_requires_the_minimum_run_count(tmp_path):
    """>=99% over >=100 runs. Ten runs is not the gate."""
    cfg = Config()
    cfg.attest.audit_log = str(tmp_path / "audit.jsonl")
    pipeline = Pipeline(cfg, MockBackend())

    async def run_turn(req: GenRequest):
        prepared = pipeline._prepare_sampling(req)
        result, _ = await pipeline.cascade.produce(prepared)
        _r, info = await pipeline._settle_tool_calls(prepared, result)
        return info

    assert not (await toolcall_gate(run_turn, runs=10)).passed


@pytest.mark.asyncio
async def test_repair_layer_rescues_the_gate(tmp_path):
    """Repair exists so a garbling model still clears sec 10.2 without prevention.

    This is the install without `[constrain]`: layer 1 is unavailable, and the gate
    has to be carried by layers 3 and 4 alone.
    """
    cfg = Config()
    cfg.attest.audit_log = str(tmp_path / "audit.jsonl")
    cfg.toolcall.constrain = False
    pipeline = Pipeline(cfg, MockBackend(fault=Fault.XML_HYBRID))

    async def run_turn(req: GenRequest):
        prepared = pipeline._prepare_sampling(req)
        result, _ = await pipeline.cascade.produce(prepared)
        _r, info = await pipeline._settle_tool_calls(prepared, result)
        return info

    result = await toolcall_gate(run_turn, runs=100)
    assert result.passed
    assert result.detail["repaired"] > 0


# --- adapter isolation (sec 4.2) --------------------------------------------


@pytest.mark.asyncio
async def test_adapter_isolation_passes_when_deltas_do_not_leak():
    def factory(names):
        return MockBackend(adapters=tuple(names))

    result = await adapter_isolation_gate(factory, ["a0", "a1-repo"])
    assert result.passed
    assert result.detail["comparisons"] > 0


@pytest.mark.asyncio
async def test_adapter_isolation_catches_a_leak():
    """A backend whose output depends on *which other* adapters are mounted."""

    class Leaky(MockBackend):
        async def generate(self, req):
            leaked = req.with_(adapter=f"{req.adapter}|{','.join(self.adapters)}")
            return await MockBackend.generate(self, leaked)

    def factory(names):
        return Leaky(adapters=tuple(names))

    result = await adapter_isolation_gate(factory, ["a0", "a1-repo"])
    assert not result.passed
    assert "leaking" in result.reason


@pytest.mark.asyncio
async def test_isolation_gate_is_vacuously_true_with_no_adapters():
    result = await adapter_isolation_gate(lambda names: MockBackend(), [])
    assert result.passed


# --- determinism gates (sec 9.3) --------------------------------------------


@pytest.mark.asyncio
async def test_g1_passes_for_equivalent_backends():
    result = await g1_backend_equivalence(MockBackend(use_tools=False), MockBackend(use_tools=False))
    assert result.passed


@pytest.mark.asyncio
async def test_g1_catches_a_divergent_kernel():
    result = await g1_backend_equivalence(
        MockBackend(use_tools=False, container="cpu-reference"),
        MockBackend(use_tools=False, container="metal-path"),
    )
    assert not result.passed
    assert "reduction order" in result.reason


@pytest.mark.asyncio
async def test_g2_passes_when_placement_does_not_change_the_model():
    """The most important gate: cache occupancy must not change the answer."""
    result = await g2_placement_invariance(lambda _bytes: MockBackend(tier=1, use_tools=False))
    assert result.passed


@pytest.mark.asyncio
async def test_g2_catches_placement_dependent_output():
    def factory(cache_bytes: int):
        # A backend whose weights differ by cache size — exactly the failure that
        # would invalidate every determinism claim in the receipt.
        return MockBackend(tier=1, use_tools=False, container=f"container@{cache_bytes}")

    result = await g2_placement_invariance(factory)
    assert not result.passed
    assert "invalid until this passes" in result.reason


# --- latency (sec 10.4, M0 Gate A) ------------------------------------------


@pytest.mark.asyncio
async def test_latency_measure_reports_per_frontier():
    samples = await measure(MockBackend(use_tools=False), frontiers=(2_000, 4_000), max_tokens=8)
    assert [s.frontier_tokens for s in samples] == [2_000, 4_000]
    assert all(s.prefill_tok_per_s > 0 for s in samples)


def test_m0_gate_a_thresholds():
    good = [
        LatencySample(2000, 0, False, ttft_s=1.0, total_s=2.0, output_tokens=64,
                      decode_tok_per_s=44.0, prefill_tok_per_s=800.0)
    ]
    assert m0_gate_a(good, toolcall_failure_rate=0.01)["pass"]

    slow = [
        LatencySample(2000, 0, False, ttft_s=6.0, total_s=9.0, output_tokens=64,
                      decode_tok_per_s=20.0, prefill_tok_per_s=300.0)
    ]
    failed = m0_gate_a(slow, toolcall_failure_rate=0.10)
    assert not failed["pass"]
    assert "async review-assist" in failed["kill_condition"]


def test_m0_gate_a_needs_warm_samples():
    cold = [
        LatencySample(2000, 0, True, ttft_s=1.0, total_s=2.0, output_tokens=64,
                      decode_tok_per_s=44.0, prefill_tok_per_s=800.0)
    ]
    assert not m0_gate_a(cold, toolcall_failure_rate=0.0)["pass"]


def test_environment_reports_unknown_as_none_never_zero():
    """Sec 10.5: an undetected hardware fact must be absent, not zero.

    A published report saying `"memory_bandwidth_gb_s": 0.0` is a claim about the
    machine; `null` is the absence of one. Bandwidth is never detected — it is a bin
    fact the operator supplies — so it pins the contract on every platform.
    """
    env = Environment.detect().as_dict()
    assert env["memory_bandwidth_gb_s"] is None
    for key in ("cpu_cores", "gpu_cores", "ram_gb", "ssd_capacity_gb"):
        assert env[key] is None or env[key] > 0, f"{key} reported a falsy non-None"


@pytest.mark.skipif(platform.system() != "Darwin", reason="IORegistry is macOS-only")
def test_environment_detects_apple_silicon_gpu_cores():
    env = Environment.detect()
    assert env.gpu_cores is not None and env.gpu_cores > 0
    assert env.cpu_cores is not None and env.ram_gb is not None
