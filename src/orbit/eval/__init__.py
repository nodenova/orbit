"""Evaluation (spec sec 10).

Benchmarks are the wrong instrument. The thesis is about merge quality, so
`merge_eval` is primary and everything else is a regression detector.
"""

from orbit.eval import gates, latency, merge_eval, regression
from orbit.eval.gates import (
    GateResult,
    adapter_isolation_gate,
    g1_backend_equivalence,
    g2_placement_invariance,
    toolcall_gate,
)
from orbit.eval.latency import Environment, LatencyReport, m0_gate_a, measure
from orbit.eval.merge_eval import Arm, EvalCase, MergeEvalReport, compare_arms
from orbit.eval.regression import Baseline, RegressionReport
from orbit.eval.regression_items import SUITE as REGRESSION_SUITE

__all__ = [
    "REGRESSION_SUITE",
    "Arm",
    "Baseline",
    "Environment",
    "EvalCase",
    "GateResult",
    "LatencyReport",
    "MergeEvalReport",
    "RegressionReport",
    "adapter_isolation_gate",
    "compare_arms",
    "g1_backend_equivalence",
    "g2_placement_invariance",
    "gates",
    "latency",
    "m0_gate_a",
    "measure",
    "merge_eval",
    "regression",
    "toolcall_gate",
]
