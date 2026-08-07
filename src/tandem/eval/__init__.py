"""Evaluation (spec sec 10).

Benchmarks are the wrong instrument. The thesis is about merge quality, so
`merge_eval` is primary and everything else is a regression detector.
"""

from . import gates, latency, merge_eval, regression
from .regression import Baseline, RegressionReport
from .regression_items import SUITE as REGRESSION_SUITE
from .gates import GateResult, adapter_isolation_gate, g1_backend_equivalence, g2_placement_invariance, toolcall_gate
from .latency import Environment, LatencyReport, m0_gate_a, measure
from .merge_eval import Arm, EvalCase, MergeEvalReport, compare_arms

__all__ = [
    "gates",
    "latency",
    "merge_eval",
    "regression",
    "Baseline",
    "RegressionReport",
    "REGRESSION_SUITE",
    "GateResult",
    "adapter_isolation_gate",
    "g1_backend_equivalence",
    "g2_placement_invariance",
    "toolcall_gate",
    "Environment",
    "LatencyReport",
    "m0_gate_a",
    "measure",
    "Arm",
    "EvalCase",
    "MergeEvalReport",
    "compare_arms",
]
