"""Gate thresholds, and the shape of a judgement against one (spec sec 11, 7.3).

A threshold has two values, always.

`SPEC_*` below is what the specification asks for. It does not vary by host and is
not configurable. The *effective* target is what a particular machine is judged
against, and an operator may relax it (`[gates]` in `tandem.toml`) so a host that
cannot meet the spec still runs end to end and reports where it is weak instead of
stopping at a red gate.

**Both travel with every result.** `judge()` emits the effective target, the spec
target, `pass` against the first and `meets_spec` against the second. A gate that
passes against a relaxed floor reports `pass: true` and `meets_spec: false`, and
names both numbers. That is the entire point of this module: the failure mode it
exists to prevent is a green gate being read as a met specification, six weeks later,
by someone who was not there when the floor was lowered. There is deliberately no way
to express "relaxed" that does not also publish the spec value it departed from.

This is a leaf — it imports nothing from the package. `config`, `eval.latency` and
`backends.mlx_tier1` all read their defaults from here so that the spec numbers exist
in exactly one place; a default that lived in both the config dataclass and the gate
function would drift, and the drift would be invisible.
"""

from __future__ import annotations

from typing import Any

# --- M0 Gate A (sec 11): can tier 0 serve interactively at all? ---------------
SPEC_GATE_A_TTFT_S = 5.0
SPEC_GATE_A_DECODE_TOK_PER_S = 30.0
SPEC_GATE_A_TOOLCALL_FAILURE_RATE = 0.05

# --- M0 Gate B (sec 11, 5.3): is the engine doing batch-union prefill? --------
# Below this the streamed tier is reading each expert per token rather than once per
# chunk, which is the difference the whole tier-1 thesis rests on.
SPEC_GATE_B_PREFILL_TOK_PER_S = 200.0

# --- sec 7.3 chat latency contract -------------------------------------------
SPEC_CHAT_TTFT_S = 2.0
SPEC_CHAT_DECODE_TOK_PER_S = 40.0


def judge(
    measured: float,
    *,
    target: float,
    spec_target: float,
    higher_is_better: bool,
    digits: int = 2,
) -> dict[str, Any]:
    """One threshold comparison, reported against both the host and the spec.

    `budget` is the effective target and is what `pass` was decided by; `spec_budget`
    is what sec 11 asks for and is what `meets_spec` was decided by. When the two are
    equal `relaxed` is False and the two verdicts always agree, which is the ordinary
    case and reads as an ordinary gate result.
    """
    ok = measured >= target if higher_is_better else measured <= target
    spec_ok = measured >= spec_target if higher_is_better else measured <= spec_target
    return {
        "worst": round(measured, digits),
        "budget": target,
        "spec_budget": spec_target,
        "pass": ok,
        "meets_spec": spec_ok,
        "relaxed": target != spec_target,
    }


def relaxations(report: dict[str, Any]) -> list[str]:
    """Names of the criteria in `report` that passed only because they were relaxed.

    Returned so a caller can say so in one line rather than expecting a reader to
    compare `budget` against `spec_budget` in every row of a nested dict. An empty
    list means every green criterion is green against the spec.
    """
    return sorted(
        name
        for name, row in report.items()
        if isinstance(row, dict)
        and row.get("relaxed")
        and row.get("pass")
        and not row.get("meets_spec")
    )
