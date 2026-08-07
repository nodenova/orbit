"""Routing profile (spec sec 6.4).

Selects the top-25% routed experts per layer for the LoRA target set (sec 4.3), and
is reused as the tier-1 pin set.

The finding this rests on: top-25% routed experts is statistically equivalent to all
experts — mean deltas within +/-1 pp, TOST equivalence at +/-2 pp in 5 of 6
model x task conditions, at 70-73% fewer trainable parameters [M, MoE-Sieve]. Random
selection at matched budget is 2-2.5 pp *worse*, and random k=16 underperforms hot
k=8: the routing signal is doing real work. So this file is not an optimisation, it
is the reason the target set is defensible.

Two details that decide whether the profile is right:

**Rank by count for Qwen, by mass for DeepSeek-style.** Count- and mass-ranked top-k
agree at J=0.920 on Qwen but only J=0.646 on fine-grained architectures, where a long
low-weight tail inflates counts [M]. Getting this backwards silently selects the
wrong quarter of the experts.

**A 10% subsample is enough** — it recovers the same top-k sets at mean Jaccard
>= 0.94 [M] — so this is one cheap gradient-free forward pass, not a training run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Architectures whose routing mass and routing count disagree enough that ranking by
# count picks the wrong experts (sec 6.4).
MASS_RANKED_ARCHITECTURES = ("deepseek", "glm", "minimax")


def ranking_mode(model_name: str) -> str:
    low = model_name.lower()
    return "mass" if any(a in low for a in MASS_RANKED_ARCHITECTURES) else "count"


@dataclass
class RoutingProfile:
    """Per-layer per-expert activation statistics, plus the derived top-k sets."""

    model_hash: str = ""
    model_name: str = ""
    corpus_hash: str = ""
    n_tokens: int = 0
    # [n_layers][n_experts]
    counts: list[list[int]] = field(default_factory=list)
    mass: list[list[float]] = field(default_factory=list)
    # percent -> [n_layers][k] expert indices
    topk: dict[str, list[list[int]]] = field(default_factory=dict)
    layer_cv: list[float] = field(default_factory=list)
    cov_at_25: float = 0.0
    rank_by: str = "count"

    @property
    def n_layers(self) -> int:
        return len(self.counts)

    @property
    def n_experts(self) -> int:
        return len(self.counts[0]) if self.counts else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_hash": self.model_hash,
            "model_name": self.model_name,
            "corpus_hash": self.corpus_hash,
            "n_tokens": self.n_tokens,
            "counts": self.counts,
            "mass": self.mass,
            "topk": self.topk,
            "layer_cv": self.layer_cv,
            "cov_at_25": self.cov_at_25,
            "rank_by": self.rank_by,
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), separators=(",", ":")) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> RoutingProfile:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def experts_for_layer(self, layer: int, percent: str = "25") -> list[int]:
        sets = self.topk.get(percent)
        if not sets or layer >= len(sets):
            return []
        return sets[layer]

    def is_target(self, layer: int, expert: int, percent: str = "25") -> bool:
        return expert in set(self.experts_for_layer(layer, percent))


def top_k_indices(scores: Sequence[float], k: int) -> list[int]:
    """Indices of the k highest scores, ties broken by index for reproducibility."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(order[:k])


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Routing skew for one layer.

    A layer with CV near zero routes uniformly and has no hot quarter worth
    selecting; a high CV is where the top-25% claim has teeth. Reported per layer so
    a profile that is flat everywhere is visible rather than silently useless.
    """
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(var) / mean


def build(
    counts: list[list[int]],
    mass: list[list[float]] | None = None,
    *,
    model_name: str = "",
    model_hash: str = "",
    corpus_hash: str = "",
    n_tokens: int = 0,
    percents: Sequence[float] = (25.0, 12.5),
) -> RoutingProfile:
    """Derive top-k sets and skew statistics from raw activation counts."""
    if not counts:
        raise ValueError("routing profile needs at least one layer of counts")
    n_experts = len(counts[0])
    if any(len(layer) != n_experts for layer in counts):
        raise ValueError("every layer must report the same number of experts")

    mode = ranking_mode(model_name)
    mass = mass or [[float(c) for c in layer] for layer in counts]
    scores = mass if mode == "mass" else [[float(c) for c in layer] for layer in counts]

    topk: dict[str, list[list[int]]] = {}
    for pct in percents:
        k = max(1, round(n_experts * pct / 100.0))
        key = f"{pct:g}"
        topk[key] = [top_k_indices(layer, k) for layer in scores]

    layer_cv = [coefficient_of_variation(layer) for layer in scores]

    # Coverage: what fraction of all activations the top-25% captures. MoE-Sieve
    # measures 37-53% for the top quarter; a profile far outside that band is a
    # signal the pass sampled the wrong corpus, not a better model.
    sel = topk.get("25", [])
    total = sum(sum(layer) for layer in counts) or 1
    covered = 0
    for layer_idx, layer_counts in enumerate(counts):
        if layer_idx < len(sel):
            covered += sum(layer_counts[e] for e in sel[layer_idx] if e < len(layer_counts))
    return RoutingProfile(
        model_hash=model_hash,
        model_name=model_name,
        corpus_hash=corpus_hash,
        n_tokens=n_tokens,
        counts=counts,
        mass=mass,
        topk=topk,
        layer_cv=[round(cv, 4) for cv in layer_cv],
        cov_at_25=round(covered / total, 4),
        rank_by=mode,
    )


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def compare(p: RoutingProfile, q: RoutingProfile, percent: str = "25") -> dict[str, Any]:
    """Mean per-layer Jaccard between two profiles' top-k sets.

    Used two ways: to check that a 10% subsample reproduces the full pass
    (expect >= 0.94), and to check that a code corpus is actually code-like — MBPP
    and CodeAlpaca agree at J=0.83, MBPP and WikiText at J=0.13 [M]. A profile built
    on the wrong corpus selects the wrong experts and the adapter underperforms for
    a reason nobody will find later.
    """
    a, b = p.topk.get(percent, []), q.topk.get(percent, [])
    n = min(len(a), len(b))
    if n == 0:
        return {"layers": 0, "mean_jaccard": 0.0}
    per_layer = [jaccard(a[i], b[i]) for i in range(n)]
    return {
        "layers": n,
        "mean_jaccard": round(sum(per_layer) / n, 4),
        "min_jaccard": round(min(per_layer), 4),
        "per_layer": [round(j, 3) for j in per_layer],
    }


def sanity(profile: RoutingProfile) -> dict[str, Any]:
    """Is this profile usable? Reported at build time, never assumed."""
    flat_layers = sum(1 for cv in profile.layer_cv if cv < 0.15)
    in_band = 0.30 <= profile.cov_at_25 <= 0.60
    return {
        "layers": profile.n_layers,
        "experts": profile.n_experts,
        "rank_by": profile.rank_by,
        "cov_at_25": profile.cov_at_25,
        "coverage_in_expected_band": in_band,
        "flat_layers": flat_layers,
        "ok": in_band and flat_layers < max(1, profile.n_layers // 4),
        "note": (
            "Top-25% coverage outside the 30-60% band measured by MoE-Sieve. Check "
            "the profiling corpus is the repository's own code, not a general mix."
            if not in_band
            else "routing skew consistent with published measurements"
        ),
    }
