"""Adapter pipeline (spec sec 6) — repo -> adapter -> served.

The differentiator. Nobody ships this. Adapter *serving* is not the defensible part;
the pipeline is — git-history extraction, DPO from review history, and the merge eval
that says whether any of it worked (sec 13).

    A0  harness   synthetic tool-call traces      universal, ships with the product
    A1  repo      merged diffs from git history   per customer repo
    A2  reviewer  pre/post-review diff pairs      per customer repo, DPO
"""

from orbit.adapters import extract_a0, extract_a1, extract_a2, profile, train
from orbit.adapters.filters import ExtractionFilters, SkipTally
from orbit.adapters.profile import RoutingProfile

__all__ = [
    "ExtractionFilters",
    "RoutingProfile",
    "SkipTally",
    "extract_a0",
    "extract_a1",
    "extract_a2",
    "profile",
    "train",
]
