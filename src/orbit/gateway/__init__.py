"""Gateway (spec sec 8)."""

from orbit.gateway.compaction import (
    TEMPLATES,
    CompactionResult,
    CompactionTemplate,
    Compactor,
    detect,
)
from orbit.gateway.context_scale import ContextScaler
from orbit.gateway.pipeline import Pipeline, TurnTrace

__all__ = [
    "TEMPLATES",
    "CompactionResult",
    "CompactionTemplate",
    "Compactor",
    "ContextScaler",
    "Pipeline",
    "TurnTrace",
    "detect",
]
