"""Gateway (spec sec 8)."""

from tandem.gateway.compaction import (
    TEMPLATES,
    CompactionResult,
    CompactionTemplate,
    Compactor,
    detect,
)
from tandem.gateway.context_scale import ContextScaler
from tandem.gateway.pipeline import Pipeline, TurnTrace

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
