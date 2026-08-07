"""Gateway (spec sec 8)."""

from .compaction import Compactor, CompactionResult, CompactionTemplate, TEMPLATES, detect
from .context_scale import ContextScaler
from .pipeline import Pipeline, TurnTrace

__all__ = [
    "Compactor",
    "CompactionResult",
    "CompactionTemplate",
    "TEMPLATES",
    "detect",
    "ContextScaler",
    "Pipeline",
    "TurnTrace",
]
