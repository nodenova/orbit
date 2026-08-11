"""Orbit — a local coding-agent runtime that runs a larger-than-memory model as a verifier.

Two model tiers on one machine: a fast resident model carrying repo-specific LoRA
adapters generates, and a model too large to hold in memory — its experts streamed
from NVMe — reads the candidates and judges them. It has no `generate` entrypoint,
because on a streamed model prefill is ~40x cheaper per token than decode.

Section references in docstrings point at docs/SPEC.md.
"""

__version__ = "0.1.0"

from orbit.types import (
    GenRequest,
    GenResult,
    Message,
    Role,
    Sampling,
    StopReason,
    ToolCall,
    ToolDef,
    ToolResult,
    TurnClass,
    Usage,
)

__all__ = [
    "GenRequest",
    "GenResult",
    "Message",
    "Role",
    "Sampling",
    "StopReason",
    "ToolCall",
    "ToolDef",
    "ToolResult",
    "TurnClass",
    "Usage",
    "__version__",
]
