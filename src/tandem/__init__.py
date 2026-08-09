"""Tandem — a local coding-agent runtime that optimises for merge quality.

Two model tiers on one machine: a fast resident model carrying repo-specific LoRA
adapters, and a large streamed model used as a verifier rather than a generator.

Section references in docstrings point at docs/SPEC.md.
"""

__version__ = "0.1.0"

from tandem.types import (
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
