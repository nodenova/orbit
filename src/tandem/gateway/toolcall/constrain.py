"""Constrained decoding — prevention (spec sec 8.5.1).

`lm-format-enforcer` by choice: pure Python (pydantic + interegular), ~1 ms/token,
and it does not drag in PyTorch. XGrammar hard-requires PyTorch and would break an
MLX-native runtime (sec 5.1), so it is not an option regardless of its merits.

Two uses:

* **Tool-bearing turns** — the model must emit a call matching one of the request's
  tools. Free-form turns are left untouched; constraining prose is how you get a
  model that answers every question with a tool call.
* **Tier-1 verdicts** — every tier-1 call is schema-constrained so output length is
  bounded by construction and 2-bit's documented JSON weakness (sec 5.2) cannot
  produce a malformed judgement.

The enforcer is optional at install time. Without it, `Constrainer.available` is
False and the runtime falls back to repair-and-retry, which is a real degradation
and is reported as such rather than passed over in silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ...types import ToolDef


def tool_call_schema(tools: Iterable[ToolDef]) -> dict[str, Any]:
    """JSON Schema for "a call to exactly one of these tools".

    A closed `enum` on the name is what makes tool invention impossible during
    decoding rather than caught afterwards by the repair layer.
    """
    tools = tuple(tools)
    branches = [
        {
            "type": "object",
            "properties": {
                "name": {"const": t.name},
                "arguments": t.parameters or {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for t in tools
    ]
    if not branches:
        return {"type": "object"}
    if len(branches) == 1:
        return branches[0]
    return {"title": "tool_call", "anyOf": branches}


@dataclass
class Constrainer:
    """Thin wrapper over lm-format-enforcer, with an honest unavailable state."""

    enabled: bool = True
    _reason: str = ""

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        return self._probe()[0]

    def status(self) -> dict[str, Any]:
        ok, reason = self._probe()
        return {
            "enabled": self.enabled,
            "available": self.enabled and ok,
            "reason": reason,
            "note": (
                "Without constrained decoding the runtime relies on repair+retry "
                "(sec 8.5.3-4), which is measurably weaker on the sec 10.2 gate."
            ),
        }

    def _probe(self) -> tuple[bool, str]:
        try:
            import lmformatenforcer  # type: ignore  # noqa: F401
        except ImportError:
            return False, "lm-format-enforcer not installed (pip install 'tandem[constrain]')"
        return True, "ok"

    def logits_processor(self, schema: dict[str, Any], tokenizer: Any) -> Any | None:
        """Build a logits processor enforcing `schema`, or None if unavailable.

        Returning None rather than raising is deliberate: an unconstrained
        generation that then goes through repair is a worse outcome than a
        constrained one, but it is a far better outcome than a failed request.
        """
        ok, _ = self._probe()
        if not (self.enabled and ok):
            return None
        from lmformatenforcer import JsonSchemaParser  # type: ignore
        from lmformatenforcer.integrations.transformers import (  # type: ignore
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = JsonSchemaParser(schema)
        return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
