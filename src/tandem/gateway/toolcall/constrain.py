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

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from tandem.types import ToolDef

# A per-request token filter: given the tokens decoded so far, the ids that may
# come next. Deliberately plain Python ints on both sides — this is the seam
# between the schema layer and whatever tensor library a backend happens to use,
# and putting an array type in it would drag MLX above `backends/base.py`.
TokenFilter = Callable[[Sequence[int]], Sequence[int]]


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
            import lmformatenforcer  # noqa: F401
        except ImportError:
            return (
                False,
                "lm-format-enforcer not installed (pip install 'tandem[constrain]')",
            )
        return True, "ok"

    def vocabulary(self, tokenizer: Any) -> Any | None:
        """LMFE's per-tokenizer preprocessing, or None if unavailable.

        **Build this once and keep it.** It walks the whole vocabulary and builds a
        prefix tree — measured on this host against the tier-0 tokenizer (248,077
        tokens): 1.1 s and ~0.6 GB. Per request that would dwarf the generation it
        constrains; per process it is noise. The caller owns the caching because the
        natural lifetime is the backend's, and a module-level cache keyed on the
        tokenizer would either leak it forever or key on `id()` and risk handing one
        model's vocabulary to another after a swap (sec 5.5 rung 2 makes that a real
        sequence, not a hypothetical).

        Every stop id goes in, not just `eos_token_id`. The enforcer only permits a
        stop token once the parser `can_end()`, so an id it does not know about is an
        id the model may never emit — and `mlx_lm.stream_generate` breaks on
        `tokenizer.eos_token_ids`, the set. Pass the singular id alone and a
        completed object cannot be terminated: the mask forbids the one token that
        would end the turn, and generation runs to `max_tokens` emitting whitespace.
        """
        ok, _ = self._probe()
        if not (self.enabled and ok):
            return None
        from lmformatenforcer import TokenEnforcerTokenizerData

        inner = getattr(tokenizer, "_tokenizer", tokenizer)
        vocab_size = len(inner)
        special = set(getattr(inner, "all_special_ids", ()) or ())
        # Prepending a known token and dropping its first character is how LMFE
        # recovers the leading space a word-start token carries; the decode of a
        # token on its own does not show it.
        token_0 = inner.encode("0")[-1]
        regular: list[tuple[int, str, bool]] = []
        for token_id in range(vocab_size):
            if token_id in special:
                continue
            after = inner.decode([token_0, token_id])[1:]
            plain = inner.decode([token_id])
            regular.append((token_id, after, len(after) > len(plain)))

        def decode(ids: list[int]) -> str:
            # A partial multi-byte character decodes to U+FFFD; feeding that to the
            # parser would fail a schema the model is still spelling correctly.
            return cast(str, inner.decode(ids).rstrip("�"))

        return TokenEnforcerTokenizerData(
            regular,
            decode,
            _stop_token_ids(tokenizer),
            False,  # bitmask output requires torch, which an MLX runtime does not have
            vocab_size,
        )

    def token_filter(
        self, schema: dict[str, Any], vocabulary: Any
    ) -> TokenFilter | None:
        """A filter enforcing `schema`, or None if unavailable.

        **Fresh per request, and that is not an oversight.** A `TokenEnforcer` keys
        its parser states on the full token tuple it has seen, so sharing one across
        requests grows a dict without bound and holds one conversation's parse states
        for the life of the process. Construction is free once `vocabulary` exists —
        measured at under a millisecond — so there is nothing to amortise.

        Returning None rather than raising keeps the degradation ordered: an
        unconstrained generation that goes through repair (sec 8.5.3-4) is worse than
        a constrained one and far better than a failed request.
        """
        ok, _ = self._probe()
        if not (self.enabled and ok) or vocabulary is None:
            return None
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer

        enforcer = TokenEnforcer(vocabulary, JsonSchemaParser(schema))

        def allowed(tokens: Sequence[int]) -> Sequence[int]:
            allowed = enforcer.get_allowed_tokens(list(tokens)).allowed_tokens
            return cast("Sequence[int]", allowed)

        return allowed


def logit_width_bound(tokenizer: Any) -> int:
    """One past the largest id a `TokenFilter` over this tokenizer can return.

    A backend needs this to decide *once* whether its logit row covers every id LMFE
    might name, instead of bounds-checking each id on every token (sec 8.5.1, F1).
    `len(tokenizer)` alone is not that bound: `vocabulary()` enumerates
    `range(len(inner))` but passes the stop ids in separately, so a stop id outside the
    enumerated range is possible and would not be covered by the length.
    """
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    return max([len(inner), *(i + 1 for i in _stop_token_ids(tokenizer))])


def _stop_token_ids(tokenizer: Any) -> list[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return sorted(int(i) for i in ids)
    single = getattr(tokenizer, "eos_token_id", None)
    return [int(single)] if single is not None else []
