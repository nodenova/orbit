"""Reported-usage scaling (spec sec 8.3).

Harnesses decide when to auto-compact by comparing reported token usage against an
assumed context window — Claude Code assumes ~200k because that is what a Claude
model has. A local model with a 32k working window overflows long before the
harness thinks it is anywhere near the limit, and the failure is ugly: the request
is truncated or rejected mid-task with no signal the harness understands.

The fix is to scale *reported usage only*. Generation, KV and the prompt itself are
untouched — this is a lie told to the harness's compaction heuristic and nowhere
else, which is why it lives in one small module with one function.

At 200k assumed over a 32k real window the factor is ~6.1; the spec's worked example
of 8.0 corresponds to a 25k working window. Both fall out of the same formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextScaler:
    enabled: bool = True
    assumed_window: int = 200_000
    real_window: int = 32_768

    @property
    def factor(self) -> float:
        if not self.enabled or self.real_window <= 0:
            return 1.0
        return self.assumed_window / self.real_window

    def scale(self, tokens: int) -> int:
        """Scale a usage count for reporting to the harness.

        Rounds up: reporting fewer tokens than the scaled truth would let the
        harness sail past the point where it should have compacted, which is the
        exact failure this exists to prevent.
        """
        if not self.enabled:
            return tokens
        return math.ceil(tokens * self.factor)

    def headroom_tokens(self) -> int:
        """Real tokens left before the harness ought to have compacted."""
        return self.real_window

    def describe(self) -> str:
        if not self.enabled:
            return "context scaling off"
        return (
            f"reporting usage x{self.factor:.2f} "
            f"({self.real_window} real window behind a {self.assumed_window} assumption)"
        )
