"""Fallback ladder rung 3 — tier 0 as its own verifier (spec sec 5.5).

    3. Tier 1 = tier 0 base *without* the adapter, as a second opinion
       (free, weak, still catches adapter overfit)

The cheapest rung and the only one that needs no second model, no swap and no
network. It runs on any machine that can run tier 0 at all, which makes it the
rung that is actually available during M0–M3 — before the 122B container exists,
and on any box that turns out not to fit both tiers at once. That includes the
baseline host, where it is the rung `orbit.toml` ships (`docs/BASELINE.md`).

**The adapter strip is the whole mechanism.** A1 is trained on the repository's own
merged diffs, so it has learned that repository's habits — including the bad ones,
and including whatever it overfit to. Asking the *adapted* model to judge its own
candidates is asking it whether it agrees with itself; the answer is yes, and it
tells you nothing. The base model has not seen this repository, so when it and the
adapter disagree, the disagreement is information: usually adapter overfit, and
occasionally the adapter being right in a way the base cannot see.

That is also the honest limit of this rung. The base model is weaker than the
122B at judging, and it is weaker than the adapter at knowing this repo. It catches
overfit. It does not replace a real verifier, and `Tier1Attestation.rung` records
which one served every verdict so a receipt never implies otherwise.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from orbit.backends.base import Backend, Delta, ToolCallRenderer
from orbit.types import GenRequest, GenResult

RUNG = "second_opinion"


class SecondOpinionBackend(Backend):
    """Tier 0, adapter forcibly unmounted, serving in the tier-1 role."""

    name = "tier0-second-opinion"
    tier = 1

    def __init__(self, tier0: Backend):
        self._tier0 = tier0

    # --- the strip ----------------------------------------------------------

    @staticmethod
    def _strip(req: GenRequest) -> GenRequest:
        """Force the base model.

        Copy-on-write, so the caller's request is untouched — the same object is
        often the one tier 0 is about to serve *with* its adapter, and mutating it
        here would silently unmount the adapter for the generation too.
        """
        if req.adapter is None:
            return req
        return req.with_(adapter=None)

    async def generate(self, req: GenRequest) -> GenResult:
        return await self._tier0.generate(self._strip(req))

    async def stream(self, req: GenRequest) -> AsyncIterator[Delta]:
        async for delta in self._tier0.stream(self._strip(req)):
            yield delta

    # --- attestation --------------------------------------------------------
    #
    # The container is tier 0's, because that is what actually ran. The adapter and
    # profile hashes are None and must stay None: this rung's entire claim is that
    # no adapter was mounted, so reporting one would invert the meaning of the
    # receipt.

    def container_hash(self) -> str | None:
        return self._tier0.container_hash()

    def adapter_hash(self, adapter: str | None) -> str | None:
        return None

    def profile_hash(self, adapter: str | None) -> str | None:
        return None

    def mounted_adapters(self) -> tuple[str, ...]:
        return ()

    # --- passthrough --------------------------------------------------------

    def render(self, req: GenRequest, render_tool_call: ToolCallRenderer = None) -> str:
        return self._tier0.render(self._strip(req), render_tool_call)

    def count_tokens(self, text: str) -> int:
        return self._tier0.count_tokens(text)

    def supports_state(self) -> bool:
        # Shares tier 0's weights and its KV cache namespace. Letting verdict
        # prefixes into the same disk cache as conversation prefixes would mix two
        # different `state_key` populations for no benefit — verifier calls are
        # one-shot and never share a prefix with the next one.
        return False

    async def close(self) -> None:
        # Not ours to close: tier 0 owns its own lifetime and is still serving
        # generation. Closing it here would take the resident model down when the
        # verifier shut down.
        return None

    def stats(self) -> dict[str, Any]:
        return {"rung": RUNG, "delegates_to": self._tier0.name}
