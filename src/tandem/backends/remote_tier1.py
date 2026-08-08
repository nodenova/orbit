"""Fallback ladder rung 4 — a remote verifier (spec sec 5.5).

    4. Tier 1 = a remote API, as an explicit choice, never a fallback

This rung is different in kind from the other three. Rungs 1-3 all run on the user's
machine; this one sends the repository's code to somebody else's. That breaks the
offline posture of sec 8.6 — which is not a preference but a *verifiable claim*, and
for the regulated buyer it is most of the reason they are here. A ladder that reaches
this rung on its own turns an airgapped runtime into an exfiltrating one in response
to a timeout.

So four things are structural rather than advisory:

* **The ladder never falls to it.** `build_tier1` returns this backend only when the
  config names the rung. Nothing degrades into it, and no error path selects it.

* **Naming the rung is not enough.** `tier1.remote_consent` has to carry the
  acknowledgement verbatim. A rung name is one word copied from a README; a sentence
  that says the code leaves the machine cannot be pasted by accident.

* **The HTTP lives outside the package.** Nothing under `src/tandem/` makes an
  outbound call and a test pins that surface, so this backend holds no client at all:
  the transport is injected and `tools/remote_tier1.py` is where the socket is. A rung
  that broke the airgap by pulling an HTTP client into the package would break it for
  every *other* rung too — including the three that are still airgapped, and including
  the deployments that never enable this one.

* **`tandem doctor` stops reporting a clean offline posture.** `OfflineReport.ok` is
  false whenever this rung is live, whatever `lsof` happens to have caught.

And one property that follows from the rung rather than being imposed on it: **a
remote verdict has no container attestation.** `container_hash()` is None and stays
None — you cannot attest to a model you do not hold, and the honest answer to "which
container produced this verdict?" is that we do not know. The receipt names the rung,
which is what tells a reader the null is a property and not a gap.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from ..types import GenRequest, GenResult, StopReason, Usage
from .base import Backend
from .tier1_call import (
    Tier1Unavailable,
    build_payload,
    read_completion,
    resolve_reasoning_control,
    validate_or_raise,
)

RUNG = "remote"

# The acknowledgement `tier1.remote_consent` must carry, verbatim.
CONSENT = "tier 1 leaves this machine"

# What the injected transport is: chat-completions payload in, parsed body out. The
# narrowest surface that keeps every socket outside the package.
Transport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class RemoteConsentMissing(ValueError):
    """The remote rung was selected without the acknowledgement it requires."""


def check_consent(consent: str) -> None:
    """Raise unless the operator wrote the acknowledgement out.

    Compared after normalising case and surrounding whitespace, and no further. A
    fuzzy match here would be a fuzzy consent gate, and the entire point of this
    string is that it cannot be arrived at inattentively.
    """
    if consent.strip().lower() != CONSENT:
        raise RemoteConsentMissing(
            "tier1.rung='remote' sends this repository's code to a third party and "
            "breaks the sec 8.6 offline claim. It is available, but only as an "
            f'explicit choice: set tier1.remote_consent = "{CONSENT}" to confirm '
            "that is what you want. Rungs 1-3 all stay on this machine."
        )


class RemoteTier1Backend(Backend):
    """A verifier reached over somebody else's API. Holds no HTTP client."""

    name = "remote-tier1"
    tier = 1

    def __init__(
        self,
        transport: Transport,
        *,
        model: str,
        endpoint_label: str = "",
        reasoning_control: str = "auto",
    ):
        if not callable(transport):
            raise ValueError(
                "RemoteTier1Backend needs a transport. The package makes no outbound "
                "call by design (sec 8.6), so the HTTP lives in tools/remote_tier1.py "
                "and is injected here."
            )
        self._transport = transport
        self.model = model
        # Host only, never the full URL with its query string, and never a key. It
        # is for the operator to recognise where verdicts went; it is not a secret
        # store and must not become one.
        self.endpoint_label = endpoint_label
        # A hosted DeepSeek-V4 endpoint reasons by default, and the two invariants
        # that breaks (sec 5.1's clamp, sec 9.3's determinism) do not become less
        # true for being someone else's engine.
        self.reasoning_control = resolve_reasoning_control(reasoning_control, model)
        self.calls = 0
        self.total_s = 0.0

    async def generate(self, req: GenRequest) -> GenResult:
        payload = build_payload(
            req, model=self.model, reasoning_control=self.reasoning_control
        )
        t0 = time.perf_counter()
        try:
            body = await self._transport(payload)
        except Tier1Unavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - the transport is not ours
            raise Tier1Unavailable(f"remote tier 1 call failed: {exc}") from exc
        total_s = time.perf_counter() - t0
        if not isinstance(body, dict):
            raise Tier1Unavailable("remote tier 1 transport did not return a JSON object")

        text, in_tok, out_tok = read_completion(body, payload, self.count_tokens)
        # Validated on this side of the boundary as well, because "the endpoint
        # supports constrained decoding" is a claim about a machine we do not
        # control and cannot check (sec 5.2).
        if req.json_schema is not None:
            validate_or_raise(text, req.json_schema)

        self.calls += 1
        self.total_s += total_s
        return GenResult(
            text=text,
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
            total_s=total_s,
        )

    # --- attestation --------------------------------------------------------

    def container_hash(self) -> str | None:
        # None by construction, not by omission. See the module docstring.
        return None

    def adapter_hash(self, adapter: str | None) -> str | None:
        return None

    def profile_hash(self, adapter: str | None) -> str | None:
        return None

    def mounted_adapters(self) -> tuple[str, ...]:
        return ()

    def supports_state(self) -> bool:
        # There is no local KV cache to snapshot, and a remote one is not ours to
        # key, name or trust.
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "rung": RUNG,
            "model": self.model,
            "endpoint": self.endpoint_label,
            "calls": self.calls,
            "total_s": round(self.total_s, 2),
            "offline": False,
        }
