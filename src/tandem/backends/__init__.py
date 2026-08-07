"""Model backends.

`build_tier0` / `build_tier1` are the only places the rest of the runtime names a
concrete engine. Everything else programs against `Backend`.

`build_tier1` is also where the sec 5.5 fallback ladder is selected. Selected, not
descended: the rung comes from the config and nothing here falls from one rung to the
next on an error. That matters most at rung 4, which sends code off the machine — a
ladder that reached it in response to a timeout would turn an airgapped runtime into
an exfiltrating one without anyone choosing it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from ..config import Config
from .base import Backend, BackendUnavailable, Delta, render_default
from .mock import Fault, MockBackend
from .remote_tier1 import RUNG as REMOTE_RUNG
from .remote_tier1 import RemoteConsentMissing, RemoteTier1Backend, check_consent
from .resident_swap import RUNG as RESIDENT_SWAP_RUNG
from .resident_swap import Occupant, ResidencySwitch, ResidentSwapBackend, SwapGuard
from .second_opinion import RUNG as SECOND_OPINION_RUNG
from .second_opinion import SecondOpinionBackend

STREAMED_RUNG = "streamed"
RUNGS = (STREAMED_RUNG, RESIDENT_SWAP_RUNG, SECOND_OPINION_RUNG, REMOTE_RUNG)

__all__ = [
    "Backend",
    "BackendUnavailable",
    "Delta",
    "Fault",
    "MockBackend",
    "Occupant",
    "RemoteConsentMissing",
    "RemoteTier1Backend",
    "ResidencySwitch",
    "ResidentSwapBackend",
    "SecondOpinionBackend",
    "SwapGuard",
    "REMOTE_RUNG",
    "RESIDENT_SWAP_RUNG",
    "SECOND_OPINION_RUNG",
    "STREAMED_RUNG",
    "RUNGS",
    "render_default",
    "build_tier0",
    "build_tier1",
]


def build_tier0(cfg: Config) -> Backend:
    if cfg.backend == "mock":
        return MockBackend(adapters=_mock_adapter_names(cfg))
    if cfg.backend == "mlx":
        from .mlx_tier0 import MLXTier0Backend

        return MLXTier0Backend(
            cfg.tier0.container_path or cfg.tier0.model,
            adapter_dir=cfg.tier0.adapter_dir,
            mtp=cfg.tier0.mtp,
            max_kv_tokens=cfg.tier0.max_kv_tokens,
        )
    raise ValueError(f"unknown backend: {cfg.backend!r} (expected 'mock' or 'mlx')")


def build_tier1(cfg: Config, tier0: Backend | None = None) -> Backend | None:
    """Tier 1, or None when it is switched off.

    None is a first-class answer: the fallback ladder (sec 5.5) ends at "no tier 1",
    and the router must degrade to tier-0-only rather than fail the turn.

    `tier0` is needed by the two rungs that involve the resident model — rung 3,
    which serves the verifier from it with the adapter unmounted, and rung 2, which
    evicts it to make room for the 80B.
    """
    if not cfg.tier1.enabled:
        return None

    rung = cfg.tier1.rung
    if rung == SECOND_OPINION_RUNG:
        if tier0 is None:
            raise ValueError(
                "tier1.rung='second_opinion' serves the verifier from tier 0, so "
                "build_tier1 needs the tier-0 backend"
            )
        return SecondOpinionBackend(tier0)
    if rung == RESIDENT_SWAP_RUNG:
        return _build_resident_swap(cfg, tier0)
    if rung == REMOTE_RUNG:
        return _build_remote(cfg)
    if rung != STREAMED_RUNG:
        raise ValueError(
            f"unknown tier1.rung: {rung!r} (expected one of {', '.join(RUNGS)})"
        )

    if cfg.backend == "mock":
        return MockBackend(name="mock-tier1", tier=1, container="mock-tier1-container-v1")
    from .mlx_tier1 import OptiqTier1Backend

    return OptiqTier1Backend(
        cfg.tier1.endpoint,
        model=cfg.tier1.model,
        container_path=cfg.tier1.container_path,
        timeout_s=cfg.tier1.request_timeout_s,
        expert_cache_bytes=cfg.tier1.expert_cache_bytes,
        pinned_version=cfg.tier1.pinned_version,
    )


# --- rung 2 -----------------------------------------------------------------


def _build_resident_swap(cfg: Config, tier0: Backend | None) -> Backend:
    """Rung 2: the 80B, swapped in over tier 0 (sec 5.5).

    Both models have to be able to leave memory and come back, which is what
    `Occupant` asks of them. On the mock backend that is a bookkeeping exercise and
    the point is to exercise the residency policy, where the concurrency bugs are.
    On MLX it is a real eviction, and the resident 80B verifier that would go on the
    other side of the switch does not exist yet — so this says so, precisely, rather
    than building a swap with nothing to swap in.
    """
    if tier0 is None:
        raise ValueError(
            "tier1.rung='resident_swap' evicts tier 0 to make room, so build_tier1 "
            "needs the tier-0 backend"
        )
    if cfg.backend != "mock":
        raise BackendUnavailable(
            "tier1.rung='resident_swap' needs a resident 80B verifier backend and an "
            "evictable tier 0 on this platform; neither is implemented for "
            f"backend={cfg.backend!r} yet (sec 5.5 rung 2). The residency policy is "
            "built and exercised under backend='mock'; rung 3 ('second_opinion') is "
            "the rung that serves a verifier today without a second model."
        )

    verifier = MockBackend(
        name="mock-tier1-resident", tier=1, container="mock-80b-container-v1"
    )
    switch = ResidencySwitch(_SwappableMock(tier0), _SwappableMock(verifier))
    return ResidentSwapBackend(verifier, switch, budget_s=cfg.tier1.swap_budget_s)


class _SwappableMock:
    """Load/unload hooks for a mock occupant.

    Records the transitions and nothing else. It deliberately does *not* pretend to
    cost ~10 s: a sleep here would make every test that touches this rung slow while
    proving nothing the policy tests do not already prove, and the budget guard reads
    a measured number, so faking the measurement would defeat it.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.name = backend.name
        self.resident = True
        self.loads = 0
        self.unloads = 0

    async def load(self) -> None:
        self.resident = True
        self.loads += 1

    async def unload(self) -> None:
        self.resident = False
        self.unloads += 1


# --- rung 4 -----------------------------------------------------------------


def _build_remote(cfg: Config) -> Backend:
    """Rung 4: somebody else's API, only ever because the config said so (sec 5.5).

    Three gates before a socket exists: the rung has to be named, the consent
    sentence has to be written out, and the transport file has to be on disk outside
    the package. Nothing degrades into this path.
    """
    check_consent(cfg.tier1.remote_consent)
    if not cfg.tier1.remote_endpoint:
        raise ValueError("tier1.rung='remote' needs tier1.remote_endpoint")

    transport = _load_transport(cfg)
    return RemoteTier1Backend(
        transport,
        model=cfg.tier1.remote_model or cfg.tier1.model,
        endpoint_label=_endpoint_label(cfg.tier1.remote_endpoint),
    )


def _load_transport(cfg: Config) -> Any:
    """Load the out-of-package transport by path.

    By path rather than by import, because the whole arrangement rests on the file
    not being part of the installed package — importable transports drift back
    inside it, and then the offline claim is a promise again.
    """
    path = Path(cfg.tier1.remote_transport)
    if not path.is_file():
        raise ValueError(
            f"tier1.remote_transport={str(path)!r} is not a file. The HTTP for rung 4 "
            "lives outside the package (sec 8.6); tools/remote_tier1.py is the one "
            "that ships with tandem."
        )
    spec = importlib.util.spec_from_file_location("tandem_remote_tier1_transport", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load a transport from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "transport_from_config", None)
    if factory is None:
        raise ValueError(
            f"{path} defines no transport_from_config(); see tools/remote_tier1.py"
        )
    return factory(
        endpoint=cfg.tier1.remote_endpoint,
        model=cfg.tier1.remote_model or cfg.tier1.model,
        api_key_env=cfg.tier1.remote_api_key_env,
        timeout_s=cfg.tier1.request_timeout_s,
    )


def _endpoint_label(endpoint: str) -> str:
    """Host only. Never a path, a query string or a key — this is for recognising
    where verdicts went, and it ends up in `stats()` and in `tandem doctor`."""
    rest = endpoint.split("://", 1)[-1]
    return rest.split("/", 1)[0]


def _mock_adapter_names(cfg: Config) -> tuple[str, ...]:
    root = Path(cfg.tier0.adapter_dir)
    if not root.is_dir():
        return ()
    return tuple(sorted(p.name for p in root.iterdir() if p.is_dir()))
