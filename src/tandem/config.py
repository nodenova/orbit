"""Runtime configuration.

TOML file plus environment overrides. Deliberately flat and explicit: every knob
that changes what the model computes also appears in the receipt (sec 9.1), so the
config surface and the attestation surface are kept in step by hand rather than by
reflection over a nested schema.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

DEFAULT_CONFIG_PATH = Path(os.environ.get("TANDEM_CONFIG", "tandem.toml"))


@dataclass
class Tier0Config:
    """Resident adapted generator (sec 4)."""

    model: str = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    container_path: str | None = None
    # Directory of adapters; every subdirectory is mounted at startup (sec 4.2) —
    # loading one mid-flight stalls every in-flight request.
    adapter_dir: str = "adapters"
    default_adapter: str | None = None
    # Multi-token-prediction head: ~1.4x decode at ~70% acceptance at depth 2 [V].
    mtp: bool = True
    max_kv_tokens: int = 32_768


@dataclass
class Tier1Config:
    """Streamed verifier (sec 5). Never generates a patch."""

    enabled: bool = False
    # Which rung of the sec 5.5 fallback ladder serves the tier-1 role.
    #
    #   "streamed"        rung 1 — the 122B design target, streamed from NVMe
    #   "resident_swap"   rung 2 — an 80B swapped into residency, evicting tier 0.
    #                     ~10 s each way, and tier 0 cannot serve while it is in.
    #   "second_opinion"  rung 3 — tier 0 with its adapter unmounted. Free, weak,
    #                     needs no second model, and still catches adapter overfit.
    #                     The rung that is actually available during M0-M3.
    #   "remote"          rung 4 — somebody else's API. Breaks the sec 8.6 offline
    #                     claim, so it is never reached by falling back and needs
    #                     `remote_consent` written out in full.
    rung: str = "streamed"
    model: str = "mlx-community/Qwen3.5-122B-A10B-OptiQ-2bit"
    container_path: str | None = None
    # Rung 2. Decline a verdict once the *measured* swap round trip costs more than
    # this many seconds; 0 leaves the guard off. Off by default because a ceiling
    # that fires on a machine nobody has measured yet disables verification for a
    # cost that was never observed.
    swap_budget_s: float = 0.0
    # Rung 4 only. The transport is a file outside the package, because nothing under
    # src/tandem/ makes an outbound call (sec 8.6) — see tools/remote_tier1.py.
    remote_endpoint: str = ""
    remote_model: str = ""
    remote_transport: str = "tools/remote_tier1.py"
    # The name of an environment variable, never the key itself: a config file is
    # committed, diffed and pasted into issues.
    remote_api_key_env: str = "TANDEM_REMOTE_API_KEY"
    # Must read "tier 1 leaves this machine". A rung name is one word copied from a
    # README; a sentence saying the code leaves the machine is not pasted by accident.
    remote_consent: str = ""
    # v1 runs mlx-optiq as a separate process behind its OpenAI endpoint (sec 5.4).
    # It is unforkable and unauditable, so it stays behind a process boundary and a
    # pinned version, and never ships as-is into a regulated deployment.
    endpoint: str = "http://127.0.0.1:8081/v1"
    pinned_version: str = ""
    stream_experts: bool = True
    expert_cache_bytes: int = 18 * (1 << 30)  # sec 2.1 budget
    request_timeout_s: float = 180.0


@dataclass
class RouterConfig:
    """Escalation policy (sec 7.2). Rule-based by design in v1."""

    # T1 best-of-N rerank on code_change turns. N=1 disables reranking.
    candidates: int = 3
    candidate_temperature: float = 0.6
    rerank_enabled: bool = True
    # T2 failure escalation, bounded to one per turn to bound the worst case.
    max_escalations_per_turn: int = 1
    # Automatic pressure valve (sec 7.3): past this, degrade to N=1 with no rerank
    # rather than let the interaction stop feeling interactive.
    degrade_after_s: float = 45.0


@dataclass
class CompactionConfig:
    """Harness compaction (sec 8.2) — the largest single latency win."""

    enabled: bool = True
    strip_tool_schemas: bool = True
    # Keep the raw system prompt on the request so --no-compact and the diff view
    # can show exactly what the harness sent.
    keep_original: bool = True


@dataclass
class CacheConfig:
    """Prompt and KV caching (sec 8.4)."""

    prompt_cache_bytes: int = 2 << 30
    disk_kv_dir: str = "var/kvcache"
    disk_kv_enabled: bool = True
    disk_kv_budget_bytes: int = 20 << 30
    # Cold-save alignment: trim a token suffix and align down to a chunk boundary
    # so a BPE boundary shift does not miss the whole entry.
    kv_chunk_tokens: int = 256


@dataclass
class ContextScaleConfig:
    """Reported-usage scaling (sec 8.3)."""

    enabled: bool = True
    # What the harness assumes the window is (Claude Code assumes ~200k).
    assumed_window: int = 200_000
    # What it actually is for the resident model's working window.
    real_window: int = 32_768


@dataclass
class ToolCallConfig:
    """Tool-call reliability (sec 8.5)."""

    constrain: bool = True
    repair: bool = True
    max_retries: int = 2
    # Tool-bearing turns run cooler; free-form turns keep the caller's temperature.
    tool_turn_temperature: float = 0.2
    replay_map_size: int = 512


@dataclass
class EvalConfig:
    """The repository's own commands (sec 10.1, 7.2).

    Three of the merge eval's five metrics and the whole of T2 failure escalation
    come down to "what does this repo run to check itself". There is no way to
    guess it — `pytest`, `npm test`, `cargo test`, a Makefile target — so it is
    declared here, once, and both callers read it.

    Empty means not measured, everywhere. `summarise` reports None, `compare_arms`
    refuses to call an unmeasured metric a pass, and the router leaves T2 dormant.
    """

    repo: str = "."
    # Run over the files a patch touches, inside the patched worktree.
    #   linters = [["ruff", "check"], ["black", "--check"]]
    linters: list[list[str]] = field(default_factory=list)
    # The suite. Runs in a throwaway worktree — never the user's checkout.
    test_command: list[str] = field(default_factory=list)
    # Run once per worktree before the suite, e.g. ["pip", "install", "-e", "."].
    setup_command: list[str] = field(default_factory=list)
    base_rev: str = "HEAD"
    test_timeout_s: float = 600.0
    lint_timeout_s: float = 120.0
    setup_timeout_s: float = 600.0
    # Outside every repository, on purpose. `repo` defaults to "." and this used to
    # default to the relative "var/worktrees", so the scratch trees landed *inside*
    # the repo under test — which `WorktreeRunner`'s own comment forbids, because
    # that repo's pytest collects them and its linters lint them, a full copy of the
    # suite rediscovered once per case. This repo gitignores `/var/`; a customer's
    # will not, so they would also watch `?? var/` sit in `git status` for hours.
    # `expanduser()` is applied at the far end (`worktree.py`).
    worktree_dir: str = "~/.cache/tandem/worktrees"

    # T2 escalation on the *served* path (sec 7.2). Off by default, and the default
    # is the interesting part: turning it on runs the repository's test command on
    # every code_change turn, which is both a latency decision and a decision to
    # execute that repository's code on every turn. The eval measures the same
    # thing offline without either cost.
    escalate_on_test_failure: bool = False
    # Whether a patch that will not apply counts as an observed failure on the
    # served path. Off by default because the served base_rev is a committed
    # revision while the user's tree usually is not, so a patch against their
    # uncommitted work fails to apply for reasons that are not the model's.
    escalate_on_apply_failure: bool = False


@dataclass
class AttestConfig:
    audit_log: str = "var/audit.jsonl"
    fsync: bool = False
    # Attach the receipt to wire responses as well as the audit log.
    attach_to_response: bool = True


@dataclass
class ServerConfig:
    # Loopback only. The offline posture (sec 8.6) is a verifiable claim, and
    # binding 0.0.0.0 would break it silently — `OfflineReport` now says so out
    # loud rather than reading lsof's `*:8080` as loopback.
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str | None = None
    # Host-header allow-list. Binding loopback answers "who can reach the socket";
    # this answers "who can drive it", which is a different question once a browser
    # is involved. Any page the operator visits can point an attacker-controlled
    # hostname at 127.0.0.1 and POST to the gateway (DNS rebinding) — and with
    # `api_key` unset, the default, that is a session against a local coding agent.
    # A browser cannot forge the Host header, so pinning it to loopback names is
    # what turns "binds loopback by default" into an actual boundary. An operator
    # who genuinely fronts this with a reverse proxy puts their hostname here.
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]", "::1")


@dataclass
class Config:
    backend: str = "mock"  # mock | mlx
    server: ServerConfig = field(default_factory=ServerConfig)
    tier0: Tier0Config = field(default_factory=Tier0Config)
    tier1: Tier1Config = field(default_factory=Tier1Config)
    router: RouterConfig = field(default_factory=RouterConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    context_scale: ContextScaleConfig = field(default_factory=ContextScaleConfig)
    toolcall: ToolCallConfig = field(default_factory=ToolCallConfig)
    attest: AttestConfig = field(default_factory=AttestConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        cfg = cls()
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if p.exists():
            _apply(cfg, tomllib.loads(p.read_text(encoding="utf-8")))
        _apply_env(cfg)
        return cfg


def _apply(target: Any, data: dict[str, Any]) -> None:
    """Merge a TOML mapping into a dataclass, rejecting unknown keys and wrong types.

    Silently ignoring a typo'd key is how a config change appears to take effect and
    does not — unacceptable when the key in question is `expert_cache_bytes` and the
    number ends up in a published measurement.

    The same argument runs one level down, so the value is checked too. `port =
    "eighty"` used to be stored verbatim and surface as a `TypeError` inside
    uvicorn; `server = "x"` used to *replace the whole dataclass* with a string and
    fail somewhere that names neither `server` nor the file it came from. A config
    error should be reported against the line that caused it.
    """
    hints = _hints(type(target))
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key: {type(target).__name__}.{key}")
        current = getattr(target, key)
        where = f"{type(target).__name__}.{key}"
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(
                    f"{where} is a section: expected a TOML table ([{key}] with its "
                    f"own keys), got {_describe(value)}"
                )
            _apply(current, value)
        else:
            expected = hints.get(key)
            if expected is not None and not _matches(expected, value):
                raise ValueError(
                    f"{where} expects {_type_name(expected)}, got {_describe(value)}"
                )
            # TOML has arrays, not tuples. A tuple-typed field means "sequence that
            # nothing mutates later", and the file cannot express that itself.
            if get_origin(expected) is tuple and isinstance(value, list):
                value = tuple(value)
            setattr(target, key, value)


@lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    """Resolved annotations for a config dataclass.

    `from __future__ import annotations` makes `Field.type` a string, and matching
    on strings would break the first time someone writes `Optional[str]` instead of
    `str | None`. Cached because `_apply` runs per key and these classes never change.
    """
    return get_type_hints(cls)


def _matches(expected: Any, value: Any) -> bool:
    origin = get_origin(expected)
    if origin in (Union, UnionType):
        return any(_matches(arm, value) for arm in get_args(expected))
    if origin is list:
        if not isinstance(value, list):
            return False
        (item,) = get_args(expected) or (Any,)
        return item is Any or all(_matches(item, v) for v in value)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):  # a TOML array is a list
            return False
        args = get_args(expected)
        if not args:
            return True
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches(args[0], v) for v in value)
        return len(value) == len(args) and all(
            _matches(a, v) for a, v in zip(args, value)
        )
    if expected is type(None):
        return value is None
    # `isinstance(True, int)` is True, so a plain isinstance would accept
    # `port = true` and `mtp = 1`. TOML distinguishes them; so does this.
    if expected is bool:
        return type(value) is bool
    if expected is int:
        return type(value) is int
    if expected is float:
        return type(value) is int or type(value) is float  # TOML `0` for a float
    return isinstance(value, expected)


def _type_name(expected: Any) -> str:
    origin = get_origin(expected)
    if origin in (Union, UnionType):
        return " or ".join(_type_name(a) for a in get_args(expected))
    if origin in (list, tuple):
        return f"a list of {_type_name(get_args(expected)[0])}"
    return {
        bool: "true or false",
        int: "an integer",
        float: "a number",
        str: "a string",
        type(None): "nothing",
    }.get(expected, getattr(expected, "__name__", str(expected)))


def _describe(value: Any) -> str:
    return f"{type(value).__name__} {value!r}"


_ENV_MAP: dict[str, tuple[str, ...]] = {
    "TANDEM_BACKEND": ("backend",),
    "TANDEM_PORT": ("server", "port"),
    "TANDEM_HOST": ("server", "host"),
    "TANDEM_API_KEY": ("server", "api_key"),
    "TANDEM_TIER0_MODEL": ("tier0", "model"),
    "TANDEM_TIER0_CONTAINER": ("tier0", "container_path"),
    "TANDEM_ADAPTER_DIR": ("tier0", "adapter_dir"),
    "TANDEM_TIER1_ENABLED": ("tier1", "enabled"),
    "TANDEM_TIER1_ENDPOINT": ("tier1", "endpoint"),
    "TANDEM_CANDIDATES": ("router", "candidates"),
    "TANDEM_NO_COMPACT": ("compaction", "enabled"),
    "TANDEM_AUDIT_LOG": ("attest", "audit_log"),
}


def _apply_env(cfg: Config) -> None:
    for env, path in _ENV_MAP.items():
        raw = os.environ.get(env)
        if raw is None:
            continue
        target: Any = cfg
        for part in path[:-1]:
            target = getattr(target, part)
        name = path[-1]
        current = getattr(target, name)
        value: Any = raw
        if isinstance(current, bool) or current is None and env.endswith("_ENABLED"):
            value = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        # TANDEM_NO_COMPACT=1 must *disable* compaction; the flag is negated.
        if env == "TANDEM_NO_COMPACT":
            value = not (raw.strip().lower() in ("1", "true", "yes", "on"))
        setattr(target, name, value)
