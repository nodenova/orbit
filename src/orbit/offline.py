"""Offline posture verification (spec sec 8.6).

The airgap claim is a feature for the target buyer and **must be verifiable**. This
module ships the check: run `lsof -i -P` during a session and assert nothing but
loopback.

It also emits the harness environment the spec names, and audits the installed
dependency set. The LiteLLM 1.82.7/1.82.8 supply-chain compromise is the standing
precedent for why the dependency list is a security surface, not a convenience —
so `audit_dependencies` reports what is actually installed under the runtime, not
what `pyproject.toml` hopes is installed.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Sec 8.6. Emitted by `env_exports()` and asserted by `check_env()`.
HARNESS_ENV: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
}

# Dependencies the runtime is allowed to have. Anything else in the import graph is
# reported, not silently tolerated.
ALLOWED_DEPENDENCIES = frozenset(
    {
        "fastapi",
        "starlette",
        "uvicorn",
        "pydantic",
        "pydantic_core",
        "httpx",
        "httpcore",
        "h11",
        "anyio",
        "sniffio",
        "idna",
        "certifi",
        "click",
        "typing_extensions",
        "annotated_types",
        "blake3",
        "lmformatenforcer",
        "interegular",
        "mlx",
        "mlx_lm",
        "orbit",
    }
)

_LSOF_LINE = re.compile(
    r"\S+\s+(\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(TCP|UDP)\s+(\S+)"
)

# `observe_connections(ALL_PROCESSES)` snapshots every socket on the machine. It is
# not the default, and the reason is that it does not answer the question: a browser
# or a package manager running alongside the gateway makes `loopback_only` false
# through no fault of orbit's, and a check that is false for reasons the operator
# cannot act on is a check the operator learns to ignore.
ALL_PROCESSES = -1


@dataclass
class Connection:
    pid: int
    proto: str
    endpoint: str
    local: str = ""
    remote: str = ""

    @property
    def is_loopback(self) -> bool:
        for host in (self.local, self.remote):
            if not host:
                continue
            if not _is_loopback_host(host):
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "proto": self.proto,
            "endpoint": self.endpoint,
            "loopback": self.is_loopback,
        }


def _is_loopback_host(host: str) -> bool:
    """Is this host unambiguously *this machine*, and nothing else?

    Three things this deliberately answers `False` to, each of which it once
    answered `True`:

    * `"*"` — how `lsof` prints a socket bound to **every** interface. A wildcard
      bind is the opposite of loopback: `ORBIT_HOST=0.0.0.0 orbit serve` puts an
      unauthenticated coding agent on the LAN, and reporting that as loopback is
      precisely the silent break `ServerConfig.host`'s comment warns about.
      `0.0.0.0` and `::` reach the same answer through `ipaddress`.
    * `""` — no host at all is not a host on this machine. Callers that mean
      "this field is absent" skip it before asking (see `Connection.is_loopback`).
    * `localhost.attacker.example` — `startswith("localhost")` matched it. Only the
      exact name and true subdomains of it resolve to loopback (RFC 6761).
    """
    host = host.strip().strip("[]").rstrip(".").lower()
    if not host or host == "*":
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost" or host.endswith(".localhost")


def endpoint_host(url: str) -> str:
    """The host of a URL — no scheme, credentials, port or path.

    Hand-parsed rather than `urllib.parse.urlsplit` because nothing under
    `src/orbit/` imports `urllib` and `tests/tools/test_export_reviews.py` walks the
    package's imports to keep it that way (sec 8.6). A host is a prefix of a URL,
    so extracting one is three splits, and it is worth that to keep the pin crisp.
    """
    rest = url.strip().split("://", 1)[-1]
    for cut in ("/", "?", "#"):
        rest = rest.split(cut, 1)[0]
    # userinfo@host — a credential in a URL must never be mistaken for the host.
    rest = rest.rsplit("@", 1)[-1]
    return _host_of(rest)


def is_loopback_endpoint(url: str) -> bool:
    """Would traffic to this URL stay on this machine?

    Used to refuse a configured endpoint before a socket exists, rather than to
    describe one that already opened.
    """
    return _is_loopback_host(endpoint_host(url))


def _split_endpoint(endpoint: str) -> tuple[str, str]:
    """Split lsof's `local->remote` NAME field into hosts."""
    local, _, remote = endpoint.partition("->")
    return _host_of(local), _host_of(remote)


def _host_of(part: str) -> str:
    part = part.strip()
    if not part:
        return ""
    if part.startswith("["):
        return part[1 : part.find("]")] if "]" in part else part
    return part.rsplit(":", 1)[0] if ":" in part else part


def observe_connections(pid: int | None = None) -> tuple[list[Connection], str]:
    """Snapshot open sockets via `lsof -i -P`. Returns (connections, note).

    Scoped to one process. `None` means *this* one — pass `ALL_PROCESSES` for the
    machine-wide snapshot, and read that constant's comment before doing so.
    """
    cmd = ["lsof", "-i", "-P", "-n"]
    if pid is None:
        pid = os.getpid()
    if pid != ALL_PROCESSES:
        # `-a` ANDs the selectors. Without it lsof ORs them, so `-i … -p PID` means
        # "every internet socket on the machine, or every file of this process" —
        # which is how the pid argument came to have no effect at all.
        cmd += ["-a", "-p", str(pid)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
    except FileNotFoundError:
        return [], "lsof not available on this machine; cannot verify offline posture"
    except subprocess.SubprocessError as exc:
        return [], f"lsof failed: {exc}"

    out: list[Connection] = []
    for line in proc.stdout.splitlines()[1:]:
        m = _LSOF_LINE.search(line)
        if not m:
            continue
        conn_pid, proto, endpoint = int(m.group(1)), m.group(2), m.group(3)
        local, remote = _split_endpoint(endpoint)
        out.append(
            Connection(
                pid=conn_pid, proto=proto, endpoint=endpoint, local=local, remote=remote
            )
        )
    return out, "ok"


@dataclass
class OfflineReport:
    loopback_only: bool = False
    connections: list[dict[str, Any]] = field(default_factory=list)
    offending: list[dict[str, Any]] = field(default_factory=list)
    env_ok: bool = False
    env_missing: list[str] = field(default_factory=list)
    unexpected_dependencies: list[str] = field(default_factory=list)
    # Fallback ladder rung 4 (sec 5.5) is configured: verdicts leave this machine.
    # A configuration fact, not an observation — `lsof` only sees a call that has
    # already happened, and the posture is wrong from the moment the rung is armed.
    remote_tier1: bool = False
    # Same reasoning, one rung down: an endpoint this process is configured to POST
    # to that is not on this machine. Rung 1's `tier1.endpoint` is the case that
    # matters — it is the *default* rung and it carries none of rung 4's four gates.
    nonloopback_endpoints: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.loopback_only
            and self.env_ok
            and not self.unexpected_dependencies
            and not self.remote_tier1
            and not self.nonloopback_endpoints
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "loopback_only": self.loopback_only,
            "offending_connections": self.offending,
            "n_connections": len(self.connections),
            "env_ok": self.env_ok,
            "env_missing": self.env_missing,
            "unexpected_dependencies": self.unexpected_dependencies,
            "remote_tier1": self.remote_tier1,
            "nonloopback_endpoints": self.nonloopback_endpoints,
            "note": self.note,
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return p


def check_env() -> tuple[bool, list[str]]:
    missing = [k for k, v in HARNESS_ENV.items() if os.environ.get(k) != v]
    return not missing, missing


def env_exports() -> str:
    """Shell to source before starting a harness against this gateway."""
    lines = [f"export {k}={v}" for k, v in HARNESS_ENV.items()]
    lines.append("# Force local API-key auth rather than an account session:")
    lines.append("#   claude --bare")
    return "\n".join(lines)


def audit_dependencies() -> list[str]:
    """Top-level packages present in the import graph but not on the allow-list.

    Reports rather than blocks: a false positive should not stop a session. But an
    unexplained package in a runtime that claims to be airgapped is exactly the
    thing to notice before it matters.
    """
    import sys

    seen: set[str] = set()
    for name in list(sys.modules):
        top = name.split(".", 1)[0]
        if top.startswith("_") or top in seen:
            continue
        seen.add(top)
    stdlib = set(getattr(__import__("sys"), "stdlib_module_names", ()))
    return sorted(
        top
        for top in seen
        if top not in ALLOWED_DEPENDENCIES
        and top not in stdlib
        and not top.startswith("_")
    )


def verify(
    pid: int | None = None,
    *,
    strict_deps: bool = False,
    remote_tier1: bool = False,
    configured_endpoints: Iterable[tuple[str, str]] = (),
) -> OfflineReport:
    """Offline posture: what this process has open, and what it is armed to do.

    `configured_endpoints` is (label, url) pairs the process would POST to — the
    same reasoning as `remote_tier1`, applied to a URL instead of a rung name. The
    claim is about what this process will do, not about what it has done so far, so
    an endpoint pointing off the machine falsifies it before any socket opens.
    """
    connections, note = observe_connections(pid)
    offending = [c for c in connections if not c.is_loopback]
    env_ok, missing = check_env()
    report = OfflineReport(
        loopback_only=not offending and note == "ok",
        connections=[c.as_dict() for c in connections],
        offending=[c.as_dict() for c in offending],
        env_ok=env_ok,
        env_missing=missing,
        unexpected_dependencies=audit_dependencies() if strict_deps else [],
        remote_tier1=remote_tier1,
        nonloopback_endpoints=[
            f"{label}={url}"
            for label, url in configured_endpoints
            if url and not is_loopback_endpoint(url)
        ],
        note=note,
    )
    if note != "ok":
        report.loopback_only = False
    if remote_tier1:
        report.note = (
            "tier1.rung='remote' (sec 5.5 rung 4) is configured: verdicts and the "
            "code they judge leave this machine. The offline claim does not hold "
            f"while it is enabled. [{note}]"
        )
    elif report.nonloopback_endpoints:
        report.note = (
            "configured to send requests off this machine: "
            + ", ".join(report.nonloopback_endpoints)
            + f". The offline claim does not hold while that is set. [{note}]"
        )
    return report
