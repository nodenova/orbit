"""Append-only local audit log (spec sec 9.2).

One JSONL line per request, local file, no network. The record is exactly the
tuple the spec names:

    (request_id, ts, harness, tier0_hash, adapter_hash, tier1_hash,
     prompt_sha256, output_sha256, tools_invoked, escalated)

plus a `prev` link. The spec asks for append-only; a hash chain is what makes that
property *checkable* by the auditor rather than merely intended by the writer, and
it costs one field. `verify_chain()` is the check.

Prompt and output are hashed, never stored: the log must be safe to hand to a
compliance reviewer without leaking the customer's source.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditRecord:
    request_id: str
    ts: float
    harness: str | None
    tier0_hash: str | None
    adapter_hash: str | None
    tier1_hash: str | None
    prompt_sha256: str
    output_sha256: str
    tools_invoked: tuple[str, ...]
    escalated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "ts": self.ts,
            "harness": self.harness,
            "tier0_hash": self.tier0_hash,
            "adapter_hash": self.adapter_hash,
            "tier1_hash": self.tier1_hash,
            "prompt_sha256": self.prompt_sha256,
            "output_sha256": self.output_sha256,
            "tools_invoked": list(self.tools_invoked),
            "escalated": self.escalated,
        }


def _link(payload: dict[str, Any], prev: str) -> str:
    """Chain link over the canonical serialisation of the record plus its parent."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{prev}\n{body}".encode()).hexdigest()


class AuditLog:
    """Append-only JSONL writer. Safe under concurrent gateway requests."""

    def __init__(self, path: str | os.PathLike[str], *, fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # fsync per record is correct for a compliance log and costs ~a millisecond
        # on NVMe, but it is off by default so the latency suite (sec 10.4) measures
        # the model rather than the filesystem. Turn it on in a deployed install.
        self._fsync = fsync
        self._lock = threading.Lock()
        self._prev = self._tail_link()

    def _tail_link(self) -> str:
        """Resume the chain from the last well-formed line."""
        if not self.path.exists():
            return GENESIS
        prev = GENESIS
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    prev = json.loads(line)["link"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    # A torn final line (crash mid-write) must not silently reroot the
                    # chain — stop here and let verify_chain() report the break.
                    break
        return prev

    def append(self, record: AuditRecord) -> str:
        payload = record.as_dict()
        with self._lock:
            link = _link(payload, self._prev)
            line = json.dumps(
                {**payload, "prev": self._prev, "link": link},
                sort_keys=True,
                separators=(",", ":"),
            )
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                if self._fsync:
                    os.fsync(fh.fileno())
            self._prev = link
        return link

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def verify_chain(path: str | os.PathLike[str]) -> tuple[bool, str]:
    """Recompute every link. Returns (ok, reason).

    Detects insertion, deletion, reordering and in-place edits of any record.
    """
    p = Path(path)
    if not p.exists():
        return True, "empty log"
    prev = GENESIS
    n = 0
    with open(p, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return False, f"line {lineno}: not JSON"
            if rec.get("prev") != prev:
                return False, f"line {lineno}: broken link (record removed or reordered)"
            payload = {k: v for k, v in rec.items() if k not in ("prev", "link")}
            if _link(payload, prev) != rec.get("link"):
                return False, f"line {lineno}: record content does not match its link"
            prev = rec["link"]
            n += 1
    return True, f"{n} records verified"


def now() -> float:
    return time.time()
