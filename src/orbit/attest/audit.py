"""Append-only local audit log (spec sec 9.2).

One JSONL line per request, local file, no network. The record is exactly the
tuple the spec names:

    (request_id, ts, harness, tier0_hash, adapter_hash, tier1_hash,
     prompt_sha256, output_sha256, tools_invoked, escalated)

plus the receipt fields the sec 9.3 determinism claim is stated in terms of —
seed, sampling, compaction template, engine commit, reduction order, candidate
counts, tier-1 rung — because a receipt the chain does not cover is a claim
nobody can check after the fact, a `seq` and a `prev` link. The spec asks for
append-only; a hash chain is what makes that property *checkable* by the auditor
rather than merely intended by the writer, and it costs two fields.
`verify_chain()` is the check, and its docstring is where the threat model lives:
what an unkeyed chain does and does not detect is not obvious.

Prompt and output are hashed, never stored: the log must be safe to hand to a
compliance reviewer without leaking the customer's source.

Format note: `seq` is part of the hashed payload, so a log written before it
existed will not verify — `verify_chain` names that case rather than guessing.
Rotate the old file; a tamper-evidence artefact that silently accepts two formats
is worth less than one that breaks loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:  # only for an annotation — see receipt_fields()
    from orbit.attest.receipt import Receipt

try:  # POSIX only. Both targets (macOS for Metal, Linux for CI) have it.
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - no Windows target (sec 2.1)
    # A bool rather than rebinding `fcntl` to None: under `# type: ignore` the name
    # keeps its module type, every `if fcntl is None` below reads as unreachable to
    # a type checker, and the Windows degradation path silently stops being checked.
    _HAVE_FCNTL = False

GENESIS = "0" * 64

# How far back to read when resuming the chain. A compliance log grows without
# bound and the tail is re-read on every append; scanning the whole file would
# make appends O(n) in the log's length. One 64 KiB window holds ~100 records,
# and the loop widens it if the last line is further back than that.
_TAIL_WINDOW = 1 << 16


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
    # Everything below is the receipt, bound into the chain (M28). Optional and
    # defaulted so a caller that only has the spec tuple still writes a valid
    # record — an honest null beats refusing to log — but the pipeline passes
    # them all, via receipt_fields(), from the same Receipt it hands the client.
    seed: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    compaction_template: str | None = None
    engine_commit: str | None = None
    reduction_order: str | None = None
    candidates_generated: int | None = None
    candidate_selected: int | None = None
    tier1_rung: str | None = None

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
            # Shaped like the receipt on purpose: a reviewer holding a response
            # and a log line should be able to compare them field for field
            # without a mapping table.
            "seed": self.seed,
            "sampling": {"temperature": self.temperature, "top_p": self.top_p},
            "compaction_template": self.compaction_template,
            "engine_commit": self.engine_commit,
            "reduction_order": self.reduction_order,
            "candidates_generated": self.candidates_generated,
            "candidate_selected": self.candidate_selected,
            "tier1_rung": self.tier1_rung,
        }


def receipt_fields(receipt: Receipt) -> dict[str, Any]:
    """The determinism-claim fields of a receipt, ready to splat into AuditRecord.

    Reads `receipt.as_dict()` — the very dict handed to the client — rather than
    the receipt's attributes, so the log and the response cannot drift apart: if a
    field is renamed or computed differently for the client, the record follows.
    """
    d = receipt.as_dict()
    return {
        "seed": d["seed"],
        "temperature": d["sampling"]["temperature"],
        "top_p": d["sampling"]["top_p"],
        "compaction_template": d["compaction_template"],
        "engine_commit": d["engine_commit"],
        "reduction_order": d["reduction_order"],
        "candidates_generated": d["candidates_generated"],
        "candidate_selected": d["candidate_selected"],
        "tier1_rung": d["tier1"]["rung"],
    }


@dataclass(frozen=True, slots=True)
class ChainTip:
    """Where the chain ends right now: `records` entries, the last one `link`.

    The pair is what an auditor anchors on. A bare link says nothing about how
    long the log was supposed to be; a count without the link is trivially
    restated. Together they pin both ends.
    """

    records: int
    link: str

    @property
    def empty(self) -> bool:
        return self.records == 0


def _link(payload: dict[str, Any], prev: str) -> str:
    """Chain link over the canonical serialisation of the record plus its parent."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{prev}\n{body}".encode()).hexdigest()


def _lock_file(fh: IO[Any], *, exclusive: bool) -> None:
    """Take an advisory whole-file lock, blocking until it is ours.

    flock(2) is what makes read-tail-then-append atomic *between processes*. The
    gateway and a `orbit` CLI run share one log; with a tip cached in memory and
    only a `threading.Lock` to guard it, the second writer chains from a link the
    first already used and the chain forks permanently — and a fork is
    indistinguishable from tampering when someone later investigates an incident
    (H12). The threading lock is still needed: flock is per open-file-description,
    so two threads sharing this process would otherwise share the lock too.

    Windows has no fcntl. Nothing here targets it (Metal, sec 2.1), so rather than
    reach for msvcrt.locking — byte-range, different semantics, untested on any
    machine we have — we degrade to the process-local lock, which is exactly right
    for the single-process case Windows would be.
    """
    if not _HAVE_FCNTL:  # pragma: no cover - no Windows target
        return
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


def _unlock_file(fh: IO[Any]) -> None:
    if not _HAVE_FCNTL:  # pragma: no cover - no Windows target
        return
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _read_tip(fh: BinaryIO) -> ChainTip:
    """Tip of the chain from an open file, reading the tail rather than the file.

    Caller must already hold the lock: the whole point is that the tip is read at
    the last possible moment before the append.
    """
    fh.seek(0, os.SEEK_END)
    size = fh.tell()
    if size == 0:
        return ChainTip(0, GENESIS)
    window = _TAIL_WINDOW
    while True:
        start = max(0, size - window)
        fh.seek(start)
        chunk = fh.read(size - start)
        lines = chunk.split(b"\n")
        if start > 0:
            lines = lines[1:]  # the first is a fragment of an earlier record
        for raw in reversed(lines):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
                seq = rec["seq"]
                if isinstance(seq, bool) or not isinstance(seq, int):
                    raise TypeError("seq is not an integer")
                return ChainTip(seq + 1, str(rec["link"]))
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                # A torn final line (crash mid-write) must not silently reroot the
                # chain — skip it, continue from the last complete record, and
                # leave the torn bytes in place for verify_chain() to report.
                continue
        if start == 0:
            # Nothing parseable anywhere. Rerooting at GENESIS here does not hide
            # the damage: the record we are about to write says prev=GENESIS while
            # earlier bytes remain, so verify_chain() reports the break.
            return ChainTip(0, GENESIS)
        window *= 4


class AuditLog:
    """Append-only JSONL writer. Safe across threads *and* across processes."""

    def __init__(self, path: str | os.PathLike[str], *, fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # fsync per record is correct for a compliance log and costs ~a millisecond
        # on NVMe, but it is off by default so the latency suite (sec 10.4) measures
        # the model rather than the filesystem. Turn it on in a deployed install.
        self._fsync = fsync
        self._lock = threading.Lock()
        # No cached tip. Caching it at construction is precisely what let a second
        # writer fork the chain (H12); the tip is re-read under the lock on every
        # append, which costs one seek and one 64 KiB read against an fsync.

    def head(self) -> ChainTip:
        """The chain's current tip, read from disk.

        Read rather than remembered, for the same reason `append` re-reads: another
        process may have written since. Callers anchor on this — put `link` in the
        receipt, a ticket, or an operator's notes, then hand it back to
        `verify_chain(expected_tip=...)`. An anchor kept only in the log file is not
        an anchor.
        """
        if not self.path.exists():
            return ChainTip(0, GENESIS)
        with open(self.path, "rb") as fh:
            _lock_file(fh, exclusive=False)
            try:
                return _read_tip(fh)
            finally:
                _unlock_file(fh)

    def append(self, record: AuditRecord) -> str:
        """Append one record, returning its link — which is the log's new tip."""
        payload_base = record.as_dict()
        # Binary + append mode: text mode cannot seek to a byte offset, and
        # O_APPEND makes the write land at the true end of the file even if
        # another process grew it while we were looking.
        with self._lock, open(self.path, "ab+") as fh:
            _lock_file(fh, exclusive=True)
            try:
                tip = _read_tip(fh)
                payload = {**payload_base, "seq": tip.records}
                link = _link(payload, tip.link)
                line = json.dumps(
                    {**payload, "prev": tip.link, "link": link},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                fh.write(line.encode("utf-8") + b"\n")
                fh.flush()
                if self._fsync:
                    os.fsync(fh.fileno())
            finally:
                # After the flush, never before: releasing the lock with the
                # record still in a buffer lets the next writer read a tip that
                # is about to move.
                _unlock_file(fh)
        return link

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def verify_chain(
    path: str | os.PathLike[str],
    *,
    expected_tip: str | None = None,
    allow_empty: bool = False,
) -> tuple[bool, str]:
    """Recompute every link, check every sequence number, and check where it ends.

    Returns (ok, reason).

    **Detected on the file alone:** an in-place edit of any field (the link stops
    recomputing), insertion, reordering, deletion of an interior record (the `prev`
    of the following record no longer matches), a gap or repeat in the sequence
    numbers, a torn or non-JSON line, and a record predating the sequenced format.

    **Detected only with `expected_tip`:** trailing truncation, and
    truncate-then-append of forged records. Both leave a chain that is internally
    perfect — a shorter log is a valid log — so the end has to be pinned from
    outside. Pass the link `AuditLog.append()` returned (or `AuditLog.head().link`)
    as it was recorded somewhere the writer does not control: the receipt that went
    back to the client, a ticket, an operator's notes. The sequence numbers make the
    arithmetic legible in the reason string ("N records, ending at seq M"), but N
    and M are both under the writer's control, so they are not themselves an anchor.

    **Not detected, by construction:** a chain rewritten from GENESIS by whoever
    holds the file. The links are unkeyed SHA-256, so anyone who can write the log
    can recompute every link after changing anything, and `AuditLog` itself is the
    tool for it. If the operator is inside the threat model, this needs a key the
    operator does not hold — HMAC-SHA256 over the same (prev, payload) bytes, or
    per-record signing — with the tip anchored off the machine. What the unkeyed
    chain buys today: a reviewer holding an anchor can prove the log they were
    handed is the log that was written, and accidental corruption — a crash, a
    truncating editor, a botched rsync, two writers racing — is always visible.

    A missing file is *not* an empty log: for a compliance artefact "no evidence"
    must not read as "verified", which is what returning True for an absent file
    used to say. `allow_empty=True` accepts a log with nothing in it yet (a gateway
    that has served nothing), and the reason string still distinguishes absent from
    empty.
    """
    p = Path(path)
    if not p.exists():
        if expected_tip is not None:
            return False, f"no audit log at {p} — the anchored tip cannot be present"
        if allow_empty:
            return True, f"no audit log at {p} — nothing written yet, and none expected"
        return False, f"no audit log at {p} — an absent log is not an empty one"

    prev = GENESIS
    n = 0
    with open(p, encoding="utf-8") as fh:
        # Shared lock: a reader that catches a writer mid-append would report a torn
        # line as tampering. Blocks only for the duration of one append.
        _lock_file(fh, exclusive=False)
        try:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return False, f"line {lineno}: not JSON"
                if rec.get("prev") != prev:
                    return (
                        False,
                        f"line {lineno}: broken link (record removed or reordered)",
                    )
                seq = rec.get("seq")
                if seq is None:
                    return (
                        False,
                        f"line {lineno}: no sequence number (log predates the format)",
                    )
                if isinstance(seq, bool) or not isinstance(seq, int) or seq != n:
                    return (
                        False,
                        f"line {lineno}: sequence number {seq!r}, expected {n}",
                    )
                payload = {k: v for k, v in rec.items() if k not in ("prev", "link")}
                if _link(payload, prev) != rec.get("link"):
                    return (
                        False,
                        f"line {lineno}: record content does not match its link",
                    )
                prev = rec["link"]
                n += 1
        finally:
            _unlock_file(fh)

    if expected_tip is not None and prev != expected_tip:
        return False, (
            f"{n} records verified but the chain ends at {prev[:12]}…, "
            f"not the anchored tip {expected_tip[:12]}… (records removed from the end)"
        )
    if n == 0:
        if allow_empty:
            return True, "0 records — the log is empty, as expected"
        return False, "0 records — an empty log is not evidence"
    anchored = ", tip anchored" if expected_tip is not None else ""
    return True, f"{n} records verified, ending at seq {n - 1}{anchored}"


def now() -> float:
    return time.time()
