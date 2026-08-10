"""Disk KV cache (spec sec 8.4).

Survives a restart, so the first turn after a reload is not a 30 s cold prefill.

Three specifics from the spec, each with a reason worth keeping written down:

* **Ordinary read/write, never mmap.** A process already mapping ~30 GB of weights
  should not add more VM mappings [V, antirez]. This is the single most likely place
  for someone to "optimise" later and regress the whole runtime, so the file format
  is deliberately a plain sequential read.
* **Keyed on the SHA-256 of the rendered byte prefix.** Not on a message list, not
  on a conversation id — on the exact bytes the model prefilled. Anything else
  restores a state that does not match the prompt.
* **Stores next-token logits alongside the token ids**, so a restored snapshot
  continues without an extra decode step.

The file also carries the tool-replay map (sec 8.5.5): clients hand back normalised
JSON rather than the model's sampled text, and re-rendering it differently breaks
the byte prefix — so the exact sampled block has to survive a restart with the
state it belongs to.

Two disciplines the format only holds up under if both are kept:

* **A corrupt entry is a miss, never an error.** That has to include the corruption
  that still parses — a truncated file, or one file's bytes under another's name.
  `read` returns short at end of file without raising, so every section is checked
  against the header and the header's digest against the name on disk. A snapshot
  that comes back short is handed to `accepts_state`, which checks the identity key
  and not the bytes, and is then prefilled as if it were the prompt.
* **A store failure is a miss too.** `put` is called on a turn the model has already
  answered — in the gateway, before the receipt and the audit record exist — so an
  exception out of it loses a completed answer and its attestation. It returns None
  and counts the reason instead.

Layout (little-endian, single sequential read):

    magic   8   b"ORBIT_KV"
    version 4   uint32 = 1
    hdr_len 4   uint32          -- JSON header length
    header  N   JSON            -- digest, token count, sizes, replay map
    tokens  4*T uint32          -- exact token ids
    logits  L   raw bytes       -- next-token logits at end of prefix
    state   S   raw bytes       -- backend-opaque KV blob
"""

from __future__ import annotations

import array
import contextlib
import json
import os
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# `align_down` used to live here with its own boundary arithmetic, and the two
# calculations disagreed: the prefix index advances a boundary forward past UTF-8
# continuation bytes, this one aligned to a strict byte multiple and backed down.
# One multi-byte character in a prompt was enough to put every stored key outside
# the set of digests lookup probes — entries written forever, never read once. The
# boundary now has a single implementation, next to the index it has to agree with,
# and is re-exported here because this is where callers import it from.
#
# The redundant-looking alias is the explicit re-export form. mypy runs with
# `no_implicit_reexport` (strict), under which plain `import align_down` is private
# to this module and every caller becomes an attr-defined error.
from orbit.gateway.cache.prompt_cache import align_down as align_down  # noqa: PLC0414

MAGIC = b"ORBIT_KV"
# v2 added `state_key`. A v1 entry carries no backend identity, so there is no way
# to tell whether restoring it would continue a conversation in a different model —
# `get` refuses any version it does not know, which turns every stale entry into a
# cache miss rather than a silent wrong answer.
VERSION = 2
_HDR = struct.Struct("<8sII")

_HEX = frozenset("0123456789abcdef")
_DIGEST_CHARS = 64
# A `.tmp` this old cannot still be in flight; anything younger might belong to
# another process publishing right now.
_TMP_MAX_AGE_S = 3600.0


@dataclass
class KVSnapshot:
    digest: str
    token_ids: list[int] = field(default_factory=list)
    next_logits: bytes = b""
    state_blob: bytes = b""
    replay: dict[str, str] = field(default_factory=dict)
    # Bytes of rendered prompt this snapshot covers.
    prefix_bytes: int = 0
    created_ts: float = 0.0
    # Backend identity this state belongs to (`Backend.state_key`). Restoring a
    # state under a different container or adapter is silently wrong, so the key
    # travels with the bytes and is checked before the state is ever used.
    state_key: str = ""

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    def size_bytes(self) -> int:
        return 4 * len(self.token_ids) + len(self.next_logits) + len(self.state_blob)


class DiskKVCache:
    """A directory of KV snapshots under a byte budget."""

    def __init__(self, root: str | os.PathLike[str], budget_bytes: int = 20 << 30):
        self.root = Path(root)
        _own_dir(self.root)
        self.budget_bytes = budget_bytes
        self._lock = threading.Lock()
        # Running size estimate, None until the first scan. See `_enforce_budget`.
        self._known_bytes: int | None = None
        # A cache that cannot write has to be visible somewhere, or it is
        # indistinguishable from one that simply never hits.
        self.store_errors = 0
        self.last_store_error: str | None = None

    def _path(self, digest: str) -> Path:
        # Two-level fan-out; a flat directory of tens of thousands of entries is
        # slow to enumerate on APFS and this costs nothing.
        #
        # The digest is validated rather than trusted: `get` and `put` are public,
        # and a key like `../../pwned` would otherwise resolve straight out of the
        # cache root. Exact-length lowercase hex is what the prefix index produces;
        # anything else is a caller bug, not an entry.
        if not _is_digest(digest):
            raise ValueError(f"not a sha-256 hex digest: {digest!r}")
        return self.root / digest[:2] / f"{digest}.tkv"

    def has(self, digest: str) -> bool:
        if not _is_digest(digest):
            return False
        return self._path(digest).exists()

    def put(self, snap: KVSnapshot) -> Path | None:
        """Publish a snapshot, or return None if it could not be stored.

        Never raises. A cache is an optimisation, and this one is written at the end
        of a turn the model has already answered: in the gateway `_remember` runs
        before the receipt is built and before the audit record is appended, so an
        exception here does not just lose a cache entry — it loses the served answer
        and erases the turn from the sec 9.2 chain. Every failure degrades to a
        later miss and is counted in `stats()`.
        """
        try:
            return self._put(snap)
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad: see the docstring. Anything that reaches here is,
            # by definition, something the caller cannot do anything about — it has
            # already produced its answer.
            self._note_store_error(f"{type(exc).__name__}: {exc}")
            return None

    def _put(self, snap: KVSnapshot) -> Path | None:
        path = self._path(snap.digest)
        _own_dir(path.parent)
        header = json.dumps(
            {
                "digest": snap.digest,
                "n_tokens": snap.n_tokens,
                "logits_bytes": len(snap.next_logits),
                "state_bytes": len(snap.state_blob),
                "prefix_bytes": snap.prefix_bytes,
                "created_ts": snap.created_ts or time.time(),
                "replay": snap.replay,
                "state_key": snap.state_key,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        try:
            tokens = array.array("I", snap.token_ids)
        except (OverflowError, TypeError, ValueError):
            # -1 and -100 are ordinary padding sentinels, and an id that does not
            # fit a uint32 is not a token id at all. Either way the snapshot cannot
            # round-trip through this format, so it is refused as a store failure
            # rather than raised: `OverflowError` is an `ArithmeticError`, which no
            # caller of a cache thinks to catch.
            self._note_store_error("token ids outside uint32")
            return None
        if sys_is_big_endian():
            tokens.byteswap()

        # A temp name unique to this writer. A shared `<digest>.tmp` is not safe
        # across processes — two `orbit serve` instances, or a CLI beside a server,
        # share `var/kvcache` by default — and both of them `os.replace` the same
        # path: one gets `FileNotFoundError` from a temp the other already renamed,
        # or publishes a file the other was still writing.
        tmp = path.parent / f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            # 0600 and O_EXCL: the name is this writer's alone, and the contents are
            # the user's prompts in cleartext — including the model's own sampled
            # tool-call blocks, so a `read_file` call naming `~/.ssh/id_rsa` is in
            # there verbatim, in a directory that defaults to inside their repo.
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            # Plain buffered write. No mmap — see module docstring.
            with open(fd, "wb") as fh:
                fh.write(_HDR.pack(MAGIC, VERSION, len(header)))
                fh.write(header)
                fh.write(tokens.tobytes())
                fh.write(snap.next_logits)
                fh.write(snap.state_blob)
                fh.flush()
                # Durability before publication. `os.replace` orders the rename
                # against other renames, not against the data behind it: without
                # this the directory entry can outlive a crash that the blocks did
                # not, which is a torn file published under a digest it does not
                # match. (The directory itself is not synced — a rename lost the
                # other way is only a miss.)
                os.fsync(fh.fileno())
            size = os.stat(tmp).st_size
            os.replace(tmp, path)
        except BaseException:
            # Includes cancellation. An abandoned temp is bytes on the disk that
            # nothing accounts for, so it goes out with the failure that made it.
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        self._enforce_budget(added_bytes=size)
        return path

    def get(self, digest: str) -> KVSnapshot | None:
        if not _is_digest(digest):
            return None
        path = self._path(digest)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                magic, version, hdr_len = _HDR.unpack(fh.read(_HDR.size))
                if magic != MAGIC or version != VERSION:
                    return None
                raw_header = fh.read(hdr_len)
                if len(raw_header) != hdr_len:
                    return None
                header = json.loads(raw_header.decode("utf-8"))
                # The name on disk is the only thing tying this file to the prefix
                # the caller asked about; the header is just bytes inside it. A file
                # whose header names a different digest is somebody else's entry —
                # a stale rename, a half-published write, a copied file — and
                # returning it restores a state for a prompt that was never
                # prefilled, under a `digest` field the caller then believes.
                if header.get("digest") != digest:
                    return None
                n_tokens = int(header["n_tokens"])
                logits_bytes = int(header["logits_bytes"])
                state_bytes = int(header["state_bytes"])
                if n_tokens < 0 or logits_bytes < 0 or state_bytes < 0:
                    return None
                token_bytes = fh.read(4 * n_tokens)
                logits = fh.read(logits_bytes)
                state = fh.read(state_bytes)
                # `read` returns short at end of file without raising, so a
                # truncated entry parses perfectly and yields a *shorter* state.
                # Nothing downstream can catch that: `accepts_state` checks the
                # identity key, not the bytes, so the torn blob is attached to the
                # request and prefilled as if it were the prompt. Every section is
                # measured against the header, and the file has to end exactly where
                # the header says — a file that grew was written by something that
                # does not own this format.
                if (
                    len(token_bytes) != 4 * n_tokens
                    or len(logits) != logits_bytes
                    or len(state) != state_bytes
                    or fh.read(1)
                ):
                    return None
                tokens = array.array("I")
                tokens.frombytes(token_bytes)
                if sys_is_big_endian():
                    tokens.byteswap()
        except (
            OSError,
            TypeError,
            ValueError,
            KeyError,
            struct.error,
            json.JSONDecodeError,
        ):
            # A corrupt entry is a cache miss, never an error to the caller.
            return None

        # Touch for LRU. Best-effort: a read-only cache directory is still usable.
        with contextlib.suppress(OSError):
            os.utime(path, None)

        return KVSnapshot(
            digest=digest,
            token_ids=list(tokens),
            next_logits=logits,
            state_blob=state,
            replay=header.get("replay", {}),
            prefix_bytes=int(header.get("prefix_bytes", 0)),
            created_ts=float(header.get("created_ts", 0.0)),
            state_key=header.get("state_key", ""),
        )

    def _enforce_budget(self, added_bytes: int = 0) -> None:
        """Evict least-recently-used entries until the directory fits the budget.

        The scan is the expensive part — one `rglob` plus a `stat` per entry, ~38 ms
        at 5k entries — and it used to run on every `put`, *before* the budget was
        even checked. That is synchronous disk I/O on the event loop, blocking
        between two tokens of every concurrent stream, to answer a question the
        answer to is almost always "no". The directory only grows through `put`, so
        a running total refreshed by each scan decides whether a scan is worth
        doing. The estimate can only be *low* — another process writing into the
        same directory is invisible to it — so the worst case is that eviction waits
        until this process's own writes push the estimate past the budget.
        """
        with self._lock:
            if self._known_bytes is not None:
                self._known_bytes += added_bytes
                if self._known_bytes <= self.budget_bytes:
                    return
            entries, total = self._scan_locked()
            self._known_bytes = total
            if total <= self.budget_bytes:
                return
            for _atime, _name, size, p in sorted(entries):
                try:
                    os.unlink(p)
                except OSError:
                    continue
                total -= size
                self._known_bytes = total
                if total <= self.budget_bytes:
                    return

    def _scan_locked(self) -> tuple[list[tuple[float, str, int, Path]], int]:
        """Every live entry with its size, plus the directory's total bytes.

        Sweeps abandoned `.tmp` files on the way past. A writer killed mid-publish
        leaves one behind, and they used to be invisible to both `stats` and the
        budget: bytes that occupy the disk the 20 GiB is measured against, and that
        nothing ever removed. Only ones older than an hour, because a younger one
        may belong to another process publishing right now.
        """
        entries: list[tuple[float, str, int, Path]] = []
        total = 0
        cutoff = time.time() - _TMP_MAX_AGE_S
        for p in self.root.rglob("*"):
            suffix = p.suffix
            if suffix != ".tkv" and suffix != ".tmp":
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if suffix == ".tmp":
                if st.st_mtime < cutoff:
                    try:
                        os.unlink(p)
                        continue
                    except OSError:
                        pass
                total += st.st_size
                continue
            # `st_atime` first so eviction is least-recently-*used*; the path is in
            # the sort key only to keep it total when two entries share a timestamp,
            # which relatime makes ordinary.
            entries.append((st.st_atime, str(p), st.st_size, p))
            total += st.st_size
        return entries, total

    def _note_store_error(self, reason: str) -> None:
        with self._lock:
            self.store_errors += 1
            self.last_store_error = reason

    def stats(self) -> dict[str, Any]:
        n = 0
        total = 0
        tmp_n = 0
        tmp_total = 0
        for p in self.root.rglob("*"):
            suffix = p.suffix
            if suffix != ".tkv" and suffix != ".tmp":
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if suffix == ".tmp":
                tmp_n += 1
                tmp_total += size
            else:
                n += 1
                total += size
        return {
            "entries": n,
            "bytes": total,
            "budget_bytes": self.budget_bytes,
            # Abandoned publishes: same disk, no entry to show for it.
            "tmp_entries": tmp_n,
            "tmp_bytes": tmp_total,
            "store_errors": self.store_errors,
            "last_store_error": self.last_store_error,
        }


def _is_digest(digest: Any) -> bool:
    return (
        isinstance(digest, str)
        and len(digest) == _DIGEST_CHARS
        and all(c in _HEX for c in digest)
    )


def _own_dir(path: Path) -> None:
    """Create a cache directory readable only by its owner.

    What is in it is the user's conversations in cleartext, and the default location
    is `var/kvcache` inside the repository under test — where 0755/0644 is every
    other local account's to read. `mkdir(mode=...)` is masked by the umask and does
    nothing at all for a directory that already exists, so the mode is set
    explicitly. Best-effort: a cache directory on a filesystem without POSIX modes
    is still a usable cache.
    """
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.chmod(0o700)


def sys_is_big_endian() -> bool:
    import sys

    return sys.byteorder == "big"
