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

Layout (little-endian, single sequential read):

    magic   8   b"TANDEMKV"
    version 4   uint32 = 1
    hdr_len 4   uint32          -- JSON header length
    header  N   JSON            -- digest, token count, sizes, replay map
    tokens  4*T uint32          -- exact token ids
    logits  L   raw bytes       -- next-token logits at end of prefix
    state   S   raw bytes       -- backend-opaque KV blob
"""

from __future__ import annotations

import array
import json
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAGIC = b"TANDEMKV"
# v2 added `state_key`. A v1 entry carries no backend identity, so there is no way
# to tell whether restoring it would continue a conversation in a different model —
# `get` refuses any version it does not know, which turns every stale entry into a
# cache miss rather than a silent wrong answer.
VERSION = 2
_HDR = struct.Struct("<8sII")


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
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget_bytes = budget_bytes
        self._lock = threading.Lock()

    def _path(self, digest: str) -> Path:
        # Two-level fan-out; a flat directory of tens of thousands of entries is
        # slow to enumerate on APFS and this costs nothing.
        return self.root / digest[:2] / f"{digest}.tkv"

    def has(self, digest: str) -> bool:
        return self._path(digest).exists()

    def put(self, snap: KVSnapshot) -> Path:
        path = self._path(snap.digest)
        path.parent.mkdir(parents=True, exist_ok=True)
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

        tokens = array.array("I", snap.token_ids)
        if sys_is_big_endian():
            tokens.byteswap()

        tmp = path.with_suffix(".tmp")
        # Plain buffered write. No mmap — see module docstring.
        with open(tmp, "wb") as fh:
            fh.write(_HDR.pack(MAGIC, VERSION, len(header)))
            fh.write(header)
            fh.write(tokens.tobytes())
            fh.write(snap.next_logits)
            fh.write(snap.state_blob)
        # Atomic publish: a torn KV file that still parses would restore a state
        # that does not match its digest, which is worse than a cache miss.
        os.replace(tmp, path)
        self._enforce_budget()
        return path

    def get(self, digest: str) -> KVSnapshot | None:
        path = self._path(digest)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                magic, version, hdr_len = _HDR.unpack(fh.read(_HDR.size))
                if magic != MAGIC or version != VERSION:
                    return None
                header = json.loads(fh.read(hdr_len).decode("utf-8"))
                n_tokens = int(header["n_tokens"])
                tokens = array.array("I")
                tokens.frombytes(fh.read(4 * n_tokens))
                if sys_is_big_endian():
                    tokens.byteswap()
                logits = fh.read(int(header["logits_bytes"]))
                state = fh.read(int(header["state_bytes"]))
        except (OSError, ValueError, KeyError, struct.error, json.JSONDecodeError):
            # A corrupt entry is a cache miss, never an error to the caller.
            return None

        # Touch for LRU. Best-effort: a read-only cache directory is still usable.
        try:
            os.utime(path, None)
        except OSError:
            pass

        return KVSnapshot(
            digest=header["digest"],
            token_ids=list(tokens),
            next_logits=logits,
            state_blob=state,
            replay=header.get("replay", {}),
            prefix_bytes=int(header.get("prefix_bytes", 0)),
            created_ts=float(header.get("created_ts", 0.0)),
            state_key=header.get("state_key", ""),
        )

    def _enforce_budget(self) -> None:
        with self._lock:
            entries: list[tuple[float, int, Path]] = []
            total = 0
            for p in self.root.rglob("*.tkv"):
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append((st.st_atime, st.st_size, p))
                total += st.st_size
            if total <= self.budget_bytes:
                return
            for _atime, size, p in sorted(entries):
                try:
                    p.unlink()
                except OSError:
                    continue
                total -= size
                if total <= self.budget_bytes:
                    return

    def stats(self) -> dict[str, Any]:
        n = 0
        total = 0
        for p in self.root.rglob("*.tkv"):
            try:
                total += p.stat().st_size
            except OSError:
                continue
            n += 1
        return {"entries": n, "bytes": total, "budget_bytes": self.budget_bytes}


def align_down(text: str, chunk_bytes: int) -> str:
    """Trim a rendered prefix down to a chunk boundary before a cold save.

    Sec 8.4: trimming a small token suffix and aligning down avoids the case where a
    BPE boundary shifts by a token or two and the whole entry misses. Cutting on a
    codepoint edge as well, since a half-character prefix is not a prompt.
    """
    data = text.encode("utf-8")
    if len(data) <= chunk_bytes:
        return ""
    end = (len(data) // chunk_bytes) * chunk_bytes
    while end > 0 and (data[end] & 0xC0) == 0x80:
        end -= 1
    return data[:end].decode("utf-8", errors="ignore")


def sys_is_big_endian() -> bool:
    import sys

    return sys.byteorder == "big"
