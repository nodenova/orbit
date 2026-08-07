"""In-memory prompt cache (spec sec 8.4).

The point is not to cache whole prompts — two turns of a conversation are never
byte-identical. It is that turn *N* shares a long prefix with turn *N-1*, so turn
*N* should prefill only the new tokens and TTFT should stop growing with
conversation length. Measured ~4x TTFT cut on turn 2 in a comparable setup [V].

The mechanism is a chunk-aligned prefix index. Because SHA-256 is streaming, one
pass over the rendered prompt yields the digest of *every* chunk-aligned prefix at
once; lookup is then "longest prefix whose digest we hold", which is a dict probe
per chunk and no rehashing. Chunk alignment is also what gives the BPE slack the
spec asks for: a cache entry never claims a prefix that ends mid-chunk, so a
retokenisation that shifts a boundary by a few tokens costs one chunk, not the hit.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

# 1 KiB ~ 256 tokens at ~4 bytes/token, matching CacheConfig.kv_chunk_tokens.
DEFAULT_CHUNK_BYTES = 1024


@dataclass
class CacheEntry:
    """A cached KV state covering a chunk-aligned prefix of some prompt."""

    digest: str
    prefix_bytes: int
    n_tokens: int
    # Backend-owned KV handle. The cache never inspects it; it only owns its
    # lifetime and its byte cost.
    state: Any = None
    size_bytes: int = 0
    # Next-token logits at the end of the prefix, so a restored snapshot continues
    # without spending an extra decode step re-deriving them (sec 8.4).
    next_logits: bytes | None = None
    # tool_id -> exact sampled block, carried with the state (sec 8.5.5).
    replay: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrefixHit:
    entry: CacheEntry
    # Bytes of the new prompt already covered — everything after this needs prefill.
    covered_bytes: int
    # Bytes that still need prefilling.
    remaining_bytes: int


def chunk_digests(text: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> list[tuple[int, str]]:
    """Digest of every chunk-aligned prefix, in one streaming pass.

    Returns [(prefix_byte_length, sha256_hex), ...] in ascending order. Boundaries
    are advanced to the next UTF-8 codepoint edge so a digest never splits a
    character, which would make the same text hash differently depending on where
    the chunking landed.
    """
    data = text.encode("utf-8")
    out: list[tuple[int, str]] = []
    h = hashlib.sha256()
    pos = 0
    n = len(data)
    while pos < n:
        end = min(pos + chunk_bytes, n)
        # Do not cut inside a multi-byte codepoint.
        while end < n and (data[end] & 0xC0) == 0x80:
            end += 1
        h.update(data[pos:end])
        out.append((end, h.hexdigest()))
        pos = end
    return out


def digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PromptCache:
    """LRU over a byte budget, indexed by chunk-aligned prefix digest."""

    def __init__(self, budget_bytes: int = 2 << 30, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
        self.budget_bytes = budget_bytes
        self.chunk_bytes = chunk_bytes
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def size_bytes(self) -> int:
        return self._bytes

    def lookup(self, rendered: str) -> PrefixHit | None:
        """Longest cached chunk-aligned prefix of `rendered`, if any."""
        marks = chunk_digests(rendered, self.chunk_bytes)
        total = len(rendered.encode("utf-8"))
        with self._lock:
            for prefix_bytes, digest in reversed(marks):
                entry = self._entries.get(digest)
                if entry is not None:
                    self._entries.move_to_end(digest)
                    self.hits += 1
                    return PrefixHit(
                        entry=entry,
                        covered_bytes=prefix_bytes,
                        remaining_bytes=total - prefix_bytes,
                    )
            self.misses += 1
            return None

    def store(self, rendered_prefix: str, entry: CacheEntry) -> None:
        """Cache a state covering `rendered_prefix`.

        The prefix is aligned *down* to a chunk boundary before storing: a state
        that covers a partial chunk is unusable as a prefix match, and storing it
        under an unaligned digest would make it dead weight against the budget.
        """
        marks = chunk_digests(rendered_prefix, self.chunk_bytes)
        if not marks:
            return
        prefix_bytes, digest = marks[-1]
        entry.digest = digest
        entry.prefix_bytes = prefix_bytes
        with self._lock:
            if digest in self._entries:
                self._bytes -= self._entries[digest].size_bytes
                del self._entries[digest]
            self._entries[digest] = entry
            self._bytes += entry.size_bytes
            self._evict_locked()

    def _evict_locked(self) -> None:
        while self._bytes > self.budget_bytes and self._entries:
            _, victim = self._entries.popitem(last=False)
            self._bytes -= victim.size_bytes
            self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "bytes": self._bytes,
            "budget_bytes": self.budget_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "evictions": self.evictions,
        }
