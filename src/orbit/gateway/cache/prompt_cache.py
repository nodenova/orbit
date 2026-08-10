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

Two rules hold the rest of it together, and both are load-bearing:

* **One boundary calculation.** A stored key is *chosen out of* the same index that
  lookup probes (`aligned_mark`), never computed alongside it. The disk cache used
  to align down to a strict byte multiple while this index advances boundaries
  forward past UTF-8 continuation bytes, so one multi-byte character — a CJK glyph,
  an emoji, an accented identifier, a box-drawing character in a pasted diff — put
  every stored key outside the set of digests lookup would ever probe. The cache
  went on writing entries and never hit one again, for the life of that
  conversation, silently.
* **Every entry is partitioned by backend identity** (`Backend.state_key`: backend,
  container and adapter). The digest describes the *bytes*, and says nothing about
  which model prefilled them. Two turns with the same prefix under adapters `a1`
  and `a2` are the same bytes and different states, so an unpartitioned index hands
  one conversation's sampled blocks and KV state to the other — fluent, plausible
  and wrong, which is what CLAUDE.md's "a `KVState` carries the identity it belongs
  to" exists to prevent. The disk path checks this and always has; this one now
  does too.
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
    # lifetime and its byte cost. An entry with `state=None` is a replay-map and
    # bookkeeping entry only: it can save no prefill, and a caller that treats it
    # as a hit skips the disk restore that would have (see `PrefixHit.has_state`).
    state: Any = None
    size_bytes: int = 0
    # Next-token logits at the end of the prefix, so a restored snapshot continues
    # without spending an extra decode step re-deriving them (sec 8.4).
    next_logits: bytes | None = None
    # tool_id -> exact sampled block, carried with the state (sec 8.5.5).
    replay: dict[str, str] = field(default_factory=dict)
    # Backend identity this entry belongs to (`Backend.state_key`), set by `store`.
    # The digest covers the prompt bytes; this covers which model produced the
    # state for them, and the two are only a valid pair together.
    state_key: str = ""


@dataclass(frozen=True, slots=True)
class PrefixHit:
    entry: CacheEntry
    # Bytes of the new prompt already covered — everything after this needs prefill.
    covered_bytes: int
    # Bytes that still need prefilling.
    remaining_bytes: int

    @property
    def has_state(self) -> bool:
        """Whether this hit can actually skip prefill.

        A hit without state saves nothing: the prefix still has to be prefilled, so
        the only thing it carries is the replay map. Callers have to distinguish the
        two, because returning early on a stateless hit shadows the disk snapshot
        that would have made the turn warm — which made turn 2 in a long-lived
        process *slower* than the same turn in a fresh one while the trace reported
        a 93% hit.
        """
        return self.entry.state is not None


def chunk_digests(
    text: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> list[tuple[int, str]]:
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


def aligned_mark(
    text: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> tuple[int, str] | None:
    """The longest prefix mark of `text` that a *longer* prompt reproduces.

    Returns `(prefix_byte_length, digest)`, or None when `text` holds less than one
    whole chunk. This is the only place a stored key is chosen — memory and disk
    both take it from here — so a stored key is by construction one that lookup will
    probe on the next turn.

    Every mark from `chunk_digests` is reproduced verbatim by any longer prompt with
    this text as a prefix, *except* possibly the last: a final short chunk ends
    where the text does, and in a longer prompt that chunk keeps going, so its
    digest is over different bytes and its boundary lands somewhere else. (The
    continuation-byte advance needs no such care: valid UTF-8 never ends mid
    codepoint, so the byte after the boundary is the same byte in both.) So the
    answer is the last mark when its chunk is full-length, and the one before it
    otherwise.
    """
    marks = chunk_digests(text, chunk_bytes)
    if not marks:
        return None
    previous_end = marks[-2][0] if len(marks) > 1 else 0
    if marks[-1][0] - previous_end >= chunk_bytes:
        return marks[-1]
    return marks[-2] if len(marks) > 1 else None


def align_down(text: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    """Trim a rendered prefix down to a chunk boundary before a cold save.

    Sec 8.4: trimming a small token suffix and aligning down avoids the case where a
    BPE boundary shifts by a token or two and the whole entry misses. Cutting on a
    codepoint edge as well, since a half-character prefix is not a prompt.

    The boundary is taken from `aligned_mark` rather than computed here. The
    separate arithmetic this used to carry (`(len(data) // chunk_bytes) *
    chunk_bytes`) disagreed with the index in two ways: it drifted out of the probe
    set on any non-ASCII prompt, and on an exact multiple it indexed one past the
    end of the buffer and raised — roughly one request in 1024, thrown *after* the
    model had answered.
    """
    mark = aligned_mark(text, chunk_bytes)
    if mark is None:
        return ""
    # Exact slice, no `errors="ignore"`: the mark is on a codepoint edge by
    # construction, and silently dropping a byte here would hand back a string
    # whose bytes are not the bytes the digest was taken over.
    return text.encode("utf-8")[: mark[0]].decode("utf-8")


def digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _refusal(entry: CacheEntry, state_key: str, prefix_bytes: int) -> str | None:
    """Why this entry must not be filed or served under `state_key`, or None.

    The partition key already separates identities, so this is defence in depth —
    but what it defends against is a KV state restored into a model that never saw
    its own prefix, or one covering bytes past the point the prompts still agree.
    Both are silent: fluent output, wrong state, and a receipt naming the adapter
    that did not produce it. Two comparisons are cheap next to that.
    """
    state = entry.state
    if state is None:
        return None
    key = getattr(state, "key", None)
    if key is not None and key != state_key:
        return "state identity does not match its partition"
    covered = getattr(state, "prefix_bytes", None)
    if isinstance(covered, int) and covered > prefix_bytes:
        return "state covers more bytes than the digest attests"
    return None


class PromptCache:
    """LRU over a byte budget, indexed by (backend identity, chunk-aligned digest)."""

    def __init__(
        self, budget_bytes: int = 2 << 30, chunk_bytes: int = DEFAULT_CHUNK_BYTES
    ):
        self.budget_bytes = budget_bytes
        self.chunk_bytes = chunk_bytes
        # Keyed on (state_key, digest), never on the digest alone: the same bytes
        # under a different adapter are a different entry, not the same one.
        self._entries: OrderedDict[tuple[str, str], CacheEntry] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # Hits that carried no state, so saved no prefill. Counted because a cache
        # reporting a 93% hit rate while saving nothing looks exactly like a
        # healthy one from the trace.
        self.hits_without_state = 0
        self.identity_refusals = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def size_bytes(self) -> int:
        return self._bytes

    def lookup(self, rendered: str, *, state_key: str = "") -> PrefixHit | None:
        """Longest cached chunk-aligned prefix of `rendered` under `state_key`, if any.

        `state_key` is `Backend.state_key(adapter)` — backend, container and adapter.
        An entry filed under a different identity is not a candidate: same bytes,
        different model.
        """
        marks = chunk_digests(rendered, self.chunk_bytes)
        total = len(rendered.encode("utf-8"))
        with self._lock:
            for prefix_bytes, digest in reversed(marks):
                key = (state_key, digest)
                entry = self._entries.get(key)
                if entry is None:
                    continue
                reason = _refusal(entry, state_key, entry.prefix_bytes)
                if reason is not None:
                    # Filed wrongly by someone. Drop it rather than serve it, and
                    # keep looking for a shorter prefix that is sound.
                    self._drop_locked(key)
                    self.identity_refusals += 1
                    continue
                self._entries.move_to_end(key)
                self.hits += 1
                if entry.state is None:
                    self.hits_without_state += 1
                return PrefixHit(
                    entry=entry,
                    covered_bytes=prefix_bytes,
                    remaining_bytes=total - prefix_bytes,
                )
            self.misses += 1
            return None

    def store(
        self, rendered_prefix: str, entry: CacheEntry, *, state_key: str = ""
    ) -> str | None:
        """Cache a state covering `rendered_prefix` under `state_key`.

        The prefix is aligned *down* to a chunk boundary before storing: a state
        that covers a partial chunk is unusable as a prefix match, and storing it
        under an unaligned digest would make it dead weight against the budget.
        Alignment is idempotent, so the caller may hand over the whole rendered
        prompt or an already-trimmed prefix — but whichever it is, the state on the
        entry has to cover the *aligned* prefix, which is what the returned digest
        attests to.

        Returns the digest the entry was filed under, or None if it was not stored.
        Never raises: a cache store runs after the model has already answered, so
        the only honest failure is a later miss.
        """
        mark = aligned_mark(rendered_prefix, self.chunk_bytes)
        if mark is None:
            return None
        prefix_bytes, digest = mark
        reason = _refusal(entry, state_key, prefix_bytes)
        if reason is not None:
            self.identity_refusals += 1
            return None
        entry.digest = digest
        entry.prefix_bytes = prefix_bytes
        entry.state_key = state_key
        key = (state_key, digest)
        with self._lock:
            self._drop_locked(key)
            self._entries[key] = entry
            self._bytes += entry.size_bytes
            self._evict_locked()
        return digest

    def _drop_locked(self, key: tuple[str, str]) -> None:
        existing = self._entries.pop(key, None)
        if existing is not None:
            self._bytes -= existing.size_bytes

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
            # A hit that carried no state prefilled the whole prompt anyway.
            "hits_without_state": self.hits_without_state,
            "evictions": self.evictions,
            "identity_refusals": self.identity_refusals,
        }
