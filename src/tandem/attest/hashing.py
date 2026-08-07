"""Content hashing for attestation (spec sec 9).

Everything that can change a model's output gets a hash: the tier-0 container, the
mounted adapter, the routing profile, the tier-1 container, the engine commit and
the compaction template. The receipt carries them; the audit log records them; the
determinism claim (sec 9.3) is stated in terms of them.

BLAKE3 throughout — Apache-2.0/CC0, fast enough that hashing a 20 GB container at
startup is not a startup cost worth caching around, and tree-hashing means a
directory digest is cheap to keep stable.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import blake3

# Read in 4 MiB blocks. Large enough that syscall overhead disappears against NVMe,
# small enough that hashing a container never spikes RSS on a memory-tight box (sec 2.1).
_CHUNK = 4 << 20

# Files that live beside weights but do not change what the model computes.
_IGNORED_NAMES = frozenset({".DS_Store", "README.md", "LICENSE", ".gitattributes"})


def hash_bytes(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def hash_text(text: str) -> str:
    return hash_bytes(text.encode("utf-8"))


def hash_file(path: str | os.PathLike[str]) -> str:
    h = blake3.blake3()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: str | os.PathLike[str]) -> str:
    """Digest of a directory: order-independent, name-sensitive, content-sensitive.

    Feeds each file as ``relpath\\0size\\0filehash\\n`` in sorted relpath order, so the
    digest is stable across filesystems that enumerate differently, and changes if a
    file is renamed, resized or edited.
    """
    root_path = Path(root)
    if root_path.is_file():
        return hash_file(root_path)
    if not root_path.is_dir():
        raise FileNotFoundError(f"no such container path: {root}")

    entries: list[tuple[str, int, str]] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name in _IGNORED_NAMES or name.startswith("."):
                continue
            full = Path(dirpath) / name
            if not full.is_file():  # dangling symlink
                continue
            rel = full.relative_to(root_path).as_posix()
            entries.append((rel, full.stat().st_size, hash_file(full)))

    h = blake3.blake3()
    for rel, size, digest in sorted(entries):
        h.update(f"{rel}\0{size}\0{digest}\n".encode())
    return h.hexdigest()


def tree_signature(root: Path) -> str:
    """A cheap change-detector for a directory: stat only, no reads.

    Deliberately *not* the directory's own (mtime, size): editing a file in place
    changes neither, so memoising on those would keep serving a stale digest for a
    hand-edited adapter — a receipt attesting to weights that are no longer on
    disk. That is precisely the silent falsehood this module exists to prevent.

    Stat-ing every file in a container is microseconds against the seconds it takes
    to read and hash it, so the cache still earns its place.
    """
    if root.is_file():
        st = root.stat()
        return f"{st.st_size}:{st.st_mtime_ns}"
    h = blake3.blake3()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            full = Path(dirpath) / name
            try:
                st = full.stat()
            except OSError:
                continue
            rel = full.relative_to(root).as_posix()
            h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


@lru_cache(maxsize=64)
def _cached_tree_hash(root: str, signature: str) -> str:
    return hash_tree(root)


def hash_artefact(path: str | os.PathLike[str] | None) -> str | None:
    """Hash a container/adapter/profile path, memoised on its stat signature.

    Model containers are immutable in practice, so re-reading 20 GB on every request
    is pure waste; but any edit anywhere in the tree changes the signature and forces
    a re-hash. Returns None for a missing path so a receipt can honestly say "not
    mounted" rather than inventing a digest.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return _cached_tree_hash(str(p.resolve()), tree_signature(p))


def short(digest: str | None, n: int = 12) -> str:
    return "—" if not digest else digest[:n]
