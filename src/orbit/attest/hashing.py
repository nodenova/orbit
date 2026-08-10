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
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import blake3

# Read in 4 MiB blocks. Large enough that syscall overhead disappears against NVMe,
# small enough that hashing a container never spikes RSS on a memory-tight box (sec 2.1).
_CHUNK = 4 << 20

# Files that live beside weights but do not change what the model computes.
_IGNORED_NAMES = frozenset({".DS_Store", "README.md", "LICENSE", ".gitattributes"})

# Directories that describe the tree rather than belong to it. Skipped wholesale:
# a .git inside an adapter directory is large, changes on every commit, and none of
# it reaches the model. This is the *only* name-based directory exclusion — the old
# "skip anything starting with a dot" rule meant a directory holding nothing but
# hidden files hashed identically to an empty one (M29).
_IGNORED_DIRS = frozenset({".git", ".hg", ".svn"})

# The provenance record is written *inside* the adapter directory so a shipped
# adapter travels with its attestation — but it carries `created_ts` and is written
# after training finishes, so hashing it would make `adapter_blake3` a function of
# the wall clock: the same weights trained twice could never share a digest, and
# rewriting the record would change the digest of bytes nobody touched (M28).
# Excluded at the tree root only; a file of that name deeper inside a container is
# not ours to interpret. The alternative — writing the record outside the directory
# — was rejected because then a shipped adapter arrives with no provenance at all,
# which is the thing sec 9.4 exists to prevent.
PROVENANCE_FILENAME = "provenance.json"


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


def _walk_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Every file under `root` as (relpath, path), symlinks followed, once each.

    Symlinked directories are descended into. Content-addressed weight stores are
    the normal layout — an adapter is often `{adapter_config.json, weights ->
    /models/blobs/xyz}` — and a walk that stopped at the link produced a digest
    blind to every byte the model actually loads (M29). Hidden files are included
    for the same reason: nothing about a leading dot means "does not affect the
    model", and excluding them made a hidden-files-only directory indistinguishable
    from an empty one.

    Following links has two consequences, both deliberate. The walk can *leave the
    tree*: a link out to a blob store is followed, because that is what the loader
    does, so a digest may cover bytes outside `root`. And the walk can *loop*, so
    directories are identified by (st_dev, st_ino) — the inode, not the path we
    arrived by — and visited once; `weights -> ..` would otherwise never terminate.

    A directory that cannot be stat-ed or enumerated is skipped, which is what
    os.walk did here before. File reads still raise: a container we cannot read is
    an error, not a digest.
    """
    seen: set[tuple[int, int]] = set()
    stack: list[tuple[Path, str]] = [(root, "")]
    while stack:
        directory, prefix = stack.pop()
        try:
            st = directory.stat()
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            continue
        seen.add(key)
        try:
            with os.scandir(directory) as it:
                entries = list(it)
        except OSError:
            continue
        for entry in entries:
            rel = f"{prefix}{entry.name}"
            try:
                if entry.is_dir():  # follows symlinks
                    if entry.name in _IGNORED_DIRS:
                        continue
                    stack.append((Path(entry.path), rel + "/"))
                    continue
                if not entry.is_file():  # dangling symlink, socket, fifo
                    continue
            except OSError:
                continue
            yield rel, Path(entry.path)


def _is_hashed(rel: str) -> bool:
    """Does this relpath contribute to the digest?

    Shared by `hash_tree` and `tree_signature` so the change-detector covers exactly
    what the digest covers — a signature that watched more files than it hashed
    would force pointless 20 GB re-reads, and one that watched fewer would serve a
    stale digest, which is H13 all over again.
    """
    if rel == PROVENANCE_FILENAME:
        return False
    return rel.rsplit("/", 1)[-1] not in _IGNORED_NAMES


def hash_tree(root: str | os.PathLike[str]) -> str:
    """Digest of a directory: order-independent, name-sensitive, content-sensitive.

    Feeds each file as ``relpath\\0size\\0filehash\\n`` in sorted relpath order, so the
    digest is stable across filesystems that enumerate differently, and changes if a
    file is renamed, resized or edited.

    Symlink targets are hashed by content but their *paths* are not part of the
    digest: a target path is an absolute local path, so hashing it would make the
    same weights digest differently on two machines and would put the operator's
    filesystem layout into every receipt.
    """
    root_path = Path(root)
    if root_path.is_file():
        return hash_file(root_path)
    if not root_path.is_dir():
        raise FileNotFoundError(f"no such container path: {root}")

    entries: list[tuple[str, int, str]] = []
    for rel, full in _walk_files(root_path):
        if not _is_hashed(rel):
            continue
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

    Per-file size and mtime lose to the same argument, one step further along: a
    same-length in-place overwrite followed by `os.utime` restores both, and the
    memoised digest then survives a weight swap forever, because `lru_cache` is
    per-process and the gateway is long-lived (H13). So the key also carries
    st_ctime_ns — set by the kernel on every inode change, with no syscall to
    backdate it — and st_ino, which catches the replace-by-rename case where a new
    inode inherits the old name, size and mtime. A false miss costs one re-read; a
    false hit costs a receipt attesting weights nobody can reproduce.

    Stat-ing every file in a container is microseconds against the seconds it takes
    to read and hash it, so the cache still earns its place.
    """
    if root.is_file():
        st = root.stat()
        return f"{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_ino}"
    rows: list[str] = []
    for rel, full in _walk_files(root):
        if not _is_hashed(rel):
            continue
        try:
            st = full.stat()
        except OSError:
            continue
        rows.append(
            f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\0{st.st_ctime_ns}\0{st.st_ino}\n"
        )
    h = blake3.blake3()
    # Sorted, because the walk order follows the filesystem and an unstable
    # signature would miss the cache on every request for an unchanged tree.
    for row in sorted(rows):
        h.update(row.encode())
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
