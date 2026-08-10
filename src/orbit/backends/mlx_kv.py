"""An `mlx_lm` prompt cache as bytes, and back (sec 8.4).

`KVState.blob` is bytes. This is what puts an MLX KV cache into them. Tier 0 is
the only caller; `mx` arrives as an argument so the module imports on a machine
with no MLX, like the rest of the tier-0 surface.

**Deliberately not `mlx_lm.save_prompt_cache`.** That writes a safetensors file
and reads it back through `mx.load`, which memory-maps it — and a process already
mapping ~30 GB of weights should not add more mappings. It is the same decision
sec 8.4 makes for the disk cache itself, one layer down, and taking the shortcut
here would undo it while leaving `kv_disk.py`'s plain read/write looking intact.
It also costs a full extra copy of the blob through the filesystem on every store
*and* every restore.

Layout (little-endian, one sequential pass):

    magic   8   b"TDMKVST0"
    version 4   uint32 = 1
    hdr_len 4   uint32
    header  N   JSON   -- cache classes, meta_state, array table, state shape
    arrays  ..  raw array bytes, concatenated in table order

`loads` returns None for anything it cannot account for, and never raises: a blob
reaches it from disk, where truncation and a half-published write are ordinary,
and sec 8.4's rule is that a corrupt entry is a miss.

Peak memory during `dumps` is twice the blob — the per-array copies and the
joined result exist together — which is what `max_bytes` is sized against, not
the stored size alone.
"""

from __future__ import annotations

import json
import struct
from typing import Any

MAGIC = b"TDMKVST0"
VERSION = 1
_HDR = struct.Struct("<8sII")

# Marks an array's slot in the JSON state shape. A cache state holds arrays,
# lists and scalars, never a dict, so there is nothing for it to collide with.
_SLOT = "@"


def dumps(cache: list[Any], mx: Any, *, max_bytes: int) -> bytes | None:
    """Serialise a prompt cache, or None if it will not fit or will not encode.

    None is a cache miss on the next turn and nothing worse, so every refusal
    here is cheaper than the alternative it avoids.
    """
    arrays: list[Any] = []
    try:
        shape = [_encode_state(c.state, arrays, mx) for c in cache]
        meta = [_plain(c.meta_state) for c in cache]
        classes = [type(c).__name__ for c in cache]
    except (AttributeError, TypeError):
        return None

    if sum(int(a.nbytes) for a in arrays) > max_bytes:
        return None

    chunks = [bytes(memoryview(mx.view(a, mx.uint8))) for a in arrays]
    # Lengths come from the bytes produced, never from `nbytes`: the table is what
    # `loads` slices the payload by, so a disagreement between the two would be
    # read back as a torn array rather than as a refusal.
    table = [
        {"d": _dtype_name(a.dtype), "s": list(a.shape), "n": len(c)}
        for a, c in zip(arrays, chunks)
    ]
    header = json.dumps(
        {"classes": classes, "meta": meta, "shape": shape, "table": table},
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join([_HDR.pack(MAGIC, VERSION, len(header)), header, *chunks])


def loads(blob: bytes, mx: Any, cache_module: Any) -> list[Any] | None:
    """Rebuild a prompt cache from `dumps`, or None if the bytes do not describe one."""
    try:
        magic, version, hdr_len = _HDR.unpack(blob[: _HDR.size])
        if magic != MAGIC or version != VERSION:
            return None
        start = _HDR.size + hdr_len
        header = json.loads(bytes(blob[_HDR.size : start]).decode("utf-8"))

        view = memoryview(blob)
        arrays: list[Any] = []
        for entry in header["table"]:
            size = int(entry["n"])
            end = start + size
            if size < 0 or end > len(blob):
                return None
            dtype = _dtype(mx, entry["d"])
            if dtype is None:
                return None
            flat = mx.array(view[start:end], dtype=mx.uint8)
            arrays.append(mx.view(flat, dtype).reshape(tuple(entry["s"])))
            start = end
        # The payload has to end where the table says. A file that grew was
        # written by something that does not own this format.
        if start != len(blob):
            return None

        classes, meta, shape = header["classes"], header["meta"], header["shape"]
        if not len(classes) == len(meta) == len(shape):
            return None
        cache: list[Any] = []
        for name, meta_state, node in zip(classes, meta, shape):
            cls = _cache_class(cache_module, name)
            if cls is None:
                return None
            cache.append(cls.from_state(_decode_state(node, arrays), meta_state))
        return cache
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        struct.error,
    ):
        return None


def cache_length(cache: list[Any]) -> int:
    """Tokens the cache covers.

    `max` rather than `[0]`: a model whose first layer holds no KV — a cache-free
    or state-space layer — reports 0 for it, and reading that as the length would
    throw away the prefix the attention layers do hold.
    """
    return max((int(c.size()) for c in cache), default=0)


def _encode_state(node: Any, arrays: list[Any], mx: Any) -> Any:
    if isinstance(node, mx.array):
        arrays.append(node)
        return {_SLOT: len(arrays) - 1}
    if isinstance(node, (list, tuple)):
        return [_encode_state(child, arrays, mx) for child in node]
    return _plain(node)


def _decode_state(node: Any, arrays: list[Any]) -> Any:
    if isinstance(node, dict):
        return arrays[int(node[_SLOT])]
    if isinstance(node, list):
        return [_decode_state(child, arrays) for child in node]
    return node


def _plain(node: Any) -> Any:
    """A meta_state as JSON, or a TypeError naming what would not survive it."""
    if isinstance(node, (list, tuple)):
        return [_plain(child) for child in node]
    if node is None or isinstance(node, (str, int, float, bool)):
        return node
    raise TypeError(f"cache state holds a {type(node).__name__}, which has no encoding")


def _dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _dtype(mx: Any, name: Any) -> Any:
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        return None
    return getattr(mx, name, None)


def _cache_class(module: Any, name: Any) -> Any:
    """The cache class `name` refers to, or None.

    The name is read from a blob on disk, so it is resolved against the module
    rather than trusted: a corrupt or stale entry naming something that is not a
    cache class is a miss, and `getattr` on an arbitrary string would otherwise
    reach any module-level callable.
    """
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        return None
    cls = getattr(module, name, None)
    if not isinstance(cls, type) or not hasattr(cls, "from_state"):
        return None
    return cls
