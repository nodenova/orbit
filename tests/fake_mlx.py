"""A model of MLX, not MLX.

`backends/mlx_tier0.py` is the largest never-executed surface in the repository:
416 lines that no CI run and no local run has ever imported, held together by the
claim that they are "correct on inspection". This module exists to make that claim
falsifiable off Apple Silicon. It installs a small, strict stand-in for the MLX
surface tier 0 touches — `mlx.core`, `mlx.nn`, `mlx_lm` — so the *real*
`MLXTier0Backend` runs, mounts adapters, and generates.

**Be exact about what that proves.** It proves the wiring: which layers are
targeted, that `mount()` reaches them, that `MultiAdapterLinear` selects on the
`ContextVar` at call time rather than at mount time, that unload/load comes back
with the same identity, and that a request's bytes are the bytes the cache keys
on. It proves nothing about numerics on real weights, quantised dtypes, Metal
determinism, or whether mlx's own `Module` API matches the one modelled here. A
green run means the plumbing is sound; `orbit gate isolation` on an M4 Max is
still what proves tier 0.

Two rules govern anything added below, both learned the hard way on `MockBackend`
(see CLAUDE.md — it has been easier to satisfy than a real backend twice):

1. **Never be more permissive than MLX.** Shapes are checked, unexpected kwargs
   raise, and `apply_chat_template` renders tool calls the way a real Qwen-family
   template does. A fake that quietly accepts what MLX would reject converts a
   hardware-day crash into a green CI run, which is worse than no test.
2. **Stay arithmetic.** The forward pass really computes `y = xW + s·(xA)B`, so an
   adapter that is mounted but not selected changes the output and a test can see
   it. A stub returning canned text could not tell adapter isolation from adapter
   amnesia.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import struct
import sys
import types
from collections.abc import Iterator
from typing import Any

# --- arrays -----------------------------------------------------------------


class Array:
    """A 2-D float matrix with the handful of ops tier 0's forward pass uses.

    Deliberately not numpy: the point is to model MLX's *interface*, and a
    dependency that broadcasts more eagerly than MLX does would hide shape bugs
    rather than surface them. Every operation here checks its shapes.

    A second, flat *byte* mode models what `mx.array(buffer, dtype=mx.uint8)`
    returns, which is the form sec 8.4's KV codec reads a serialised cache back
    through. Only `mx.view` accepts one.
    """

    __slots__ = ("dtype", "raw", "rows")

    def __init__(self, rows: Any, dtype: str = "float32"):
        if isinstance(rows, (bytes, bytearray, memoryview)):
            if dtype != "uint8":
                raise TypeError("a buffer only builds a uint8 array")
            self.raw = bytes(rows)
            self.rows = []
            self.dtype = "uint8"
            return
        if not rows or not all(isinstance(r, (list, tuple)) for r in rows):
            raise ValueError("Array takes a non-empty list of rows")
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("ragged rows")
        self.rows = [[float(v) for v in r] for r in rows]
        self.dtype = dtype
        self.raw = b""

    @property
    def shape(self) -> tuple[int, ...]:
        if self.dtype == "uint8":
            return (len(self.raw),)
        return len(self.rows), len(self.rows[0])

    @property
    def nbytes(self) -> int:
        if self.dtype == "uint8":
            return len(self.raw)
        return len(self.rows) * len(self.rows[0]) * 4

    def reshape(self, shape: Any) -> Array:
        dims = tuple(int(v) for v in shape)
        if len(dims) != 2:
            raise ValueError(f"fake arrays are 2-D; cannot reshape to {dims}")
        flat = [v for row in self.rows for v in row]
        if dims[0] * dims[1] != len(flat):
            raise ValueError(f"cannot reshape {len(flat)} values into {dims}")
        return Array(
            [flat[i * dims[1] : (i + 1) * dims[1]] for i in range(dims[0])], self.dtype
        )

    def astype(self, dtype: str) -> Array:
        return Array(self.rows, dtype)

    @property
    def T(self) -> Array:
        n, m = self.shape
        return Array(
            [[self.rows[i][j] for i in range(n)] for j in range(m)], self.dtype
        )

    def __matmul__(self, other: Array) -> Array:
        if not isinstance(other, Array):
            return NotImplemented
        n, k = self.shape
        k2, m = other.shape
        if k != k2:
            raise ValueError(f"shape mismatch: {self.shape} @ {other.shape}")
        out = [
            [
                sum(self.rows[i][t] * other.rows[t][j] for t in range(k))
                for j in range(m)
            ]
            for i in range(n)
        ]
        return Array(out, self.dtype)

    def __add__(self, other: Array) -> Array:
        if not isinstance(other, Array):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"shape mismatch: {self.shape} + {other.shape}")
        return Array(
            [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(self.rows, other.rows)],
            self.dtype,
        )

    def __mul__(self, scalar: float) -> Array:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Array([[v * scalar for v in r] for r in self.rows], self.dtype)

    __rmul__ = __mul__

    def tolist(self) -> list[list[float]]:
        return [list(r) for r in self.rows]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Array({self.rows!r}, dtype={self.dtype!r})"


def _digest_floats(seed: str, n: int) -> list[float]:
    """`n` deterministic floats in [-1, 1) from a string."""
    out: list[float] = []
    counter = 0
    while len(out) < n:
        block = hashlib.blake2b(f"{seed}:{counter}".encode(), digest_size=32).digest()
        for i in range(0, len(block), 4):
            if len(out) == n:
                break
            word = int.from_bytes(block[i : i + 4], "big")
            out.append((word / 2**31) - 1.0)
        counter += 1
    return out


# --- nn ---------------------------------------------------------------------


class Module:
    """Modelled on `mlx.nn.Module` in the two ways tier 0 depends on.

    `children()` returns sub-modules only — never plain attributes — and includes
    lists of modules, which is how a real transformer exposes its layer stack and
    the branch `_walk_linears` has to get right.
    """

    def children(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in vars(self).items():
            if isinstance(value, Module) or (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(v, Module) for v in value)
            ):
                out[key] = value
        return out

    def parameters(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in vars(self).items():
            if isinstance(value, Array):
                out[key] = value
            elif isinstance(value, Module):
                out[key] = value.parameters()
            elif (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(v, Module) for v in value)
            ):
                out[key] = [v.parameters() for v in value]
        return out


class Linear(Module):
    """`y = x @ W.T + b`, with W shaped [out, in] as in MLX."""

    def __init__(
        self, input_dims: int, output_dims: int, bias: bool = True, *, seed: str = ""
    ):
        self.input_dims = input_dims
        self.output_dims = output_dims
        flat = _digest_floats(
            f"W:{seed}:{input_dims}x{output_dims}", input_dims * output_dims
        )
        self.weight = Array(
            [flat[r * input_dims : (r + 1) * input_dims] for r in range(output_dims)]
        )
        self.bias = Array([_digest_floats(f"b:{seed}", output_dims)]) if bias else None

    def __call__(self, x: Array) -> Array:
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


# --- a model with the layer names §4.3 targets ------------------------------


class _Attention(Module):
    def __init__(self, dim: int, tag: str):
        self.q_proj = Linear(dim, dim, seed=f"{tag}.q")
        self.k_proj = Linear(dim, dim, seed=f"{tag}.k")
        self.v_proj = Linear(dim, dim, seed=f"{tag}.v")
        self.o_proj = Linear(dim, dim, seed=f"{tag}.o")

    def __call__(self, x: Array) -> Array:
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _MLP(Module):
    def __init__(self, dim: int, tag: str):
        self.gate_proj = Linear(dim, dim, seed=f"{tag}.g")
        self.up_proj = Linear(dim, dim, seed=f"{tag}.u")
        self.down_proj = Linear(dim, dim, seed=f"{tag}.d")

    def __call__(self, x: Array) -> Array:
        return self.down_proj(self.gate_proj(x) + self.up_proj(x))


class _Block(Module):
    def __init__(self, dim: int, tag: str):
        self.self_attn = _Attention(dim, tag)
        self.router = Linear(dim, dim, seed=f"{tag}.r")
        self.mlp = _MLP(dim, tag)

    def __call__(self, x: Array) -> Array:
        return self.mlp(self.router(self.self_attn(x)))


class FakeModel(Module):
    """Two blocks and an untargeted head.

    `lm_head` is here on purpose: §4.3 does not target it, so a change to
    `_is_target` that starts wrapping everything has something to fail against.
    """

    def __init__(self, dim: int = 4, n_layers: int = 2):
        self.dim = dim
        self.layers = [_Block(dim, f"l{i}") for i in range(n_layers)]
        self.lm_head = Linear(dim, dim, seed="head")

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = _normalise(layer(x))
        return self.lm_head(x)


def _normalise(x: Array) -> Array:
    """Keep activations bounded over a long decode.

    Without it a 128-token generation overflows to `inf` and every token after the
    overflow is identical — which would make an isolation failure invisible, since
    two different adapters both produce a wall of the same token.
    """
    peak = max((abs(v) for row in x.rows for v in row), default=0.0)
    return (
        x
        if peak <= 1.0
        else Array([[v / peak for v in row] for row in x.rows], x.dtype)
    )


# --- tokenizer --------------------------------------------------------------

_TEMPLATE_KEYS = {"role", "content", "tool_calls", "tool_call_id"}


class FakeTokenizer:
    """A chat template that renders what a real one renders.

    This is the strictest thing in the file and the reason it earns its keep. A
    fake that rendered only `role` and `content` would let a backend drop the
    entire tool history and still look byte-stable, and the disk KV cache keys on
    these bytes (sec 8.4). Two conversations that differ only in their tool calls
    must not render identically. That is how tier 0's dropped-tool-calls bug was
    found, and it is the property that stops it coming back.

    `tool_call_id` is accepted and deliberately **not** rendered, which is what a
    real Qwen-family template does with it. Emitting it would make this template
    discriminate between conversations a real one collides — flattering the cache
    key in exactly the direction that hides a bug until the hardware arrives.
    """

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        if tokenize:
            raise NotImplementedError(
                "fake tokenizer renders text only; pass tokenize=False"
            )
        parts: list[str] = []
        if tools:
            parts.append(
                "<|tools|>" + json.dumps(tools, sort_keys=True) + "<|/tools|>\n"
            )
        for msg in messages:
            unknown = set(msg) - _TEMPLATE_KEYS
            if unknown:
                raise ValueError(
                    f"chat template got keys it cannot render: {sorted(unknown)}. A real "
                    "template would drop them silently; this one refuses so the drop is "
                    "not discovered on the hardware."
                )
            parts.append(f"<|im_start|>{msg['role']}\n{msg.get('content', '')}\n")
            for call in msg.get("tool_calls") or ():
                parts.append(
                    f"<|tool_call|>{json.dumps(call, sort_keys=True)}<|/tool_call|>\n"
                )
            parts.append("<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __len__(self) -> int:
        """One past the largest id `encode` can produce, as a real tokenizer reports.

        Honest rather than convenient: `_token_id` hashes an unknown piece into a
        24-bit space, so this is ~16.8 M and every logit row in the suite is narrower.
        That means `logit_width_bound` never licenses the F1 fast path here, which is
        the correct outcome — a fake reporting a small length would let the backend
        skip its bounds guard while still emitting ids far outside the row.
        """
        return _ID_LIMIT

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Whitespace-ish tokens. Never zero — a zero input count would report an
        infinite prefill rate, and tier 0's usage feeds the latency suite.

        One id per whitespace-separated piece, but the id is a digest of the piece
        rather than its length: sec 8.4 restores a KV cache by comparing token ids,
        so a tokenizer that mapped "cat" and "dog" to the same id would let a state
        built for one prompt be accepted for another and the test could not tell.

        `add_special_tokens` is accepted and adds nothing, because this tokenizer
        has no BOS to add — which is also why it reports none.
        """
        return [_token_id(p) for p in text.split()] or [1]


# --- prompt cache (sec 8.4) -------------------------------------------------


def _f32(value: float) -> float:
    """`value` as it survives a float32 round trip.

    The KV codec stores raw bytes, so a cache restored from disk holds exactly
    float32. Rows are folded to float32 as they are appended, which is what lets a
    warm start and a cold start reach byte-identical cache contents — the property
    the restore path is worth having only if it holds.
    """
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


class KVCache:
    """Modelled on `mlx_lm.models.cache.KVCache` in the parts sec 8.4 touches.

    One row per token rather than the real one's `[batch, heads, seq, dim]`, and
    the sequence axis is 0 instead of 2. What is reproduced exactly is the
    contract the codec depends on: `state` slices to `offset` so a trimmed cache
    serialises to its trimmed length, `from_state` rebuilds the offset from the
    array it is handed, and `trim` moves the offset without touching storage.
    """

    def __init__(self) -> None:
        self.keys: list[list[float]] = []
        self.values: list[list[float]] = []
        self.offset = 0

    def append(self, token: int, dim: int) -> None:
        # Rows past the offset are what a trim left behind. The real cache
        # overwrites them in place; dropping them here is the same thing, and
        # keeping the append O(1) is what makes a 2000-token prompt testable.
        del self.keys[self.offset :]
        del self.values[self.offset :]
        row = [_f32(v) for v in _digest_floats(f"kv:{token}", dim)]
        self.keys.append(row)
        self.values.append(row[::-1])
        self.offset += 1

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self) -> tuple[Array, Array]:
        # An unfed cache raises rather than returning something empty, which is
        # what the real one does when it reaches `.shape` on a None it never
        # allocated. The codec reads that as "nothing to store".
        if not self.offset:
            raise AttributeError("cache has no state: nothing has been fed to it")
        return Array(self.keys[: self.offset]), Array(self.values[: self.offset])

    @state.setter
    def state(self, value: Any) -> None:
        keys, values = value
        self.keys = [list(r) for r in keys.rows]
        self.values = [list(r) for r in values.rows]
        self.offset = len(self.keys)

    @property
    def meta_state(self) -> str:
        return ""

    @meta_state.setter
    def meta_state(self, value: Any) -> None:
        if value:
            raise ValueError("KVCache has no meta_state but one was set")

    @classmethod
    def from_state(cls, state: Any, meta_state: Any) -> KVCache:
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj


class RecurrentCache(KVCache):
    """A linear-attention layer's state: fixed size, and **not rewindable**.

    Modelled on `mlx_lm.models.cache.ArraysCache`, which is what the baseline
    container builds for 30 of its 40 layers — `full_attention_interval = 4` in
    its `text_config`. Two inherited properties are the ones that matter: `size()`
    reports 0, because a recurrent state has no token count, and `is_trimmable()`
    is False, because a state folded over N tokens cannot be unfolded back to K.

    It is here because without it every cache this file could build was
    trimmable, and sec 8.4's export would have refused on the only container it
    ships against while the suite stayed green.
    """

    def size(self) -> int:
        return 0

    def is_trimmable(self) -> bool:
        return False

    def trim(self, n: int) -> int:
        return 0


def make_prompt_cache(model: FakeModel, max_kv_size: int | None = None) -> list[Any]:
    """The cache the model asks for, one entry per layer.

    Hybrid by default and in the container's own proportion, so the path that
    runs on hardware is the path the tests take. A test that wants the
    all-attention arrangement sets `cache_kinds` on the model.
    """
    kinds = getattr(model, "cache_kinds", None) or ("recurrent", "kv")
    return [
        KVCache() if kinds[i % len(kinds)] == "kv" else RecurrentCache()
        for i in range(len(model.layers))
    ]


def can_trim_prompt_cache(cache: list[Any]) -> bool:
    return all(c.is_trimmable() for c in cache)


def trim_prompt_cache(cache: list[Any], num_tokens: int) -> int:
    if not can_trim_prompt_cache(cache) or not cache:
        return 0
    # Copied from mlx_lm verbatim, and `next()` is not the same function: the list
    # comprehension trims *every* cache and reports the first one's count, while a
    # generator would trim only the first and leave the rest at their old offset.
    return [c.trim(num_tokens) for c in cache][0]  # noqa: RUF015 - trims all, not the first


def _feed(cache: list[KVCache], token: int, dim: int) -> None:
    for entry in cache:
        entry.append(token, dim)


def _cache_seed(cache: list[KVCache]) -> str:
    """The whole cached sequence, as a string to derive a hidden state from.

    Generation is seeded from this rather than from the tokens fed on this call,
    so a turn that restores a prefix and prefills the rest lands on the same
    hidden state as one that prefilled all of it. A fake that seeded from the fed
    tokens alone would make every warm start produce different text from its cold
    equivalent, and the test that matters here could never be written.
    """
    head = cache[0]
    body = repr(head.keys[: head.offset])
    return hashlib.blake2b(body.encode(), digest_size=16).hexdigest()


# --- generation -------------------------------------------------------------

_VOCAB = (
    "<eos>",
    "return",
    "self",
    "value",
    "if",
    "None",
    "raise",
    "config",
    "path",
    "result",
    "token",
    "adapter",
    "cache",
    "error",
)

_SEED = 0
_VOCAB_IDS = {word: i for i, word in enumerate(_VOCAB)}
_HASH_BYTES = 3
# One past the largest id `_token_id` can return. Derived from the same two values
# it is built from rather than written out, because `FakeTokenizer.__len__` returns
# this and under-reporting it is the dangerous direction: a backend that trusts the
# length skips its out-of-range guard (sec 8.5.1, F1) and scatters outside the row.
_ID_LIMIT = len(_VOCAB) + 2 ** (8 * _HASH_BYTES)


def _token_id(piece: str) -> int:
    """The id this tokenizer gives a piece of text.

    A generated word and the same word read back out of a rendered conversation
    have to get the *same* id, because sec 8.4 restores a cache by comparing the
    ids it holds against the ids of the next prompt — and turn N+1 carries turn
    N's reply as text. A tokenizer whose `encode` disagreed with what generation
    sampled would refuse every continuation, which is the one case the cache is
    for. Real tokenizers round-trip; this one has to as well.
    """
    known = _VOCAB_IDS.get(piece)
    if known is not None:
        return known
    # Past the vocabulary, so a word the model can emit is never collided with.
    return len(_VOCAB) + int.from_bytes(
        hashlib.blake2b(piece.encode(), digest_size=_HASH_BYTES).digest(), "big"
    )


class Step:
    """`mlx_lm.stream_generate` yields these; tier 0 reads `.text`."""

    __slots__ = ("text", "token")

    def __init__(self, text: str, token: int):
        self.text = text
        self.token = token


def make_sampler(temp: float = 0.0, top_p: float = 1.0):
    """Modelled on `mlx_lm.sample_utils.make_sampler`.

    Greedy at temp=0 means a pure function of the logits — which is what makes the
    isolation gate meaningful, since any difference in output is then a difference
    the adapter deltas caused and nothing else.
    """
    if not isinstance(temp, (int, float)) or not isinstance(top_p, (int, float)):
        raise TypeError("make_sampler takes numeric temp and top_p")

    def sample(logits: Array) -> int:
        vals = [round(v, 6) for v in logits.rows[0]]
        seed = repr(vals) if temp == 0 else f"{vals!r}:{temp}:{top_p}:{_SEED}"
        return int(hashlib.blake2b(seed.encode(), digest_size=8).hexdigest(), 16) % len(
            _VOCAB
        )

    return sample


class TokenSequence:
    """What `generate_step` hands a logits processor as its first argument.

    An `mx.array` of the tokens sampled so far. Only `.tolist()` is modelled,
    because that is all the protocol needs, and a richer stand-in would invite a
    processor to depend on array behaviour this file does not reproduce.
    """

    __slots__ = ("_ids",)

    def __init__(self, ids: list[int]):
        self._ids = list(ids)

    def tolist(self) -> list[int]:
        return list(self._ids)


def stream_generate(
    model: FakeModel,
    tokenizer: FakeTokenizer,
    *,
    prompt: str | list[int],
    max_tokens: int = 256,
    sampler: Any = None,
    logits_processors: list[Any] | None = None,
    prompt_cache: list[Any] | None = None,
    **unexpected: Any,
) -> Iterator[Step]:
    """Modelled on `mlx_lm.stream_generate`, including its processor contract.

    `logits_processors` is **applied, not merely accepted**. A fake that took the
    kwarg and ignored it would reproduce exactly the bug this parameter exists to
    fix — sec 8.5.1 constrained decoding was computed, passed down, and dropped on
    the floor by the real backend for the life of the repository, and a green suite
    could not see it because nothing ever checked that the mask reached the logits.

    The call order matches `generate_step`: each processor receives every token
    sampled so far and the logits for the next one, and returns the logits to
    sample from. The prompt's tokens are deliberately absent — the real one prefills
    all but the last token without invoking processors.

    `prompt_cache` is likewise used rather than accepted, and used the way
    `generate_step` uses it: whatever it already holds is *not* re-prefilled, the
    tokens passed in are appended to it, and so are the tokens generated. It is the
    caller's job to pass only the suffix the cache does not cover — the real one
    does not check, and a fake that recovered from a bad offset would hide the sec
    8.4 restore returning fluent text against the wrong prefix.
    """
    if unexpected:
        raise TypeError(f"stream_generate got unexpected kwargs: {sorted(unexpected)}")
    if isinstance(prompt, str):
        ids = tokenizer.encode(prompt)
    elif isinstance(prompt, (list, tuple)):
        ids = [int(t) for t in prompt]
    else:
        raise TypeError("stream_generate takes a string or a list of token ids")
    if sampler is None:
        raise TypeError("stream_generate needs a sampler")
    if model is None or tokenizer is None:
        raise RuntimeError("model is not resident — load() before generating")
    for processor in logits_processors or ():
        if not callable(processor):
            raise TypeError("each logits processor must be callable(tokens, logits)")
    if not ids:
        # `generate_step` raises on this rather than generating from the cache
        # alone, which is why the backend always leaves itself a token to feed.
        raise ValueError("Either input_embeddings or prompt (or both) must be provided")
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model)
    if not isinstance(prompt_cache, list) or not all(
        isinstance(c, KVCache) for c in prompt_cache
    ):
        raise TypeError("prompt_cache must be the list make_prompt_cache returns")

    for token_id in ids:
        _feed(prompt_cache, token_id, model.dim)
    x = Array([_digest_floats(_cache_seed(prompt_cache), model.dim)])
    sampled: list[int] = []
    for _ in range(max_tokens):
        logits = model(x)
        for processor in logits_processors or ():
            logits = processor(TokenSequence(sampled), logits)
            if not isinstance(logits, Array):
                raise TypeError("a logits processor must return logits, not a mask")
        index = sampler(logits)
        word = _VOCAB[index]
        if word == "<eos>":
            return
        sampled.append(index)
        _feed(prompt_cache, index, model.dim)
        yield Step(text=word + " ", token=index)
        x = _normalise(logits + Array([_digest_floats(word, model.dim)]))


def load(model_path: str) -> tuple[FakeModel, FakeTokenizer]:
    return FakeModel(), FakeTokenizer()


# --- mlx.core ---------------------------------------------------------------


def _mx_load(path: str) -> dict[str, Array]:
    """Stand-in for `mx.load` on a safetensors file.

    Reads JSON off disk instead, so a test can write adapter weights without a
    safetensors writer. The filename is still `adapters.safetensors`, because
    `mount_all` looks for exactly that name.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {key: Array(rows) for key, rows in raw.items()}


def _mx_view(a: Array, dtype: str) -> Any:
    """Reinterpret an array's bits, modelled on `mx.view`.

    The byte side is a plain `bytes` rather than a uint8 `Array`, which is the one
    place this file's representation departs from MLX's. It has to: what the KV
    codec does with the result is `bytes(memoryview(...))`, and on Python 3.11 a
    pure-Python object cannot offer a buffer at all (PEP 688 is 3.12). `bytes`
    supports exactly the surface the codec uses of a real uint8 array and nothing
    wider, so the direction of the departure is toward stricter, not looser.
    """
    if not isinstance(a, Array):
        raise TypeError("view takes an array")
    if dtype == "uint8":
        if a.dtype == "uint8":
            return a.raw
        flat = [v for row in a.rows for v in row]
        return struct.pack(f"<{len(flat)}f", *flat)
    if a.dtype != "uint8":
        raise TypeError(f"fake view only reinterprets uint8 bytes, not {a.dtype}")
    if dtype != "float32":
        raise TypeError(f"fake arrays are float32; cannot view bytes as {dtype}")
    if len(a.raw) % 4:
        raise ValueError("byte count is not a whole number of float32 values")
    return Array([list(struct.unpack(f"<{len(a.raw) // 4}f", a.raw))], dtype)


def _mx_eval(*args: Any) -> None:
    return None


def _mx_clear_cache() -> None:
    return None


def _seed(value: int) -> None:
    global _SEED
    _SEED = int(value)


# --- installation -----------------------------------------------------------


def _build_modules() -> dict[str, types.ModuleType]:
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    nn = types.ModuleType("mlx.nn")
    mlx_lm = types.ModuleType("mlx_lm")
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    models = types.ModuleType("mlx_lm.models")
    cache_mod = types.ModuleType("mlx_lm.models.cache")

    random = types.SimpleNamespace(seed=_seed)
    core.load = _mx_load
    core.eval = _mx_eval
    core.clear_cache = _mx_clear_cache
    core.random = random
    core.array = Array
    core.view = _mx_view
    core.uint8 = "uint8"
    core.float32 = "float32"

    nn.Module = Module
    nn.Linear = Linear

    mlx.core = core
    mlx.nn = nn

    mlx_lm.load = load
    mlx_lm.stream_generate = stream_generate
    mlx_lm.sample_utils = sample_utils
    mlx_lm.models = models
    models.cache = cache_mod
    sample_utils.make_sampler = make_sampler
    cache_mod.KVCache = KVCache
    cache_mod.RecurrentCache = RecurrentCache
    cache_mod.make_prompt_cache = make_prompt_cache
    cache_mod.can_trim_prompt_cache = can_trim_prompt_cache
    cache_mod.trim_prompt_cache = trim_prompt_cache

    return {
        "mlx": mlx,
        "mlx.core": core,
        "mlx.nn": nn,
        "mlx_lm": mlx_lm,
        "mlx_lm.sample_utils": sample_utils,
        "mlx_lm.models": models,
        "mlx_lm.models.cache": cache_mod,
    }


@contextlib.contextmanager
def install() -> Iterator[None]:
    """Put the fake MLX on `sys.modules` for the duration of the block.

    Restores whatever was there before, including nothing — so a developer who
    genuinely has MLX installed still runs against theirs outside the block, and a
    test that forgets to clean up cannot leak the fake into the next one.
    """
    modules = _build_modules()
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


# --- fixtures for adapters on disk ------------------------------------------


def write_adapter(
    directory: Any,
    keys: list[str],
    *,
    rank: int = 2,
    dim: int = 4,
    alpha: int = 4,
    salt: str = "",
) -> Any:
    """Write an adapter `mount()` can load: A [in, r], B [r, out], plus config."""
    directory.mkdir(parents=True, exist_ok=True)
    weights: dict[str, list[list[float]]] = {}
    for key in keys:
        a = _digest_floats(f"{salt}:{key}:a", dim * rank)
        b = _digest_floats(f"{salt}:{key}:b", rank * dim)
        weights[f"{key}.lora_a"] = [a[r * rank : (r + 1) * rank] for r in range(dim)]
        weights[f"{key}.lora_b"] = [b[r * dim : (r + 1) * dim] for r in range(rank)]
    (directory / "adapters.safetensors").write_text(
        json.dumps(weights), encoding="utf-8"
    )
    (directory / "adapter_config.json").write_text(
        json.dumps({"lora_parameters": {"rank": rank, "alpha": alpha}}),
        encoding="utf-8",
    )
    return directory
