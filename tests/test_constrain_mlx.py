"""Constrained decoding, from the schema down to the logits (spec sec 8.5.1).

These tests exist because of a specific silent failure. `Pipeline` computes a JSON
Schema for every tool-bearing turn and attaches it to the request; `MockBackend`
honoured it; the real MLX tier-0 backend never read the field. So on hardware the
constraint was computed, carried the whole way down, and dropped — leaving tool-call
correctness entirely to repair and retry. Measured on the M4 Max before the fix:
**0 clean first-attempt tool calls in 100 turns.**

The suite could not see it, and the reason is worth keeping in front of whoever
edits this file. `MockBackend` was *stricter* than the hardware, which inverts the
rule in CLAUDE.md — the mock must never be easier to satisfy than a real backend.
When the two disagree, the green run belongs to the one nobody ships.

What is covered here, and by what:

* the token filter really enforces a schema — real `lm-format-enforcer`, against a
  tokenizer small enough to assert on exactly;
* the mask arithmetic — a double under CI, and real MLX where it is installed, so
  the array contract is checked against the library and not only against a model
  of it;
* the backend passes the processor down at all — `fake_mlx`, which now applies
  processors rather than accepting and ignoring them.
"""

from __future__ import annotations

import json
import string

import pytest

from tandem.gateway.toolcall.constrain import Constrainer, tool_call_schema
from tandem.types import ToolDef

lmformatenforcer = pytest.importorskip(
    "lmformatenforcer", reason="constrained decoding is the '[constrain]' extra"
)


# --- a tokenizer small enough to assert on ----------------------------------


class TinyTokenizer:
    """A character tokenizer with the surface `Constrainer.vocabulary` reads.

    **No whitespace piece, deliberately.** JSON permits whitespace almost
    everywhere, so a vocabulary containing a space lets a first-allowed walk emit
    it forever without ever violating the schema — which is what the real model
    does under a mask when it is asked to pick the lowest-id allowed token. Leaving
    it out is what makes "walk greedily and assert on the result" a test of the
    schema rather than a test of tie-breaking.
    """

    def __init__(self) -> None:
        self.pieces = [
            "<eos>",
            "<pad>",
            *list('{}[]":,.-_'),
            *list(string.ascii_lowercase),
            *list(string.digits),
        ]
        self._ids = {piece: i for i, piece in enumerate(self.pieces)}
        self.all_special_ids = [0, 1]
        self.eos_token_id = 0
        self.eos_token_ids = {0}

    def __len__(self) -> int:
        return len(self.pieces)

    def encode(self, text: str) -> list[int]:
        return [self._ids[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[i] for i in ids)


READ_FILE = ToolDef(
    name="read_file",
    description="Read a file",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
)


@pytest.fixture
def tokenizer() -> TinyTokenizer:
    return TinyTokenizer()


@pytest.fixture
def vocabulary(tokenizer):
    return Constrainer().vocabulary(tokenizer)


def _filter(vocabulary, schema):
    return Constrainer().token_filter(schema, vocabulary)


def _walk(tokenizer, allowed_fn, *, limit: int = 200) -> tuple[str, list[int]]:
    """Decode greedily under the mask, preferring to close an open string.

    **The filter must be driven one token at a time**, and every helper here obeys
    that. `TokenEnforcer` reconstructs parser state from the prefix it has already
    been shown: hand it a sequence whose parent it has never seen and it treats the
    whole thing as a fresh root, reporting the allowed set for an empty generation.
    That is not a quirk to work around — it is the contract `generate_step` already
    satisfies, since it calls processors once per sampled token.

    The closing-quote preference is what makes the walk terminate. `path` is an
    unbounded string, so a strict lowest-id rule spells a legal but endless value;
    the schema is satisfied either way, which is the point, but only the closing
    rule produces a document to assert on.
    """
    quote = tokenizer._ids['"']
    seq: list[int] = list(tokenizer.encode("0"))
    out: list[int] = []
    for _ in range(limit):
        allowed = list(allowed_fn(seq))
        live = [i for i in allowed if i not in tokenizer.eos_token_ids]
        if not live:
            break
        pick = quote if quote in live else min(live)
        seq.append(pick)
        out.append(pick)
    return tokenizer.decode(out), out


def _advance(tokenizer, allowed_fn, text: str) -> set[int]:
    """The allowed set after `text`, fed token by token as generation would."""
    seq: list[int] = list(tokenizer.encode("0"))
    allowed = allowed_fn(seq)
    for token_id in tokenizer.encode(text):
        seq.append(token_id)
        allowed = allowed_fn(seq)
    return set(allowed)


# --- the filter enforces the schema -----------------------------------------


def test_a_greedy_walk_under_the_mask_is_valid_json_for_the_schema(
    tokenizer, vocabulary
):
    schema = tool_call_schema([READ_FILE])
    text, _ = _walk(tokenizer, _filter(vocabulary, schema))

    parsed = json.loads(text)
    assert parsed["name"] == "read_file"
    assert "path" in parsed["arguments"]


def test_the_first_token_cannot_be_one_that_does_not_open_an_object(
    tokenizer, vocabulary
):
    allowed = _advance(
        tokenizer, _filter(vocabulary, tool_call_schema([READ_FILE])), ""
    )

    assert tokenizer._ids["{"] in allowed
    for illegal in ("}", "]", ",", ":", "a"):
        assert tokenizer._ids[illegal] not in allowed


def test_a_const_name_admits_only_the_letters_that_spell_it(tokenizer, vocabulary):
    """The `const` in `tool_call_schema` is what makes tool invention impossible.

    Repair can rescue a misspelled name afterwards by resolving it against the
    offered tools; prevention means the wrong letter is never sampled. Asserting on
    the letter after `read_` is asserting on that difference.
    """
    allowed = _advance(
        tokenizer, _filter(vocabulary, tool_call_schema([READ_FILE])), '{"name":"read_'
    )

    assert allowed == {tokenizer._ids["f"]}


def test_a_stop_token_is_refused_until_the_object_is_complete(tokenizer, vocabulary):
    midway = _advance(
        tokenizer, _filter(vocabulary, tool_call_schema([READ_FILE])), '{"name":'
    )
    assert not midway & tokenizer.eos_token_ids

    text, _ = _walk(tokenizer, _filter(vocabulary, tool_call_schema([READ_FILE])))
    complete = _advance(
        tokenizer, _filter(vocabulary, tool_call_schema([READ_FILE])), text
    )
    assert complete & tokenizer.eos_token_ids, (
        "a completed object must admit a stop token, or the turn runs to max_tokens"
    )


def test_two_tools_stay_selectable_and_a_third_name_does_not(tokenizer, vocabulary):
    write = ToolDef(
        name="write_file",
        description="Write a file",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    token_filter = _filter(vocabulary, tool_call_schema([READ_FILE, write]))

    after_quote = _advance(tokenizer, token_filter, '{"name":"')
    assert after_quote == {tokenizer._ids["r"], tokenizer._ids["w"]}


def test_every_stop_id_reaches_the_enforcer_not_only_the_singular_one(tokenizer):
    """`stream_generate` breaks on the *set*; a mask built from the scalar deadlocks.

    If the enforcer only knows one stop id and the model's chat template ends turns
    with another, the completed object admits no terminator: the mask forbids the
    one token that would end the turn and generation runs to `max_tokens`.
    """
    tokenizer.eos_token_ids = {0, 1}
    tokenizer.all_special_ids = []  # keep id 1 a regular token so it must come from the set
    vocabulary = Constrainer().vocabulary(tokenizer)

    assert set(vocabulary.eos_token_id) == {0, 1}


def test_an_unavailable_enforcer_degrades_to_none_rather_than_raising(tokenizer):
    """Repair is a real degradation; a failed request is a worse one (sec 8.5.3-4)."""
    disabled = Constrainer(enabled=False)

    assert disabled.vocabulary(tokenizer) is None
    assert disabled.token_filter(tool_call_schema([READ_FILE]), None) is None
    assert disabled.available is False


def test_a_filter_cannot_be_built_without_a_vocabulary(tokenizer):
    assert Constrainer().token_filter(tool_call_schema([READ_FILE]), None) is None


# --- the mask arithmetic ----------------------------------------------------


class _Array:
    """The `mx.array` surface `build_logits_processor` touches, and no more."""

    def __init__(
        self, values: list[float], shape: tuple[int, ...], dtype: str = "float32"
    ):
        self.values = list(values)
        self.shape = shape
        self.dtype = dtype

    def __setitem__(self, index: _Array, value: float) -> None:
        if not isinstance(index, _Array):
            raise TypeError("scatter takes an index array")
        for i in index.values:
            self.values[int(i)] = float(value)

    def __add__(self, other: _Array) -> _Array:
        # Real MLX broadcasts a [vocab] mask against [1, vocab] logits; a double
        # that refused would fail code the library accepts.
        width = self.shape[-1]
        if other.shape[-1] != width:
            raise ValueError(f"shape mismatch: {self.shape} + {other.shape}")
        return _Array(
            [a + b for a, b in zip(self.values, other.values)], self.shape, self.dtype
        )


class _MX:
    int32 = "int32"

    @staticmethod
    def full(shape: tuple[int, ...], value: float, dtype: str = "float32") -> _Array:
        return _Array([value] * shape[-1], shape, dtype)

    @staticmethod
    def array(values: list[int], dtype: str = "int32") -> _Array:
        return _Array([float(v) for v in values], (len(values),), dtype)


def _processor(allowed):
    from tandem.backends.mlx_tier0 import build_logits_processor

    return build_logits_processor(lambda _tokens: allowed, _MX())


def _logits(width: int) -> _Array:
    return _Array([float(i) for i in range(width)], (1, width))


class _Tokens:
    @staticmethod
    def tolist() -> list[int]:
        return [7]


def test_the_mask_leaves_allowed_ids_alone_and_kills_the_rest():
    out = _processor([1, 3])(_Tokens(), _logits(5))

    assert out.values[1] == 1.0 and out.values[3] == 3.0
    for banned in (0, 2, 4):
        assert out.values[banned] == float("-inf")


def test_the_highest_scoring_token_under_the_mask_is_an_allowed_one():
    """The point of the mask: argmax moves off the model's preference onto a legal one."""
    out = _processor([1])(_Tokens(), _logits(5))

    assert max(range(5), key=lambda i: out.values[i]) == 1


def test_an_empty_allowed_set_passes_the_logits_through_untouched():
    """All `-inf` becomes NaN after the log-softmax and samples garbage silently."""
    logits = _logits(5)
    out = _processor([])(_Tokens(), logits)

    assert out is logits


def test_ids_past_the_logit_width_are_dropped_not_scattered_out_of_range():
    """`len(tokenizer)` and the model's output width are routinely different."""
    out = _processor([2, 99])(_Tokens(), _logits(5))

    assert out.values[2] == 2.0
    assert out.values[0] == float("-inf")


def test_an_allowed_set_entirely_past_the_width_passes_through():
    logits = _logits(5)

    assert _processor([99, 100])(_Tokens(), logits) is logits


def test_each_step_masks_from_its_own_allowed_set():
    """No caching between steps — a stale mask would permit last token's choices.

    `build_logits_processor` carried an identity-keyed mask cache briefly; it
    measured 27.1 tok/s against 27.6 without and was removed. This pins the
    behaviour that matters either way: what the mask permits tracks the parser,
    step by step.
    """
    from tandem.backends import mlx_tier0

    sets = [[1], [3]]
    processor = mlx_tier0.build_logits_processor(lambda _t: sets.pop(0), _MX())

    first = processor(_Tokens(), _logits(5))
    second = processor(_Tokens(), _logits(5))

    assert first.values[1] == 1.0 and first.values[3] == float("-inf")
    assert second.values[3] == 3.0 and second.values[1] == float("-inf")


# --- the backend actually passes it down ------------------------------------


@pytest.fixture
def tier0(tmp_path):
    """A real `MLXTier0Backend` on the fake MLX, as `test_mlx_tier0.py` builds one."""
    import fake_mlx

    container = tmp_path / "qwen3.6-35b-a3b-4bit"
    container.mkdir()
    (container / "config.json").write_text(
        '{"model_type": "qwen3_moe"}', encoding="utf-8"
    )
    with fake_mlx.install():
        from tandem.backends.mlx_tier0 import MLXTier0Backend

        yield MLXTier0Backend(str(container))


def _request(schema=None):
    from tandem.types import GenRequest, Message, Role

    return GenRequest(
        messages=[Message(role=Role.USER, content="find the importers")],
        json_schema=schema,
    )


def test_a_free_form_turn_is_left_unconstrained(tier0):
    """Constraining prose is how you get a model that answers everything with a call."""
    assert tier0._logits_processors(_request()) == []


def test_a_schema_bearing_turn_gets_exactly_one_processor(tier0, tokenizer):
    tier0._constrain_vocab = Constrainer().vocabulary(tokenizer)

    processors = tier0._logits_processors(_request(tool_call_schema([READ_FILE])))

    assert len(processors) == 1 and callable(processors[0])


def test_the_vocabulary_is_built_once_and_reused(tier0, tokenizer, monkeypatch):
    """~1.1 s and ~0.6 GB against the real tokenizer — per turn that is not viable."""
    tier0.tokenizer = tokenizer
    builds = []
    real = Constrainer.vocabulary

    def counting(self, tok):
        builds.append(tok)
        return real(self, tok)

    monkeypatch.setattr(Constrainer, "vocabulary", counting)
    for _ in range(3):
        tier0._logits_processors(_request(tool_call_schema([READ_FILE])))

    assert len(builds) == 1


def test_unloading_drops_the_vocabulary_with_the_tokenizer_it_describes(
    tier0, tokenizer
):
    """A stale prefix tree would constrain against a vocabulary nobody is sampling."""
    import asyncio

    tier0._constrain_vocab = Constrainer().vocabulary(tokenizer)
    asyncio.run(tier0.unload())

    assert tier0._constrain_vocab is None


def test_a_missing_enforcer_is_probed_once_and_then_remembered(tier0, monkeypatch):
    calls = []

    def unavailable(self, tok):
        calls.append(tok)

    monkeypatch.setattr(Constrainer, "vocabulary", unavailable)
    for _ in range(3):
        assert tier0._logits_processors(_request(tool_call_schema([READ_FILE]))) == []

    assert len(calls) == 1


@pytest.mark.parametrize("width", [5, 64])
def test_the_mask_matches_real_mlx_where_mlx_is_installed(width):
    """The double models MLX; this checks the model against the library.

    Skips off Apple Silicon, which is where the whole class of bug lives — a
    contract checked only against a stand-in is a contract nobody has tested.
    """
    mx = pytest.importorskip("mlx.core", reason="MLX is Apple-Silicon-only")
    from tandem.backends.mlx_tier0 import build_logits_processor

    allowed = [1, width - 2]
    processor = build_logits_processor(lambda _tokens: allowed, mx)
    logits = mx.array([[float(i) for i in range(width)]])
    out = processor(mx.array([7]), logits)

    assert out.shape == (1, width)
    assert int(mx.argmax(out).item()) in allowed
    values = out.tolist()[0]
    for i in range(width):
        assert (values[i] == float(i)) if i in allowed else (values[i] == float("-inf"))
