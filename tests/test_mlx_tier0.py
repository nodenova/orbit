"""Tier 0 — the resident adapted generator, executed (spec sec 4).

Every line of `backends/mlx_tier0.py` had never run. Not "was untested" — had
never been imported by anything, on any machine, in the life of the repository.
It was held up by the claim that it is correct on inspection, and inspection is
exactly the method that misses a chat template quietly dropping half a
conversation.

`fake_mlx` makes the module executable off-target. Read its docstring for what
that does and does not establish; the short version is that these tests prove the
*wiring* and prove nothing about numerics on real weights. The isolation gate on
an M4 Max is still what proves tier 0. But the wiring is where the silent
wrong-answer bugs in sec 4.2 live, and the wiring is now checkable in 8 seconds
instead of after a hardware purchase.
"""

from __future__ import annotations

import asyncio

import pytest

import fake_mlx
from tandem.eval.gates import adapter_isolation_gate
from tandem.types import (
    GenRequest,
    Message,
    Role,
    Sampling,
    StopReason,
    ToolCall,
    ToolDef,
    ToolResult,
)

A1_KEYS = [
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.mlp.down_proj",
    "layers.1.self_attn.q_proj",
]
# Deliberately a different, overlapping layer set: §4.3 targets the top-25% of
# routed experts, so two adapters trained on the same repo do not carry the same
# key set, and a mount path that assumed they did would work only by luck.
A2_KEYS = [
    "layers.0.self_attn.q_proj",
    "layers.1.router",
    "layers.1.mlp.up_proj",
]


@pytest.fixture
def mlx():
    with fake_mlx.install():
        yield fake_mlx


@pytest.fixture
def container(tmp_path):
    path = tmp_path / "qwen3.6-35b-a3b-4bit"
    path.mkdir()
    (path / "config.json").write_text('{"model_type": "qwen3_moe"}', encoding="utf-8")
    return path


@pytest.fixture
def adapters(tmp_path):
    root = tmp_path / "adapters"
    fake_mlx.write_adapter(root / "a0-harness", A1_KEYS, salt="a0")
    fake_mlx.write_adapter(root / "a1-myrepo", A2_KEYS, salt="a1")
    return root


def _backend(mlx, container, *, adapter_dir=None):
    from tandem.backends.mlx_tier0 import MLXTier0Backend

    return MLXTier0Backend(str(container), adapter_dir=adapter_dir)


def _req(text: str = "Fix the pagination helper.", **kw) -> GenRequest:
    kw.setdefault(
        "sampling", Sampling(temperature=0.0, top_p=1.0, seed=0, max_tokens=32)
    )
    return GenRequest(messages=[Message(role=Role.USER, content=text)], **kw)


# --- it runs at all ---------------------------------------------------------


def test_the_backend_constructs_and_wraps_its_targets(mlx, container):
    backend = _backend(mlx, container)
    assert backend.mounted_adapters() == ()
    assert backend.container_hash() is not None
    # 2 blocks x (4 attention + 1 router + 3 mlp).
    assert len(backend._targets) == 16


def test_targeting_follows_4_3_and_leaves_the_head_alone(mlx, container):
    """A change that starts wrapping everything is a rank-32 delta on the output
    embedding, which is both wasteful and not what the profile trained."""
    from tandem.backends.mlx_tier0 import _is_target

    backend = _backend(mlx, container)
    assert "layers.0.self_attn.q_proj" in backend._targets
    assert "layers.1.router" in backend._targets
    assert "layers.0.mlp.down_proj" in backend._targets
    assert "lm_head" not in backend._targets

    assert _is_target("model.layers.7.self_attn.o_proj") is True
    assert _is_target("model.layers.7.mlp.shared_expert.gate_proj") is True
    assert _is_target("lm_head") is False
    assert _is_target("model.embed_tokens") is False


@pytest.mark.asyncio
async def test_the_base_model_generates(mlx, container):
    backend = _backend(mlx, container)
    result = await backend.generate(_req())
    assert result.text
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens == len(result.text.split())


# --- mounting (sec 4.2) -----------------------------------------------------


def test_mount_all_picks_up_every_adapter_directory(mlx, container, adapters):
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    assert backend.mounted_adapters() == ("a0-harness", "a1-myrepo")
    assert backend.adapter_hash("a0-harness") is not None
    assert backend.adapter_hash("a1-myrepo") != backend.adapter_hash("a0-harness")
    assert backend.adapter_hash("not-mounted") is None


def test_an_adapter_matching_no_layer_is_a_mount_failure(mlx, container, tmp_path):
    """The key-naming mismatch between trainer and served model. Silence here
    means a receipt naming an adapter that contributed nothing."""
    bad = fake_mlx.write_adapter(
        tmp_path / "wrong-names", ["transformer.h.0.attn.c_attn"]
    )
    backend = _backend(mlx, container)
    with pytest.raises(ValueError, match="matched no target layer"):
        backend.mount("wrong-names", bad)


def test_an_adapter_covers_only_its_own_layers(mlx, container, adapters):
    """§4.3 targets a subset, so most wrapped layers carry no delta for a given
    adapter. Falling through to base there is correct, not an error."""
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    q0 = backend._targets["layers.0.self_attn.q_proj"]
    k0 = backend._targets["layers.0.self_attn.k_proj"]
    assert set(q0.deltas) == {"a0-harness", "a1-myrepo"}
    assert k0.deltas == {}


def test_the_scale_comes_from_the_adapter_config_not_a_default(
    mlx, container, tmp_path
):
    path = fake_mlx.write_adapter(tmp_path / "a1", A1_KEYS, rank=2, alpha=8)
    backend = _backend(mlx, container)
    spec = backend.mount("a1", path)
    assert spec.rank == 2
    assert spec.scale == 4.0


def test_a_missing_routing_profile_hashes_to_none_rather_than_inventing_one(
    mlx, container, adapters
):
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    assert backend.profile_hash("a0-harness") is None


# --- selection (sec 4.2, the ContextVar) ------------------------------------


@pytest.mark.asyncio
async def test_an_adapter_changes_the_output(mlx, container, adapters):
    """If it does not, every test below is vacuous and so is the product."""
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    base = await backend.generate(_req())
    adapted = await backend.generate(_req(adapter="a1-myrepo"))
    assert base.text != adapted.text


@pytest.mark.asyncio
async def test_two_adapters_produce_two_different_outputs(mlx, container, adapters):
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    a0 = await backend.generate(_req(adapter="a0-harness"))
    a1 = await backend.generate(_req(adapter="a1-myrepo"))
    assert a0.text != a1.text


@pytest.mark.asyncio
async def test_an_unknown_adapter_name_falls_through_to_base(mlx, container, adapters):
    """It generates from base rather than failing — and, importantly, reports no
    adapter hash, so the receipt cannot claim an adapter that did not run."""
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    base = await backend.generate(_req())
    unknown = await backend.generate(_req(adapter="a9-nonexistent"))
    assert unknown.text == base.text
    assert backend.adapter_hash("a9-nonexistent") is None


@pytest.mark.asyncio
async def test_selection_is_read_at_call_time_and_is_per_task(mlx, container, adapters):
    """Sec 4.2's silent wrong-answer bug, reproduced as directly as it can be.

    A module global here races: task B sets the adapter while task A is mid-decode
    and A finishes in B's adapter with a receipt attesting to its own. The
    `ContextVar` is what makes that impossible, and the property only shows up
    when two tasks genuinely interleave — so the two tasks below hand off through
    an event in the middle of the forward pass.

    The full backend cannot be driven into this shape under the fake, because its
    decode loop has no true suspension point; what interleaves is the layer the
    race would actually corrupt.
    """
    from tandem.backends.mlx_tier0 import ACTIVE_ADAPTER

    backend = _backend(mlx, container, adapter_dir=str(adapters))
    wrapper = backend._targets["layers.0.self_attn.q_proj"]
    x = fake_mlx.Array([[0.5, -0.25, 0.75, 0.125]])
    both_set = asyncio.Event()

    async def under(name: str, wait: bool):
        token = ACTIVE_ADAPTER.set(name)
        try:
            first = wrapper(x)
            if wait:
                both_set.set()
                await asyncio.sleep(0)
            else:
                await both_set.wait()
            # After the other task has set its own adapter, this one must still be
            # computing under the name it set.
            assert wrapper(x).tolist() == first.tolist()
            return first.tolist()
        finally:
            ACTIVE_ADAPTER.reset(token)

    a, b = await asyncio.gather(under("a0-harness", True), under("a1-myrepo", False))
    assert a != b
    assert ACTIVE_ADAPTER.get() is None


@pytest.mark.asyncio
async def test_the_contextvar_is_reset_when_a_stream_is_abandoned(
    mlx, container, adapters
):
    """A client disconnecting mid-stream is the ordinary case, not the exotic one.

    A binding leaked on the way out makes the *next* request on this task adapted
    without asking, and no receipt would catch it: the request never named an
    adapter, so there is nothing for the attestation to contradict.
    """
    from tandem.backends.mlx_tier0 import ACTIVE_ADAPTER

    backend = _backend(mlx, container, adapter_dir=str(adapters))
    stream = backend.stream(_req(adapter="a1-myrepo"))
    await stream.__anext__()
    await stream.aclose()
    assert ACTIVE_ADAPTER.get() is None


# --- the blocking gate, run for real (sec 4.2) ------------------------------


@pytest.mark.asyncio
async def test_adapter_isolation_gate_passes_against_the_real_tier0(
    mlx, container, adapters
):
    """`tandem gate isolation`, against `MLXTier0Backend` rather than the mock.

    This is the sec 4.2 blocking gate — greedy output under adapter *i* with N
    mounted must be byte-identical to output with only *i* mounted — driven
    through the real `MultiAdapterLinear` and the real `ContextVar` binding. It
    does not retire the gate on hardware: real weights, real quantised deltas and
    Metal are all still untested. It does mean an isolation bug in this code is a
    CI failure rather than a discovery in month two.
    """

    def factory(names):
        backend = _backend(mlx, container)
        for name in names:
            backend.mount(name, adapters / name)
        return backend

    result = await adapter_isolation_gate(factory, ["a0-harness", "a1-myrepo"])
    assert result.passed, result.reason
    assert result.detail["comparisons"] == 6


@pytest.mark.asyncio
async def test_the_isolation_gate_catches_a_leak(mlx, container, adapters):
    """A gate nobody has seen fail is a gate nobody has tested.

    The leak is the one sec 4.2 names: a wrapper that applies every delta it holds
    instead of the selected one. Under N mounted it sums both adapters; solo it
    applies one. That is precisely the difference the gate compares, so it must
    come back failed — and with the mismatches named, because "failed" alone does
    not tell anyone which adapter leaked into which.

    `_wrap_targets` builds a fresh `MultiAdapterLinear` class per backend, so
    patching the class the wrappers came from stays inside this one factory.
    """

    def leaky_factory(names):
        backend = _backend(mlx, container)
        for name in names:
            backend.mount(name, adapters / name)

        def leaky(self, x):
            y = self.base(x)
            for a, b, scale in self.deltas.values():  # never reads ACTIVE_ADAPTER
                y = y + scale * ((x.astype(b.dtype) @ a.astype(b.dtype)) @ b)
            return y

        type(next(iter(backend._targets.values()))).__call__ = leaky
        return backend

    result = await adapter_isolation_gate(leaky_factory, ["a0-harness", "a1-myrepo"])
    assert result.passed is False
    assert result.detail["mismatches"]
    assert "leaking" in result.reason


# --- residency (sec 5.5 rung 2) ---------------------------------------------


@pytest.mark.asyncio
async def test_unload_then_load_comes_back_with_the_same_identity(
    mlx, container, adapters
):
    """Rung 2 evicts tier 0 to admit the 80B. A tier 0 that came back with a
    different adapter set would produce receipts naming what did not run."""
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    before = await backend.generate(_req(adapter="a1-myrepo"))
    hashes = {n: backend.adapter_hash(n) for n in backend.mounted_adapters()}

    await backend.unload()
    assert backend.model is None
    assert all(not spec.weights for spec in backend._adapters.values())

    await backend.load()
    assert backend.mounted_adapters() == ("a0-harness", "a1-myrepo")
    assert {n: backend.adapter_hash(n) for n in backend.mounted_adapters()} == hashes
    after = await backend.generate(_req(adapter="a1-myrepo"))
    assert after.text == before.text


@pytest.mark.asyncio
async def test_load_on_a_resident_backend_is_a_no_op(mlx, container, adapters):
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    model = backend.model
    await backend.load()
    assert backend.model is model


# --- rendering (sec 8.4, 8.5.5) ---------------------------------------------


def test_render_goes_through_the_model_s_own_chat_template(mlx, container):
    backend = _backend(mlx, container)
    rendered = backend.render(_req("hello"))
    assert "<|im_start|>user" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_a_backend_with_its_own_template_does_not_render_canonically(mlx, container):
    """`Pipeline.render` asks this to decide whether to wrap the request in the
    replay map. Getting it wrong is a cache-key bug."""
    backend = _backend(mlx, container)
    assert backend.renders_canonically() is False


def test_the_system_prompt_and_tools_are_in_the_rendered_bytes(mlx, container):
    tool = ToolDef(name="edit_file", parameters={"type": "object", "properties": {}})
    rendered = _backend(mlx, container).render(
        _req("go", system="You are terse.", tools=(tool,))
    )
    assert "You are terse." in rendered
    assert "edit_file" in rendered


def test_tool_calls_and_results_reach_the_prompt(mlx, container):
    """Sec 8.4 and 8.5.5, and the reason this file was worth writing.

    Two failure modes from one omission, both silent. The model cannot see what it
    already called, so it calls it again; and the disk KV cache keys on the SHA-256
    of these bytes, so two conversations differing only in their tool history
    collide on one key and a restored state describes a different conversation.
    """
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})
    req = GenRequest(
        messages=[
            Message(role=Role.USER, content="what is in a.py?"),
            Message(role=Role.ASSISTANT, tool_calls=(call,)),
            Message(
                role=Role.TOOL,
                tool_results=(ToolResult(tool_call_id="call_1", content="print(1)"),),
            ),
        ],
        sampling=Sampling(temperature=0.0, max_tokens=8),
    )
    rendered = _backend(mlx, container).render(req)
    assert "read_file" in rendered
    assert "a.py" in rendered
    assert "print(1)" in rendered


def test_two_conversations_differing_only_in_tool_calls_render_differently(
    mlx, container
):
    """The cache-key half of the bug above, stated as the property that matters."""
    backend = _backend(mlx, container)

    def with_call(path: str) -> str:
        return backend.render(
            GenRequest(
                messages=[
                    Message(role=Role.USER, content="look"),
                    Message(
                        role=Role.ASSISTANT,
                        tool_calls=(
                            ToolCall(
                                id="c1", name="read_file", arguments={"path": path}
                            ),
                        ),
                    ),
                ],
                sampling=Sampling(temperature=0.0, max_tokens=8),
            )
        )

    assert with_call("a.py") != with_call("b.py")


def test_the_model_s_own_sampled_bytes_go_back_into_the_prompt(mlx, container):
    """Sec 8.5.5, and the reason the renderer is a parameter rather than a lookup.

    Handing the template a *parsed* call lets the template choose the
    serialisation, and the template's choice is not what the model sampled. The
    prefix then diverges at the first tool call in the conversation and every turn
    after it rebuilds from scratch — a 2 s turn becoming a 60 s one, on every
    tool-using turn, which is all of them.
    """
    sampled = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
    req = GenRequest(
        messages=[
            Message(role=Role.USER, content="look"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=(
                    ToolCall(id="c1", name="read_file", arguments={"path": "a.py"}),
                ),
            ),
        ],
        sampling=Sampling(temperature=0.0, max_tokens=8),
    )
    rendered = _backend(mlx, container).render(req, lambda call: sampled)
    assert sampled in rendered


def test_a_failed_tool_result_does_not_render_as_a_successful_one(mlx, container):
    """Same text, opposite meaning. The model's next move differs."""
    backend = _backend(mlx, container)

    def render_result(is_error: bool) -> str:
        return backend.render(
            GenRequest(
                messages=[
                    Message(
                        role=Role.TOOL,
                        tool_results=(
                            ToolResult(
                                tool_call_id="c1",
                                content="no such file",
                                is_error=is_error,
                            ),
                        ),
                    )
                ],
                sampling=Sampling(temperature=0.0, max_tokens=8),
            )
        )

    assert render_result(True) != render_result(False)


def test_an_empty_tool_list_is_not_rendered_as_an_empty_tool_block(mlx, container):
    """`tools=[]` and `tools=None` must not be different bytes for the same
    conversation, or every toolless turn misses the cache on alternate runs."""
    rendered = _backend(mlx, container).render(_req("hi"))
    assert "<|tools|>" not in rendered


# --- generation accounting --------------------------------------------------


@pytest.mark.asyncio
async def test_hitting_the_ceiling_reports_max_tokens(mlx, container):
    """Clamped below the turn's natural length, so the ceiling is genuinely hit
    rather than the model happening to stop first."""
    backend = _backend(mlx, container)
    free = await backend.generate(_req())
    assert free.usage.output_tokens > 1, "need a turn long enough to truncate"

    ceiling = free.usage.output_tokens - 1
    result = await backend.generate(
        _req(sampling=Sampling(temperature=0.0, top_p=1.0, seed=0, max_tokens=ceiling))
    )
    assert result.usage.output_tokens == ceiling
    assert result.stop_reason.value == "max_tokens"


@pytest.mark.asyncio
async def test_a_stop_sequence_ends_the_turn_and_is_reported_as_one(mlx, container):
    backend = _backend(mlx, container)
    free = await backend.generate(_req())
    marker = free.text.split()[1]

    stopped = await backend.generate(
        _req(
            sampling=Sampling(
                temperature=0.0, top_p=1.0, seed=0, max_tokens=32, stop=(marker + " ",)
            )
        )
    )
    assert stopped.stop_reason.value == "stop_sequence"
    assert stopped.usage.output_tokens < free.usage.output_tokens


@pytest.mark.asyncio
async def test_streaming_deltas_reassemble_into_the_final_text(mlx, container):
    backend = _backend(mlx, container)
    chunks: list[str] = []
    final = None
    async for delta in backend.stream(_req()):
        if delta.text:
            chunks.append(delta.text)
        if delta.done:
            final = delta.result
    assert final is not None
    assert "".join(chunks) == final.text


@pytest.mark.asyncio
async def test_greedy_generation_is_reproducible(mlx, container, adapters):
    """The receipt's determinism claim, at the only layer that can be checked
    here. G1/G2 on hardware are what check it against Metal."""
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    first = await backend.generate(_req(adapter="a1-myrepo"))
    second = await backend.generate(_req(adapter="a1-myrepo"))
    assert first.text == second.text


def test_count_tokens_uses_the_real_tokenizer_not_the_byte_estimate(mlx, container):
    backend = _backend(mlx, container)
    assert backend.count_tokens("one two three") == 3


# --- constrained decoding reaches the logits (sec 8.5.1) --------------------


@pytest.mark.asyncio
async def test_a_logits_processor_is_called_once_per_sampled_token(mlx, container):
    """Guards the fake against the bug the real backend actually had.

    `MLXTier0Backend` accepted `json_schema` and never applied it, and the suite
    stayed green because nothing checked that a constraint reached the logits. A
    fake that took `logits_processors` and ignored it would reproduce exactly that
    blindness one layer down.
    """
    backend = _backend(mlx, container)
    seen: list[list[int]] = []

    def record(tokens, logits):
        seen.append(tokens.tolist())
        return logits

    backend._logits_processors = lambda req: [record]
    result = await backend.generate(_req())

    assert seen[0] == [], (
        "the prompt's tokens are prefilled without invoking processors"
    )
    assert [len(s) for s in seen] == list(range(len(seen))), "one token added per step"
    # One call more than there are emitted tokens: the last one masked the step that
    # sampled the stop token. That is the call that has to happen — a mask which
    # stops running once the schema is satisfied is a mask that cannot permit the
    # terminator, and the turn would run to `max_tokens` instead of ending.
    assert len(seen) == len(result.text.split()) + 1
    assert result.stop_reason is StopReason.END_TURN


@pytest.mark.asyncio
async def test_a_processor_that_changes_the_logits_changes_the_output(mlx, container):
    """`stay arithmetic`: the mask must reach the sampler, not just the signature."""
    backend = _backend(mlx, container)
    unconstrained = await backend.generate(_req())

    def shift(tokens, logits):
        return logits + fake_mlx.Array([[1.0] * logits.shape[1]])

    backend._logits_processors = lambda req: [shift]
    constrained = await backend.generate(_req())

    assert constrained.text != unconstrained.text


def test_the_fake_refuses_a_processor_that_is_not_callable(mlx, container):
    with pytest.raises(TypeError):
        list(
            fake_mlx.stream_generate(
                fake_mlx.FakeModel(),
                fake_mlx.FakeTokenizer(),
                prompt="hi",
                sampler=fake_mlx.make_sampler(),
                logits_processors=["not a processor"],
            )
        )


# --- KV state (sec 8.4) -----------------------------------------------------
#
# The container this ships against builds a *hybrid* cache: 30 of its 40 layers
# are linear attention, whose recurrent state cannot be rewound. `fake_mlx`
# reproduces that proportion, so unless a test says otherwise it is exercising
# the arrangement the hardware has — the one where `can_trim_prompt_cache` is
# False and a snapshot has to carry the whole turn instead of a trimmed prefix.


def _conversation(*texts: str, **kw) -> GenRequest:
    kw.setdefault(
        "sampling", Sampling(temperature=0.0, top_p=1.0, seed=0, max_tokens=32)
    )
    return GenRequest(
        messages=[Message(role=Role.USER, content=t) for t in texts], **kw
    )


def _continued(req: GenRequest, reply: str, follow_up: str) -> GenRequest:
    """The next turn of the same conversation: the reply, then a new message.

    The shape the disk cache exists for. A follow-up built as two *user* messages
    instead diverges from the stored prefix at the first one's end, which is a
    cache miss for a good reason and proves nothing about restoring.
    """
    return req.with_(
        messages=[
            *req.messages,
            Message(role=Role.ASSISTANT, content=reply.strip()),
            Message(role=Role.USER, content=follow_up),
        ]
    )


def _all_attention(backend):
    """Drop the recurrent layers, leaving a cache that can be rewound."""
    backend.model.cache_kinds = ("kv",)
    return backend


def _shared_prefix(a: str, b: str) -> str:
    """The bytes two rendered prompts have in common."""
    n = 0
    for left, right in zip(a, b):
        if left != right:
            break
        n += 1
    return a[:n]


def _exported(backend, req, prefix=None):
    """Run a turn and snapshot it, the way `Pipeline._remember` does."""
    result = asyncio.run(backend.generate(req))
    return result, backend.export_state(req, prefix or backend.render(req), result)


def test_a_turn_can_be_snapshotted_and_the_snapshot_names_its_model(mlx, container):
    backend = _backend(mlx, container)
    assert backend.supports_state() is True

    req = _req()
    result, state = _exported(backend, req)
    assert state is not None
    assert state.key == backend.state_key(None)
    assert state.blob

    prompt = backend._encode(backend.render(req))
    # Nothing rewound this cache, so the state names the reply as well as the
    # prompt — and names every token of it, or it could not be restored at all.
    assert state.token_ids[: len(prompt)] == tuple(prompt)
    assert state.n_tokens == len(prompt) + result.usage.output_tokens


def test_export_state_without_a_turn_behind_it_returns_nothing(mlx, container):
    """`export_state` is called on every turn, including ones no backend ran."""
    backend = _backend(mlx, container)
    assert backend.export_state(_req(), "prefix", None) is None


def test_a_warm_start_reproduces_the_cold_answer(mlx, container):
    """The property the whole path is worth having only if it holds.

    A restored prefix must put the model in the state a full prefill would have.
    If the two answers differ, the cache is not an optimisation — it is a second,
    quieter model.
    """
    backend = _backend(mlx, container)
    first = _req()
    reply, state = _exported(backend, first)
    follow_up = _continued(first, reply.text, "Now add a test.")

    cold = asyncio.run(backend.generate(follow_up))
    assert cold.usage.cached_input_tokens == 0

    warm = asyncio.run(backend.generate(follow_up.with_(warm_state=state)))
    assert warm.usage.cached_input_tokens == state.n_tokens
    assert warm.usage.input_tokens == cold.usage.input_tokens
    assert warm.text == cold.text


def test_a_warm_start_reproduces_the_cold_answer_when_the_cache_can_be_trimmed(
    mlx, container
):
    """The same guarantee on the other arrangement, which trims instead."""
    backend = _all_attention(_backend(mlx, container))
    first = _req()
    follow_up = _conversation("Fix the pagination helper.", "Now add a test.")
    prefix = _shared_prefix(backend.render(first), backend.render(follow_up))

    cold = asyncio.run(backend.generate(follow_up))
    _reply, state = _exported(backend, first, prefix=prefix)
    warm = asyncio.run(backend.generate(follow_up.with_(warm_state=state)))

    assert warm.usage.cached_input_tokens == state.n_tokens
    assert warm.text == cold.text


def test_a_state_from_another_adapter_is_refused(mlx, container, adapters):
    """Same container, different adapter: fluent output from the wrong model."""
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    first = _req(adapter="a0-harness")
    reply, state = _exported(backend, first)
    assert state is not None

    follow_up = _continued(first, reply.text, "Now add a test.")
    wrong = follow_up.with_(adapter="a1-myrepo", warm_state=state)
    assert asyncio.run(backend.generate(wrong)).usage.cached_input_tokens == 0


def test_a_state_whose_tokens_diverged_is_refused(mlx, container):
    """The identity key says "same model". It says nothing about the bytes."""
    backend = _backend(mlx, container)
    _reply, state = _exported(backend, _req("Fix the pagination helper."))

    other = _req("Rewrite the retry policy instead.").with_(warm_state=state)
    assert asyncio.run(backend.generate(other)).usage.cached_input_tokens == 0


def test_a_corrupt_blob_is_a_miss_and_not_an_error(mlx, container):
    """States come back off disk, where a truncated entry is ordinary (sec 8.4)."""
    from dataclasses import replace

    backend = _backend(mlx, container)
    first = _req()
    reply, state = _exported(backend, first)
    follow_up = _continued(first, reply.text, "Now add a test.")

    # The control: intact, this state restores. Without it every assertion below
    # would hold for a state that was never going to be used.
    warmed = follow_up.with_(warm_state=state)
    assert asyncio.run(backend.generate(warmed)).usage.cached_input_tokens > 0

    for blob in (state.blob[:-8], state.blob + b"tail", b"", b"not a snapshot"):
        warmed = follow_up.with_(warm_state=replace(state, blob=blob))
        assert asyncio.run(backend.generate(warmed)).usage.cached_input_tokens == 0


def test_a_repeat_of_a_fully_cached_turn_still_has_a_token_to_feed(mlx, container):
    """`generate_step` rejects an empty prompt, so the last token is always fed."""
    backend = _all_attention(_backend(mlx, container))
    req = _req()
    rendered = backend.render(req)
    _reply, state = _exported(backend, req, prefix=rendered)

    repeat = asyncio.run(backend.generate(req.with_(warm_state=state)))
    assert repeat.usage.cached_input_tokens == len(backend._encode(rendered)) - 1


def test_a_state_over_the_byte_budget_is_not_stored(mlx, container):
    """Refusing costs one cold prefill; allocating it costs the machine (T9)."""
    from tandem.backends.mlx_tier0 import MLXTier0Backend

    backend = MLXTier0Backend(str(container), max_state_bytes=8)
    _result, state = _exported(backend, _req())
    assert state is None


def test_export_state_lets_go_of_the_cache_it_snapshotted(mlx, container):
    """A KV cache the size of the prompt must not outlive the turn."""
    backend = _backend(mlx, container)
    req = _req()
    result = asyncio.run(backend.generate(req))
    assert result.kv_handle is not None

    backend.export_state(req, backend.render(req), result)
    assert result.kv_handle is None
    assert backend.export_state(req, backend.render(req), result) is None


def test_a_trimmable_cache_is_rewound_to_the_prefix_it_is_keyed_by(mlx, container):
    """Where it can rewind it should: the state then serves any conversation
    sharing that prefix, not only this one's continuation."""
    backend = _all_attention(_backend(mlx, container))
    req = _req()
    rendered = backend.render(req)
    prefix = rendered[: len(rendered) // 2]

    _result, state = _exported(backend, req, prefix=prefix)
    assert state is not None
    full = backend._encode(rendered)
    assert 0 < state.n_tokens < len(full)
    assert state.token_ids == tuple(full[: state.n_tokens])


def test_a_recurrent_cache_is_not_silently_dropped(mlx, container):
    """The failure this arrangement exists to catch.

    A snapshot that refused whenever the cache could not be rewound would be a
    no-op on the container that ships — every layer of it — while the suite that
    only ever built trimmable caches stayed green.
    """
    from mlx_lm.models import cache as kv

    backend = _backend(mlx, container)
    req = _req()
    result = asyncio.run(backend.generate(req))
    assert kv.can_trim_prompt_cache(result.kv_handle.cache) is False

    state = backend.export_state(req, backend.render(req), result)
    assert state is not None and state.n_tokens > 0


def test_the_state_key_separates_containers_and_adapters(mlx, container, adapters):
    """A state restored under a changed adapter gives fluent output from the wrong
    model and a receipt naming the adapter that did not produce it."""
    backend = _backend(mlx, container, adapter_dir=str(adapters))
    base = backend.state_key(None)
    assert base != backend.state_key("a1-myrepo")
    assert backend.container_hash() in base


@pytest.mark.asyncio
async def test_the_disk_cache_makes_a_restart_warm(mlx, container, tmp_path):
    """Tier 0 through the whole sec 8.4 loop, not just its two halves.

    A second `Pipeline` over the same directory is a fresh process in every way
    that matters. Until `supports_state` returned True this path was dead on the
    mlx backend: `_probe_cache` and `_remember` both check it first, so every turn
    re-prefilled from scratch and the disk cache held nothing to hit.
    """
    from tandem.config import Config
    from tandem.gateway.pipeline import Pipeline

    cfg = Config()
    cfg.attest.audit_log = str(tmp_path / "audit.jsonl")
    cfg.cache.disk_kv_dir = str(tmp_path / "kv")
    cfg.compaction.enabled = False
    body = "def handler(request):\n    return process(request)\n\n" * 60

    first = Pipeline(cfg, _backend(mlx, container))
    opening = _conversation(body)
    reply, _trace = await first.run(opening)
    assert first._disk_kv_stats()["entries"] > 0

    second = Pipeline(cfg, _backend(mlx, container))
    result, trace = await second.run(_continued(opening, reply.text, "and the tests?"))

    assert trace.cache["source"] == "disk"
    assert trace.cache["restored_tokens"] > 0
    assert result.usage.cached_input_tokens > 0
    assert second.disk_kv_hits == 1
