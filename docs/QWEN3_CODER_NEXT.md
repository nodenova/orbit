# Qwen3-Coder-Next on this host — it serves, and it is the first thing here to pass Gate B at spec

`mlx-community/Qwen3-Coder-Next-4bit`, 44.84 GB on disk, experts streamed from SSD.
**Measured 2026-08-10.** Gate B reads **258.0 tok/s** against sec 11's 200 floor, so
`meets_spec` is **true** — the first configuration on this machine to report it.

Three results, and the third is the one that keeps rung 3:

1. **It serves through `optiq serve`**, which `DeepSeek-V4-Flash-0731-OptiQ-2bit` has never
   done — §3. Same engine, same flags, same stack.
2. **It is 1.68× the 122B on Gate B at a third of the resident footprint**: 258.0 against
   153.7, 1.36 GB against 3.46 — §4.
3. **It is still ~3.5× slower than rung 3 at what tier 1 does**, and its published coding
   scores do not beat the tier 0 it would judge — §6. The gap narrowed from 6–9×; it did
   not close.

This document owns the model's own numbers, the load accounting and the buffer-ceiling
finding. `HANDOFF.md` T7 owns the decision; `PROCESSES.md` §7 owns how to bring it up.

---

## 1. What it is

From its `config.json` — 48 layers, hidden 2048, vocab 151,936, **512 routed experts with
10 active** plus 1 shared, `moe_intermediate_size` 512. `Qwen3NextForCausalLM`,
`model_type` `qwen3_next`.

| Field | Value | Why it matters |
|---|---|---|
| active parameters | **3 B** of 80 B | Prefill is compute-bound on active parameters, and this is the whole case for the model (§4). |
| `full_attention_interval` | 4 | 12 full-attention layers of 48; the other 36 are gated-delta linear attention. |
| `num_key_value_heads` / `head_dim` | 2 / 256 | 24 KiB/token of KV across the 12 full-attention layers. |
| `max_position_embeddings` | 262,144 | Not reachable here — §5. |
| `decoder_sparse_step` / `mlp_only_layers` | 1 / `[]` | Every layer is MoE, so the sweep is the whole tower. |

**It is the same architecture family as tier 0**, which is `qwen3_5_moe` — hybrid
gated-delta attention, `switch_mlp` experts, one size up. mlx-lm 0.31.3 implements it at
`models/qwen3_next.py` and mlx-optiq carries **no patch for it**, so the streaming path
runs against stock upstream code.

**The quantisation is uniform where DeepSeek's is not.** Top level is `bits 4,
group_size 64, mode affine`; the only per-path overrides are the 96 router gates and
shared-expert gates at 8-bit. **No `switch_mlp` path is overridden**, so the case
`ds4_headers.py` asserts against — `load_streaming` reading `mode` from the top level
while the config overrides it per path (`BASELINE.md` §4.6) — does not arise.

## 2. Shape, off the headers, without downloading it

`tools/qcn_headers.py`, nine HTTP Range requests, no weights and no GPU. Every figure
below was predicted before the download and reproduced exactly by the engine's own
startup lines (§3).

| Component | GB | Fate |
|---|---|---|
| routed expert `.weight` | 38.65 | streamed |
| routed expert `.scales` + `.biases` | 4.84 | streamed iff over the budget |
| everything else | **1.36** | **resident** |
| **total** | **44.84** | |

144 `.switch_mlp.` tensors, exactly 48 layers × 3 projections, named
`model.layers.N.mlp.switch_mlp.{gate,up,down}_proj` — the segment
`moe_stream._EXPERT_SEGMENTS` matches. **No `mtp`/`nextn` keys**, so nothing is dropped by
`sanitize()` and nothing is paid for.

Experts are stacked one tensor per (layer, projection), so a per-expert *read* is a slice:

| Read | Stride | Count per expert-layer |
|---|---|---|
| `{gate,up,down}_proj.weight` | **512 KiB** | 3 |
| `{gate,up,down}_proj.{scales,biases}` | **32 KiB** | 6 |

576 KiB per routed expert per layer; **849 MB per decoded token** at top-10 across 48.
Against DeepSeek-V4's 1,826 MB that is 2.15× less traffic — but at a quarter of the block
size, which §4 shows costs more than the byte count suggests.

## 3. It loads, and it serves

*2026-08-10, host passing `PROCESSES.md` §3.1 at 325/346 GB/s and 11.97 TFLOP/s.
mlx-optiq 0.4.18, mlx 0.32.0, mlx-lm 0.31.3.*

```
[moe_stream] expert scales/biases 4.8 GB vs budget 3.9 GB -> STREAM (off SSD)
[moe_stream] swapped 144 expert projections; resident 1.36 GB (load peak 1.36 GB)
```

Both numbers are §2's header sums to the digit, and `swapped 144` is the 48 × 3 the
headers predict. **Load peak equals steady state** — there is no 3.7× spike of the kind
making DeepSeek's scales resident produces (`BASELINE.md` §4.7).

| `OPTIQ_STREAM_SCALES_BUDGET_GB` | meta | resident | load peak |
|---|---|---|---|
| unset (engine prints a 3.9 GB budget) | streamed | **1.36 GB** | **1.36 GB** |
| 6 | resident | 6.22 GB | 11.02 GB |

**The first request returns.** 10 prompt tokens, 8 completion tokens, 3.5 s, engine still
listening, pageouts unmoved. That is the whole difference from T7's model, which loads on
this engine and aborts during prompt processing before any token, every time.

## 4. The numbers

### 4.1 Gate B

*`orbit bench tier1`, `--prefill-step-size 8192`, three consecutive runs, one session.
Filler is the gate's own `prefill_filler`, so these compare row-for-row with
`BASELINE.md` §4.1a's six runs on the 122B.*

| input tokens | run 1 | run 2 | run 3 |
|---|---|---|---|
| 5,130 | 266.0 | 259.2 | 256.4 |
| 10,431 | 262.6 | 257.9 | 253.5 |
| 21,008 | 284.5 | 282.3 | 276.4 |
| **worst of run** | **262.6** | **257.9** | **253.5** |

Mean **258.0**, spread **3.5%**. Every run `pass: true` and **`meets_spec: true`**.

| | 122B-A10B | DeepSeek-V4 | **Qwen3-Coder-Next** |
|---|---|---|---|
| Gate B, worst of run | 153.7 | 48 | **258.0** |
| `meets_spec` (sec 11's 200) | false | false | **true** |
| decode tok/s | 4.05 | 2.6 | **8.96** |
| resident | 3.46 GB | 6.49 GB | **1.36 GB** |
| disk | 45.94 GB | 92.49 GB | 44.84 GB |
| serves via `optiq serve` | yes | **no** | **yes** |

### 4.2 The scales budget is worth 1.08×, not the 1.36× the sweep alone predicts

*Same three frontiers, `OPTIQ_STREAM_SCALES_BUDGET_GB=6`.*

| input tokens | tok/s |
|---|---|
| 5,130 | 286.2 |
| 10,431 | **282.6** (worst of run) |
| 21,008 | 299.6 |

The lever costs 4.86 GB of residency and buys **1.08×** end to end. `qcn_streambench.py`
prices the sweep at 8.0 s/chunk streamed against 5.8–5.9 resident — **1.36×** — and the
difference between those two ratios is the compute term, which the lever does not touch.
**Read a sweep improvement as a fraction of prefill, never as prefill.** Unlike DeepSeek,
where the same lever needs 9.3 GB and raises the load peak to 23.80 GB, 4.84 GB here is
cheap enough that the only question is whether anything else wants the memory.

### 4.3 What the SSD actually does at these strides

*`tools/qcn_streambench.py`, sustained, `F_NOCACHE`, `iostat`-bracketed. `dev/req` held at
0.95 across every window, so these are storage rather than page cache.*

| | Measured |
|---|---|
| decode, 432 sequential barriers at qd 10 | **3.85 tok/s** — I/O only, compute not charged |
| prefill sweep, k = 512 at qd 24 | **8.0 s/chunk** at 5.40 GB/s |
| same, meta resident | **5.8–5.9 s/chunk** at 6.6 GB/s |
| weight 512 KiB, qd 10 / qd 24 | 4.80 / **5.55 GB/s** |
| meta 32 KiB, qd 10 / qd 24 | 0.90 / **1.27 GB/s** |

**The 32 KiB reads are the whole reason the lever pays.** They are 11% of the bytes and
run at 23% of the weight rate, which is `BASELINE.md` §2.1's "small blocks cost more than
their share" at a stride that section never measured.

The measured decode of **8.96 tok/s** is 2.3× this table's 3.85 because the OS page cache
holds hot experts across requests, and the page cache is the only expert cache in this
engine (`BASELINE.md` §4.2). Treat 3.85 as the cold floor and 8.96 as what a warm session
sees; neither is within 2.8× of `gate_a_decode_tok_per_s`, which is why this is a verifier.

## 5. What broke, and it is a ceiling this repo recorded as never having bitten

A probe sized its prompts in *words* and this tokenizer runs ~11.8 tokens per generated
identifier, so the third call sent **52,008 tokens** against `--max-context 32768`. The
engine aborted:

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
    [metal::malloc] Attempting to allocate 23622320128 bytes which is greater than
    the maximum allowed buffer size of 22613000192 bytes.
```

That is `max_buffer_length`, **21.06 GiB**, which `BASELINE.md` §1 recorded as "has not
bitten, because MLX loads safetensors shard by shard" and as foreclosing only a
single-tensor-per-model loader. **It also binds prefill**, and it binds it at a context
the config advertises as supported: `max_position_embeddings` is 262,144 and a single
allocation exceeds the ceiling somewhere between 21,008 tokens (measured, fine) and
52,008 (measured, fatal).

Two things follow:

- **The usable context ceiling here is a buffer limit, not a memory limit.** Resident was
  1.36 GB and headroom never moved; `total − active` is the wrong instrument for this
  failure and would have reported the machine as healthy, which it was.
- **`--max-context` is not a guard.** The engine accepted a prompt 59% over it and died
  inside the forward pass rather than refusing the request. Anything sizing prompts for
  this model must count tokens, not characters or words — which is also T31, where Gate B's
  own filler overshoots its frontier by ~36–39%.

Before the abort the same session produced three `curl`-side prefill readings, recorded
because they are the only ones taken against hand-built rather than gate-built prompts:

| prompt tokens | wall s | prefill tok/s |
|---|---|---|
| 5,898 | 17.92 | 329.2 |
| 19,008 | 59.55 | 319.2 |
| 52,008 | 181.68 | 286.3 |

These read ~25% above §4.1's gate figures at comparable sizes, the same direction and
rough size as the `curl`-versus-gate gap `BASELINE.md` §4.1a records for the 122B. Compare
the two tables by shape, not row against row.

## 6. What this settles, and what it does not

| Question | Answer |
|---|---|
| Does it load inside the working set? | **Yes** — 1.36 GB resident, 1.36 GB load peak, the smallest of any streamed model here |
| Does it serve through `optiq serve`? | **Yes**, unlike DeepSeek-V4 |
| Does it pass Gate B? | **Yes, and at spec** — 258.0 against 200, `meets_spec: true` |
| Is it a better verifier than the 122B on throughput? | **Yes** — 1.68× prefill, 2.2× decode, 2.5× less resident |
| Is it better than rung 3? | **No.** A 30k review is ~116 s of prefill against rung 3's ~33 s |
| Does it beat tier 0 on coding quality? | **No** — published SWE-bench Verified 70.6–74.2 against 73.4, Terminal-Bench 2.0 **36.2 against 51.5** |
| Do its verdicts decorrelate from tier 0's? | **Unmeasured.** This is the only question that matters now |
| What does `container_hash` cost over 44.84 GB? | Unmeasured |

**The rung-3 decision does not change**, and the reason is unchanged in kind but smaller in
size: rung 3 is tier 0 with the adapter stripped, costs no memory, and is ~3.5× faster at
the only thing tier 1 does. The 122B made that gap 6–9×. What this model changes is that
**the cost of verdict independence is now ~3.5× rather than ~8×**, and T5's open question —
whether independence is worth anything — is worth asking against a cheaper answer.

**Two derivations that did not survive contact, both recorded because the reasoning looked
sound:**

1. **Prefill was projected at ~350 tok/s and measured 258** — 26% optimistic. The
   projection scaled `BASELINE.md` §4.2's `tokens/218` compute term linearly from the 122B's
   10 B active parameters to 3 B, giving a 727 tok/s asymptote. It does not scale: 36 of 48
   layers are gated-delta linear attention and the router picks 10 of 512 experts, and
   neither cost shrinks with active parameters. **Scale the sweep term by expert bytes;
   do not scale the compute term by active parameters.**
2. **The model was predicted to take the batchable path and does not.** `make_cache()`
   returns `ArraysCache` and `KVCache`, both of which carry `merge`, so
   `is_batchable` should hold — yet the abort traceback in §5 runs through
   `server.py:814 _generate` → `_serve_single`, the **sequential** path, the same one
   DeepSeek takes. So being on the sequential path is *not* what distinguishes the model
   that crashes from the one that serves, and `DEEPSEEK_V4.md` §2's "the 122B avoids the
   whole area by being batchable" is left without a mechanism. **What separates the two is
   still open** — it is now known not to be the path.

## 7. Reproduction

```bash
# 2, the shape with no weights, no GPU, ~1 min. Retarget REPO for another quant.
python3 tools/qcn_headers.py

# 4.3, the read pattern replayed against local blobs. No weights, ~30 s, reads ~124 GB.
python3 tools/qcn_streambench.py

# 3, serve it. PROCESSES.md 7 first. Holds 1.36 GB resident, 6.22 with the lever.
S=~/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5
HF_HUB_OFFLINE=1 .venv-optiq/bin/optiq serve --stream-experts --model "$S" \
    --host 127.0.0.1 --port 8081 --max-context 32768 --prefill-step-size 8192

# 4.1, Gate B. Needs a config with tier1.rung = "streamed" pointing at the snapshot
# above; orbit.qcn.toml.example is that config.
.venv/bin/orbit --config orbit.qcn.toml.example bench tier1

# 4.2, the scales lever. Same command, engine restarted with the budget above 4.84 GB.
OPTIQ_STREAM_SCALES_BUDGET_GB=6 HF_HUB_OFFLINE=1 .venv-optiq/bin/optiq serve ...
```

**Do not send this engine more tokens than `--max-context`** — §5 is what happens, and it
takes the process with it.
