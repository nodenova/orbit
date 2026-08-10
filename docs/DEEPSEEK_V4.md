# DeepSeek-V4-Flash on this host — it runs, and it is slower than what we already have

`mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`, 92.49 GB on disk, experts streamed from
SSD. **It generated its first token on 2026-08-10** (`tools/ds4_probe.py`), after a year of
this repository recording it as loading-but-never-generating (`HANDOFF.md` T7,
`BASELINE.md` §4.7, `PROCESSES.md` §6).

Two results, and the second is the one that decides anything:

1. **The crash is the server's, not the model's.** `optiq serve` still aborts the process on
   the first request. The identical call chain driven on the **main thread** runs to
   completion, so what blocked this was thread-scoped MLX streams inside mlx-lm's server —
   §2.
2. **As a verifier it is worse than the 122B on both axes**: prefill **48 tok/s** against
   165, decode **2.6** against 4.05, at 2× the disk. Every reason T7 was deferred on cost
   holds, and now it is measured rather than derived — §3.

This document owns the model's own numbers and the crash mechanism. `HANDOFF.md` T7 owns the
decision; `PROCESSES.md` §6 owns how to bring it up.

---

## 1. What it is

From its `config.json` — 43 layers, hidden 4096, vocab 129,280, 256 routed experts with 6
active plus 1 shared, `expert_dtype fp4`. The parts that matter here are not the MoE:

| Field | Value | Why it matters |
|---|---|---|
| `max_position_embeddings` | **1,048,576** | with `rope_scaling` YARN factor 16 over an original 65,536 |
| `compress_ratios` | per layer, `4` or `128` | compressed attention: most layers keep a pooled state, not a full KV |
| `sliding_window` | 128 | the local half of each layer's attention |
| `index_topk` / `index_n_heads` | 512 / 64 | a sparse-attention indexer, its own cache |
| `num_nextn_predict_layers` | 1 | an MTP head ships with the model (see T27 — `tier0.mtp` is parsed and never read) |

`optiq/mlx_lm_patches/deepseek_v4.py` implements it, `make_cache()` returns
`RotatingKVCache` and `CacheList(RotatingKVCache, PoolingCache…)` per layer, and
`PoolingCache` is why the context ceiling is not where you would expect it (§3).

## 2. Why `optiq serve` aborts, and why driving it directly does not

The failure, unchanged since 2026-08-09 and reproduced twice on 2026-08-10:

```
INFO:root:Prompt processing progress: 0/11
libc++abi: terminating due to uncaught exception of type std::runtime_error:
    There is no Stream(gpu, 1) in current thread.
```

`PYTHONFAULTHANDLER=1` turns that abort into a Python traceback, which is the whole
diagnosis:

```
mlx_lm/server.py:814  _generate                 ← the SEQUENTIAL path
mlx_lm/server.py:976  _serve_single
mlx_lm/generate.py:716 stream_generate
mlx_lm/generate.py:433 generate_step            ← inside `with mx.stream(generation_stream)`
optiq/mlx_lm_patches/deepseek_v4.py:999→974→929→456
mlx_lm/models/switch_layers.py:188
optiq/runtime/moe_stream.py:263  __call__       ← np.array(indices) forces the first eval
```

**MLX 0.32's stream scoping, measured directly rather than assumed:**

| stream | created on | used on another thread |
|---|---|---|
| `mx.default_stream(device)` → `Stream(gpu, 0)` | main | **RuntimeError: there is no Stream(gpu, 0) in current thread** |
| `mx.new_thread_local_stream(device)` | main | **works** |
| `mx.default_stream(device)` resolved inside the thread | worker | works, and is a *different* index |
| no stream context at all | worker | works |

So a plain `Stream` is bound to the thread that resolved it and a `ThreadLocalStream` is
not. That **disproves** the mechanism this repo had recorded as likely — "a thread-local
`generation_stream` against optiq's 24-thread reader pool" — because the thread-local kind
is precisely the portable one. `mlx_lm.generate`'s module-level
`generation_stream = mx.new_thread_local_stream(...)` is correct and is not the fault.

A patch replacing it with `mx.default_stream()` was tried and **made it worse in the
diagnostic direction**: the abort became `There is no Stream(gpu, 0)`, confirming the table
above. `tools/ds4_serve.py` keeps that experiment and its result; it is not a fix.

**What is still unknown:** which frame resolves the plain `Stream(gpu, 1)` that the
original abort names. `mlx_lm/server.py:692` resolves a default stream inside `_generate`
and hands it to the `BatchGenerator`, which is the only plain-stream capture in either
package, and nothing in optiq calls `mx.new_stream`. The 122B avoids the whole area by
being *batchable* — `is_batchable = all(hasattr(c, "merge") for c in
make_prompt_cache(model))` — which routes it through `BatchGenerator` on the stream the
server resolved for itself. DeepSeek-V4 takes the sequential path.

**And none of it is needed to use the model.** `optiq.runtime.moe_stream.load_streaming` +
`mlx_lm.generate.stream_generate` on the main thread has never failed: single-threaded, no
server, no HTTP. That is how every number below was taken, and it is the configuration in
which this model has ever worked at all.

## 3. The numbers

Streamed experts, default scales budget (so the meta streams too), `--grad-checkpoint` n/a,
greedy, prompts built from this repository's own source. Host at 325/347 GB/s.

| Input tokens | Prefill | Prefill tok/s | Decode tok/s | Peak | Pageouts |
|---|---|---|---|---|---|
| 102 | 12.0 s | 8.5 | 2.78 | 12.19 GB | +86 |
| 1,607 | 33.0 s | 48.7 | 2.66 | 13.43 GB | +144 |
| 6,267 | 126.9 s | 49.4 | 2.54 | 14.11 GB | +137 |
| **12,776** | **266.9 s** | **47.9** | **2.57** | **14.15 GB** | +1,076 |

Load is **1.1–2.2 s** with the page cache warm, resident **6.49 GB**, load peak 6.49 GB —
all three matching `BASELINE.md` §4.7's earlier load-only measurement exactly.

**Prefill is flat at ~48 tok/s from 1.6k to 12.8k**, and the 102-token row is not an
outlier: at 8.5 tok/s it is one expert sweep amortised over almost nothing. This is the
same batch-union shape §4.2 found on the 122B, except the per-chunk cost here rises with
the chunk's expert union rather than staying constant — 12 s for 102 tokens against 33 s for
1,607 in the same single chunk.

**Memory is not the context constraint.** Peak moved 14.11 → 14.15 GB between 6.3k and
12.8k tokens, i.e. **~0.04 GB per 6.5k tokens**, because `compress_ratios` keeps most layers
on a pooled state rather than a growing KV. Against a 28.08 GiB ceiling there is room for
far more context than time allows.

**Time is the constraint, and it is severe:**

| Context | Prefill at 47.9 tok/s |
|---|---|
| 8k | 2.8 min |
| 16k | 4.5 min (**measured** at 12,776 tokens: 4.4 min) |
| 32k | ~11 min |
| 64k | ~22 min |
| 128k | ~45 min |

So the usable window is **whatever the caller will wait for**, not what the model or the
machine supports. `tier1.request_timeout_s = 300` puts the practical ceiling at **~14k
tokens**, and sec 11's Gate B floor of 200 tok/s is **4.2×** away — this host's own relaxed
floor of 150 is 3.1× away. Both fail.

## 4. What it does with a real task

Given ~1,600 tokens of this repository's source and asked whether adding
`prefill_step_size=512` to a `generate_step` call changes a greedy generation's *output* or
only its speed, with `enable_thinking=false`, it answered in 55 tokens and stopped cleanly:

> "The change only alters the speed of greedy generation, not its output. The
> `prefill_step_size` parameter controls how many tokens are generated in a single forward
> pass, which affects computational efficiency but does not change the deterministic
> token-by-token selection logic of greedy decoding."

Fluent, well-formed, correctly scoped — **and wrong on this platform**, which we know
because it was measured the same day. `HANDOFF.md` §3.12: re-chunking one 8,190-token
prompt moves logits by up to **2.031** and exceeds the greedy top1−top2 margin at **7 of 65
steps**. The answer is the conventional one and would be right on most stacks; it is also
exactly the failure mode a verifier exists to avoid, since it reasons from convention rather
than from the code it was handed. (Its description of the parameter is also imprecise — it
governs prompt processing, not generation.)

**Thinking is on by default and the template is not where you would look for it.** This
model ships no template in `tokenizer_config.json`; it ships `chat_template.jinja`, which
sets `enable_thinking = true` when undefined and also honours `reasoning_effort`. With
thinking on, 24-token samples were pure reasoning preamble — no answer at all. That is the
request-side half `tier1.reasoning_control = "deepseek_v4"` sends, and it confirms the
dialect choice for this model family. The response-side guard (`refuse_reasoned_answer`,
§3.11) is what catches the case where the flag is not honoured.

## 5. What this settles, and what it does not

| Question | Answer |
|---|---|
| Does it load inside the working set? | **Yes** — 6.49 GB resident, 6.49 GB load peak, 14.15 GB peak under a 12.8k prompt |
| Does it generate? | **Yes, single-threaded.** Never through `optiq serve` |
| Is it a better verifier than the 122B? | **No.** 48 vs 165 tok/s prefill, 2.6 vs 4.05 decode |
| Does it pass Gate B? | **No** — 48 tok/s against a 150 host floor and a 200 spec floor |
| Do 2-bit routed experts produce usable verdicts? | **Partly answered, and cautionary** — §4: fluent, well-formed, and wrong on the one question we had ground truth for |
| What does `container_hash` cost over 92.49 GB? | Still derived, ~60 s. Unmeasured |
| Does `OPTIQ_STREAM_SCALES_BUDGET_GB=12` pay? | Still arithmetic only (~33% of decode). Now testable, since the model runs |
| Why `Stream(gpu, 1)` specifically? | **Open** — §2 |

**The rung-1 decision does not change.** Rung 3 still ships: it is tier 0 with the adapter
stripped, costs no memory, and is 6–9× faster than any streamed verifier here. What changed
is that "we cannot run this model" is no longer true, so the argument against it is a
measured cost rather than an unexplained crash.

## 6. Reproduction

```bash
# PROCESSES.md §3.1 first. Nothing else may hold the GPU; this holds 6.49 GB resident
# and peaks at 14.15 GB under a 12.8k prompt.
S=~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-0731-OptiQ-2bit/snapshots/0edd7d3e70d562a0fc1d1574943ca4fe2b2c1e36

# §3, laddered. The 16k frontier is ~4.5 min of prefill: background it.
HF_HUB_OFFLINE=1 .venv-optiq/bin/python tools/ds4_probe.py \
    --frontiers 0,2000,8000 --decode-tokens 24 --out var/ds4-8k.json

# §4, a real verdict with thinking off
HF_HUB_OFFLINE=1 .venv-optiq/bin/python tools/ds4_probe.py \
    --frontiers 2000 --decode-tokens 160 --no-thinking --out var/ds4-task-nothink.json

# §2, the crash. PYTHONFAULTHANDLER is what turns the abort into a traceback.
HF_HUB_OFFLINE=1 PYTHONFAULTHANDLER=1 .venv-optiq/bin/optiq serve --stream-experts \
    --model "$S" --host 127.0.0.1 --port 8081 --max-context 32768 --prefill-step-size 8192 &
curl -s localhost:8081/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"ds4","messages":[{"role":"user","content":"Say ok"}],"max_tokens":4}'
# → the engine aborts. It does not need killing; it is gone.

# §2's stream table, no weights, seconds
.venv-optiq/bin/python tools/ds4_serve.py --help    # the experiment and why it is not a fix
```
