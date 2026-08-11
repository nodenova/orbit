# Baseline platform — MacBook Pro M4 Max, 36 GB, 1 TB

| | |
|---|---|
| **Purpose** | The reference machine every budget in this repository derives from, and what it measurably does. |
| **Answers** | "Will this model run here?" · "Why is tier-1 rung 1 off?" · "Why does `orbit.toml` relax a gate?" |
| **Does not answer** | What should be running right now (`operations.md`) · how the system is put together (`architecture.md`). |
| **Hardware** | `Mac16,6`, Apple M4 Max (14 CPU / 32 GPU cores), 36 GiB unified, 1 TB SSD (926 GiB usable), macOS 26.5.2 (25F84), Metal 4 |
| **Stack** | MLX 0.32.0, mlx-lm 0.31.3, mlx-optiq 0.4.18, Python 3.13.5 |
| **Provenance** | Every figure was produced on this machine unless marked *published*. Reproduction commands: §8. |

**This is the baseline, not a compromise against a larger one.** Every budget, gate
threshold and model choice in the repository is derived from the three ceilings in §1
and the throughput in §2 — measured here, not scaled from a bigger box.

What that settles, so it is not re-litigated:

| Decision | Consequence |
|---|---|
| Tier 0 and a streamed tier 1 do not co-reside — **and as of 2026-08-10 that is a missing rung, not a memory ceiling.** The `~12 GiB` this row used to charge the verifier was `expert_cache_bytes`, a host-sizing input that reaches no engine (T34); **measured residency is 3.46 GiB for the 122B and 1.27 GiB for `Qwen3-Coder-Next`**, so 23.0 + 1.27 fits inside 28.08 with room to spare. | The verifier is still served by **rung 3**. What forbids co-residency is that no rung implements it — rung 2 evicts rather than co-resides (T5, T23) — and that a serving gateway holding both is a different question from Gate B, which builds no tier 0. Do not quote the arithmetic in this row as the reason; quote the missing rung. |
| Expert streaming from SSD reaches **165 tok/s of prefill** by `curl` and **153.7 ± 1.1 by Gate B itself** over six runs, 1.3× under the spec floor (§4, §4.1a) — **and 258.0 on a 3 B-active model, which clears it (§4.8).** | Rung 1 is slow, not dead, and how slow depends on the model rather than the mechanism. It is off because rung 3 is faster (6–9× against the 122B, **~3.5× against the fastest measured**) and costs no memory, not because it cannot serve. |
| Active parameters, not total, bind decode (§5). | Every viable model here is a low-active-parameter MoE. A 27 B dense model fits in memory and is still unusable. |

The v1 specification budgets a larger machine, so a few of its figures (`sec 2`'s
expert cache, the 32k KV frontier) are re-derived here rather than copied. Where they
differ, this file is what `orbit.toml` implements and the spec figure is recorded
beside it.

---

## 1. Ceilings

From `mx.device_info()`. Every budget in the repo should derive from these, and the
one that binds is **not** 36 GB.

| Limit | GiB | Bounds |
|---|---|---|
| `max_recommended_working_set_size` | **28.08** | weights + KV + activations, all of it |
| `max_buffer_length` | 21.06 | any *single* MLX array |
| `memory_size` | 36.00 | the machine |

Tier 0 (`Qwen3.6-35B-A3B-OptiQ-4bit`) is **23.0 GiB** of the working set — 82% —
leaving ~5.1 GiB for KV and activations.

**The 21.06 GiB single-buffer limit has bitten — 2026-08-10, and not where this
section expected it.** It forecloses any single-tensor-per-model loader, and MLX loading
safetensors shard by shard is why it never bit on *load*. It binds **prefill**:
`Qwen3-Coder-Next-4bit` at 52,008 input tokens aborted the engine with
`[metal::malloc] Attempting to allocate 23622320128 bytes which is greater than the
maximum allowed buffer size of 22613000192 bytes`, while resident sat at 1.36 GB and
headroom never moved (`platform.md` §4.8).

Two consequences worth carrying to any model:

- **`total − active` is the wrong instrument for this failure.** §1.1 answers "is there
  room", and there was; the allocation that failed was one array, not the working set.
- **`--max-context` does not guard it.** That run was 59% over a configured 32,768 and
  the engine died inside the forward pass rather than refusing the request.

### 1.2 The array that fails, and how to shrink it

**Measured 2026-08-11**, `tools/probe/qcn_context.py`, mlx 0.32.0 / mlx-lm 0.31.3 /
mlx-optiq 0.4.18. The frontier is not a context length — it is a *product*:

```
worst single array = num_attention_heads × prefill_step_size × context × 2 B
                   = 16 × Lq × Lk × 2      must stay ≤ 22,613,000,192
```

`head_dim` is **256**, outside the head dims MLX 0.32.0 fuses. Full-attention layers
therefore take the unfused path and materialise a bf16 score matrix per prefill chunk;
`head_dim` 128 measures **zero** transient on the same call, and 256 measures 2.06–2.19 B
per score element regardless of whether the mask is absent, `"causal"`, or an array. The
abort reproduces byte-for-byte: 16 × 8192 × 90,112 × 2 = **23,622,320,128**, the same
count as 2026-08-10.

So the reachable context is inversely proportional to the prefill step, and **the native
262,144 is reachable** — it was never a hardware limit:

| `--prefill-step-size` | max prompt | reaches native 262,144 |
|---|---|---|
| 8192 | 89,791 | no |
| 4096 | 176,047 | no |
| **2048** | **346,106** | **yes** |
| 1024 | 690,176 | yes |

2,695 is the largest step that still reaches 262,144. Seven probes, every one matching
the formula's verdict:

| prompt | step | worst array | prefill tok/s | result |
|---|---|---|---|---|
| 32,768 | 2048 | 2.00 GiB | 241.9 | ok |
| 65,536 | 2048 | 4.00 GiB | 221.8 | ok |
| 131,072 | 2048 | 8.00 GiB | 175.7 | ok |
| **262,144** | 2048 | **15.99 GiB** | **124.4** | **ok** |
| 40,960 | 8192 | 10.00 GiB | 317.5 | ok |
| 49,152 | 8192 | 12.00 GiB | 290.2 | ok |
| 131,072 | 8192 | 32.00 GiB | — | **abort at Lk 90,112** |

Prefill rate falls with context (242 → 124 tok/s from 32k to 262k) and rises with the
step, so the step trades reach against speed in both directions. The 262,144 run held
1.36 GB resident and cost 2,107 s and ~6,200 pageouts.

**One thing this does not explain.** The 2026-08-10 abort at 52,008 tokens implies
Lq × Lk = 738,197,504, which for Lk ≤ 52,008 needs Lq ≥ 14,194 — larger than the 8192
step that run recorded. The flags in effect then cannot have been exactly as written down,
so treat that run's *configuration* as unverified; its byte count is reproduced above.

### 1.1 Judging headroom before a load

| Question | Read | Do not read |
|---|---|---|
| Is there room? | `hw.memsize` − `Pages active` | `Pages free` |
| Is it thrashing? | `Pageouts` (from `memory_pressure`) | `vm.swapusage used` |

macOS reports most of a healthy machine as inactive or speculative, both reclaimable
on demand; `Pages free` alone refuses runs that would have been fine. **Tier 0 needs
~27 GB by the `total − active` measure.** Below it, close applications.

Loading tier 0 legitimately moves swap 0.46 → 3.27 GB with **223 total pageouts**,
because the compressor holds 5.12 GB of pages in 1.00 GB of RAM. A guard on swap
usage aborts runs that were never in trouble; a guard on pageouts does not.

---

## 2. Throughput

### 2.1 Machine

| Test | Result |
|---|---|
| fp16 elementwise read+write | **247 GB/s** (spec bin 410, *published*) |
| fp16 read-only reduction | 242 GB/s |
| fp16 GEMM 4096³ | **10.80 TFLOP/s** (peak; 2048³ 9.74, 8192³ 6.48) |
| SSD random read, 1 MB blocks, qd 24, `F_NOCACHE` | **6.93 GB/s** (burst) |
| SSD random read, same blocks, qd 1 | 1.70 GB/s |
| SSD random read, 2 MiB blocks, qd 24, sustained over 628 GB | **4.9 GB/s** |
| SSD random read, 128 KiB blocks, qd 24 / qd 6 | 4.40 / **2.50 GB/s** |
| SSD random read, 512 KiB blocks, qd 24 / qd 10 | 5.55 / 4.80 GB/s |
| SSD random read, **32 KiB** blocks, qd 24 / qd 10 | **1.27 / 0.90 GB/s** |

**247 GB/s is the number to plan with** — 60% of the spec bin, normal for MLX
elementwise work and what decode actually sees.

> **⚠️ These figures are a floor, and this host does not always meet them.** On
> 2026-08-09 the same benchmark read **23 GB/s** r+w and **5.29 TFLOP/s** at 4096³ after
> 15 h of uptime — ~9% and ~50% — with nothing resident; a reboot restored it to **323 /
> 352 GB/s and 11.9 TFLOP/s**, which is *above* this table. Neither the degradation nor
> the 1.3× headroom over the recorded figures is explained.
> **Run `tools/bench/mlxbench.py` and require ≥ ~250 GB/s before trusting any new measurement
> on this machine** — `operations.md` §3.1 is the gate, `operations.md` §3.1 the finding.

Three SSD facts govern any streaming path:

- **Queue depth is worth 4×** at expert-sized blocks (0.5–2 MB): 1.70 GB/s at qd 1
  against 6.93 at qd 24. This is why mlx-optiq ships `pread` at queue depth 24.
- **Small blocks cost more than their share.** 128 KiB reads run at 2.50 GB/s at qd 6
  against 6.04 for 2 MiB — so a workload where 11% of the bytes are 128 KiB spends 23%
  of its time on them (§4.4). Block size matters more than total bytes. **The 32 KiB row
  is the same fact one stride further down and is the worst measured here**: 1.27 GB/s at
  qd 24, 23% of the 512 KiB rate beside it, which is why `Qwen3-Coder-Next`'s 4.84 GB of
  meta is worth making resident (`platform.md` §4.8). A quant with smaller
  experts is not automatically cheaper to stream: that model moves 2.15× fewer bytes per
  token than DeepSeek-V4 and reads every one of them at a quarter of the block size.
- **Burst and sustained differ by ~30%, and the burst figure is what §2.1 used to
  quote alone.** 6.93 GB/s is the first few GB; 628 GB read continuously at the same
  block size decays to 4.88, monotonically and mildly (−4% over the run).

Two measurement traps, both of which have produced a wrong number in this repo:

- Any streaming benchmark that does not set `F_NOCACHE` is measuring the page cache —
  an earlier run without it reported 5.6 GB/s purely from RAM.
- **`F_NOCACHE` is not sufficient.** It stops *that fd* from populating the cache; it
  does not stop a read being served from pages something else already cached. Bracket
  every window with `iostat -Id disk0` and print device_bytes / requested_bytes. An
  earlier unbracketed run reported 50.6 GB/s at 8 MB blocks — which no consumer NVMe
  does, and which is how this section came to claim that blocks above 2 MB "get worse".
  **That claim was an artifact and has been withdrawn**; `ds4_streambench.py` reaches
  dev/req 0.92 only after ~600 GB, so treat anything measured below that as optimistic.

### 2.2 Tier 0 by context frontier

Recorded in `var/gate-a.json`.

| ctx | TTFT cold | TTFT warm | decode tok/s | prefill tok/s |
|---|---|---|---|---|
| 2 k | 10.65 s | 1.58 s | 85.9 | 1160 |
| 4 k | 2.63 s | 3.10 s | 83.3 | 1185 |
| 8 k | 5.59 s | 6.28 s | 79.7 | 1168 |
| 16 k | 13.25 s | 13.42 s | 75.1 | 1094 |
| 32 k | 31.56 s | **30.47 s** | **69.6** | 963 |

**Decode passes its budget by 2.3×. TTFT fails at long context, and the failure is
arithmetic.** TTFT ≈ context ÷ prefill rate, so 32k at ~960 tok/s *is* ~32 s. Hitting
2 s at 32k would need 16,000 tok/s of prefill — 2.4× more FLOP/s than this GPU has at
100% efficiency. **Unreachable on any M4 Max at any memory size.**

The only mechanisms that can close it are compaction and the prompt cache, both upstream
of what this suite measures. `mlx_tier0.supports_state()` returns True, but on this
container a restore covers **0 tokens** (`platform.md` §2.2), so today every turn
still re-prefills. Read a red TTFT as "the mitigations are not being measured", not as a
model problem.

The measurement chain is self-consistent: prefill of ~1,100 tok/s at 3 B active
parameters is 2 × 3e9 × 1100 = 6.6 TFLOP/s, 61% of GEMM peak. That is why the
projections in §5 can be trusted.

### 2.3 Constrained decoding

| Turn type | decode tok/s | first-attempt tool calls |
|---|---|---|
| free-form | **65.6** *(90.1 on a fresh boot — see below)* | n/a |
| tool-bearing, constrained | **27.1–27.6** *(29.5 with F1+F2)* | **100/100** |
| tool-bearing, unconstrained | 65.6 | **0/100** — all prose |

> **Re-measured 2026-08-09 with F1+F2 in the tree, on a host reading 321 GB/s.** Free-form
> is **90.1 tok/s** and constrained is **29.5**, so the ratio *widened* to 2.38× even
> though the constrained arm got faster. **The two arms do not scale together**: a faster
> GPU moves free-form and leaves constrained where it is, because constrained decode is
> host-bound. Treat the 65.6/27.1 pair as one host-day rather than as a property of the
> model, and never quote a ratio across two days.
> Full six-variant attribution, including what the sync and the plumbing actually cost:
> `constrained-decoding.md` §8.

**The sec 10.2 gate has been re-run against F1+F2 and passes: 100/100 well-formed, rate
1.00, 115.9 s** (`orbit gate toolcall --runs 100`, 2026-08-09). Load peak 23.61 GB, steady
headroom 27.1 GB, pageouts +134 — no thrashing.

~21 ms/token of overhead, of which only ~6 ms is building the mask (LMFE 3.30 ms, id
list 1.77 ms, `mx.array` 0.34 ms, scatter 0.59 ms at a 248k vocabulary). The rest is
the synchronisation `tokens.tolist()` forces on a decode loop MLX otherwise
pipelines, and it is inherent to any Python-side constrained decode — the mask depends
on the token just sampled. One-time cost: 1.1 s and ~0.6 GB per tokenizer.

**The 2.4× is real; that split and that attribution are withdrawn (2026-08-09 —
`constrained-decoding.md`, `constrained-decoding.md` §5).** The sync alone costs nothing — a synthetic decode loop that syncs and does no other
work runs at 13.24 ms/token against 13.12 unconstrained. What costs is host work
*serialised against an idle GPU*, and the profile above is the median of a bimodal
distribution: the 20 of 43 tokens whose parser state allows 246,908 ids cost ~10–13 ms
each, not ~6. Most of it is removable in Python — `[int(t) for t in allowed …]` converts
`int` to `int` for 7.39 ms/token, and dispatching the forward pass before building the
mask hides the remainder. **"Inherent to any Python-side constrained decode" is the part
that was wrong.**

**"Most of it is removable in Python" is now wrong too — measured 2026-08-09.** Removing
the `int()` conversion is worth **2.99 ms/token** of a ~26 ms/token penalty, and the mask
cache another 0.08–0.54; together they buy **11–15%**, not "most". The paragraph above is
right that the sync is free (a null processor doing only `tokens.tolist()` costs 0.62
ms/token against real weights) and wrong about what that leaves: **22.15 ms/token sits
inside the filter-and-mask path, where the no-weights bench prices the same work at 4.0.**
That 5× is now explained, and the explanation is not about having a model resident —
`constrained-decoding.md` §8.4. **~97% of the cost is one parser state: the gap between a
backslash and its escape character, ~425 ms every time it is occupied.** Two occurrences
in the measured call were 849 of its ~1,000 ms of LMFE time.

Whether a newline occupies that state is decided by the *tokenizer*: `"…(req)\n"` splits
into `')\'` + `'n'` and pays; `"line0\nline1"` emits `'\n'` whole and pays nothing. So the
cost of a tool call scales with **split escapes, not tokens** — measured dead linear at
~428 ms each — and a twenty-line code edit whose lines end in `)` costs ~8.5 s of host
time on its own. **This is the workload Orbit is for, and the 2.38× decode ratio does not
express it.**

**The paragraph above read 2.5 ms/token because its fixture had no escape at all.**
`tools/bench/constrained_decode_bench.py escape` reproduces the hardware figure to within 2%
with no weights, and is the cheaper way to ask anything further about where the time goes.

**Rung 0 now reproduces this without weights — `constrained-decoding.md` §3.2, §4.** Run
against the escaped fixture, the no-weights bench reads **21.02 ms/token against the
22.15 measured here** (LMFE alone within 0.5%), where the same harness on the unescaped
fixture reads 2.97. So the 5× that stood in this section was the fixture and never the
machine, and anything further about where constrained-decode time goes costs a minute
rather than 20.6 GiB. The measured shape is **~2.9 ms × tokens + ~419 ms × split escapes**,
which also corrects F4: overlapping host work with the forward pass is worth **~1.11×** on
a call with escapes, not the ~1.5× read off the average (§8.3).

**It stays paid: the state is cacheable and not by us — `constrained-decoding.md` §8.5.**
Occurrences inside one property are the same set, so a cache would collapse 75% of the
cost (1,228 of 1,638 ms over four escapes), but the set is fixed by the enclosing object
stack rather than by the backslash — a key on the escape alone permits a comma where the
object must close. The sound key is stack-wide, LMFE does not implement one in its current
release, and F4 cannot substitute: hiding 11 ms/token behind the forward pass still leaves
a 410 ms stall. Treat the escape cost as a property of the workload when reading any
latency number here.

**That 2.4× is the price of the sec 10.2 gate passing at 1.00 instead of failing at
0.81.** It is worth paying, and it is why `gate_a_decode_tok_per_s` is relaxed rather
than the constraint being dropped (§7).

**Warm up before timing anything.** The first generation after a load pays Metal
kernel compilation — ~9 s, enough to read 4.1 tok/s where the truth is 27.

### 2.4 Determinism, measured in logits rather than in text

*2026-08-10, `tools/probe/determinism_probe.py`, mlx 0.32.0, one load of 22.14 GB, host at
325/347 GB/s. `var/determinism-device.json`, `var/determinism-chunk-8k.json`.*

Sec 9.3's G1 and G2 compare *text* and answer yes/no. Measured one layer down — per step,
the full logit vector and the greedy top1−top2 margin — the answer has a size, and the size
is what decides whether a receipt's determinism claim survives. Arms are teacher-forced to
the reference's tokens, so every step compares logits computed from identical inputs.

| Reference: Metal, `prefill_step_size` 2048 | max Δlogit | median Δ | Δ ≥ that step's margin | argmax flips |
|---|---|---|---|---|
| the same configuration again | **0.0, bitwise** | 0.0 | 0 of 65 | 0 |
| `prefill_step_size` 512, 8,190-token prompt | **2.031** | 0.688 | **7 of 65** | 0 of 65 |
| CPU instead of Metal, 37-token prompt | **4.375** | 3.406 | 1 of 5 | **1 of 5, the first token** |

Three things follow, and only the first is comfortable:

1. **Same configuration reproduces bitwise**, not merely as identical text. The 90-item
   regression suite reproducing exactly (§7) is confirmed at logit granularity, so nothing
   below is flakiness — each figure is a property of the path.
2. **Re-chunking one prompt changes the logits by up to 2.031**, and at 7 of 65 greedy steps
   that exceeds the gap between the top two tokens. No argmax flipped in that window, so this
   is exposure rather than an observed failure — 21 of 65 steps carry less margin than the
   run's own worst Δ. This is the mechanism `platform.md` §2.4 hypothesised for a restored KV
   state changing an answer at 8k, and the knob that produces it (`--prefill-step-size`) is
   recorded in no receipt (T13).
3. **CPU-vs-Metal flips the first generated token.** 30 of this model's 40 layers are linear
   attention, and `mlx_lm/models/gated_delta.py:281` dispatches on
   `mx.default_device() != mx.gpu` to a different algorithm — so G1's byte-identity is not
   available on this platform and its "pin the reduction order" remedy does not apply. At the
   single-op level, no weights needed, a 4-bit `quantized_matmul` differs by 2.3e-05.

**The CPU arm costs ~3,200× Metal**: 450.5 s for a 37-token prefill plus four tokens against
0.14 s. It needs no extra memory — unified memory means both devices read the same buffers —
but `mlx_lm.generate` pins its generation stream to the default device *at import*, so a
caller's `mx.stream(mx.cpu)` is silently ignored and the arm returns Metal's own logits in
0.1 s. That first run looked like a clean G1 pass and measured nothing; the device has to be
swapped globally with that module's stream rebound alongside it.

---

## 3. KV budget

From the model's own `config.json` and `model.safetensors.index.json`:

- 40 layers, of which **10 are full attention** (indices 3, 7, 11, …, 39) and 30 are
  linear attention.
- Full-attention layers: `num_key_value_heads = 2`, `head_dim = 256`.
- KV per token = 10 × 2(K,V) × 2 × 256 × 2 B = **20 KiB/token**.
- The 30 linear-attention layers carry a *fixed* ~30 MiB recurrent state that does not
  grow with context.

| `max_kv_tokens` | KV fp16 | with the shipped `kv_config.json` |
|---|---|---|
| 32,768 | 640 MiB | 228 MiB |
| **65,536** *(current)* | **1.25 GiB** | 0.45 GiB |
| 131,072 | 2.50 GiB | 0.89 GiB |
| 262,144 *(native max)* | 5.00 GiB | 1.79 GiB |

Headroom is 5.08 GiB, so even the native maximum fits with the shipped quantisation
(7 layers at 4-bit, 3 at 8-bit, ~0.36× of fp16).

**The ceiling that actually binds is prefill time, not KV memory**: 131k tokens is
~136 s of wall clock at 963 tok/s. Raise `max_kv_tokens` further against a
`mx.get_peak_memory()` reading on a loaded model, not against this table.

---

## 4. Tier 1

`Qwen3.5-122B-A10B-OptiQ-2bit`, 45.94 GB on disk, experts streamed from SSD by
`optiq serve --stream-experts`.

**Gate B measures prefill, not decode.** The two differ by 41× on this hardware, and
that difference is the premise tier 1 is built on (`architecture.md` §1, fact 1). A decode
rate compared against `gate_b_prefill_tok_per_s` divides the premise away.

### 4.1 Measured

*2026-08-09, mlx-optiq 0.4.18, `--stream-experts --max-context 32768`. Prompts from
`mlx_tier1.prefill_filler`, unique prefix per request, `cached_tokens` 0 on every row.
`max_tokens=1` for prefill rows. Pageouts moved 1,291 → 2,311 across the whole session
— no thrashing, so the timings stand.*

| `--prefill-step-size` | prompt_tok | chunks | wall s | tok/s |
|---|---|---|---|---|
| 2048 *(default)* | 353 | 1 | 8.92 | 39.6 |
| 2048 | 1,380 | 1 | 14.66 | 94.1 |
| 2048 | 2,786 | 2 | 28.70 | 97.1 |
| 2048 | 5,576 | 3 | 49.26 | 113.2 |
| 8192 | 1,380 | 1 | 14.65 | 94.2 |
| 8192 | 5,576 | 1 | 33.86 | **164.7** |
| 8192 | 11,296 | 2 | 72.11 | 156.6 |
| 8192 | 16,992 | 3 | 121.29 | 140.1 |

| | |
|---|---|
| **Prefill, best** (`--prefill-step-size 8192`) | **164.7 tok/s** |
| **Decode** | **4.05 tok/s** |
| Resident while streaming | 3.46 GB |
| Gate B floor | 200 tok/s spec · 150.0 this host (was 20.0 until §4.1a measured it) |

**Rung 1 is 1.3× under the spec floor.** The figures above come from `curl` against the
engine; the gate's own are below.

#### 4.1a Gate B itself — 2026-08-10

*Same engine, same flags, **six** consecutive runs of `orbit bench tier1`, one session at
~11 h uptime. Prompts from the gate's `prefill_filler`, not the hand-built `curl` ones.*

| input tokens | mean tok/s | min–max | spread | sd |
|---|---|---|---|---|
| 5,433 | 157.0 | 153.9–159.1 | 3.3% | 1.99 |
| 11,035 | 154.0 | 153.0–156.7 | 2.4% | 1.22 |
| 22,208 | 155.4 | 152.1–159.7 | 4.9% | 2.61 |

Worst frontier of each run: **152.1 · 153.2 · 153.4 · 153.7 · 153.9 · 155.7** — mean
**153.7**, sd **1.07**, full spread **2.34%**. Every run `pass: true`, `meets_spec: false`.

**So the figure is 153–154 tok/s, and it is a distribution rather than a sample.** It reads
under §4.1's best `curl` number (164.7 at 5,576 tokens) and above its fall-off (140.1 at
16,992): the gate's filler is identifier-diverse by construction, so its per-chunk expert
union is not the same as the `curl` prompts'. Compare the two tables by shape, not row
against row.

This spread is what sets `gate_b_prefill_tok_per_s = 150.0` — 3σ below the mean, under every
run observed (`platform.md` §7). One session on one healthy host is the narrowest part of
the estimate; re-measure before reading a single red as a regression.

`meets_spec` is **false** in every run: ~153 against sec 11's 200 is 1.3× under, and that is
what a Gate B result is quoted by.

> **Read that red against §4.2 before concluding anything from it — `platform.md` §4.8.**
> The 200 is a proxy for "is the engine amortising the expert sweep across the chunk?", and
> §4.2 answers that directly and affirmatively: the 8.0 s sweep is constant per chunk and
> independent of how full the chunk is. What sinks the number is the `tokens/218` compute
> term, which makes **218 tok/s this host's asymptote** — so the spec floor sits at 92% of
> it, and at `--prefill-step-size 8192` the ceiling for a single full chunk is **179.7
> tok/s**, below 200 at any engine quality. The gate is failing on this machine's compute,
> not on the architecture it was written to detect. `gate_b_prefill_tok_per_s` moved 20.0 → **150.0** on
the strength of these runs (`platform.md` §7).

Two defects had to be fixed before this could run at all — an eager 23.0 GiB tier-0 build in
`cmd_bench`, and a reasoning guard reading a key this engine does not send. §4.1a and §4.7.

Decode is unusable for generation, which is what the design requires — it is why tier 1
has no `generate`.

### 4.2 The engine amortises the expert sweep, and the chunk is a knob

Both step sizes fit one model:

> **prefill seconds ≈ 8.0 × chunks + tokens / 218**

The 8.0 s per chunk is constant across both step sizes and independent of how full the
chunk is — 1,380 tokens costs the same 14.65 s at step 2048 and step 8192. That is a
full expert-set sweep, once per chunk:

- 44.6 GB of routed experts ÷ 8.0 s = **5.6 GB/s effective**, against 6.93 GB/s at
  1 MB blocks (§2.1). Lower because experts are 3.63 MB reads and §2.1 already records
  that blocks above 2 MB get worse. Three measurements, mutually consistent.
- `tokens / 218` is compute: 2 × 10e9 × 218 = 4.36 TFLOP/s, **40% of the 10.80
  TFLOP/s GEMM peak** (§2.1). Ordinary for prefill.

As the chunk grows the fixed sweep spreads thinner and the rate converges on ~218
tok/s. The 16,992-token row already falls off (140 measured against 165 predicted)
because attention is O(n²). **No chunk size exceeds that asymptote.**

**The chunk size is `mlx_lm.server`'s `--prefill-step-size`, default 2048**, reachable
because `optiq serve` forwards arguments it does not recognise. Raising it to 8192 is
worth **1.75×**. There is no `optiq` flag for it, so `optiq serve --help` does not list
it — read the callee's parser.

**`--stream-experts-cache` is accepted and discarded** — it is a per-projection
*count*, not a byte budget, and `_ShardWeightReader.__init__` never stores it, so the
only cache is the OS page cache. This is why `Tier1Attestation` records
`expert_cache_configured_bytes` and why `expert_cache_provenance` says the value did not
reach the engine rather than printing it beside a throughput. The startup log confirms
the split independently: `expert scales/biases 7.2 GB vs budget 3.9 GB -> STREAM`, then
`swapped 144 expert projections; resident 3.46 GB`.

### 4.3 What this costs the verifier contract

At the measured rates, and against rung 3 (§2.2: 963–1,185 tok/s prefill, 65–87 decode):

| Verifier turn | rung 3 | rung 1, at measured rates |
|---|---|---|
| rerank, 5k in / 8 out | ~7 s | ~34 s |
| review, 30k in / 128 out | ~33 s | **~214 s** |

`request_timeout_s = 300.0`, so a 30k review fits with ~86 s of margin. **Rung 1 is
slow, not dead** — inside the timeout at both ends of the documented input range.

**Rung 3 ships because it is 6–9× faster and costs no memory**, not because rung 1
cannot serve. What rung 1 would buy is verdict *independence*, which `platform.md` §4.5
names and nobody has measured.

### 4.4 DeepSeek-V4-Flash, measured without downloading it

*2026-08-09. `mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed`, 92.83 GB, ungated
(HTTP 200 — the 401 is a property of the OptiQ-branded repos, not the model).*

> **Superseded as the target, later the same day: the OptiQ repos are no longer 401.**
> `mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit` is ungated, and now on disk — §4.6.
> Everything below still stands as a measurement *of this quant*, and it is the only
> DeepSeek-V4 number anyone has; it is no longer the quant Orbit would serve.

**Nothing here required the weights.** The shape came off the 18 safetensors headers by
HTTP Range request, and the throughput came from replaying the loader's exact read
pattern against local blobs (`tools/bench/ds4_streambench.py`). Both are reproducible in
about three minutes.

**The engine will stream it, and that was the open question.** `--stream-experts`
detection is *name-keyed, not brand-keyed*: `moe_stream._EXPERT_SEGMENTS` matches
`.switch_mlp.`, and this quant names its routed experts
`model.layers.N.ffn.switch_mlp.{gate,up,down}_proj`. `DeepseekV4MoE.switch_mlp` resolves
in optiq's vendored decoder, so `_walk` finds every one; `sanitize()` drops every `mtp.*`
key, so the MTP block never loads. "Non-OptiQ quant" was never the risk it read as.

| Component | GB | Fate |
|---|---|---|
| routed expert `.weight` | 69.26 | streamed |
| routed expert `.scales` + `.biases` | 8.66 | **streamed** — see below |
| MTP / DSpark block | 8.43 | dropped by `sanitize()` |
| everything else | 6.49 | resident |
| **total** | **92.83** | |

All 43 layers are MoE: the config carries no `first_k_dense_replace` at all, closing the
±3-layer uncertainty every previous per-token figure carried. `num_hash_layers = 3`.
Routed experts take no per-path quantisation override, so they are the top-level
2-bit / group 128; the 545 overrides are attention at 6-bit and shared experts at 8-bit.

**This host streams the meta.** `_scales_budget_bytes()` is `max(2 GB, 10 % RAM)` —
the engine prints it as 3.9 GB — against 9.26 GB of scales and biases, so `stream_meta`
fires. It is why the read pattern is nine `os.pread`s per expert per layer, six of them
128 KiB:

> **Correction, 2026-08-09.** This paragraph used to end "and the proxy never did — the
> 122B kept its ~1 GB resident and paid none of this", and §4.4's cost model treated the
> streamed meta as the thing distinguishing this model from the measured proxy. Serving
> the 122B prints `expert scales/biases 7.2 GB vs budget 3.9 GB -> STREAM`: it streams its
> meta too, and always has. The ~1 GB figure was inferred from its 3.46 GB resident total,
> never measured. The per-block arithmetic below is unaffected — what is withdrawn is the
> claim that the proxy's numbers omit a cost this model pays (§4.7).

| Read | Stride | Count per expert-layer |
|---|---|---|
| `{gate,up,down}_proj.weight` | **2 MiB** exactly | 3 |
| `{gate,up,down}_proj.{scales,biases}` | **128 KiB** exactly | 6 |

7.078 MB per routed expert per layer; **1.826 GB per decoded token** at top-6 across 43.

**Measured, replaying that pattern** — sustained, `F_NOCACHE`, `iostat`-bracketed:

| | Measured | Note |
|---|---|---|
| decode, 387 sequential barriers at qd 6 | **2.67–2.79 tok/s** | I/O only; compute not charged |
| prefill sweep, k = 256 at qd 24 | **11.1 s/chunk** | the whole 77.91 GB routed set, once |
| soak | 628 GB continuous, −4% decay | dev/req 0.86 → 0.92 |

The sweep is *better* than the 13.9 s/chunk previously derived by scaling the 122B. The
decode figure is *worse* than the 3.1–3.8 tok/s previously derived, because that
derivation charged neither the streamed meta nor the queue depth: `read()` is called
three times per projection and blocks on each, so decode runs at qd 6, not 24.

**Gate B, with the I/O term measured and the compute term scaled from the proxy**
(218 tok/s at 10 B active → **168 tok/s** at 13 B, and that charges nothing for CSA's
top-512 index, HCA, or 20 Sinkhorn iterations per layer, so it is a ceiling):

| `--prefill-step-size` | derived tok/s | spec ≥ 200 |
|---|---|---|
| 2,048 *(default)* | 88 | fail |
| 8,192 | 137 | fail |
| ∞ | **168** | **fail** |

**Gate B fails on compute, and no knob reaches it.** Amortisation solved the bandwidth
half — the sweep spreads thinner as the chunk grows — but 168 tok/s is what 13 B active
parameters cost on this GPU.

**`--prefill-step-size 8192` is required for this model, not an optimisation.** At the
2048 default a 30k review costs 15 sweeps: 345 s of prefill plus 47 s of decode is
**~393 s against `request_timeout_s = 300`** — it times out. At 8192 it is 4 sweeps,
223 s + 47 s ≈ **270 s**, inside the timeout with ~30 s of margin.

| Verifier turn | rung 3 | rung 1, 122B | rung 1, DeepSeek-V4 |
|---|---|---|---|
| rerank, 5k in / 8 out | ~7 s | ~34 s | ~44 s |
| review, 30k in / 128 out | ~33 s | ~214 s | **~270 s** |

**An unrecorded lever, worth ~33% of decode: `OPTIQ_STREAM_SCALES_BUDGET_GB`.** The meta
is 11% of the bytes and 23% of decode time, because 128 KiB at qd 6 runs at 2.50 GB/s
against 6.04 for 2 MiB. Setting the budget above 9.3 GB keeps it resident: **15.15 GB
resident against a 30.15 GB working set** — comfortable, given nothing else co-resides —
and decode goes 2.7 → ~3.6 tok/s, the sweep 11.1 → ~9.3 s/chunk, a 30k review 270 → 251 s.
Untested against the real model; the arithmetic is from the measured per-block rates.

**So the case for this model is not throughput, and never was.** It is 8× slower than
rung 3 at the only thing tier 1 does. What it buys is verdict *independence* — decided
against §4.5, which was free and strictly cheaper than fetching 92.83 GB.

### 4.5 Rung 3 is not a no-op, and the number that shows it is not the obvious one

*2026-08-09. `tools/quality/rung3_agreement.py`, 12 fixed tasks × 5 candidates at temperature
0.6, `Qwen3.6-35B-A3B-OptiQ-4bit`, no adapter mounted. `var/rung3-agreement.json`.*

`platform.md` §4.5 asked one question before spending 92.83 GB on an independent verifier:
**how often does rung 3 disagree with tier 0's own top candidate?**
`Cascade._code_change_turn` falls back to candidate 0 when a rerank fails, because that
is what a no-tier-1 install would have returned — so that fallback rate bounds what the
whole tier-1 apparatus is worth.

| | Measured | Under a uniform-random reranker |
|---|---|---|
| disagreement with candidate 0 | **0.75** (9/12) | **0.80** |
| choice histogram over 5 slots | 3 / 3 / 3 / 2 / 1 | flat |
| **same candidate when the order is rotated** | **0.833** (10/12) | **0.20** |
| candidates actually distinct | 5 of 5, every task | — |
| rerank latency, p50 | 3.13 s | — |

**The headline number is uninformative and the control carries the entire signal.** 0.75
disagreement is what picking at random gives you (0.80) — quoting it alone would have
reported a verifier that works when a coin would have scored the same. What separates
them is rotating the candidate list and asking whether the verifier follows the
*content* or the *slot*: 10 of 12 followed the content, against 0.2 by chance,
**p ≈ 4.5 × 10⁻⁶**. Two tasks answered slot 0 in both orders and are the only positional
cases.

This is `architecture.md` §5, trap 6 in a new costume — an arithmetic comparison whose inputs
were all correct and which compared the wrong pair. The lesson generalises: **a
disagreement rate needs its null stated beside it**, and for an N-way choice the null is
1 − 1/N, which is high.

Three things this does *not* say, and the third is the next question:

- Not that rung 3 picks the **better** candidate — only that it makes a reproducible,
  content-determined choice. Measuring better needs a quality signal, which is the merge
  eval (`orbit eval merge`), not another verifier.
- Not that an adapted model would behave this way. `adapters/` does not exist, so
  `SecondOpinionBackend._strip` is a no-op and this is tier 0 judging its own samples
  with identical weights — **the degenerate case, and therefore a floor**. The rung's
  actual mechanism, stripping A1, is still unexercised.
- **It does settle the download.** Rung 3 is not measuring nothing, so the case for
  92.83 GB of independent verifier rests on decorrelation alone, at 8× the latency, and
  is not urgent. Revisit once A1 exists and the strip does something.

---

### 4.6 The OptiQ quant is no longer gated, and it is the one to serve

*2026-08-09, requested by the owner. `mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`,
**92.48 GB**, 42 shards, 2765 tensors, ungated. **Downloaded and verified — 42/42 shards,
92.49 GB referenced by the snapshot; `platform.md` §4.6 has the resolved path. Never
loaded.***

**§4.4's parenthetical is out of date and this supersedes its choice of quant.** The
OptiQ-branded repos answered 401 when §4.4 was written; this one now answers 200 on the
metadata API and **206 on a range request for shard bytes**, which is the check that
distinguishes an open repo from a gated one whose metadata is public. It is the quant to
serve because tier 1 runs `optiq serve --stream-experts`, so the OptiQ build is what that
engine is built for; `2.4bit-mixed` was only ever the fallback the 401 forced.

Shape by the same method as §4.4 — 42 shard headers over HTTP Range, no weights, no GPU
(`tools/probe/ds4_headers.py` with `REPO` repointed):

| Component | GB | Fate |
|---|---|---|
| routed expert `.weight` | 69.26 | streamed |
| routed expert `.scales` + `.biases` | 8.66 | streamed iff over the budget |
| MTP / DSpark block | 8.08 | dropped by `sanitize()` |
| everything else | **6.49** | **resident** |
| **total** | **92.48** | |

Architecture matches §4.4 exactly — 43 layers, all MoE (no `first_k_dense_replace`), 256
routed experts, top-6, 3 hash-routed layers, 1 shared. Streaming will engage: **414
`.switch_mlp.` keys** named `model.layers.N.ffn.switch_mlp.{gate,up,down}_proj`, the
segment `moe_stream._EXPERT_SEGMENTS` matches. Per-token cost is **7.078 MB per routed
expert per layer → 1.826 GB per decoded token at top-6**, and 77.91 GB per prefill chunk
if the chunk reaches every expert.

**One difference from `2.4bit-mixed` that looks like a risk and is not.** This quant
carries 678 per-path quantisation overrides against 545, and unlike the mixed quant
**129 of them are main-tower `switch_mlp` paths** — the case `ds4_headers.py` explicitly
asserts against, because `load_streaming` reads `mode` from the top level only and has no
per-path path. They are harmless here: every one reads `bits=2, group_size=128`, identical
to the top level, so the streamed reader and the config agree by value even though it
reaches them by a different route. Recorded because the assertion will fire on the next
DeepSeek quant and the answer is not "ignore it" — it is "check the values match".

**Nothing above is a measurement of the model** — it is all read off the headers. It has
since been *loaded*, twice, and it has never *generated*: §4.7.

### 4.7 The engine loads DeepSeek-V4 and then dies on the first request

> **Superseded in part, 2026-08-10 — the model generates; `optiq serve` is what cannot run
> it.** Driven single-threaded it measures **47.9–49.4 tok/s of prefill, 2.54–2.87 decode,
> peak 14.15 GB at 12,776 tokens**, and `platform.md` §4.7 owns those figures and the crash
> mechanism. What stays true and lives here is the *load* accounting below, which the
> single-threaded runs reproduced exactly (6.49 GB resident, 6.49 GB load peak).

*2026-08-09, on a host passing §3.1 at 320/348 GB/s and 11.95 TFLOP/s. mlx-optiq 0.4.18,
mlx 0.32.0, mlx-lm 0.31.3. `optiq serve --stream-experts --max-context 32768
--prefill-step-size 8192`, one `curl` completion of ≤11 prompt tokens.*

**Loading works and answers the question §4.4 left open.** Both configurations load:

| `OPTIQ_STREAM_SCALES_BUDGET_GB` | meta | resident | **load peak** |
|---|---|---|---|
| unset (engine prints a 3.9 GB budget) | streamed | 6.49 GB | **6.49 GB** |
| 12 | resident | 15.21 GB | **23.80 GB** |

Both are under the 30.15 GB working set, so `fast_quantized_load` holds — but note the
peak is **3.7× the steady state** when the scales are made resident, and 23.80 GB is much
closer to the ceiling than the 15.21 GB figure suggests. Plan against the peak.
`swapped 129 expert projections` both times, exactly the 43 × 3 the headers predict.

**Generation does not work.** Every run dies during prompt processing, before the first
token:

```
INFO:root:Prompt processing progress: 0/11
libc++abi: terminating due to uncaught exception of type std::runtime_error:
  There is no Stream(gpu, 1) in current thread.
```

The process exits and frees the port. Three configurations, all identical: scales
streamed, scales resident, and `--max-concurrent 1`. **It is not memory** — headroom sat
flat at ~23 GB across the whole run and pageouts did not move.

**The control is what makes this a finding rather than a symptom.** `Qwen3.5-122B-A10B-OptiQ-2bit`
on the same engine, same flags, same stack, minutes apart, returns a completion in 15 s.
So the failure is specific to this model, not to streaming, not to the host, and not to a
stack regression.

That control also **falsifies the reason §4.4 gave for expecting trouble**: the 122B was
said to keep its meta resident and so never exercise the streamed-meta path. It does not
— it prints `expert scales/biases 7.2 GB vs budget 3.9 GB -> STREAM`. Both models stream
their meta, and DeepSeek crashes with the meta *resident* too, so the meta path is not
implicated at all.

**Mechanism — a pointer, not a diagnosis.** `mlx_lm/generate.py:226` builds
`generation_stream = mx.new_thread_local_stream(mx.default_device())` and wraps generation
in `with mx.stream(generation_stream)`, while optiq's `moe_stream` reads and evaluates
expert weights on a 24-worker `ThreadPoolExecutor`. A *thread-local* stream does not exist
on a pool thread, and `Stream(gpu, 1)` is the non-default stream that call creates. What
is **not** established is why the 122B survives the same arrangement. Do not quote this as
the cause until something has distinguished the two decoders.

**Consequence for T7.** The question was never throughput — it was whether ~8× rung 3's
latency is worth verdict independence. That is now moot at a lower level: **there is no
rung-1 deployment of this model to decide about.** Fixing it is upstream work in an engine
Orbit deliberately does not spawn (sec 5.4), and the fallback ds4 (`operations.md` §6.4)
has never been brought up here.

### 4.8 A streamed rung 1 that serves, and passes at spec

*2026-08-10. `mlx-community/Qwen3-Coder-Next-4bit`, 44.84 GB, 3 B active of 80 B.*
**`platform.md` §4.8 owns this model's numbers**; what belongs here is what it changes
about the tier-1 picture above.

| | 122B-A10B (§4.1a) | DeepSeek-V4 (§4.7) | Qwen3-Coder-Next |
|---|---|---|---|
| Gate B, worst of run | 153.7 | 48 | **258.0** |
| `meets_spec` (sec 11's 200) | false | false | **true** |
| decode tok/s | 4.05 | 2.6 | 8.96 |
| resident / load peak | 3.46 / 3.46 GB | 6.49 / 6.49 GB | **1.36 / 1.36 GB** |
| serves via `optiq serve` | yes | **no** | yes |

**This is the first `meets_spec: true` on this machine**, and it is worth being precise
about what that does and does not overturn. It does not overturn **T32**: 200 tok/s is a
proxy for per-chunk expert amortisation, §4.2 already answered that question affirmatively
for the 122B, and this model passing the number is consistent with — not evidence for —
the proxy being well chosen. What it does overturn is the reading that **218 tok/s is the
host's asymptote**: that figure is 10 B active parameters' worth of compute, not the
machine's, and a 3 B-active model clears it.

**§4.3's verifier-contract table gains a column, and rung 3 still wins it:**

| Verifier turn | rung 3 | rung 1, 122B | rung 1, DeepSeek-V4 | rung 1, Qwen3-Coder-Next |
|---|---|---|---|---|
| rerank, 5k in / 8 out | ~7 s | ~34 s | ~44 s | **~20 s** |
| review, 30k in / 128 out | ~33 s | ~214 s | ~270 s | **~130 s** |

So the price of a decorrelated verdict falls from ~6–9× to **~3.5×**. That is a change of
size, not of kind — **rung 3 still ships** — but T5's question is now being asked against a
cheaper answer, and this is the first candidate for which the answer might be worth it.

---

## 5. What runs here, and what merely fits

Projections use the two laws §2 validates:

> decode tok/s ≈ 247 GB/s ÷ (active_params × bpw ÷ 8) × 0.55
> prefill tok/s ≈ 6.6 TFLOP/s ÷ (2 × active_params)

**The prefill law is for resident models and does not transfer to a streamed one — it
over-predicted `Qwen3-Coder-Next` by 26%.** Applied to §4.2's streamed form it says the
`tokens/218` compute term scales inversely with active parameters, giving 727 tok/s at
3 B against the 122B's 218 at 10 B, and a Gate B of ~350. Measured: **258.0**. What does
not scale is everything the router and the attention pay regardless of how few experts
fire — 36 of that model's 48 layers are gated-delta linear attention, and picking 10 of
512 experts costs the same whatever their size. **For a streamed model, scale the sweep
term by expert bytes and treat the compute term as measured-or-unknown.**

| Model (OptiQ / MLX) | Disk | Active | decode | Verdict |
|---|---|---|---|---|
| `Qwen3.6-35B-A3B-OptiQ-4bit` | 24.7 GB | 3 B | **70–87 meas.** | current tier 0 |
| `gpt-oss-20b-OptiQ-4bit` | 11.7 GB | ~3.6 B | ~65 | leaves room for a desktop |
| `gemma-4-26B-A4B-it-OptiQ-4bit` | 18.8 GB | 4 B | ~60 | pairs with a verifier |
| `Qwen3.5-9B-OptiQ-4bit` | 8.2 GB | 9 B dense | ~27 | independent verifier |
| `Qwen3.6-27B-OptiQ-4bit` | 20.0 GB | 27 B dense | **~9** | 77.2 SWE-bench; **fits, unusable** |
| `gemma-4-31B-it-qat-OptiQ-4bit` | 23.6 GB | 31 B dense | ~8 | **fits, unusable** |
| `Qwen3.5-122B-A10B-OptiQ-2bit` | 46.9 GB | 10 B | **4.05 meas.** | streams; **prefill 165 meas.** — the one row where decode is the wrong column |
| `DeepSeek-V4-Flash-0731-2.4bit-mixed` | 92.8 GB | 13 B | **2.7 meas.** | streams, 6.49 GB resident; **prefill ≤168 derived** — §4.4. Fits; Gate B fails on compute |
| `DeepSeek-V4-Flash-0731-OptiQ-2bit` | 92.5 GB | 13 B | **never generated a token** | **Loads, then dies on the first request** — §4.7. Resident 6.49 GB streamed / 15.21 GB with scales resident, load peak **23.80 GB**, all measured. Throughput figures remain inherited from the row above and are now unreachable on this engine |
| `Qwen3-Coder-Next-4bit` | 44.8 GB | 3 B | **8.96 meas.** | streams, **1.36 GB resident**; **Gate B 258.0 meas., `meets_spec: true`** — the only row here that clears sec 11's 200. Serves through `optiq serve`. `platform.md` §4.8 |

**Active parameters, not total, are the constraint.** The rows marked *fits, unusable*
are the finding: on a bandwidth-limited host "does it fit in memory" is the wrong
question, and answering it produces configurations that load successfully and then
miss the latency contract by 4–8×. Every viable configuration here is a
low-active-parameter MoE.

Frontier open weights (Kimi K3 at 2.8 T, DeepSeek V4 Pro at 1.6 T, GLM-5.2 at 744 B)
are 400 GB – 1.4 TB at 2-bit and would stream at well under 1 tok/s. They are the
reason rung 4 exists, not candidates.

---

## 6. Environment

Two virtualenvs, on purpose: mlx-optiq pins `transformers<5.13` and orbit's stack
resolves 5.14.1. They never share an interpreter anyway — sec 5.4 puts the engine
behind a process boundary and `backends/mlx_tier1.py` imports no MLX.

```bash
uv venv --python 3.13 .venv        && uv pip install -e '.[dev,constrain,mlx]'
uv venv --python 3.13 .venv-optiq  && uv pip install --python .venv-optiq/bin/python 'mlx-optiq>=0.4.12'
```

**Corporate TLS interception breaks every Python HTTPS download** with
`CERTIFICATE_VERIFY_FAILED`. System `curl` works because it trusts the Keychain;
Python ships certifi and does not. Export the Keychain roots and point Python at them
— do **not** disable verification, because the sec 8.6 offline claim is about what this
machine talks to and a runtime that skips certificate checks cannot make it.

```bash
B=~/.config/certs/macos-ca-bundle.pem
security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain >  "$B"
security find-certificate -a -p /Library/Keychains/System.keychain                        >> "$B"
security find-certificate -a -p ~/Library/Keychains/login.keychain-db                     >> "$B"
export SSL_CERT_FILE="$B" REQUESTS_CA_BUNDLE="$B"
```

The bundle is a snapshot; re-run the three commands if the gateway rotates its root.

---

## 7. Gate thresholds on this host

`[gates]` in `orbit.toml` sets what this machine is judged against. Code defaults are
the sec 11 figures (`orbit.thresholds`), so an absent block judges the spec.

| Knob | Spec | Here | Measured | Why it moved |
|---|---|---|---|---|
| `gate_a_ttft_s` | 5.0 | **35.0** | 30.47 s @32k | Arithmetic, not tuning: TTFT = context ÷ prefill rate. Meeting 5.0 s at 32k needs 16,000 tok/s — 2.4× more FLOP/s than the GPU has. §2.2. |
| `gate_a_decode_tok_per_s` | 30.0 | **25.0** | 27.1 constrained / 65.6 free | The constrained arm costs 2.4× and buys 100/100 first-attempt tool calls against 0/100. §2.3. |
| `gate_b_prefill_tok_per_s` | 200.0 | **150.0** | **153.7 mean, sd 1.07** (gate, 6 runs) — **and 258.0 on `Qwen3-Coder-Next`, where this floor is 42% low and catches nothing; §4.8, `orbit.qcn.toml.example`** | Was 20.0, set against a projected ~26 tok/s and wrong by ~6× in the pessimistic direction — it passed anything and measured nothing. Raised once the gate actually ran (§4.1a). 150.0 is 3σ below the six-run mean and under every run observed, so it does not flap but still catches a few-percent host degradation (T18); a fallback to 2048-token chunks lands near 118. `meets_spec` still reports the honest 1.3× shortfall — and **T32** on why that red is not the architecture verdict sec 11 intended. |
| `contract_chat_ttft_s` | 2.0 | **35.0** | 30.47 s @32k | Same fact as `gate_a_ttft_s`, relaxed to the same value so one report does not disagree with itself about one number. |
| `contract_chat_tok_per_s` | 40.0 | *unchanged* | 27.1 constrained | Left failing on purpose. `gate_a_decode_tok_per_s` already records the decision about the constrained arm; two relaxations for one fact would read as two hosts' worth of shortfall. |
| `gate_a_toolcall_failure_rate` | 0.05 | *unchanged* | 0/100 | Not a hardware fact. A host that cannot meet it has a bug in the tool-call layer, which is what the gate is for. |

**Relaxing hides nothing.** Every report carries `spec_budget` beside `budget` and
`meets_spec` beside `pass`, and `relaxed_criteria` names each row that is green only
because of this block. Delete the section before quoting a gate result off this
machine.

---

## 8. Reproduction

```bash
# §1 Metal ceilings
.venv-optiq/bin/python -c "import mlx.core as mx; print(mx.device_info())"

# §2.1 bandwidth and GEMM · SSD random read with the page cache bypassed
.venv-optiq/bin/python tools/bench/mlxbench.py    # it is also operations.md 3.1's gate
python3 tools/bench/ssdbench_verified.py

# §4.6 / §4.4 DeepSeek-V4 without loading it: shape off the safetensors headers by HTTP
# Range, then the loader's exact read pattern replayed against local blobs. ~3 min,
# no weights, no GPU. Re-run the second one from a fresh boot; it reads ~1 TB.
python3 tools/probe/ds4_headers.py                                       # §4.6, the OptiQ quant
python3 tools/probe/ds4_headers.py mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed   # §4.4
python3 tools/bench/ds4_streambench.py

# §2.2 Gate A
.venv/bin/orbit bench latency --out var/gate-a.json

# §2.3 the tool-call gate on real weights
.venv/bin/orbit gate toolcall --runs 100

# §2.3's escape finding. No weights, ~1 min, reproduces the hardware cost to within 2%.
HF_HUB_OFFLINE=1 .venv/bin/python tools/bench/constrained_decode_bench.py escape

# §2.3's caching decision: which escapes share a state, and what a key would collapse.
HF_HUB_OFFLINE=1 .venv/bin/python tools/bench/constrained_decode_bench.py statekey

# §2.4 determinism in logits. One load of 22.14 GB serves every arm, because on MLX a
# CPU arm reads the same unified-memory buffers. The chunk arm is ~30 s; the CPU arm is
# ~450 s for four tokens, so ladder it and do not raise --cpu-tokens casually.
HF_HUB_OFFLINE=1 .venv/bin/python tools/probe/determinism_probe.py \
  --prompt-tokens 8000 --chunk-a 2048 --chunk-b 512 --decode-tokens 64 \
  --out var/determinism-chunk-8k.json
HF_HUB_OFFLINE=1 .venv/bin/python tools/probe/determinism_probe.py \
  --decode-tokens 4 --cpu-tokens 4 --out var/determinism-device.json

# §4.5 does rung 3 ever disagree with candidate 0. Loads tier 0; ~6 min for 12 tasks.
python tools/quality/rung3_agreement.py --config orbit.toml --prompts 12 --candidates 5 \
  --out var/rung3-agreement.json

# §4.1 tier-1 streaming. The engine is started by hand and Orbit never spawns it
# (sec 5.4 is a process boundary), so stopping it is also by hand — a stale optiq
# outlives every Orbit restart and shows up only as memory pressure.
HF_HUB_OFFLINE=1 .venv-optiq/bin/optiq serve --stream-experts \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-122B-A10B-OptiQ-2bit/snapshots/<hash> \
  --host 127.0.0.1 --port 8081 --max-context 32768 --prefill-step-size 8192
# --prefill-step-size is mlx_lm.server's, not optiq's; optiq forwards what it does
# not recognise. It is the 1.75x in §4.2 and there is no optiq flag for it.

# §1.1 headroom, read as total - active
python3 -c "import subprocess,re;
v=subprocess.run(['vm_stat'],capture_output=True,text=True).stdout
a=int(re.search(r'Pages active:\s+(\d+)',v).group(1))*16384
t=int(subprocess.run(['sysctl','-n','hw.memsize'],capture_output=True,text=True).stdout)
print(f'{(t-a)/2**30:.1f} GB free of {t/2**30:.0f}')"
```

**All four are committed as of 2026-08-09.** Three of them lived in gitignored
`specs/bench/` while the sections above cited them for reproduction — the same trap as
`specs/NEXT_STEPS.md` and `specs/CONSTRAINED_DECODE_SYNC.md` (`architecture.md` §5, trap 11): a
number is only reproducible if the script that produced it survives a clone. Their
measurement discipline is described in §2.1.

---
