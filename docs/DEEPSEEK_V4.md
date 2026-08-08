# DeepSeek-V4-Flash-0731 as tier 1

**Question asked:** can the streaming mechanism tier 1 already uses run
DeepSeek-V4-Flash-0731 locally, MacBook first — and does this repository have what
that architecture needs?

**Answer:** yes, in the role of tier 1, on the engine tier 1 already talks to, with
no new backend. The architecture itself never crosses our line — sec 5.4 put the
engine behind a process boundary, so mlx-optiq implements CSA, HCA, mHC and the MoE
router and we implement none of it. What this repository had to grow was smaller and
in a different place than "support the architecture" suggests: **the model reasons by
default, and a reasoning verifier breaks two invariants at once, silently.** That, a
Gate B instrument that would have flattered this model class, and a set of numbers
that need re-deriving for 64 GB, were the real gaps. All three are closed here.

Nothing below has met the hardware. Every number is labelled measured (by someone,
somewhere, cited), derived (arithmetic from those, stated inputs), or unverified.

---

## 1. The architecture

`deepseek-ai/DeepSeek-V4-Flash-0731`, released 31 July 2026, MIT, ungated.

| | |
|---|---|
| Parameters | 284B total, 13B active |
| Layers | 43 |
| MoE | 1 shared + 256 routed experts per layer, intermediate 2048, **top-6** routed |
| Routing | first three MoE layers use **hash routing**; the rest learned |
| Attention | hybrid: 5 layers sliding-window local, 20 KV-compressed (HCA), 21 KV-compressed + top-k selection (CSA) |
| Residuals | mHC (Manifold-Constrained Hyper-Connections), expansion 4, 20 Sinkhorn-Knopp iterations — replaces the plain residual stream |
| Context | 1M, max output 384K |
| Speculative | DSpark / MTP head shipped in-checkpoint |
| Reasoning | three modes — non-think, think-high, think-max |

Two notes on the numbers. The HF card totals **304B** where the architecture writeups
say 284B; the ~20B gap is consistent with the in-checkpoint DSpark module being
counted in one and not the other, but this is inference, not something the card says
— it does not change anything below, because the DSpark head is not loaded for a
verifier. And `first_k_dense_replace` — how many of the 43 layers are dense before
the MoE stack begins — is **unverified**; §5 carries it as a ±3-layer uncertainty.

### What of it reaches this repository

Almost none, and that is the design working rather than a gap.

`Backend` asks for three things: `render`, `generate`/`stream`, `container_hash`.
`OptiqTier1Backend` is an httpx client. CSA's FP4 lightning indexer, HCA's 128×
compression, the Sinkhorn-Knopp iterations in mHC, hash routing on the first three
MoE layers — all of it lives on the far side of a socket, in mlx-optiq's vendored
`deepseek_v4` decoder. We do not implement it, cannot get it wrong, and do not need a
stand-in for it.

What *does* reach us is everything the architecture exposes at the API: the reasoning
mode, the DSML tool-call dialect, the token accounting, and the cost model that
decides which tier this model can serve. That is the actual surface, and §3 audits it.

---

## 2. Role and engine

Both were open. Both are decided here, on numbers.

### Role: tier 1, and only tier 1

| Role | Verdict |
|---|---|
| **Tier 1 verifier** | **Chosen.** Input-dominated by construction — rerank emits an integer, review a verdict, both over 5–30k tokens. That is the one shape SSD streaming serves. |
| Tier 0 generator | **Rejected.** Gate A needs ≥30 tok/s warm decode and carries a kill condition. Measured decode for this model streaming on a 48 GB Mac is 4.5–5 tok/s — 6× short, and the gap is SSD bandwidth, not tuning. |
| Both tiers, one model | **Rejected.** Inherits tier 0's decode wall, and drops the repo-derived adapters, which are the product thesis. |
| Rung 2 resident-swap occupant | **Rejected.** Rung 2 needs a model that fits resident after evicting tier 0. 92.5 GB does not fit in 64 GB at any eviction. |

The role question answers itself once the asymmetry is stated: this model reads two
orders of magnitude faster than it writes on this hardware, and tier 1 is the role
this codebase already built for a model shaped that way.

### Engine: mlx-optiq, with ds4 as the measured alternative

| | mlx-optiq | ds4 (DwarfStar) |
|---|---|---|
| DeepSeek-V4 support | `deepseek_v4` decoder since v0.4.12, vendored from mlx-lm PR #1192 | purpose-built for this model |
| Quant for 64 GB | `DeepSeek-V4-Flash-0731-OptiQ-2bit-mixed`, 92.5 GB disk / 6.5 GB resident | `ds4f-q2` GGUF, ~87–98 GB, `--ssd-streaming-cache-experts 32GB` documented for 64 GB MacBooks |
| **Schema-constrained output** | **`response_format` + `guided_json` since v0.2.7** | **not documented** |
| Chat template | auto-generated, pinned against DeepSeek's own encoder | own DSML handling |
| Integration cost | **none — it is already rung 1** | new backend |

mlx-optiq wins on the one criterion that is an invariant rather than a preference.
Sec 5.2 exists because 2-bit's documented failure mode is broken JSON and invented
schema fields; `response_format` is what makes that failure not bite, and
`validate_or_raise` is the second line, not the first. An engine with no constrained
decoding makes the second line the only line.

It also wins on cost of being wrong: it is already the rung-1 transport, so choosing
it is a config change, and choosing against it later is one too.

**ds4 is not dismissed.** It is the engine with published fast-prefill numbers, it
speaks all three of our wire protocols, and it independently implements exact DSML
replay maps for KV reuse — the same idea as sec 8.5.5, arrived at separately, which
is worth reading before Phase 2. If Gate B fails on mlx-optiq, ds4 is the first thing
to measure, and §6 puts that in the bring-up order rather than leaving it as a
sentiment. What ds4 would then have to answer for is the schema gap.

---

## 3. Conformance audit

Every surface in this repository that could care about the model, and whether it does.

| Surface | Verdict |
|---|---|
| `Backend` interface | **Fits.** Three methods, all satisfied by the existing httpx client. |
| `render` / `renders_canonically` | **Not reached.** Tier 1 posts an OpenAI messages array; the engine templates it. DeepSeek ships no Jinja template (a Python `encoding_dsv4.py` reference encoder instead) and mlx-optiq generates one pinned against it. Only matters if this model ever renders locally — it does not, as tier 1. |
| DSML tool calls | **Not reached, by design.** `build_payload` sends `{role, content}` only; a verifier is given no tools and emits none. The dialect matters for a tier-0 DeepSeek, which §2 rejected. |
| `rerank_schema(n)` / `REVIEW` / `PLAN_CRITIQUE` | **Fit unchanged.** `additionalProperties: false` and the `maximum` bound are exactly the sec 5.2 mitigations this model needs at 2-bit. |
| `CALL_BUDGETS` output clamp | **Broken by reasoning.** Fixed — §4.1. |
| Greedy sampling / sec 9.3 determinism | **Broken by reasoning.** Fixed — §4.1. |
| `measure_prefill` / Gate B | **Instrument flattered this model class.** Fixed — §4.2. |
| `expert_cache_bytes` = 18 GB | **Holds, with headroom, and was worth re-deriving** — §5. |
| `request_timeout_s` = 180 s | **Too tight at the Gate B floor** — §4.3. |
| `container_hash` over 92.5 GB | **Works, cost unmeasured** — §4.4. |
| `count_tokens` byte estimate | **Adequate.** Only a fallback for when the engine reports no `usage`; both candidate engines report it. |
| `context_scale.real_window` | **Untouched.** It describes tier 0's window. The 1M context is not reported to the harness and must not be. |
| `MASS_RANKED_ARCHITECTURES` | **Already correct.** `"deepseek"` is in the tuple; profiles rank DeepSeek-style by mass, not count. Only reached if this model were tier 0. |
| Offline posture (sec 8.6) | **Holds.** `_require_loopback_endpoint` already refuses a non-loopback rung-1 endpoint; a local `optiq serve` is exactly what it is for. Weights are MIT and ungated, so the download is one-off and auditable. |
| `SourceKind`, `RegressionReport` | **Untouched.** No new source of training data, no new score. |

---

## 4. What was fixed

### 4.1 Tier 1 must not think — and must be caught when it does

The headline gap, and the one worth the most.

DeepSeek-V4-Flash reasons by default. Two invariants break if it does, and **neither
one reports anything**:

1. **The sec 5.1 clamp stops bounding what it exists to bound.** `CALL_BUDGETS` caps
   the completion at 128 tokens for a rerank, and a `<think>` block is part of the
   completion. The model spends the budget reasoning and is cut off before the JSON
   verdict exists. `validate_or_raise` then correctly refuses it — and every rerank
   degrades to a failed `Verdict`. Best-of-N silently stops reranking on every turn:
   the router's documented rung-5.5 degradation, reached by accident, reported as
   nothing.
2. **`temperature` stops being honoured.** DeepSeek's API documentation states that
   thinking mode does not support `temperature`, `top_p`, `presence_penalty` or
   `frequency_penalty`, and that setting them is *not an error*. `build_payload` asks
   for greedy so two runs of the same rerank agree. Under thinking mode that request
   is accepted and discarded — and the receipt goes on attesting to a greedy
   judgement that was a sample.

Failure 2 is the worse one. Failure 1 is loud enough to notice eventually, because
verification stops working. Failure 2 produces a working system whose attestation is
false, which is the exact shape of thing HANDOFF §5 exists to prevent.

The fix has two halves, and the second is the one that holds:

* **Request:** `tier1.reasoning_control` (`auto` | `deepseek_v4` | `none`) sends
  thinking off in both documented spellings — DeepSeek's top-level `thinking` object
  and the vLLM recipe's `chat_template_kwargs`. Two authorities document different
  keys and the engines disagree about which they read; sending one is a coin flip.
  `auto` recognises the family from the model name, including ds4's `ds4f-*` aliases.
  **There is deliberately no value that turns reasoning on** — that would be a knob
  for disabling the clamp.
* **Response:** `refuse_reasoned_answer` rejects any verdict that arrives with
  `reasoning_content` or non-zero `reasoning_tokens`, on **every transport and every
  model**, ungated by the config. The request-side flag is a guess about what the
  engine reads; this is an observation of what it did. An engine that ignored the
  flag, a model the name match missed, an operator who configured thinking on at the
  engine — all three land here, loudly, on the first call.

That split is the point. A name-matching guess that silently did nothing for an
unrecognised model would reintroduce failure 2 for exactly the models nobody thought
about.

### 4.2 Gate B's filler would have flattered a streamed MoE

`measure_prefill` built its filler by repeating one 23-character line to the target
length, on the reasoning that "content is irrelevant, length is the variable". For a
dense model that is true. For this one it is not, and the direction it is wrong in is
the dangerous one.

Streamed prefill costs the **union** of experts a chunk routes to, read off SSD once
each. Identical tokens route to identical experts — and on the first three MoE layers,
which use hash routing, they route identically by construction. One line repeated to
16k tokens collapses the union to a handful of experts, the engine's cache serves the
whole sweep out of RAM, and Gate B reports a throughput no real prompt will reach.

Gate B is a floor test deciding whether the in-house streaming loader is a three-week
option or M-blocking. A floor test must not fail in the flattering direction.

`prefill_filler` now emits deterministic, identifier-diverse, code-shaped text across
four block shapes with per-block literals — the token diversity real source has. It
is still fixed-seed, and still sized in characters so a sample labelled 16k carries
16k. It does not claim to reproduce a real repository's expert distribution; nothing
synthetic can. It claims only to stop understating the union, which is what the old
filler did.

`gate_b_report` now also carries the model and `expert_cache_bytes`, because streamed
throughput is a function of how much of the expert set is already resident and a rate
quoted without it is not reproducible (sec 10.5).

### 4.3 The request timeout and the Gate B floor are in conflict

Not changed — flagged, because the right value depends on a measurement not yet taken.

`request_timeout_s` defaults to 180 s. Tier 1's design point is ~1,100 tok/s prefill,
at which a 12k review is the ~25 s the verifier docstring quotes. Gate B's floor is
200 tok/s — 5.5× slower. At the floor, a 30k-token review costs **150 s of prefill
alone**, plus ~46 s to emit 512 tokens at ~11 tok/s: **~196 s, past the timeout.**

So a `review` at the top of its documented input range times out at exactly the
throughput Gate B calls a pass. It degrades to a failed `Verdict` rather than
breaking anything, which is why this is a flag and not a defect — but it means a
passing Gate B does not by itself imply reviews complete. The DeepSeek block in
`tandem.toml.example` carries the arithmetic and a raised value.

### 4.4 Hashing a 92.5 GB container

`hash_artefact` BLAKE3s the whole tree on first use, memoised on a stat signature.
Its docstring justifies this with "hashing a 20 GB container at startup is not a
startup cost worth caching around". This container is **4.6× that**, and the read
competes with the expert stream for the same SSD.

Derived, unverified: ~60 s at ~1.5 GB/s single-threaded. Measure it on day one
(`time tandem doctor` with the container configured). If it hurts, `blake3`'s
`max_threads` is the lever; **not** mmap, which sec 8.4 forbids for the KV cache and
which would be a bad habit to start here.

---

## 5. Budget on the M4 Max, 64 GB, 1 TB

Derived from the mlx-optiq quant's published 92.5 GB / 6.5 GB split.

**Memory**

| | GB |
|---|---|
| macOS floor | ~8 |
| Tier 0, Qwen3.6-35B-A3B 4-bit | ~20 |
| Tier 0 KV at 32k + prompt cache | ~4 |
| DeepSeek-V4 non-routed, resident | 6.5 |
| DeepSeek-V4 activations at ~16k context | ~4 |
| **Available for the expert cache** | **~21** |

The existing 18 GB default fits, with ~3 GB of headroom. That is a coincidence worth
naming: 18 GB was sec 2.1's budget for the 122B, and it survives re-derivation for a
model 2.3× its size only because this one keeps so little resident. Do not read it as
tuned — §6 tunes it against a measurement.

18 GB of an ~86 GB routed-expert set is **~21% resident**.

**Disk:** 92.5 + ~20 + 20 (disk KV budget) + adapters ≈ 135 GB of 1 TB. Comfortable —
and per HANDOFF §3, capacity here is a *performance* spec, not just a capacity one.

**Bandwidth — the number Gate B actually turns on**

Inputs: 86 GB streamed set; 256 experts; top-6; ~40 MoE layers (43 minus an
unverified dense prefix, ±3); ~8 MB per expert per layer; M4 Max SSD read taken at
5–6 GB/s, itself unverified.

* **Decode:** 6 × 40 × 8 MB ≈ **2.0 GB per token**. At 5.5 GB/s and a 21% cache hit,
  ~0.29 s/token ≈ **3.5 tok/s**. Sanity check: the independent 48 GB MLX
  implementation with a ~31 GB cache measures 4.5–5 tok/s, and this model predicts
  ~4.2 for it. Calibrated to within ~15%.
* **Prefill, if the engine sweeps by chunk:** a chunk of C tokens reaches full expert
  coverage at C ≈ 260 (coupon-collector on 256 experts drawn 6 at a time), so beyond
  that the sweep reads the uncached ~68 GB **once**, not once per position. At
  5.5 GB/s that is ~12.4 s per chunk:

  | Prefill chunk | Derived tok/s | Gate B (≥200) |
  |---|---|---|
  | 2,048 | ~165 | fail |
  | 4,096 | ~330 | pass |
  | 8,192 | ~660 | pass |

* **Prefill, if it does not:** every position pays its own experts and prefill equals
  decode — ~3.5 tok/s, **60× under the floor**.

That gap is the whole risk, and it is not a hardware question. Both failure modes are
already on the record: the 48 GB MLX engine measures prefill and decode at parity
(~200 ms/token) and says so plainly — "every prompt position also pays for its
experts" — and HANDOFF §3 already notes ds4 measuring GLM-5.2 streaming prefill at
3–5 t/s, ~100× below the bandwidth bound.

**So Gate B is one question with a named knob: does the engine amortise the expert
sweep across a prefill chunk, and is that chunk ≥ ~4k tokens?** Everything else about
this deployment follows from the answer.

---

## 6. Bring-up order

Hardware is in hand, so this is a measurement plan. Each step can invalidate the ones
after it.

```bash
# 0. Engine and weights. MIT, ungated, ~92.5 GB.
pip install 'mlx-optiq>=0.4.12'
optiq serve --model mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit-mixed \
            --stream-experts --port 8081

# 1. Point tier 1 at it — see the DeepSeek block in tandem.toml.example.
tandem doctor          # rung, container hash, offline posture. Time this: §4.4.

# 2. Gate B. The decision. Everything else waits on it.
tandem bench tier1
```

**If Gate B passes** (≥200 tok/s at 4k/8k/16k) — sweep the expert cache at 12/18/24 GB
to find where the curve flattens, then re-derive §5 from the measurement rather than
the arithmetic. Then G2 (expert cache at 0 vs max, byte-identical). G2 matters more
here than anywhere: an LRU expert cache that changes the output makes every
determinism claim in the receipt false, and this is the first deployment where
placement is dynamic.

**If Gate B fails** — measure `--prefill-chunk` first, because §5 says a 2k chunk
fails and a 4k chunk passes on the same hardware. If the engine has no such knob, or
the number does not move, it is not doing a batch-union sweep at all: bring up ds4
with `--ssd-streaming-cache-experts 18GB` and run the same gate against it. If both
fail, the in-house streaming loader is M-blocking rather than optional, which is
exactly the schedule fact Gate B exists to surface on day three.

**Regardless of Gate B:** DSpark/MTP stays **off** for tier 1. It is opt-in and
experimental in ds4, it disables native batching there, and it buys nothing for a
verifier writing ≤128 tokens — while threatening the byte-identity G1 and G2 assert.
Worth noting for the record that it is the one mechanism that could rescue *decode*
if this model were ever asked to generate: verifying K speculated tokens in one
forward pass amortises the expert sweep across all K, which is the prefill trick
applied to decode. That is a Phase 2 question, not this one.

---

## 7. Open questions

- **`first_k_dense_replace`.** Unverified; ±3 layers on every per-token figure in §5.
  Read it off the served `config.json` on day one.
- **M4 Max SSD read bandwidth under the engine's access pattern.** Taken at 5–6 GB/s.
  8 MB random reads at queue depth are not sequential reads, and every number in §5
  scales linearly with the real figure.
- **Whether `tier1.model` should change default.** Left as the Qwen. Flipping it is a
  one-line change once Gate B has an answer; flipping it before would put an
  unmeasured model in front of every new deployment.
- **The 284B/304B discrepancy.** Does not affect a verifier, but should be resolved
  before anyone quotes a parameter count in a receipt or a slide.
- **ds4's DSML replay maps.** An independent implementation of sec 8.5.5's idea,
  including surviving restarts via an appended KTM section. Worth reading before
  Phase 2 whether or not ds4 is ever the engine.
