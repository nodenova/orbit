# Handoff — start here

| | |
|---|---|
| **Purpose** | What a session needs to know that the code and git history do not carry. |
| **Audience** | A session starting from a cold clone with no memory of how the code got this way. |
| **Answers** | "What state is this in?" · "What do I do next, and in what order?" · "What is open?" · "Why is this written the strange way?" |
| **Verified against** | `main` at the commit that last touched this file — `git log -1 --format='%h %ad' docs/HANDOFF.md`. |
| **Rule** | **If the code and this document disagree, the code is right and this needs fixing.** Say so, and fix it in the same commit. |

Written as a lookup rather than a pasted hash: a pasted one is stale the moment anything
else lands, and a handoff nobody trusts is a handoff nobody reads.

This file is **committed on purpose**. It lived at `specs/NEXT_STEPS.md` until `/specs/`
became the ignored drop point for the v1 specification, and it had to move rather than
follow it into the ignore rule — an untracked file dies with the container it was written
in, and a cold clone is the only reader this file has.

### What this file owns

**State, the plan, and the tracker.** All three existed in two or three copies —
`ROADMAP.md` and §4 held the same plan, §5 and `BASELINE.md` §9 held two trackers with
overlapping IDs, and `STATUS.md` §3 held a third partial gap list. They had already
drifted: two of them still called rung 1 "arithmetically dead" after it was measured at
165 tok/s, and four places asserted `supports_state()` was False after it had been True for
a week. They are merged here.

| File | Owns |
|---|---|
| `BASELINE.md` | every number, and the gate thresholds derived from them |
| `STATUS.md` | specification section → code, and the deliberate deviations |
| `PROCESSES.md` | what may hold the GPU, and the host-health gate |
| `CONSTRAINED_DECODE.md` | the constrained-decode cost model: F1/F2 built, F3 rejected on measurement, F4 open |
| `../README.md` · `../CLAUDE.md` | what this is and how to run it · working conventions |

---

## 1. What this is

**A local coding-agent runtime that optimises for merge quality.** Two model tiers on one
machine: a fast resident model carrying repo-specific LoRA adapters, and a large streamed
model used as a **verifier rather than a generator**.

Three facts the whole design turns on:

| # | Fact | Consequence |
|---|---|---|
| 1 | **Streamed models are ~40× cheaper per input token than per output token.** Decode streams top-k experts per token (**4.05 tok/s measured**). Prefill with a batch-union sweep reads each expert once per chunk (**165 tok/s measured**). | Every published number in the field is single-request decode, so the field concluded streamed models are too slow — correct *for generation*, wrong for anything input-dominated. |
| 2 | **The input-dominated tasks are exactly the ones that decide merge quality.** Reranking N candidates emits an integer; reviewing a diff emits a verdict. Both read 5–30k tokens and write 10–300. | Tier 1 is a verifier, structurally: the interface has no `generate`. |
| 3 | **Merge quality, not benchmark score, is the binding constraint.** METR: ~half of SWE-bench-passing PRs would not be merged by maintainers. | A local 35B at 73.4% SWE-bench Verified is not short of capability — it is short of *this repository's* conventions, which is what a repo-derived adapter encodes. |

Fact 1's multiplier is measured on the streamed 122B, not borrowed from tier 0's resident
prefill. Quote 41×, and `BASELINE.md` §4 for both terms.

**Baseline platform: MacBook Pro M4 Max, 36 GB, 1 TB** — `BASELINE.md`, and every budget in
the repository derives from it. The one structural consequence: tier 0 and a streamed tier
1 cannot co-reside, so the verifier is served by rung 3 (tier 0 with its adapter stripped).
That constrains which *rung* serves the verifier, which the codebase already models as a
config choice; it does not touch the thesis.

**The v1 specification is not in this repository.** Code docstrings reference it by section
(`sec 8.2`, `sec 6.3`, …). Without it those references dangle — get a copy and drop it in
`/specs/`, which is gitignored for exactly that. `STATUS.md` maps every section to the
code, which covers most day-to-day needs.

---

## 2. State

| | |
|---|---|
| Branch | `main`, green CI (Python 3.11 and 3.14; jobs `ruff`, `mypy`, `tests`, `cli and gateway smoke`). **Red for seven consecutive builds on 2026-08-09 and green again** — one Markdown code block's comment alignment, because `ruff format` formats Python inside ```` ```python ```` fences and only the `.py` files were being formatted. Trap 14. |
| Toolchain | ruff 0.16.2 (lint + format, 413-rule default) and mypy 2.3 strict over `src/`, `tests/` and `tools/`, both blocking. Config and rationale in `pyproject.toml`. |
| Tests | **570**, passing on both `[dev,constrain]` and `[dev]`-only installs |
| Tier 0 | **Run against real weights** — `Qwen3.6-35B-A3B-OptiQ-4bit`, 23.0 GiB, 36 GB M4 Max |
| DeepSeek-V4-Flash | **It generates, 2026-08-10 — single-threaded, never through `optiq serve`.** 48 tok/s prefill, 2.6 decode, 6.49 GB resident, peak 14.15 GB at 12.8k tokens. **Slower than the 122B on both axes**, so rung 3 still ships. `DEEPSEEK_V4.md`, T7 |
| A1 / step 3 | **blocked on more than a repo** — two silent defects in `build_sft_command`, a `nan`-producing default pair, and a LoRA step that peaks at 31.10 GB against a 30.15 GB ceiling. `A1_TRAINING.md` |
| Tier 1 | **One engine has served requests, one cannot.** `optiq serve --stream-experts` on `Qwen3.5-122B-A10B-OptiQ-2bit` answers by `curl`, not yet by Tandem — prefill 165 tok/s, decode 4.05. The same engine **loads `DeepSeek-V4-Flash-0731-OptiQ-2bit` and dies on the first request** (`BASELINE.md` §4.7, T7). Rung 3 serves the verifier here. |
| Gates run on hardware | Gate A (`var/gate-a.json`), sec 10.2 tool-call (1.00 — **re-run against F1+F2 on 2026-08-09, still 100/100**) |
| Rung 3 measured end to end | Reranks on real weights, 3.13 s p50, and its choices are content-determined — `var/rung3-agreement.json`, `BASELINE.md` §4.5 |
| Tier-0 KV state | **serialises; does not fire.** `backends/mlx_kv.py` round-trips a cache, and on the real container a follow-up restores **0 tokens** — the template's `<think>` tail, §3.9. Worth 20.7× TTFT at 8k when it does fire, and it changed the answer there. |
| Regression detector | **recorded, proven on hardware, and now committed** — 90 items at `baselines/regression-baseline.json` (T29), a no-change re-run clean at 0/0/90, a clamped run reporting 35 regressions (§4.5). Greedy decode reproduces exactly on this container. |
| Gate B (sec 11) | **run 2026-08-10**, six times against the live streamed engine — 153.7 tok/s mean (sd 1.07), `pass` on this host's floor, `meets_spec: false` against 200 (§3.3). Read T32 before quoting that red |
| Determinism (sec 9.3) | **measured 2026-08-10 — and G1's answer is red.** Same config reproduces *bitwise* in logits; CPU-vs-Metal diverges **4.375 against a 1.625 margin and flips the first token**; re-chunking one prompt diverges **2.031**, past the margin at 7 of 65 steps. Platform properties, not bugs — §3.12, T33/T34. The gates themselves still have not run, and neither original shape could have answered validly |
| Gates never run on hardware | isolation (sec 4.2) — **now within the memory ceiling**, one arm at a time instead of two (T21); G1/G2 (sec 9.3), each blocked on its own shape rather than on hardware (T33, T34) |
| Host health | **Healthy, re-confirmed 2026-08-09 22:45** — 320/343 GB/s, 11.97 TFLOP/s at 4096³, ~4 h uptime, **measured immediately after a 23.6 GB tier-0 load** (T18's onset test — it did not fire). It sat at 23 GB/s for most of the previous day. Re-check before any measurement: `PROCESSES.md` §3.1. |
| Headroom right now | **~31 GB by `total − active`** with nothing resident, against the ~27 GB a tier-0 load needs (T9). **Enough for everything, and the gate/bench/eval backlog that was waiting on a fresh boot is unblocked** — `gate toolcall` has since run at a 23.6 GB peak with 27.1 GB of steady headroom left. |

```bash
pip install -e '.[dev,constrain]'
pytest -q            # 570 passed
tandem doctor        # runtime status, offline posture, configured tier-1 rung
```

Working end to end against `MockBackend`: three wire protocols, harness compaction (~41×
measured), incremental streaming, prompt + disk KV caching across a restart, the five-layer
tool-call path, best-of-N with tier-1 rerank, T2 escalation through a git worktree, the
full merge eval, the regression detector, attestation with a hash-chained audit log, and
the A0/A1/A2 extractors.

All four rungs of the sec 5.5 ladder are selectable by name. Rung 2's residency policy is
built and tested; its two `Occupant`s need hardware. Rung 4 is complete, including the
gates that keep it from ever being reached by falling back.

### 2.1 Both MLX backends execute off-target — with the halves kept apart

- **`backends/mlx_tier1.py` imports no MLX.** It was on the Apple-Silicon-only list by
  association and did not belong there — sec 5.4 puts mlx-optiq behind a *process
  boundary*, so the file is an httpx client and a `MockTransport` stands exactly where the
  socket does. Covered directly: the payload, the sec 5.1 clamp on the wire, the schema
  refusal, Gate B's arithmetic.
- **`backends/mlx_tier0.py` genuinely needs MLX**, so `tests/fake_mlx.py` models the
  surface it touches and the real `MLXTier0Backend` runs against it — mounting adapters,
  generating, passing the sec 4.2 isolation gate for real rather than vacuously. **This
  proves the wiring and nothing else.**

The previous version of this claim said both files were *unexecutable* off Apple Silicon.
That was half wrong, and the wrong half hid two silent bugs for months.

---

## 3. What the hardware has said so far

Full numbers and their derivations: `BASELINE.md`.

### 3.1 Constrained decoding was a silent no-op, and now is not

`Pipeline` computes a JSON Schema for every tool-bearing turn and attaches it to the
request. `MockBackend` honoured it. `MLXTier0Backend` never read the field, so on hardware
the schema was computed, carried the whole way down, and dropped — tool-call correctness
rested entirely on repair and retry. **The mock being *stricter* than the hardware is what
kept a fully green suite from seeing it**, and it inverts the rule in `CLAUDE.md` that the
mock must never be easier to satisfy than a real backend.

`lm-format-enforcer` ships no MLX integration, so the fix is a bridge over its
`TokenEnforcer` core: `Constrainer.vocabulary` does the per-tokenizer preprocessing once,
`Constrainer.token_filter` builds a per-request filter, and
`mlx_tier0.build_logits_processor` turns that into the `(tokens, logits) -> logits`
callable `mlx_lm.stream_generate` wants.

| sec 10.2 gate, 100 runs | before | after |
|---|---|---|
| well-formed rate | 0.81 — **fails** | **1.00 — passes** |
| outright failures | 19 | 0 |
| prose (no call attempted) | 62 | 0 |
| needed a retry | 19 | 0 |
| first-attempt tool calls | **0** | **100** |
| wall clock | ~20 min | 1 min 56 s |

The cost is decode throughput on tool-bearing turns only: **27 tok/s constrained against 65
unconstrained**. Free-form turns are untouched. That 2.4× is Python waste and loop structure
rather than a language boundary, and it is removable — `CONSTRAINED_DECODE.md`, T20.

**"Removable" is now measured, and it is mostly not — 2026-08-09.** F1 and F2, the two
Python fixes that ladder derived, buy **11–15%**: 2.55× → 2.38×, 27.0 → 29.5 tok/s. The
gate still passes at 1.00 over 100 runs, so the fixes are free in correctness terms and
nearly free in throughput terms.

**The 2.4× survives, and it is not Python waste.** ~97% of it is one LMFE parser state —
the gap between a backslash and its escape character — at ~425 ms per occurrence, and the
cost of a turn therefore scales with how many escapes in its string arguments the
tokenizer happens to split, not with tokens decoded. T26; `CONSTRAINED_DECODE.md` §8.4.
**Nor is it ours to remove**: the state recurs and a cache would collapse 75% of it, but
the only sound key is one upstream declines to provide, and the obvious local key is a
silently wrong constraint — §8.5.

### 3.2 Gate A — decode passes, TTFT does not

Recorded in `var/gate-a.json`. Worst warm figures across the five frontiers:

| Criterion | Measured | Spec | Verdict |
|---|---|---|---|
| decode tok/s | 69.6 (free-form) / 27.1 (constrained) | ≥30 | pass free-form, fail constrained |
| TTFT | 30.47 s @32k | <5 s | **fail** |
| tool-call failure rate | 0/100 | <5% | pass |

**The TTFT failure is arithmetic, not misconfiguration.** TTFT ≈ context ÷ prefill rate, so
32k at ~960 tok/s *is* ~32 s. Meeting 5 s at 32k needs 16,000 tok/s of prefill — 2.4× more
FLOP/s than this GPU has at 100% efficiency. Unreachable on any M4 Max at any memory size.
The only mechanisms that can close it are compaction and the prompt cache, and the prompt
cache **restores 0 tokens on this container** (§3.9), so today every turn still re-prefills.

**Read a red TTFT as "the mitigations are not being measured", and only as the kill
condition once they exist and it is still red.**

### 3.3 Gate B — run, 2026-08-10

**`tandem bench tier1` has now run against a live `optiq serve --stream-experts` on the
122B**, three times, and T2 is closed. These are the gate's own numbers — `measure_prefill`,
the sec 5.1 clamp on the wire and `gate_b_report` — not `curl`'s.

**Six runs**, worst frontier of each: **152.1 · 153.2 · 153.4 · 153.7 · 153.9 · 155.7** —
mean **153.7**, sd **1.07**, full spread **2.34%**. Every one `pass: true` against this
host's floor and **`meets_spec: false`** against sec 11's 200.

Per frontier, across the six:

| input tokens | mean tok/s | min–max | spread |
|---|---|---|---|
| 5,433 | 157.0 | 153.9–159.1 | 3.3% |
| 11,035 | 154.0 | 153.0–156.7 | 2.4% |
| 22,208 | 155.4 | 152.1–159.7 | 4.9% |

**The figure is 153–154 tok/s, not a single sample**, and the spread is what set the floor:
`gate_b_prefill_tok_per_s` is **150.0**, three sigma below the mean and under every observed
run (T14). All six are one session at ~11 h uptime on a healthy host, which is the narrowest
part of the estimate.
It sits slightly under `BASELINE.md` §4.1's best `curl` reading of 164.7 and well above that
table's fall-off at 16,992 tokens (140.1) — the gate's filler is identifier-diverse, so its
expert union differs from the `curl` prompts'.

**Rung 1 is 1.3× under the spec floor and comfortably over this host's.** A 30k review costs
~214 s against a 300 s timeout, so it fits. Rung 1 is slow, not dead. **Rung 3 still ships**
because it is 6–9× faster and costs no memory — a preference with a measured price rather
than a wall.

**Two things blocked this run and neither was the hardware:**

1. `cmd_bench` built tier 0 before building tier 1 — 23.0 GiB, eagerly, for a rung that
   reaches its engine over a socket and never reads it. `PROCESSES.md` §6.1 recorded this as
   the reason Gate B "cannot be run on this host as the CLI is written". It is now built only
   for the rungs that serve from it, and the rung-3 refusal that used to cost 23.0 GiB
   answers in **0.06 s**. `tests/test_mlx_tier1.py` pins it.
2. The gate's probe sends **no schema**, and the 122B thinks by default — §3.11.

### 3.4 Gate thresholds are host-relative

`tandem.thresholds` holds the sec 11 numbers as code defaults; `[gates]` in `tandem.toml`
sets what a given host is judged against. Every report carries `budget`, `spec_budget`,
`pass`, `meets_spec` and `relaxed_criteria`.

This exists so a machine that cannot meet a floor still runs end to end and surfaces the
*next* weakness, rather than stopping at a known red. **It is not a way to make gates
green.** A pass against a relaxed floor means "this host cleared its own floor";
`meets_spec` is what a result is quoted by. The four values this host sets, with the
measurement behind each, are in `BASELINE.md` §7.

### 3.5 Two traps found while measuring

- **Warm up before timing.** The first constrained call read 4.1 tok/s. That was Metal
  kernel compilation, not the mask — the same effect Gate A records as cold@2k 10.65 s
  against warm@2k 1.58 s.
- **An identity-keyed mask cache is not worth it** *as it was written*. It measured 27.1
  tok/s against 27.6 — nothing. The reason is not that caching cannot pay: LMFE ≥ 0.11
  returns a fresh list object per call, so an `id()` key **cannot hit at all** (0 of 42
  consecutive positions). A content key hits 40% of the time and compares in 0.12 ms.
  `CONSTRAINED_DECODE.md` §3.4.

### 3.6 The engine amortises the expert sweep, and the chunk is a knob

Prefill fits `8.0 s × chunks + tokens / 218`. The 8.0 s is a full expert-set sweep,
constant per chunk and independent of how full the chunk is; 44.6 GB ÷ 8.0 s = 5.6 GB/s,
consistent with `BASELINE.md` §2.1's SSD curve at 3.63 MB blocks. The `tokens / 218` term
is compute and is the asymptote no chunk size beats.

The chunk is `mlx_lm.server`'s **`--prefill-step-size`, default 2048**, reachable because
`optiq serve` forwards arguments it does not recognise. Raising it to 8192 is worth
**1.75×**, and `optiq serve --help` does not list it — trap 7 in §7.

This is what the streamed-tier-1 question reduces to, and it needed no download: it is a
property of the engine, not the model.

### 3.7 A large streamed MoE can be costed without downloading it

`BASELINE.md` §4.4, in full. The method is the transferable part, because it cost about
three minutes and answered questions that had been carried as blocking for weeks:

| Question | Answered by |
|---|---|
| Does the engine stream *this* quant? | Reading `moe_stream._EXPERT_SEGMENTS` and the quant's tensor names. Detection is **name-keyed, not brand-keyed** — "non-OptiQ quant" was never the risk it read as. |
| What is resident, streamed, dropped? | 18 safetensors headers by HTTP Range request. No download. |
| How fast will it stream? | Replaying the loader's exact read pattern (2 MiB weights, 128 KiB meta, qd 6 decode / qd 24 prefill) against local blobs. |

Two things that derivation got wrong before, both because it worked from totals rather than
from the loader's actual calls:

- **Decode is worse than derived** (2.7 against 3.1–3.8). `StreamingQuantizedSwitchLinear`
  reads weight, then scales, then biases, blocking on each, so decode runs at **queue depth
  6, not 24** — 387 sequential barriers per token. And on this host the scales stream too.
  ~~which the 122B proxy never paid~~ — **withdrawn 2026-08-09**: serving the proxy prints
  `expert scales/biases 7.2 GB vs budget 3.9 GB -> STREAM`, so it streams its meta as well
  and always did. Its ~1 GB of resident meta was inferred from a total, never measured. The
  queue-depth reasoning stands; what falls away is the claim that the proxy's figures
  omitted a cost this model pays (`BASELINE.md` §4.7).
- **Prefill is better than derived** (11.1 s/chunk against 13.9), because the sweep is one
  large-block pass at full queue depth.

Net: unchanged verdict, better-founded. The generalisation worth keeping is that **a
streamed model's cost is a property of the loader's call pattern, not of its size** — 11%
of the bytes being 128 KiB costs 23% of decode.

### 3.8 The KV state serialises — and the container cannot be rewound

`supports_state()` was False, so `_probe_cache` and `_remember` both short-circuited and the
disk cache held nothing on the mlx backend. It is now True, and three things had to be
decided rather than looked up.

**The blob is not `mlx_lm.save_prompt_cache`.** That writes a safetensors file and reads it
back through `mx.load`, which maps it — undoing sec 8.4's no-mmap rule one layer above
`kv_disk.py`, where it would still look intact. `backends/mlx_kv.py` is a plain byte
container instead: a JSON header of cache classes, `meta_state` and an array table, then the
raw arrays via `mx.view(a, mx.uint8)`. Verified against real `mlx_lm` cache classes —
`KVCache`, `QuantizedKVCache` with its nested triples, and `RotatingKVCache` — round-tripping
bit-exact, with truncation, growth, a bad magic and an over-budget cache all degrading to a
miss.

**Coverage is counted in tokens, and the gateway keys in bytes.** A chunk boundary lands
mid-token, so `rendered_prefix` re-encodes to something that agrees with the turn's ids only
up to that boundary. The state carries its own ids and `_warm_start` re-checks them against
the next prompt; the byte figure is the key and the reported number, never the authority.

**The finding, and it would have shipped as a no-op.** The first version trimmed the cache
back to the keyed prefix, which is correct and is what every cache the off-target fake could
build supports. The baseline container does not: 30 of its 40 layers are linear attention
(`full_attention_interval = 4`), whose state is recurrent, and `ArraysCache.is_trimmable()`
is False — so `can_trim_prompt_cache` is False for the whole hybrid and every export would
have refused. Silently, on the only model this ships against, under a green suite. So there
are two paths: trim where the cache allows it, and otherwise store what the cache actually
holds — prompt *and* reply — which is a prefix of the next turn's prompt whenever the reply
re-renders to the bytes it was sampled as. `fake_mlx` now builds the hybrid by default, so
the tested path is the one the hardware takes.

| | |
|---|---|
| ceiling on a snapshot | `tier0.max_state_bytes`, 1 GiB — ~49k tokens, measured |
| snapshot size | **20.00 KiB/token + a fixed 64.4 MB.** The marginal term is exactly the 10 full-attention layers `tandem.toml` derives; the constant is the 30 recurrent ones |
| why a ceiling at all | `dumps` peaks at twice the blob |
| over it | the turn stores nothing and the next prefills — slow, never wrong |

### 3.9 …and on the real container it never fires. Measured.

`tools/kv_state_bench.py`, `var/kv-state-bench.json`. Tier 0 at 20.62 GiB, 7.36 GiB
headroom, `Pageouts` +2164 over the run (not thrashing). **The feature is inert in the
scenario it was built for, and the cause is two tokens.**

| Turn 2 of a real conversation | 512 | 2048 | 8192 |
|---|---|---|---|
| TTFT cold | 3.514 s | 8.186 s | 37.73 s |
| TTFT with the state restored | 3.519 s | 8.605 s | 37.75 s |
| tokens actually restored | **0** | **0** | **0** |

`apply_chat_template(..., add_generation_prompt=True)` ends the prompt with
`<|im_start|>assistant\n<think>\n`. Turn N+1 carries turn N's reply in that slot instead, so
turn N's prompt is **not** a prefix of turn N+1's — 8037 of 8039 ids shared, diverging on
the `<think>\n` tail. The cache cannot be rewound past it (§3.8), so `_warm_start` refuses
the whole state. It refuses *correctly*: restoring would continue from a prefix the prompt
does not have. It just never restores.

**The prize is real, and large.** Rendering around the tail so the ids do line up:

| Restore that fires | 512 | 2048 | 8192 |
|---|---|---|---|
| TTFT cold → warm | 3.42 → 1.61 s | 9.23 → 1.70 s | 66.66 → 3.21 s |
| speed-up | 2.1× | 5.4× | **20.7×** |
| same answer as cold | yes | yes | **no** |

**The fix is a seam, not a rewrite.** `generate_step` calls
`prompt_progress_callback(processed, total)` after each prefill chunk and leaves the last
token to `_step`, so at the final callback the cache holds exactly `processed` tokens.
Snapshotting *there*, at a cut the gateway chooses, gives a state that stops before the
generation-prompt tail — the one place a non-rewindable cache can be cut.

**Two findings that outrank the speed-up.**

* **A restore changed the answer at 8k.** The control says that is the restore, not the
  model: two cold runs of the same prompt are byte-identical, and the warm one is not. The
  blob round-trips bit-exact, so the difference is upstream of it — the stored KV was
  prefilled in one chunk arrangement and the cold run's in another, and MLX's batched matmul
  does not promise the same rounding across shapes. **This is a determinism claim failing,
  which is what G1/G2 exist for**, and it makes the cache unsafe to enable for attestation
  until it is characterised. T16.
* **Decode read 2.9 tok/s where `BASELINE.md` records ~65.** Not caused by this work, and
  now explained: it was §3.10's host-wide degradation, not Gate A's and not the KV bench's.
  T17, closed.

### 3.10 The 22× was host-wide, and a reboot cleared it

*2026-08-09. `tools/mlxbench.py`.*

§3.9 handed 2.9 tok/s to "whoever re-runs Gate A". It was neither Gate A's nor the KV
bench's, and not a call-shape artifact: **`mlxbench.py` loads no model at all and reported
23 GB/s against the 247 in `BASELINE.md` §2.1.** The whole GPU was slow.

| `mlxbench.py` | recorded | degraded, 15 h uptime | after a reboot |
|---|---|---|---|
| fp16 elementwise r+w | 247 GB/s | **23** | **323** |
| fp16 read-only reduction | 242 GB/s | **11** | **352** |
| GEMM 4096³ | 10.80 TFLOP/s | **5.29** | **11.9** |

**The reboot is the finding.** The degraded readings were taken at 17:29 on 15 h 47 m of
uptime; the healthy ones at 19:59 on a fresh boot — three consecutive passes, two
independently written scripts agreeing to within 1%. Nothing else changed; a 92 GB download
was in flight during both. So the state is **boot-scoped**, which is what the runbook
predicted and what nobody had yet tested.

Two consequences, and the second costs money if ignored:

- **The measurement freeze lifts.** Every gate is quotable again *once `PROCESSES.md` §3.1
  passes on the day it is run* — not on the strength of this paragraph.
- **The healthy figures are ~1.3× above the recorded baseline** (323 against 247). So
  `BASELINE.md` §2.1's numbers are a floor, and any figure taken between them and 23 GB/s is
  a partly-degraded host rather than a clean measurement.

Ruled out during the degradation, and still unexplained: paging (`Pageins` +411 on a slow
run), thermal (`pmset -g therm` clean), low-power (`powermode 0`), a resident neighbour
(`api/ps` empty, 8080/8081 clear), competing I/O (a `SIGSTOP`ed download changed nothing),
DVFS wake-up (8 back-to-back GEMMs held 4.30–5.60 TFLOP/s with no ramp), and measurement
error (two array sizes, `mx.synchronize()`, three stable passes). **The asymmetry is the
strongest surviving clue: bandwidth fell to ~9% of spec while compute held ~50%.**
Proportional throttling would move both together. T18.

### 3.11 The verifier's thinking guard read a key the engine does not send

*2026-08-10, mlx-optiq 0.4.18, `Qwen3.5-122B-A10B-OptiQ-2bit` — the model `tier1.model`
names. Found by running Gate B, which is the first thing to put an unschema'd call through
this path.*

`refuse_reasoned_answer` is the half of the reasoning-control design that is meant to be
load-bearing: `tier1_call.py` says in as many words that guessing the dialect from the model
name is safe *only* because a model the guess misses "fails loudly on the first call instead
of quietly on every one". **It read `message.reasoning_content` and
`usage.completion_tokens_details.reasoning_tokens`. This engine sends neither** — it spells
it `message.reasoning` and reports no `completion_tokens_details` — so the guard never fired
and the reasoned answer arrived upstream as `text = ""`.

What actually happens, measured in all four combinations against the live engine:

| call | `auto` (no disable keys) | `deepseek_v4` (keys sent) |
|---|---|---|
| **schema-constrained** — every real tier-1 call | verdict | verdict |
| **no schema** — Gate B's `measure_prefill` | **reasoning block, no `content`** | content |

**The schema is what was covering this**, and that is why nothing had caught it: `rerank`,
`review` and `plan_critique` are all schema-constrained, and the engine's structured-output
grammar forces JSON from the first token, so thinking never manifests. The one unschema'd
tier-1 call in the tree is Gate B's own probe, and it is what exposed it.

So the blast radius was narrow — **no verdict was ever wrong** — but the guard protecting
every future one was inert, and it is the stated reason the name guess is allowed to be
narrow. Fixed to read both spellings; with the fix, that same unschema'd call now refuses
with *"403 reasoning tokens"* instead of returning an empty string. Proven on hardware, both
directions.

**Qwen3 was deliberately not added to the `auto` name guess.**
`test_a_non_reasoning_model_gets_no_extra_keys` records why: an unknown key is a 400 on a
strict engine, and widening the guess takes tier 1 down for every deployment that never
asked for it. A measured failure on one host does not license that trade for all of them.
This host sets `tier1.reasoning_control = "deepseek_v4"` explicitly instead — the value names
the *dialect of "off"*, not the vendor. **T30** is whether the library default should change,
and it is the owner's call, not a bug.

### 3.12 G1 and G2 asked for byte-identity, and the platform's answer is a number

*2026-08-10 third session, `Qwen3.6-35B-A3B-OptiQ-4bit`, mlx 0.32.0, host at 325/347 GB/s.
`tools/determinism_probe.py`, one load of 22.14 GB, `var/determinism-*.json`.*

Both determinism gates compare *text* and report a boolean. The measurement below is the
same comparison one layer down — per step, the full logit vector and the greedy top1−top2
margin — and it says the boolean was never the interesting part. **Same config repeated is
bitwise identical**, so nothing here is flakiness; every number is a property of the path.

| Arm compared against a 2048-token-chunk Metal reference | max Δlogit | median Δ | steps where Δ ≥ that step's margin | argmax flips |
|---|---|---|---|---|
| the same config again | **0.0, bitwise** | 0.0 | 0 of 65 | 0 |
| `prefill_step_size` 512 instead of 2048, 8,190-token prompt | **2.031** | 0.688 | **7 of 65** | **0 of 65** |
| CPU instead of Metal, 37-token prompt | **4.375** | 3.406 | **1 of 5** | **1 of 5 — the first token** |

**This is T16's mechanism, measured.** T16 recorded that a restored KV state changed the
answer at 8k and reasoned that "batched matmul does not promise identical rounding across
shapes". It does not: re-chunking the *same* prompt moves logits by up to 2.03, and at 7 of
65 steps that is more than the gap between the top two tokens. No argmax actually flipped in
this window, so the honest reading is exposure rather than failure — but it is exposure on
32% of steps (21 of 65 carry less margin than the run's own worst Δ), and the one knob that
produces it is the one **T13** says Tandem cannot see or record. That promotes T13 from
cosmetic to load-bearing: a receipt claiming determinism must pin `--prefill-step-size`,
because it is a parameter that demonstrably changes the logits.

**G1 fails on hardware, and this is the first result it has ever produced.** Not narrowly:
the divergence is 4.375 against a 1.625 margin — 2.5× — and the **argmax flips on the very
first generated token**, so the device changes the answer rather than merely the bits behind
it. Everything below is why that red is a platform property and not a bug to fix, and why
the gate as written could not have reported it:

1. **Two backends is the wrong shape.** The signature takes two live `Backend`s and gathered
   them; on this host that is 2 × 23.0 GiB against 28.08 GiB. Unified memory means one load
   would serve both devices, which is what the probe does — the gate's shape, not the
   hardware, was the obstacle.
2. **The device cannot be varied per call.** `mlx_lm.generate` binds a module-level
   `generation_stream = mx.new_thread_local_stream(mx.default_device())` **at import** and
   wraps `generate_step`'s body in it, so a caller's `mx.stream(mx.cpu)` is overridden from
   the inside. The first CPU arm here returned in **0.1 s with bitwise-identical logits** —
   it had silently run on Metal, which reads as a clean G1 pass and measures nothing. This
   probe fell into the trap it was written to detect, and the fix is to swap the default
   device *and* rebind that global — backend-global state of exactly the kind §6 forbids
   under concurrency. **So a real G1 is two processes compared by recorded output, not two
   backends in one process** (T33).
3. **Byte-identity is not the platform's to give.** 30 of this model's 40 layers are
   `linear_attention`, and `mlx_lm/models/gated_delta.py:281` dispatches on
   `mx.default_device() != mx.gpu` to `gated_delta_ops` instead of its Metal kernel — a
   *different algorithm* for 75% of the model, by design. Even at the single-op level, no
   weights needed, one 4-bit `quantized_matmul` differs CPU-vs-Metal by 2.3e-05.
   `g1_backend_equivalence`'s red says "pin the reduction order (sec 9.3)"; there is no
   reduction order to pin between two algorithms, and nothing a caller can do from Python
   would make them agree.

Cost, for anyone tempted to run G1 as specified: the CPU arm took **450.5 s** for a 37-token
prefill plus four tokens, against 0.14 s on Metal — **~3,200×**. The gate's own defaults
(4 prompts × 128 tokens) are a multi-hour run on this host, not a CI job.

**What the red does and does not say.** It does not say the receipt's determinism claim is
false: same device and same chunk arrangement reproduce bitwise, twice measured. It says the
claim is *conditional on the execution path*, and that the two conditions which change the
answer — device and chunk arrangement — are both absent from the receipt today (T13). The
disclosure, not the kernel, is the work.

**G2 was worse than never-run: it was a guaranteed pass.** Its two arms differ only in
`tier1.expert_cache_bytes`, and `expert_cache_provenance` in our own tree already records
that this value reaches no engine — nothing sends it, and `--stream-experts-cache` is a
per-projection *count* that mlx-optiq 0.4.18 accepts and drops before its shard reader. So
both arms were one engine at one placement, and the comparison was green by construction on
the gate whose docstring calls it "the most important gate in the product". Run against the
pre-fix code the new test gets exactly `passed=True, reason='ok'`. It now reports **not
measured** and builds no arm at all, keyed off that same function so the fact lives in one
place (T34). The placement axis this host *can* vary is the engine's `--stream-experts` /
`--no-stream-experts`, on tier 0's own 35B — recorded as the way to actually measure it.

---

## 4. What to do next, in order

**The order is load-bearing.** Each step can invalidate the ones after it, so running them
out of order risks tuning against a premise already falsified. Step 3 is a gate on the
thesis: if it fails, steps 4 and 5 are not merely delayed, they are the wrong work.

**A step is done only when its exit criterion has been *observed*, with the artefact that
shows it named.** A step that is coded but unmeasured is *written*, never done.

| # | Step | Must prove | Status | Blocked on |
|---|---|---|---|---|
| 0 | Host health (`PROCESSES.md` §3.1) | `mlxbench.py` ≥ ~250 GB/s on the day of the run | **passing, 2026-08-09 22:45** — 320/343 GB/s, and that reading was taken *after* a 23.6 GB load. Re-check, never assume | nothing |
| 1 | Serialise tier-0 KV state (§4.3) | a follow-up reports `usage.cached_input_tokens > 0` **and** the answer a cold prefill gives | **measured — and the criterion is not met** | nothing |
| 2 | Record a regression baseline (§4.5) | the file exists, recorded against the untouched model, and a deliberate perturbation is detected against it | **done, 2026-08-09 — both halves observed.** 90 items, **committed at `baselines/regression-baseline.json`** since T29; a no-change re-run is clean (0/0/90, exit 0) and a clamped-generation run reports **35 regressions, 0 fixes** | nothing |
| 3 | Train A1, re-run the merge eval (§4.7) | A1 beats base on **≥3 of 5** merge-eval metrics | **open, and no longer only on inputs — `A1_TRAINING.md`** | a real repo (this one has 43 commits against a 500-pair floor), an `[eval]` block, **two silent defects in `build_sft_command`**, and a targeting/sequence decision the measured memory forces |
| 4 | Prove adapter isolation on hardware (§4.4) | the gate passes on `backend = "mlx"` with ≥2 real adapters mounted | **open** | step 3 — there is no adapter to isolate |
| 5 | Re-ask the DeepSeek question (§4.9) | — | **blocked upstream** — deferred by decision, and now the engine cannot serve it at all (`BASELINE.md` §4.7, T7) | an engine that runs it; no longer step 3 |

**Step 3 is now the front of the ordered plan** — steps 0, 1 and 2 are settled (1 as a
measured negative), and 4 waits on 3. Everything below it is either blocked upstream or
cheap-and-unordered, so a session with no repo to point step 3 at should take the three
constrained-decode items instead. They need no hardware, no headroom and about a minute
each:

| Do this | Why now | Where |
|---|---|---|
| ~~**Re-run §3.2 and §4 against `ESCAPED_CALL`**~~ | **Done 2026-08-10 — `filter` runs both fixtures and §3.2/§4 are rebuilt on it.** Rung 0 now predicts rung 1 to within 5%, and it corrected F4's ceiling to ~1.11× | T28 |
| ~~**Decide T29**~~ | **Done 2026-08-10 — committed at `baselines/regression-baseline.json`, which is now the CLI default.** It was in the wrong directory, not an exception to the ignore rule | §4.5 |

**All three are now done, and so is §4.2 (Gate B, 2026-08-10 — §3.3).** A session arriving
here with a real repo should go straight to step 3; one without should take §4.6 (A0 and the
restated E1) or §4.8 (G1/G2), neither of which needs a repo. What is *not* on this list is
anything further on constrained decoding: the remaining lever is upstream (§8.5) and F4 is
worth ~1.11×.
| ~~**Decide T26's fix**~~ | **Done 2026-08-09 — `CONSTRAINED_DECODE.md` §8.5, and the answer is not to write one.** The 5,798-vs-5,795 check was the whole item: the set is fixed by the enclosing object stack, so a key on the backslash alone is a silently wrong constraint. Sound key is stack-wide, does not exist upstream, and is a contribution rather than an upgrade | T26 |

Cheap and unordered beside them: A0 and the restated E1 (§4.6), G1/G2 and E2–E4 (§4.8).

| # | Item | Status |
|---|---|---|
| 4.1 | Gate A | **done** — §3.2 |
| 4.2 | Gate B | **done, 2026-08-10** — §3.3. Six runs, 153.7 tok/s mean, `pass` against this host's floor and `meets_spec: false` against sec 11's 200. Closes T2; T14's floor moved 20.0 → 150.0 on the strength of it. T32 records why the red is not an architecture verdict. |
| 4.2a | **Does rung 3 ever disagree with tier 0's own top candidate?** | **done** — yes, 9 of 12, and the choice tracks content rather than slot. `BASELINE.md` §4.5. **This closes the tier-1 download question**: rung 3 is not a no-op, so neither streamed model is urgent. |
| 4.3 | Tier-0 KV state serialisation | **built, measured, and inert** — §3.8, §3.9. Reopened: the state never restores on this container. |
| 4.4 | Isolation gate on hardware | open — needs a trained adapter |
| 4.5 | Regression baseline | **done** — recorded, re-run clean, and proven to fire (§4.5) |
| 4.6 | A0, then re-gate tool calls | open — **E1 needs restating first** |
| 4.7 | A1 and the merge eval | open — the product thesis |
| 4.8 | G1/G2, then E2–E4 | **the determinism question is measured (§3.12): G1 red on hardware, T16's mechanism quantified. The gates are not run** — T33, T34. E2–E4 open |
| 4.9 | The DeepSeek-V4 OptiQ download | **complete, 92.49 GB.** The model has since been loaded twice and **has never generated a token** — `BASELINE.md` §4.7 |

### 4.3 Serialise the tier-0 KV state — built, and measured not to fire

`supports_state()` returns True. `backends/mlx_kv.py` puts an `mlx_lm` prompt cache into
`KVState.blob` and back, `stream` feeds token ids so a restored cache can cover a prefix of
them, and `export_state` snapshots the turn. §3.8 is what that cost and what it found.

**That measurement has now been taken, and it is negative** — §3.9. A real follow-up
restores 0 tokens at every context size, because the chat template ends the prompt with
`<think>\n` and a recurrent cache cannot be rewound past it. TTFT is unchanged to within
noise. The step is **not** done.

Two things to do, in this order:

1. **Snapshot at the prefill seam.** `prompt_progress_callback(processed, total)` fires
   after each chunk and the last token is handled separately, so the cache holds exactly
   `processed` tokens at the final callback. Cutting there — at a boundary the gateway
   picks, before the generation-prompt tail — is the only cut a non-rewindable cache admits.
2. **Then settle determinism, before enabling it for anything attested.** A restore changed
   the answer at 8k while cold-vs-cold was byte-identical. Until that is characterised (it
   is G1/G2's question), a cache hit can change a receipt's answer. T16.

Two decisions here were made rather than looked up, and both are in §6: the blob is
hand-serialised rather than routed through `save_prompt_cache`, and `export_state` does not
require a trimmable cache.

### 4.4 Prove the adapter mounting

```bash
tandem gate isolation --adapters a0 a1-myrepo
```

Greedy output under adapter *i* with N mounted must be byte-identical to output with only
*i* mounted. **Until it passes on hardware, treat every receipt naming an adapter as
unproven** — a failure means adapter deltas leak between concurrent requests, which is a
silent wrong-answer bug.

The *wiring* is no longer unexercised: `tests/test_mlx_tier0.py` runs this same gate against
the real `MLXTier0Backend` under the fake MLX, and the suite carries a deliberately leaky
wrapper to prove the gate can still fail. What that cannot reach is real quantised deltas
and Metal, which is most of what the gate is for.

> [!NOTE]
> **This gate builds tier 0 once per adapter plus one, and holds exactly one live**
> (2026-08-10). It used to gather the N-mounted arm against each solo arm, which put
> 2 × 23.0 GiB against a 28.08 GiB ceiling and made it unrunnable here; it now records the
> N-mounted arm's ≤128-token outputs, releases the weights through the rung-2 `unload()`
> seam, and builds the solo arms afterwards. `test_isolation_gate_holds_one_arm_at_a_time`
> pins that and fails against the old shape. Still read `PROCESSES.md` §4 for the pre-flight
> — one tier 0 is 23.0 GiB and wants ~27 GB of headroom.

### 4.5 Record a regression baseline *before* changing anything

```bash
tandem eval regression            # --baseline defaults to baselines/regression-baseline.json
```

It is a detector; it detects nothing without a reference point. Record it on the untouched
model, then again after each step below. A baseline taken after the first change is a
baseline of the wrong thing — and step 3 is the first change. A baseline that has never
caught anything is a file rather than a detector, so perturb the model deliberately and
check that it fires.

Loads 23.0 GiB: `PROCESSES.md` pre-flight first, and `RegressionReport` has no score field
on purpose (§6).

**Done 2026-08-09, and both halves were observed.** 90 items (30 each of maths, reasoning,
code_localisation), 2 min 18 s, headroom trough 19.9 GB, pageouts +378 across four runs.

| Run | Result |
|---|---|
| First — records the reference | 90 items written, `baseline_written: true` |
| Re-run, nothing changed | **clean: 0 regressed, 0 fixed, 90 unchanged, exit 0** |
| Generation clamped to `max_tokens = 8`, same container and adapter | **35 regressed, 0 fixes, 55 unchanged, exit 1** |

The clean re-run is the half that is easy to skip and the more informative of the two:
the suite runs greedy at `temperature = 0.0, seed = 0`, and it reproduces exactly, so the
detector has **no false-positive rate on this container**. Without that, every later
regression report would be unreadable.

The perturbation's shape is what makes it credible rather than merely non-zero:
code_localisation 16 and reasoning 18 regress against maths 1, because an 8-token budget
truncates prose and a maths answer is a short number. A detector that fired uniformly
across categories would be reporting the clamp, not the behaviour.

> **Decided 2026-08-10 (T29): the baseline is committed, and it is not under `var/`.**
> It now lives at **`baselines/regression-baseline.json`** and is the CLI's default, so a
> cold clone carries the reference point rather than re-deriving it.
>
> The argument for committing was already recorded: it is 2,195 bytes — 90 booleans plus
> `adapter`, `container_hash` and `engine_commit` — and re-recording it after a change
> compares the change against itself. What settled *where* is that the objection ("it
> changes what `var/` means") is correct: everything in `var/` is reproducible output,
> and this is the reference that output is read against. So the file was in the wrong
> directory, and an exception to the ignore rule was never the fix. A `!` negation could
> not have worked anyway — `.gitignore` excludes the `/var/` **directory**, and git
> cannot re-include a file whose parent directory is excluded.
>
> **Safe by the code, not by convention:** `cli.py` treats `check_comparable`'s warning as
> disqualifying — a baseline from another container or adapter is set to `None`, the
> report is marked `comparable = False`, and a fresh one is recorded. A committed baseline
> therefore cannot report someone else's container as a regression; the worst it does is
> get replaced, which shows up as a dirty working tree and is the correct signal.
>
> `var/regression-baseline.json` is the byte-identical copy this was taken from. Nothing
> points at it any more and it is safe to delete.

### 4.6 A0, then re-gate tool calls — and restate E1

Train the harness adapter, re-run `tandem gate toolcall --runs 100`. **E1 was "A0 should
beat base on tool-call validity". On this host base now scores 1.00**, so that experiment
has no headroom left and a green result would say nothing. Constrained decoding closed the
gap the adapter was meant to close.

What is still worth asking, and what E1 should become:

- Does A0 hold 1.00 with constrained decoding **off** (`[dev]`-only install)? That is the
  honest test of whether the adapter learned the harness, and it decides whether prevention
  is load-bearing or merely convenient.
- Does A0 reduce the *repair* layer's work? The hardware gate counts every current success
  as `repaired`, because tier 0 returns raw text and extraction happens a layer up. A
  backend-parsed `wellformed` outcome would be the real improvement.
- Does A0 recover any of the 27 → 65 tok/s constrained-decode cost by making the mask
  redundant on most turns? Note that `CONSTRAINED_DECODE.md` proposes recovering most of it
  without an adapter at all, so run that comparison against F1/F2 rather than against today.

### 4.7 A1 and the merge eval — the product thesis

> **`A1_TRAINING.md` is the prerequisite list, and it is longer than "find a repo".**
> Measured 2026-08-10: `build_sft_command` passes `epochs` where mlx-lm wants `--iters`, so
> a 1,000-pair corpus trains on **3 examples** and exits 0; the corpus has no `valid.jsonl`
> and mlx-lm treats that as an empty set rather than an error; `mask_prompt` plus a prompt
> over `max_seq_length` yields a `nan` loss, 0 trained tokens and a saved adapter. And the
> configured targeting (`--num-layers -1`) peaks at **31.10 GB against a 30.15 GB ceiling**
> at the shortest sequence that exists, while real-length pairs page the machine. None of it
> is visible to the suite, which only tests `--dry-run`.

```bash
tandem extract a1 --repo <real repo> --holdout 25 --out corpus/a1
tandem train sft --corpus corpus/a1/train.jsonl --out adapters/a1-x --name a1-x --repo <real repo>
tandem eval merge --repo <real repo> --a1 a1-x --out var/merge-eval.json
```

Needs an `[eval]` block in `tandem.toml` naming the repo's linters and test command, or
three of five metrics report "not measured" and `compare_arms` correctly refuses to call it
a pass. **That is the guard working, not a failure to route around.** `tandem extract` exits
**2** on a corpus under 500 pairs — also an answer, not an error.

**If A1 does not beat base on ≥3 of 5 metrics, stop and re-plan before building tier 1.**
That is the entire reason M3 sits at week 8 — before the expensive part.

It also answers the question the DeepSeek memo opened and could not close: whether the
verifier picks the *better* candidate rather than merely a different one. No second verifier
can answer that — it needs a quality signal.

### 4.8 Determinism, then the experiments

**Done 2026-08-10, one layer below the gates — §3.12.** `tools/determinism_probe.py` answered
what G1 and G2 were built to ask, by recording per-step logits and the greedy margin instead
of comparing text: same config is bitwise identical, **CPU-vs-Metal flips the first token**
(4.375 divergence against a 1.625 margin), and **re-chunking one prompt diverges 2.031** —
which is T16's mechanism, measured. Both are platform properties; the work they imply is
disclosure (T13), not a kernel fix.

What is left is the gates themselves, and each is blocked on its own shape rather than on
hardware:

- **G1 (T33)** needs two processes compared by recorded output. Two backends cannot hold two
  devices in one process — `mlx_lm` binds its generation stream to the default device at
  import — and the arms are 2 × 23.0 GiB besides. Byte-identity is also the wrong assertion:
  30 of 40 layers run a different algorithm on CPU by design, so the useful criterion is
  margin-over-divergence, which `GateResult` has no field for.
- **G2 (T34)** needs a placement axis that reaches the engine. It had none — the gate now
  reports *not measured* rather than the guaranteed green it used to. The axis this host can
  vary is `optiq serve --stream-experts` against `--no-stream-experts` on tier 0's own 35B,
  run sequentially and compared by recorded output.

Then E2 (int8 vs bf16 adapter deltas), E3 (expert coverage at 35B), E4 (candidate count).
All three run on the baseline platform as configured.

### 4.9 The DeepSeek-V4 OptiQ download

**Complete, 2026-08-09.** All 42 shards present, **92.49 GB referenced by the snapshot**
against 92.48 declared, every symlink resolving. Started, interrupted twice and resumed in
place; re-running the same command continues from disk.

```bash
export SSL_CERT_FILE=~/.config/certs/macos-ca-bundle.pem \
       REQUESTS_CA_BUNDLE=~/.config/certs/macos-ca-bundle.pem
.venv/bin/hf download mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit
```

Resolved snapshot, which `container_path` needs and which is the thing that takes an hour to
find again:

```
~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-0731-OptiQ-2bit/snapshots/0edd7d3e70d562a0fc1d1574943ca4fe2b2c1e36
```

- **The cert exports are not optional**, and their absence is the one failure that looks
  like a network fault: Python raises `CERTIFICATE_VERIFY_FAILED` through the corporate TLS
  interception while system `curl` succeeds on the same URL. `BASELINE.md` §6 has the bundle
  procedure; the bundle already exists on this host.
- **Counting `.incomplete` blobs is the wrong completeness check, and it fails open on
  exactly this repo.** T6 closed on "zero `.incomplete`" because that download was never
  interrupted. This one was killed twice, and `hf download` leaves the abandoned partials
  behind rather than reaping them: this fetch finished with **16 orphans holding 15.87 GB
  beside all 42 complete shards.** The naive check therefore reports a finished download as
  incomplete forever.
  Check what the snapshot *references* instead — it follows the symlinks into `blobs/` and
  is indifferent to orphans:

  ```bash
  S=~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-0731-OptiQ-2bit/snapshots/0edd7d3e70d562a0fc1d1574943ca4fe2b2c1e36
  ls "$S"/model-*.safetensors | wc -l                      # expect 42, and no broken links
  python3 -c "import glob,os;print(f'{sum(os.path.getsize(os.path.realpath(f)) for f in glob.glob(os.environ[\"S\"]+\"/*\") if os.path.isfile(os.path.realpath(f)))/1e9:.2f} GB')"
  ```

  **Those orphans were deleted on 2026-08-09** at the owner's request, reclaiming 15.87 GB;
  the model verified intact afterwards at 42/42 shards and 92.49 GB referenced. Deleting
  them is safe *once the overlap is proven empty* — compare the set of snapshot symlink
  targets against the `.incomplete` set and require zero intersection, rather than trusting
  the suffix. A partial that a running fetch is still appending to carries the same suffix,
  so check no `hf download` is live first.
- **Observed throughput varied 10–42 MB/s and the cause was never isolated.** An earlier
  note here blamed the link or the intercepting proxy on the strength of two ~10 MB/s runs;
  the resume then ran at 33–42 MB/s with no configuration change, which falsifies that. Do
  not plan around either figure.
- Disk is not a constraint — ~580 GB free with the full download on it.

**Finishing the download unblocked nothing on its own, and the load has now confirmed why.**
It was loaded twice on 2026-08-09, on a host passing the §3.1 gate at 320 GB/s: it loads
(resident 6.49 GB streamed, 15.21 GB with the scales resident, load peak up to 23.80 GB)
and then **dies during prompt processing on the first request**, in every configuration
tried, while a 122B control on the same engine answers in 15 s. `BASELINE.md` §4.7 is the
record; the procedure and the warning are `PROCESSES.md` §6.

So the cost argument that deferred step 5 — Gate B derives to ≤168 tok/s against it, and it
buys verdict independence at ~8× rung 3's latency and nothing else — is still true and no
longer the binding constraint. **The next move on T7 is an engine, not a decision.**

---

## 5. Tracker

**One tracker.** This was two — §5 here and `BASELINE.md` §9 — with overlapping IDs for the
same facts, plus a third partial list in `STATUS.md` §3. `T*` ids are stable and are what
other documents cite.

Old `BASELINE.md` §9 item numbers map here: 1→T15, 2→T4, 3→T19, 4→T1, 5→T1, 6→T21, 7→T23,
8→T5, 9→T2, 10→T13, 11→T7, 12→T7, 13→T5, 14→T18, 15→T20.

| ID | Item | Status | Next action |
|---|---|---|---|
| **T18** | **The host ran at ~9% of its recorded bandwidth, and a reboot cleared it** | **cleared, cause unexplained — watch** | 23 GB/s r+w at 15 h uptime; **323 GB/s on a fresh boot**, three passes, two scripts (§3.10). Boot-scoped, so the remedy is known and the trigger is not. Ruled out: paging, thermal, low-power, a resident neighbour, competing I/O, DVFS ramp, measurement error. The unexplained part is the asymmetry — bandwidth at ~9% while compute held ~50%. **Run `PROCESSES.md` §3.1 before every measurement session and after any large load**; if it recurs, capture `mlxbench.py` immediately after the load that triggered it. **The onset test has now been run once and came back negative: 2026-08-09, `mlxbench.py` immediately after a 23.6 GB tier-0 load read 320/343 GB/s against 321/347 before it.** So "a large load triggers the degradation" is falsified for a single tier-0 load at ~4 h uptime — which does not clear the multi-hour hypothesis, since the degraded readings came at 15 h. **2026-08-10, third session: 325/347 GB/s at 14 h 44 m of uptime, and 323/346 at 15 h 21 m after six tier-0 loads and a 450 s CPU generation.** Those are the first readings taken *inside* and then *past* the ~15 h mark where the degraded readings were originally seen, and both are healthy — so uptime alone does not trigger it, which was the last hypothesis with any support. Four negatives now, and what differs between that day and this one is unidentified. The row stays open on that basis rather than on the symptom. |
| **T4** | Gate A `chat_ttft_s` fails: 30.47 s at 32k | **open, partly unreachable** | Unreachable as specified (§3.2). The cache that would move it restores 0 tokens (T15), so this cannot be read as a model problem until T15 is fixed. The number to read then is `prefix_hit` against `usage.cached_input_tokens` — gateway found an entry, backend accepted it — and the gap between them is what nobody has yet. |
| **T15** | The chat template's `<think>` tail defeats the KV state | **open, and it is the whole of step 1** | `add_generation_prompt=True` ends every prompt with `<\|im_start\|>assistant\n<think>\n`; turn N+1 puts the reply in that slot, so turn N's prompt is not a prefix of it (8037 of 8039 ids shared) and a recurrent cache cannot be rewound past the tail. Restores 0 tokens at every context size — §3.9. Fix: snapshot at the prefill seam (`prompt_progress_callback`), cutting before the tail. `supports_state()` itself is closed — it returns True. |
| **T16** | A restored KV state changed the answer at 8k | **mechanism measured 2026-08-10, still blocks attested use** | Cold-vs-cold is byte-identical, so it is the restore, not the model. The blob round-trips bit-exact, so it is upstream: the stored KV was prefilled under one chunk arrangement and the cold run's under another, and batched matmul does not promise identical rounding across shapes. **That last clause is now a number rather than a hypothesis (§3.12).** Re-chunking the same 8,190-token prompt — `prefill_step_size` 512 against 2048, nothing else changed — moves logits by **up to 2.031, median 0.688**, while the same config repeated is bitwise identical. At **7 of 65** greedy steps that divergence exceeds the step's own top1−top2 margin, and at 21 of 65 it exceeds the run's worst margin; no argmax flipped in that window, so this is measured *exposure*, not a measured flip. **What it settles:** the restore does not need a bug to change an answer, so T16 is a platform property and the remedy is not a fix but a disclosure — pin the chunk arrangement in the receipt (**T13**) and do not enable a restore for anything attested until it is pinned. Reproduce: `tools/determinism_probe.py --prompt-tokens 8000 --chunk-a 2048 --chunk-b 512 --decode-tokens 64`. |
| **T20** | Constrained decoding's 2.4× is Python waste, not a language boundary | **closed — F1 and F2 landed and measured, F3 rejected, and the residual is now T26** | Full scope: `CONSTRAINED_DECODE.md`. The `tokens.tolist()` sync is **free** (13.24 vs 13.12 ms/token synthetic); the cost is host work serialised against an idle GPU. **Landed 2026-08-09:** F1 hoists the out-of-range guard to a per-backend fact (`constrain.logit_width_bound`, which is `max(len(inner), max(stop_ids)+1)` — the length alone is *not* the bound, because stop ids are passed to LMFE separately), and F2 is a single-slot mask cache keyed on `prev is not ids and prev == ids`. **F3 is dead** — §6.1: it proposed sharing `JsonSchemaParser`, which costs **0.035 ms**, while the 78 ms is three uncached prefix-tree walks that LMFE will not cache for a JSON schema (`JsonSchemaParser.cache_key()` is None, and `allowed_token_cache` is observably empty after a full call). Sharing the enforcer instead measured 98.3 → 97.2 ms while `prefix_states` grew 43 → 86: unbounded growth for nothing. So the floor without F4 is **2.7 ms/token, not 0.9**. **Validated end to end 2026-08-09, and the result is small.** The ladder is complete on a host passing §3.1 at both ends (321 GB/s before, 320 after): F1 is worth **2.99 ms/token**, F2 **0.08–0.54**, together **11–15%** — 2.55× → **2.38×**, 27.0 → **29.5 tok/s**, reproducible across three runs. `tandem gate toolcall --runs 100` **passes at 1.00, 100/100, 115.9 s**, so the fixes cost no correctness. **What the validation found is bigger than what it confirmed** — T26 and T28. The 2.4× is *not* mostly Python waste after all: ~97% of it is a single LMFE parser state, so this row's title is the part that did not survive. |
| **T26** | **~97% of constrained decoding's cost is one LMFE parser state, at ~425 ms per occurrence** | **explained, reproducible, and the fix is decided: not ours to write** | **The gap between a backslash and its escape character** inside a JSON string argument: ~5,798 of 248,077 tokens may follow, LMFE walks the prefix tree to find them, and nothing caches the answer (`JsonSchemaParser.cache_key()` is None). Two occurrences were **849 of the ~1,000 ms** of LMFE time in a 49-token generation; the 246,908-token string-interior states cost 1.19 ms, exactly what the no-weights bench predicted. **Whether a newline occupies that state is the tokenizer's choice**: `"…(req)\n"` splits to `')\'`+`'n'` and pays, `"line0\nline1"` emits `'\n'` whole and pays nothing — measured dead linear at ~428 ms per *split* escape and flat at zero without. Three candidates were tested and killed first: GPU contention (a stopwatch inside the processor accounts for 21.50 of 24.37 ms), a long prompt prefix (`mlx_lm` passes 49 tokens, not 455), and CPU down-clocking beside a busy GPU (2.55 vs 2.39 ms). **Reproduce with `tools/constrained_decode_bench.py escape` — no weights, ~1 min, within 2% of hardware.** `CONSTRAINED_DECODE.md` §8.4; the fixture re-measure is its §9 item 7. **Caching decided 2026-08-09 — §8.5, and the deciding fact is that it is *not* one state.** The two occurrences returned 5,798 and 5,795 tokens because the set is fixed by the enclosing object stack, not by the backslash: 42 more characters of string content change nothing, one unwritten property changes exactly 3 tokens — `\",`, `/",`, `"",`, each of which closes the string and emits a comma in one token. **So a key on "after a backslash" permits a comma where the object must close**, which is a wrong constraint rather than a slow one, and §6.1's principle now has a counterexample instead of only an argument. A sound key is stack-wide *and* must model `num_consecutive_whitespaces` plus a `last_parsed_string` read through a `_Context` that `get_allowed_characters` reassigns during the walk. Nothing to pick up upstream either — **0.11.3 is the current release (24 Aug 2025) and `main` still defines only `shortcut_key()`** — so the route is a contribution, with `tools/constrained_decode_bench.py statekey` as reproducer and the 3-token difference as its acceptance test. The prize is real and unclaimed: four escapes in one property are **one** distinct set, so a sound cache collapses **1,228 of 1,638 ms (75%)**, and a twenty-line edit ~8.5 s → ~0.43 s. **F4 does not substitute** — hiding 11 ms/token behind the forward pass leaves a 410 ms stall at 410 ms. |
| **T28** | **Every rung-0 conclusion rests on one fixture that lacks the feature dominating the cost** | **closed 2026-08-10 — every section re-measured, and rung 0 now predicts rung 1** | `TARGET_CALL` in `tools/constrained_decode_bench.py` is a realistic-*looking* `edit_file` call with no escape sequence in any string argument. It is the fixture behind §3, §4 and §6.1, and it under-measures the real cost by 5× (T26). **§6.1 has since been re-run against `ESCAPED_CALL` and F3 stays rejected** — sharing an enforcer moves 930.6 → 970.6 ms, recovering nothing at 9.5× the stake, with `allowed_token_cache` empty exactly as its mechanism predicts. **Closed 2026-08-10: `filter` now runs both fixtures**, reporting per parser state, pre-F1 and landed, so §3.2 and §4 are measurements rather than derivations. **The payoff is that rung 0 became a predictor**: on `ESCAPED_CALL` it reads **21.02 ms/token against hardware's 22.15** (LMFE alone within 0.5%), where the same harness on `TARGET_CALL` reads 2.97 — so the 5× was the fixture, never the machine, and a question about where constrained-decode time goes no longer needs 20.6 GiB. F1 (3.38 vs 3.15 ms/token) and F2 (1.00 vs 1.05) are confirmed fixture-independent, and the escape state is **87% of an escaped call in two tokens of 46**. The re-measure also **corrected §8.3**: F4's ceiling was `max()` taken over an *average*, and against the real distribution F4 is **~1.11×** on a call with escapes and ~1.25× on one without, not ~1.5×. The general lesson stands (§8.4): a fixture for a coding agent must contain multi-line code in string arguments, because that is what a coding agent emits. |
| **T27** | `tier0.mtp` is parsed, plumbed to the backend and never read | **open, cosmetic but misleading** | `config.py:35` → `backends/__init__.py:75` → `mlx_tier0.py:227` `self.mtp = mtp`, and nothing reads it. `tandem.toml` sets `mtp = true`, so the file claims a multi-token-prediction setting the runtime does not have. Harmless for every number measured so far — it is off in both arms of every comparison — but it is a config knob that reads as load-bearing and is not. Either wire it or delete it; a receipt implying MTP was on would be wrong. |
| **T29** | The regression baseline is the one reference point that does not survive a clone | **closed 2026-08-10 — committed at `baselines/regression-baseline.json`** | Every other gitignored artefact is reproducible output; this one is a *reference*, and re-recording it after a change compares the change against itself. **It was in the wrong directory, so the fix is a move rather than an exception**: 2,195 bytes of booleans plus `adapter`, `container_hash` and `engine_commit`, now tracked and the CLI default. A `!` negation inside `var/` could not have worked — `.gitignore` excludes the `/var/` *directory*, and git cannot re-include a file under an excluded parent, which is the trap this decision walked past rather than into. Safety is in the code: `cli.py` discards a baseline `check_comparable` rejects and records a fresh one, so a foreign container is replaced rather than reported as 90 regressions. **Same shape as trap 11**, and closed the same way `NEXT_STEPS.md` was — by moving the fact out of a directory designed not to keep facts. |
| **T2** | Gate B — decide | **closed, 2026-08-10** | `tandem bench tier1` ran against the live engine, six times: worst-of-run **152.1–155.7 tok/s**, mean **153.7**, sd **1.07**. `pass: true` against this host's floor, **`meets_spec: false`** against sec 11's 200 — rung 1 is **1.3× under the spec**, not 1.2×, because the gate's own filler reads slightly lower than `BASELINE.md` §4.1's `curl` prompts. The verdict does not change the rung decision: rung 3 still ships, and rung 1 remains slow-not-dead. §3.3. |
| **T14** | `gate_b_prefill_tok_per_s = 20.0` passes anything | **closed, 2026-08-10** | Raised **20.0 → 150.0**, which is this row's original suggestion arrived at from data rather than adopted on faith. Six runs: worst-of-run 152.1–155.7, mean **153.7**, sd **1.07**, spread 2.34%. 150.0 is 3σ below the mean and under every run observed, so a run must lose ~2.4% to trip it. Set close to the measurement on purpose — the engine falling back to 2048-token chunks is 1.75× and lands near 118, but a partially degraded host (T18) shows as a few percent, and a cushioned floor would swallow exactly that. The old 20.0 came from a published ~26 tok/s and was wrong by ~6× in the pessimistic direction. **Caveat: all six runs are one session at ~11 h uptime.** **And it is calibrated to one model**: on `Qwen3-Coder-Next` the same floor sits 42% under a measured 258.0 and would pass a chunk-size regression that lands near 190, so `tandem.qcn.toml.example` keeps 150.0 deliberately and says why rather than retuning off three runs (T7). |
| **T30** | Should `auto` recognise Qwen3 as a thinking family? | **open — owner's call, not a bug** | `tier1.model` is a Qwen3 and it thinks by default, but widening the name guess sends unknown keys to every deployment, and `test_a_non_reasoning_model_gets_no_extra_keys` records that as a 400 on a strict engine. This host sets `reasoning_control = "deepseek_v4"` explicitly and the repaired guard makes any missed model fail loudly, so nothing is silently broken either way. §3.11. |
| **T32** | **A red Gate B on this host is not the finding sec 11 designed it to report** | **open — read this before quoting `meets_spec: false`** | The 200 tok/s floor is a *proxy*, and `thresholds.py` says what for: "is the engine doing batch-union prefill? Below this the streamed tier is reading each expert per token rather than once per chunk." **§3.6 has already answered that question directly, and the answer is yes** — prefill fits `8.0 s × chunks + tokens/218`, and the 8.0 s is constant per chunk *independent of how full the chunk is*, which is per-chunk amortisation by definition. So the engine passes the test the number stands in for while failing the number. It fails because `tokens/218` is compute, ~40% of GEMM peak, making 218 tok/s this host's asymptote: **the spec's 200 is 92% of it, and at the configured `--prefill-step-size 8192` the ceiling for one full chunk is 179.7 tok/s — unreachable at any engine quality.** 200 needs a single ~19.4k-token chunk, and §4.1's fall-off at 16,992 tokens (140.1) says attention would eat that. Structurally the same trap as `gate_a_ttft_s`, where §3.2 already says to read a red as "the mitigations are not being measured". **Whether 200 is right on the larger machine the spec was written for is unknown and not checkable here** — sec 11 is not in this repository (`specs/README.md`), so nothing above is a claim about the spec's reasoning, only about what its number does on this host. |
| **T33** | **G1 is red on hardware, and cannot be a gate over two backends in one process** | **open — the answer is measured, the gate's shape is the blocker, and it is decided which way out** | **The result first: CPU-vs-Metal diverges 4.375 logits against a 1.625 greedy margin and the argmax flips on the first generated token** (§3.12) — so this gate fails here, and it fails for a reason no fix of ours addresses. Three measured reasons the gate could not have reported that itself: two live tier-0 arms are 2 × 23.0 GiB against 28.08 GiB; `mlx_lm.generate` binds `generation_stream` to the default device **at import**, so a caller's stream context is silently overridden (the first CPU arm returned Metal's logits in 0.1 s, a clean-looking vacuous pass); and swapping devices means mutating that module global, which is backend-global state §6 forbids. **Route: two processes, compared by recorded output** — the same record-then-compare shape §4.5's regression baseline uses, and the shape the isolation gate was just converted to. Note what a red would then mean: CPU and Metal run *different algorithms* for 30 of 40 layers (`gated_delta.py:281`), so byte-identity is unavailable and the useful assertion is margin-over-divergence, which the probe reports and `GateResult` has no field for. Cost of the specified defaults on this host: ~3,760× Metal, so 4 prompts × 128 tokens is multi-hour. |
| **T34** | **G2's two arms were the same arm** | **open — the gate now refuses instead of passing; the axis it needs is named** | `g2_placement_invariance` varied `tier1.expert_cache_bytes`, which `expert_cache_provenance` already records as reaching no engine: nothing sends it, and `--stream-experts-cache` is a per-projection count that mlx-optiq 0.4.18 drops before `_ShardWeightReader`. Both arms were therefore one engine at one placement — a **guaranteed pass** on the gate whose failure would invalidate every other claim, which is strictly worse than never running it. Against the pre-fix code the new test reads `passed=True, reason='ok'`. Fixed to report **not measured**, built off that same provenance function so one place records the fact and the gate starts measuring the day the plumbing lands. **To actually measure it here:** tier 0's model is itself an MoE, so run `optiq serve` on `Qwen3.6-35B-A3B-OptiQ-4bit` twice — `--stream-experts` against `--no-stream-experts`, greedy, sequentially (23 GB resident, so never co-resident) — and compare recorded output. Caveat to record with the result: those two arms differ in *kernel* as well as placement (`StreamedSwitchLinear` against the fused path), so a red would not by itself mean RAM-vs-disk changed the answer. |
| **T31** | Gate B measures ~1.38× the frontier it asks for | **open, low** | `measure_prefill(4_000)` sent **5,433** tokens; 8,000 → 11,035; 16,000 → 22,208. `prefill_filler` sizes by characters at 4 chars/token and this filler runs ~2.9, so every frontier overshoots ~36–39%. The *rate* is unaffected — it divides by the engine's own reported `prompt_tokens` — so no published number is wrong. What it risks is a frontier chosen to sit under a context cap silently exceeding it: at `--max-context 32768` a 24k frontier would deliver ~33k and the engine rotates the KV window rather than failing, which would corrupt the measurement invisibly. §3.3. |
| **T13** | Prefill step size is invisible to Tandem | **open — promoted 2026-08-10 from cheap to load-bearing for the determinism claim** | `--prefill-step-size 8192` is worth 1.75× and lives only on the engine's command line. `gate_b_report` records model and expert-cache size so a streamed rate is reproducible; without this one it still is not. **On DeepSeek-V4 it is a correctness knob rather than an optimisation** — at the 2048 default a 30k review exceeds `request_timeout_s` (`BASELINE.md` §4.4). **And it is now the one parameter measured to change the model's logits** (§3.12, T16): chunk arrangement alone moves them up to 2.031, past the greedy margin at 7 of 65 steps. A receipt that claims determinism while omitting the chunk size omits the knob most likely to break the claim — so this is no longer only a reproducibility nicety. Tier 0's own `prefill_step_size` is `mlx_lm`'s 2048 default and is equally unrecorded. |
| **T7** | A larger streamed MoE for rung 1 | **answered again 2026-08-10, and the second answer is the useful one: it is not DeepSeek. `Qwen3-Coder-Next-4bit` serves, and passes Gate B at spec — `QWEN3_CODER_NEXT.md`** | **Gate B 258.0 tok/s, `meets_spec: true`** — the first configuration on this host to clear sec 11's 200 floor, against the 122B's 153.7 and DeepSeek's 48. Three runs, worst-of-run 262.6 / 257.9 / 253.5, spread 3.5%. **Resident 1.36 GB, load peak 1.36 GB** (no spike; making the 4.84 GB of meta resident costs 6.22/11.02 and buys 1.08×). Decode 8.96 tok/s. 44.84 GB on disk, 3 B active of 80 B, `qwen3_next` — the same architecture family as tier 0, one size up, on stock mlx-lm with no optiq patch. **It serves through `optiq serve`, which DeepSeek never has.** What it does *not* do: beat rung 3 (a 30k review is ~130 s against ~33 s, so the price of independence falls from ~6–9× to **~3.5×**, and rung 3 still ships), or beat tier 0 on published coding quality (SWE-bench Verified 70.6–74.2 against 73.4; **Terminal-Bench 2.0 36.2 against 51.5**). **The one question that now matters is verdict decorrelation, and it is unmeasured** — T5. Two derivations died on contact and are recorded in §6 of that doc: prefill was projected 26% high by scaling the compute term with active parameters, and the model was predicted batchable but takes the sequential path anyway, which leaves `DEEPSEEK_V4.md` §2's mechanism without an explanation. |
| **T7 (DeepSeek)** | The 2026-08-10 DeepSeek record | **superseded as the answer to T7, still the record for that model: `DEEPSEEK_V4.md`** | **The model runs.** `optiq.runtime.moe_stream.load_streaming` + `stream_generate` on the **main thread** has never failed; the abort is `optiq serve`'s, and the identical call chain (`stream_generate` → `deepseek_v4` → `moe_stream`) completes off the server. Measured: prefill **47.9–49.4 tok/s** flat from 1.6k to 12.8k tokens, decode **2.54–2.87**, resident 6.49 GB, peak **14.15 GB** at 12,776 tokens. **Against the 122B's 165 tok/s prefill and 4.05 decode that is 3.4× and 1.6× worse at 2× the disk**, so every cost argument that deferred this row survives and is now measured rather than derived — and Gate B would read 48 against this host's 150 floor. **Memory is not the context ceiling**: peak moved 0.04 GB between 6.3k and 12.8k because `compress_ratios` keeps most layers on a pooled state. Time is: ~48 tok/s puts `request_timeout_s = 300` at ~14k tokens. Its verdict quality got its first datapoint and it is cautionary — a fluent, well-formed, *wrong* answer on the one question this session had ground truth for (`DEEPSEEK_V4.md` §4). **The crash mechanism is half solved and the recorded hypothesis is disproven**: MLX 0.32 binds a plain `Stream` to the thread that resolved it while a `ThreadLocalStream` is portable (measured, four cases), so mlx-lm's thread-local `generation_stream` is *correct* and not the fault; what remains open is which frame resolves the plain `Stream(gpu, 1)` the abort names. The 122B escapes it by being batchable. **Rung 3 still ships.** |
| **T7 history** | The 2026-08-09 record, kept for the cost arithmetic that is still the reason this is not urgent | **superseded wherever it disagrees with the row above** | `optiq serve --stream-experts` aborted during prompt processing on the *first* request, before any token, in three configurations (scales streamed, scales resident, `--max-concurrent 1`), identical each time; **not memory** — headroom flat at ~23 GB, pageouts unmoved. A `Qwen3.5-122B-A10B-OptiQ-2bit` control on the same engine, flags and stack answered in 15 s. Loading settled: resident **6.49 GB** streamed / **15.21 GB** with scales resident, load peak **6.49 / 23.80 GB**, so `fast_quantized_load` holds and the resident configuration's peak is 3.7× its steady state. **Two claims here did not survive 2026-08-10:** the mechanism pointer ("a thread-local `generation_stream` against optiq's 24-thread reader pool") is *disproven* — a `ThreadLocalStream` is the portable kind — and "next action is not ours" was wrong, since driving the model single-threaded needed no engine upgrade. **The cost argument, still true:** `…-2.4bit-mixed` was costed without downloading it (`BASELINE.md` §4.4) at decode **2.7 tok/s** and Gate B **≤168 tok/s**, and the measured OptiQ-2bit is worse than that derivation on prefill (48). A 30k review is ~270 s against rung 3's ~33 s. **8× for verdict independence alone.** Still unmeasured: what `container_hash` costs over 92 GB (derived ~60 s) and whether `OPTIQ_STREAM_SCALES_BUDGET_GB` pays its derived ~33% of decode — both now testable, since the model generates. |
| **T35** | **`max_buffer_length` binds prefill, and nothing guards it** | **open, new 2026-08-10 — a ceiling `BASELINE.md` §1 recorded as never having bitten** | A 52,008-token prompt to `Qwen3-Coder-Next` aborted the engine: `[metal::malloc] Attempting to allocate 23622320128 bytes which is greater than the maximum allowed buffer size of 22613000192 bytes` — Metal's **21.06 GiB single-array limit**, which §1 had foreclosing only a single-tensor-per-model loader. **It is not a memory-pressure failure**: resident was 1.36 GB, headroom never moved, and `total − active` would have reported the machine healthy, which it was. Three things make this worth a row rather than a footnote. **`--max-context` is not a guard** — the engine accepted a prompt 59% over a configured 32,768 and died inside the forward pass instead of refusing it. **The frontier is unbisected**: 21,008 tokens is fine and 52,008 is fatal, and nothing between has been tried. **And the same shape as T31 is how it got sent** — sizing prompts by characters or words rather than tokens, where this tokenizer runs ~11.8 tokens per generated identifier. Anything that raises a context frontier on a streamed model should bisect this first; the model's own `max_position_embeddings` is 262,144 and is not reachable here. |
| **T5** | No rung supports two co-resident backends, and rung 3's verdicts are not independent | **open, downgraded — the prerequisite is measured, and as of 2026-08-10 the price is ~3.5× rather than ~8× (T7)** | Rung 3 disagrees with tier 0's own top candidate in 9 of 12 tasks and its choices survive rotating the candidate order (10/12 against 0.2 by chance), so a fifth rung for decorrelated verdicts is **not urgent** — `BASELINE.md` §4.5. Three caveats keep it open: the measurement is the degenerate case (no `adapters/`, so `_strip` is a no-op); reproducible ≠ **better**, which only the merge eval can answer (§4.7); and co-residency (e.g. `gemma-4-26B-A4B` at 17.5 GiB + `Qwen3.5-9B` at 7.6 = 25.7 of 28.08) is a new rung rather than a config change, since rung 2 evicts rather than co-resides. |
| **T21** | `tandem gate isolation` has never run on hardware | **open — but no longer over the memory ceiling, 2026-08-10** | Needs mounted adapters, and `adapters/` does not exist, so the gate short-circuits — which is why it is harmless today. Real quantised deltas and Metal are exactly what it is for and exactly what the fake cannot reach. **Until it passes, treat every receipt naming an adapter as unproven.** **The 2 × 23.0 GiB warning is now stale**: the gate ran the N-mounted arm against each solo arm inside one `asyncio.gather`, so two were live at once against a 28.08 GiB ceiling and it could not have run here at all. It now runs the N-mounted arm to completion, records ≤128-token strings, releases it via the rung-2 `unload()` seam, and only then builds each solo arm — same comparisons, **one arm live**, pinned by `test_isolation_gate_holds_one_arm_at_a_time` (which fails against the old code). Step 4 is therefore no longer blocked on memory, only on an adapter existing. |
| **T22** | G1/G2 determinism gates have never run on hardware | **the question is answered, the gates are not run — split into T33 (G1) and T34 (G2), 2026-08-10** | Same-config repetition was already known deterministic (90 greedy items reproducing exactly, §4.5); that is confirmed one layer down — repeated arms are **bitwise identical in logits**, not merely in text. What G1/G2 vary has now been measured directly with `tools/determinism_probe.py` (§3.12) rather than through the gates: **device costs 4.375 logits against a 1.625 margin** and **chunk arrangement costs 2.031**, both against a reference that reproduces itself exactly. So the determinism claim's real content is known: it holds per configuration and does *not* survive a change of execution path, which is a disclosure question rather than a bug. **What remains is the gates themselves**, and each has its own blocker: T33 for G1 (needs two processes), T34 for G2 (needs a placement axis that reaches the engine). Neither is "nobody got to it" — the original shapes could not have produced a valid answer on this platform, and G2's would have produced an invalid green. |
| **T19** | Gate A scores a tool-call failure rate it never measures | **open, cheap** | `--toolcall-failure-rate` defaults to 0.0, which passes. Feed it `tandem gate toolcall`'s number, or report *not measured*. |
| **T11** | Gate A's decode budget did not know constrained decoding exists | **closed by config, open by measurement** | Free-form decodes at 65 tok/s and passes; tool-bearing turns run at 27 and would not. `gate_a_decode_tok_per_s` now makes the budget explicit per host, but **Gate A still measures only the unconstrained arm**. The real fix is measuring both and saying which the budget applies to. Do not "fix" it by relaxing further — the 27 tok/s buys 100 first-attempt tool calls against 0. |
| **T1** | `Environment` reported `gpu_cores: 0` and `memory_bandwidth_gb_s: 0.0` in every receipt | **half closed** | `gpu_cores` is detected from the IORegistry (`ioreg -k gpu-core-count`, ~20 ms), keyed on the property rather than the `AGXAccelerator` class name; unknowns are `null`, never `0`, because `0.0` read as a measurement of zero. `var/gate-a.json` predates the fix and still shows `0`. **Still open:** bandwidth is not queryable and there is no config path to supply the measured figure. Deliberately not inferred from the core count — a lookup table would be indistinguishable from a measurement in the report. |
| **T24** | The routing-profile forward pass | **open** | `tandem profile` builds and validates the sidecar from an activation dump; producing that dump needs the real model. |
| **T23** | Rung 2's second occupant | **open** | Needs a resident 80B verifier that does not exist. `build_tier1` says so on `backend="mlx"` rather than building a swap with nothing to swap into. |
| **T25** | Experiments E1–E4 have not run | **open** | E1 needs restating first (§4.6) — base already scores 1.00, so the experiment as written has no headroom. |
| **T9** | Little memory margin at rest | **watch — binding for tier 0, not for a streamed rung 1** | Tier 0 at 23.0 GiB against 28.08 leaves no room for a browser, and needs ~27 GB by the `total − active` measure. Headroom read **22.7 GB** with the DeepSeek download running and **23.6 GB** after it stopped — still short, so anything that builds tier 0 wants a fresh boot, with free memory recorded at the start. **A streamed rung-1 load is a different question and this does not block it** — confirmed by running one at 23.4 GB of headroom: optiq's default scales budget (the engine prints 3.9 GB) is under this quant's 9.3 GB of scales and biases, so the meta streams and the steady state is **6.49 GB measured**. The load peak is measured too — **6.49 GB streamed, 23.80 GB with the scales made resident** (`BASELINE.md` §4.7). Plan against the peak, not the steady state. **Tier 0's own peak is now measured directly rather than guarded against: `/usr/bin/time -l` reports a 23.6 GB peak memory footprint for `tandem gate toolcall`, and across the 100-run gate the `total − active` headroom troughed at 21.2 GB and held 27.1 GB in steady state from a 31.3 GB start.** So the ~27 GB rule of thumb is the right *precondition* and 23.6 GB is what it is buying room for. |
| **T10** | Hardware ceiling | **open** | Rung 1 is not closed on the baseline platform: 165 tok/s of prefill, and a 30k review fits inside `request_timeout_s`. Rung 3 ships because it is 6–9× faster and costs no memory. A bigger machine would make rung 1 fast, not possible. |
| **T8** | Streaming reads must use queue depth ≥12 | **informational** | If any in-house streaming loader is written: queue depth is worth 4× (1.70 → 6.93 GB/s at 1 MB), and any benchmark must set `F_NOCACHE` **and** bracket with `iostat`, or it measures the page cache. |
| **T3** | `max_kv_tokens` was ~50× more conservative than the budget allows | **closed** | Raised 16384 → **65536** (1.25 GiB fp16). Stopped at 4× rather than 8× because prefill time binds before KV memory does. Raise further against a `mx.get_peak_memory()` reading, not against a comment. |
| **T6** | `Qwen3.5-122B-A10B-OptiQ-2bit` download | **closed** | 45.94 GB declared, 45.94 GB present. It loads and serves, and is the proxy every tier-1 streaming number comes from. **The "zero `.incomplete` blobs" check this closed on is unsound** and happened to work only because that download was never interrupted — §4.9 and trap 12. Re-verify by what the snapshot references if it ever matters again. |
| **T12** | Reasoning-off dialects | **closed by fix** | On mlx-optiq 0.4.18 both spellings `_disable_thinking` sent were ignored; only `chat_template_kwargs.enable_thinking` works. Fixed, measurement in the docstring. `refuse_reasoned_answer` is what held the invariant meanwhile — the response-side half §6 forbids removing. |
| **T17** | Decode measured 2.9 tok/s against `BASELINE.md`'s ~65 | **closed** | Not a call-shape artifact and not Gate A's: it was T18's host-wide degradation, and a reboot restored 323 GB/s. §3.10. |

---

## 6. Things not to undo

Each looks simplifiable and is not. Every one has a comment in the code saying so, and
every one fails as a **silently wrong answer rather than an error**.

### Adapters and state

- **Adapters are never merged into the base** (sec 4.2). Merging costs ~20 GB per adapter
  instead of ~250 MB, kills multi-tenancy, and leaves the receipt unable to name what
  produced a change.
- **Adapter selection and restored KV state are per-request, never on the backend.**
  Backend-global state races under concurrency.
- **A KV state carries the identity it belongs to** (`Backend.state_key`). Restoring one
  built under a different container or adapter gives fluent output from the wrong model and
  a receipt naming the adapter that did not produce it.
- **The disk KV cache uses plain `read`/`write`, never mmap** (sec 8.4). A process already
  mapping ~30 GB of weights should not add more VM mappings. **This extends to the blob
  itself**: `backends/mlx_kv.py` serialises a prompt cache by hand rather than through
  `save_prompt_cache`, whose `mx.load` maps a file. Routing the state through it would undo
  the rule while leaving `kv_disk.py` looking untouched.
- **A KV state names every token it holds, and the backend re-checks them against the next
  prompt.** The identity key answers "same model, same adapter" and says nothing about the
  bytes. Restoring on the key alone continues a conversation from a prefix that was never in
  it.
- **`export_state` does not require a trimmable cache.** 30 of the container's 40 layers are
  recurrent and cannot be rewound, so the version that trimmed or refused was a no-op on the
  only model this ships against — §3.8. Whatever replaces it must keep a path for the cache
  it cannot rewind.

### Tier 1

- **Tier 1 has no `generate` entrypoint** and clamps `max_tokens` per call type (sec 5.1).
  The model that reranks five candidates in 18 s takes six minutes to write one.
- **Tier 1 never reasons, and a reasoned verdict is refused rather than read.** Both halves
  are load-bearing and not redundant. The request-side flag is a guess about which spelling
  the engine reads, and on mlx-optiq 0.4.18 two of the three documented spellings were
  measured doing nothing; the response-side check is an observation of what the engine did.
  Dropping it because the flag "already handles it" restores the failure for exactly the
  models nobody thought about — a reasoned verdict spends the sec 5.1 clamp before the JSON
  exists, and thinking mode discards `temperature` without erroring, so the receipt attests
  greedy to what was a sample.
- **`rerank_schema(n)` bounds the choice to the candidates on offer.** Without the `maximum`,
  a constrained decode can still name a candidate that does not exist.
- **Rung 3 strips the adapter.** An adapted model judging its own candidates is asking
  whether it agrees with itself.
- **The rung is selected, never descended.** No error path picks a different rung from the
  one the config names. A ladder that reached rung 4 on a timeout would turn an airgapped
  runtime into an exfiltrating one with nobody choosing it.
- **Rung 4 needs the consent sentence, and its HTTP lives outside the package.**
  `tier1.remote_consent` must read "tier 1 leaves this machine" verbatim — a rung name is one
  word copied from a README.
- **Rung 2 guards tier 0** (`Pipeline` wraps it in `SwapGuard`). Residency is exclusive; an
  unguarded tier 0 is asked to generate from evicted weights, and on the mock that failure is
  invisible.
- **Rung 2's swap back is lazy**, and its budget guard reads a measured round trip rather
  than a configured estimate. Eager restore adds ~10 s to the tail of every verified turn;
  declining on a guess disables verification for a cost nobody observed.

### Rendering and tool calls

- **A backend handed `json_schema` must apply it to the logits, and the fake must apply the
  processors it is handed.** Both halves, because dropping either is silent. This failure
  already happened once and cost the sec 10.2 gate 0.19 of its rate and every first-attempt
  tool call — §3.1.
- **`Backend.renders_canonically()` is asked of the object, not the type.** A wrapper that
  delegates `render` changes none of the bytes; a `type(...).render` check reads it as a real
  chat template and silently drops replay-aware rendering (sec 8.5.5).
- **Every backend is *handed* the replay renderer**, including the ones with their own chat
  template. A backend left to find the map itself cannot reach it, and tier 0 did exactly
  what that leads to: it dropped tool calls out of the prompt entirely.
- **Tool calls go into message content, not a structured `tool_calls` field.** Handed a
  parsed call, a chat template picks its own serialisation, and the template's bytes are not
  the bytes the model sampled — the prefix diverges at the first tool call and never
  recovers.
- **The tool-replay map is not optional** (sec 8.5.5).

### Reporting

- **`SourceKind` is a closed enum** with no member for another model's outputs.
- **The regression report has no score field** (sec 10.3) — asserted by a test.
- **A relaxed gate threshold never removes the spec figure** (`tandem.thresholds`). There is
  deliberately no way to express "relaxed" that does not also publish the spec value it
  departed from.
- **`tools/` is outside the package** — `export_reviews.py` and `remote_tier1.py` both — so
  the offline claim is structural rather than a promise. A test pins the package's network
  surface to one file and a second import there is a test failure, not a discovery.
- **A gate's comparison arms run in sequence and the first one is *recorded*, never gathered.**
  `asyncio.gather` over two arms reads as the obvious concurrency win and is what made both
  the isolation gate and G2 unrunnable on the baseline platform: two live tier 0s are
  2 × 23.0 GiB against a 28.08 GiB ceiling, which wedges the machine rather than failing.
  Recording costs a dict of ≤128-token strings. Two tests pin one-live-arm and fail against
  the gathered shape.
- **A gate that cannot vary its own independent variable reports *not measured*, never a
  pass.** G2 varied `expert_cache_bytes`, which reaches no engine, so both arms were one
  engine at one placement and the byte comparison was green by construction — on the gate
  whose docstring calls it the most important in the product. It now refuses, keyed off
  `expert_cache_provenance` so the fact lives in exactly one place and the gate starts
  measuring the day the plumbing does. A vacuous green is worse than a gate nobody ran.

Three documented deviations from the spec, each preserving its intent, are in `STATUS.md`
§2 — the largest being that merge commits are not skipped unconditionally, because on a
merge-commit repository that drops ~90% of usable history (measured: 10 pairs vs 131 on
`pallets/click`).

---

## 7. Traps this repo has already fallen into

The first four were caught by CI or by running the thing, never by local unit tests. The
fifth was caught by nothing at all for months, which is the point of it.

| # | Trap | Lesson |
|---|---|---|
| 1 | **Unanchored `.gitignore` patterns.** `adapters/` matched `src/tandem/adapters/` and silently excluded the entire adapter pipeline from the first commit — on disk, importing fine, simply not in the repository. | Anchor new patterns with a leading `/`. |
| 2 | **The mock must never be easier to satisfy than the real thing.** Failed twice: ignoring `const`/`anyOf` (inventing tool names a constrained model could not), then ignoring `minimum`/`maximum` (picking candidate indices that did not exist). Then a third time in the other direction — the mock honoured `json_schema` and the *hardware* did not (§3.1). | When extending `MockBackend`, ask whether the change makes it more permissive than a real backend — and whether a real backend actually does what the mock assumes. |
| 3 | **Run both install states.** `[dev,constrain]` and `[dev]` alone exercise genuinely different paths — prevention versus repair. | CI runs the first; run the second before claiming both work. |
| 4 | **Not every 403 is a rate limit.** The exporter retried policy refusals five times over ~60 s and then blamed throttling, when the answer was in the first response body. | Read the body before classifying the status. |
| 5 | **"Correct on inspection" is not a state a file can stay in.** `mlx_tier0.py` was 416 lines nothing had ever imported, and inspection had signed off on a `render` that dropped every tool call and tool result on the floor — the model blind to what it had already called, and two conversations differing only in tool history hashing to one KV cache key. | Where a real dependency is unavailable, model it and run the code; where it is not needed at all (`mlx_tier1.py` imports no MLX), notice that before filing the file under untestable. |
| 6 | **A projection that skips the premise.** Gate B was closed on a "40× gap" built by comparing a *decode* rate to a *prefill* floor — the two quantities §1 fact 1 exists to distinguish. Every input was measured and correct; they were divided into each other wrongly, and three published figures were cited as confirming it. They confirmed the decode rate, which was never in doubt. | Arithmetic inherits none of the credibility of its inputs. Name the two quantities and check they are the same kind, and check what a citation actually confirms. |
| 7 | **A wrapper's `--help` is not its argument surface.** `optiq serve` forwards what it does not recognise to `mlx_lm.server`, so `--prefill-step-size` was always there and worth 1.75×. Its absence from `optiq --help` was recorded as evidence the engine did no batch-union sweep at all. | When a tool documents "extra arguments are forwarded", read the callee's parser. An absent flag is not an absent mechanism. |
| 8 | **A rate quoted without its null.** T5 asked for rung 3's disagreement rate as *the* number deciding a 92.83 GB download. It measures **0.75** — and a reranker choosing uniformly at random over 5 candidates scores **0.80**. The headline number could not have distinguished a working verifier from a coin, and it was the metric the plan named. What actually settled it was a control nobody had asked for: rotate the candidate order and see whether the choice follows content or slot (10/12 against 0.2 by chance). | Write the null beside the measurement *before* running it. For an N-way choice the null disagreement rate is 1 − 1/N, which is high enough to look like success. Related to trap 6: both are correct inputs compared against the wrong reference. |
| 9 | **A stand-in that could only build the easy case.** Sec 8.4's export was written to trim a KV cache back to the keyed prefix, and every cache `fake_mlx` could build was trimmable, so the suite was green. The container it ships against is hybrid — 30 of 40 layers recurrent, `is_trimmable()` False — and the export would have refused on every turn, on hardware, silently. Found by reading the container's own `config.json`, which costs nothing and loads no weights. | The rule that the mock must not be *easier* than the real thing has a second half: it must not be **narrower**. Ask what shapes the real dependency produces that the stand-in cannot, and go and read the artefact rather than the class it is nominally an instance of. |
| 10 | **A slow number read as a code regression.** 2.9 tok/s against a recorded 65 was carried for a day as "whoever re-runs Gate A owns this". It was the host: a benchmark that loads no model at all read 23 GB/s against 247, and a reboot restored it. | A throughput reading that disagrees with a recorded one by more than ~2× is a host-state question **before** it is a code question. `PROCESSES.md` §3.1 is one command and answers it. |
| 11 | **A committed document citing a gitignored file.** `BASELINE.md` pointed at `specs/CONSTRAINED_DECODE_SYNC.md` for "full scope, measurements and design" — a reference to something that does not survive a clone. The same mistake as `specs/NEXT_STEPS.md`, made again eight commits later, and **a third time** in `BASELINE.md` §8: three benchmark scripts named as the way to reproduce §2.1, §4.4 and §4.6 while living only in `specs/bench/`. All three are now `tools/`, and `/specs/` holds no file any committed document cites. | If a committed document needs it, commit it. Prose is not the only kind of citation — **a reproduction command is one**, and a measurement whose script died with the container is not reproducible. |
| 13 | **A difference inferred from a total, then used to explain why a measurement would not transfer.** The DeepSeek costing said the streamed *meta* was what made this model unlike the 122B proxy every tier-1 number came from — "the proxy kept its ~1 GB of meta resident and never paid this" — and built a 23%-of-decode correction on it. Nobody had measured the proxy's meta; it was inferred from its 3.46 GB resident total. Starting the engine prints the answer in one line: **7.2 GB, streamed**. The two models were alike in the respect the analysis was built on being different. | An inferred quantity is not evidence, however reasonable the inference, and it is most dangerous when it explains a gap you already expect. The check cost one command. **Run the control before writing the theory** — the same 15-second 122B run also turned "streaming is broken on this host" into "this model does not run", which is a different problem with a different owner. |
| 14 | **A green local run and a red CI, for seven consecutive builds.** `ruff format` formats Python inside ```` ```python ```` fences, so `.md` files are part of the format check — and `CLAUDE.md` says to format *touched paths*, which everyone read as the `.py` ones. One doc's comment alignment failed `ruff format --check --diff` from 2026-08-09 19:51 onward while `ruff check`, `mypy`, both test matrices and the smoke job stayed green the whole time. Nobody looked, because the local `pytest`/`mypy`/`ruff check` triple was clean and the failing step was the fourth. | Run the *whole* command CI runs, not the per-file one the hook runs — `ruff format --check` with no path argument takes a second. And when a job goes red, read which **step** failed before assuming it is the code: six of the seven runs had every other job green. |
| 15 | **A new import passing locally because the platform-only extra brings it in.** `tools/determinism_probe.py` imports numpy, which arrives transitively under the Apple-only `[mlx]` extra — so it resolves on the M4 Max and is simply absent on every CI box. Local `mypy` was clean and both CI `mypy` jobs failed on `import-not-found`. The numpy override already existed but carried `follow_imports` only, on the explicitly stated grounds that *nothing in the repo imported numpy*; the new file made that comment false and the override insufficient in the same commit. | Before adding an import to `tools/`, ask which extra installs it and whether CI has that extra. This is trap 14's shape one layer over: the local run and the CI run are not checking the same tree, and here they could not be — the machine that can run the code is the one where the check passes. When a comment states the premise of a config choice, changing the premise is part of the edit. |
| 12 | **A completeness check that can never pass.** "Done means zero `.incomplete` blobs" was written into §4.9 as *the* status to quote, on the strength of T6 — where it had worked. It works only for a download nobody interrupted: `hf download` does not reap the partials a killed run leaves, so the 92.48 GB fetch finished with **16 orphans holding 15 GB** beside all 42 complete shards, and the check called it unfinished permanently. The replacement asks what the snapshot *references* — follow the symlinks into `blobs/`, count the shards, sum the realpath sizes — which is indifferent to litter. | A check inherits its soundness from the failure it was tested against, not from the one case where it passed. Ask what the check does when the thing it measures is *dirty*, not only when it is clean — and prefer measuring what is present over the absence of what is not. |

---

## 8. Repo facts

- Remote: `github.com/nodenova/orbit`, **public**. `main` is the only branch and the
  default; commit and push directly to it. Do not open a PR unless asked.
- **A session cannot delete a branch or change repo settings.** The git proxy refuses
  `push --delete`, and the REST API answers *"Write access to this GitHub API path is not
  permitted through this proxy"*. Pushing commits works fine. Anything else needs a local
  clone or the GitHub UI. Ask rather than burning a turn rediscovering it.
- **`/specs/` is the ignored directory.** Put a local copy of the v1 specification there and
  nothing else. Anything a future session needs belongs here or in `docs/`, written so that
  publishing it is fine — this repository is public.
- **`var/` is output; `baselines/` is reference.** Anything under `var/` is reproducible by
  re-running the thing that made it and is gitignored. `baselines/` holds what a run is
  *compared against*, which is not reproducible by definition, so it is tracked — T29. A
  file cannot be re-included from inside `var/`: git will not un-exclude a path whose parent
  directory is excluded.
- **Each fact has one home** (`docs/README.md`). Numbers live in `BASELINE.md`, the spec→code
  map in `STATUS.md`, what is safe to start in `PROCESSES.md`, and state, plan and tracker
  here. A fact stated in two places is a fact that will disagree with itself, and it did:
  three copies of the gap list, two trackers, four assertions that `supports_state()` was
  False after it had been True for a week.
- **Keep this file true.** It is the only thing standing between a fresh session and
  re-deriving every decision above from scratch. A doc that is 80% right is worse than none,
  because the wrong 20% is indistinguishable from the rest.
