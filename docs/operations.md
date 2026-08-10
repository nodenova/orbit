# Process and memory reference

| | |
|---|---|
| **Purpose** | What may hold the GPU on this host, and the checks to run before anything loads weights. |
| **Answers** | "Is it safe to start this?" · "What is already resident?" · "Which commands load 23.0 GiB?" |
| **Does not answer** | Why the numbers are what they are (`platform.md`). |
| **Applies to** | The baseline platform (`platform.md`). A machine with more unified memory changes the arithmetic in §1 and §3. |

**The operating rule, and nothing enforces it:**

> **One process holding model weights at a time.** Two is not slower, it is a wedged
> machine — tier 0 is 23.0 GiB of a 28.08 GiB Metal working set, 82%, and nothing else
> fits beside it.

No component checks whether another one already holds the GPU. The check is
procedural and belongs to whoever starts the second process.

---

## 1. Inventory

| Process | Port | Started by | Orbit talks to it? | Idle cost | Serving cost |
|---|---|---|---|---|---|
| `ollama serve` | 11434 | Ollama.app, at login | **No** | ~0 (no model resident) | 17–23 GB, whichever model is asked for |
| Ollama.app helper | 49157 | Ollama.app | No | ~0 | — |
| `orbit serve` | 8080 | you | it *is* Orbit | — | 23.0 GiB, loaded at startup |
| `mlx-optiq --stream-experts` | 8081 | you, manually | rung 1 only | — | 3.46 GiB resident + streamed experts (122B) |

Resident cost is a property of the model, not of streaming: the 122B measures 3.46 GB,
DeepSeek-V4-Flash is 6.49 GB, and 15.15 GB if its expert scales are made resident
(`platform.md` §4.4). None of them co-resides with tier 0's 23.0 GiB.

Ports 8080 and 8081 come from `orbit.toml` (`[server] port`, `tier1.endpoint`).

### 1.1 ollama is a neighbour, not a dependency

Nothing under `src/orbit/` opens a connection to 11434, and nothing can: the package
makes no outbound network call at all, which `tests/test_export_reviews.py` pins. Tier
0 loads through `mlx_lm.load()` on a snapshot directory (`backends/mlx_tier0.py`) and
tier 1 rung 1 is an httpx client against `tier1.endpoint` on loopback.

It is still the most likely way to wedge a run, because it is resident at login and
answers anyone who asks:

- Idle it holds nothing, and the daemon itself is a few MB.
- One request loads a model into the same 28.08 GiB pool Orbit wants, and holds it
  for `OLLAMA_KEEP_ALIVE` (unset here, so the 5-minute default) after the last token.
- **The requester need not be you.** An editor extension, a shell alias, a background
  tool configured against 11434 — any of them can pull 17–23 GB in while tier 0 loads.

```bash
curl -s http://127.0.0.1:11434/api/ps      # {"models":[]} means idle — fine, leave it
ollama stop <model>                        # evict one without killing the daemon
osascript -e 'quit app "Ollama"'           # or take the whole thing down
```

Idle is fine. Do not kill it reflexively; do check it.

### 1.2 mlx-optiq must NOT be running here

`orbit.toml` sets `rung = "second_opinion"` (rung 3), which serves the verifier from
tier 0's own weights with the adapter stripped. It costs no memory and needs no second
process. `tier1.container_path` and `tier1.endpoint` are still filled in deliberately —
a resolved snapshot takes an hour to find again — but they are inert at rung 3.

Starting mlx-optiq on 8081 anyway gets ~12 GiB resident that nobody reads, next to
tier 0's 23.0. **Orbit never spawns this process** (`backends/mlx_tier1.py` is a
client), so a stale mlx-optiq from an earlier experiment survives every Orbit restart
and shows up only as memory pressure.

Rung 1 is off because rung 3 is 6–9× faster and costs no memory, not because it cannot
serve: streamed prefill measures 165 tok/s by `curl` and 153–154 by Gate B itself, against
sec 11's 200 floor. `platform.md` §4 and §4.1a.

**The one time it should be running is Gate B**, which needs the engine on 8081 and — since
2026-08-10 — loads no tier 0 beside it. Whole procedure, ~5 min:

```bash
S=~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-122B-A10B-OptiQ-2bit/snapshots/1081c89e171447884d329abd7232566ebd38737c
HF_HUB_OFFLINE=1 .venv-optiq/bin/optiq serve --stream-experts --model "$S" \
  --host 127.0.0.1 --port 8081 --max-context 32768 --prefill-step-size 8192 &
# expect: `swapped 144 expert projections; resident 3.46 GB (load peak 3.46 GB)`

sed 's/^rung = "second_opinion"$/rung = "streamed"/' orbit.toml > var/gate-b.toml
HF_HUB_OFFLINE=1 .venv/bin/orbit --config var/gate-b.toml bench tier1
pkill -f "optiq serve"          # it does not exit with Orbit; Orbit never spawned it
```

Only the engine is resident, so this runs in ~3.5 GB against a 28.08 GiB ceiling — it is
**not** a two-model rung. Takes ~4.5 min: three frontiers, the largest ~145 s of prefill.
`tier1.reasoning_control` must name a dialect the engine reads, or the gate's unschema'd
probe comes back as a reasoning trace and is refused — `platform.md` §4.7.

---

## 2. What loads weights

`MLXTier0Backend.__init__` calls `mlx_lm.load()` eagerly — there is no lazy path — so
every command that builds tier 0 pays the full 23.0 GiB before it prints anything.

**Loads weights** (`backend = "mlx"`):

| Command | Why |
|---|---|
| `orbit serve` | `gateway/app.py` builds tier 0 at startup |
| `orbit doctor` | builds tier 0 to report its container hash — **not a cheap probe** |
| `orbit gate toolcall` | builds tier 0, then runs N turns |
| `orbit gate isolation` | builds tier 0 **more than once** — §4 |
| `orbit eval merge`, `orbit eval regression`, `orbit bench` | build tier 0, some also tier 1 |
| `orbit train` | the trainer holds the base model |

**Free** — no weights, safe to run alongside anything:

`pytest -q` · `orbit extract` · `orbit profile` · `orbit audit verify` ·
`orbit offline-env` · `orbit serve --backend mock` · any command with a config whose
`backend = "mock"`.

### 2.1 Asking the cheap questions

That `orbit doctor` costs a full load is worth internalising: it reads like `git
status` and behaves like a model load. For offline posture, constrained-decoding
availability and environment, point it at a mock config.

```bash
printf 'backend = "mock"\n[tier1]\nenabled = true\nrung = "second_opinion"\n' > /tmp/mock.toml
orbit --config /tmp/mock.toml doctor
```

`--config` is a global flag and goes **before** the subcommand.

---

## 3. Pre-flight checks

| Task | Check |
|---|---|
| Editing, tests, mock backend | Nothing. ollama may stay up; it is idle. |
| **Anything whose output is a number** | §3.1 first, every time. A gate run on a degraded host answers nothing in either direction. |
| **Gateway against real weights** | `curl -s http://127.0.0.1:11434/api/ps` → `{"models":[]}`; `lsof -nP -iTCP:8081 -sTCP:LISTEN` → nothing (rung 3 needs no mlx-optiq); then `HF_HUB_OFFLINE=1 orbit serve` |
| **A gate or an eval** | The same two, plus no `orbit serve` already holding 8080. One gate at a time — they each want the whole GPU. |
| **In doubt about headroom** | Read `total − active`, never `Pages free`. Tier 0 needs ~27 GB by that measure. Script in `platform.md` §8. |

Thrashing is `Pageouts`, not `vm.swapusage used` — loading tier 0 legitimately moves
swap 0.46 → 3.27 GB with 223 pageouts, because the compressor holds pages that were
never on disk. A guard on swap aborts runs that were never in trouble.

### 3.1 The host-health gate — run it before believing any throughput number

This host has been observed running at **~9% of its recorded bandwidth** with nothing
resident and compute at ~50% — `operations.md` §3.1. A gate that is green on such a host is a
meaningless pass and a red one is a meaningless fail, so this check comes before any
measurement, and again after any large model load.

```bash
.venv-optiq/bin/python tools/mlxbench.py     # loads no weights, ~30 s
```

| Reading | Healthy | Degraded, as observed |
|---|---|---|
| bandwidth r+w | **≥ ~250 GB/s** | 23 |
| read-only reduction | ≥ ~240 GB/s | 11 |
| GEMM 4096³ | ≥ ~10.8 TFLOP/s | 5.3 |

**Below ~200 GB/s, stop and reboot.** A reboot is the only remedy that has ever worked,
and it did: 2026-08-09 17:29 read 23 GB/s at 15 h 47 m of uptime, and 19:59 on a fresh
boot read **323 / 352 GB/s and 11.9 TFLOP/s** — three consecutive passes, two independent
scripts agreeing to within 1%. That is *above* `platform.md` §2.1's recorded figures, so
treat those as a floor rather than a target.

A reboot that *fails* to clear it is the more valuable result — it kills the boot-scoped
hypothesis T18 rests on — so record which it was either way.

Once it passes, the queue that was waiting on it, in order:

1. **`mlxbench.py` again immediately after the next large model load.** The onset
   hypothesis is that a big load triggers the degradation, and this is the one cheap
   chance to catch it in the act rather than infer it a session later.
2. Gate A re-run (T4, T19), then `orbit bench tier1` (T2).
3. T20's constrained-decode fixes, which need a healthy host to show 6.4 → 0.9 ms/token
   end to end.

---

## 4. The isolation gate multiplies the load

`orbit gate isolation` is the one command that asks for tier 0 several times over.
Building a backend with a subset of adapters mounted is the point of the test, so it
cannot reuse one instance (`eval/gates.py:178-180`). On the mlx backend each
`factory()` call constructs a fresh `MLXTier0Backend`, and each loads 23.0 GiB:

| Load | Purpose |
|---|---|
| 1 | enumerate the adapter names (`cli.py`, `build_tier0(cfg).mounted_adapters()`) |
| 2 | `all_mounted` |
| 3…n | one per adapter, for `solo` |

**Sequential since 2026-08-10, so the peak is one load rather than two.** `all_mounted`
and the current `solo` used to be live across one `asyncio.gather`, which put the floor
for a **single** adapter at **2 × 23.0 = 46.0 GiB against a 28.08 GiB ceiling** — the gate
could not have run on this host at all, and this section called it the likeliest way to
wedge the machine. The gate now runs `all_mounted` to completion, records its ≤128-token
outputs, releases the weights through the rung-2 `unload()` seam, and only then builds each
`solo`. Same comparisons, same verdict, one arm live;
`test_isolation_gate_holds_one_arm_at_a_time` fails against the old shape.

So the count above is still n+1 loads *in sequence* — each one still 23.0 GiB and still
wanting ~27 GB of headroom, so run §3.1 first and expect roughly a load's worth of wall
clock per adapter. Today none of it bites: `adapters/` does not exist, so enumeration
returns empty and the gate short-circuits on `"no adapters mounted; nothing to isolate"` —
one wasted load to answer nothing.

---

## 5. Downloads

Two caches, both large, and they overlap:

```bash
hf cache list      # HF snapshots — what tier 0 loads from
ollama list        # GGUF blobs — unrelated to this project
```

Check both before pulling anything, and **say which format you need**. They do not
substitute: `qwen3.6:latest` in ollama and `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`
in the HF cache are the same 35B-A3B weights at the same 4-bit class, ~23 GB each,
pulled four weeks apart, and neither can serve the other — tier 0 needs a snapshot
directory of OptiQ safetensors, ollama stores opaque Q4_K_M blobs with no
`config.json`. The MLX copy is the one this project loads.

**Check the model can run here before spending an hour fetching it.** The 46.9 GB
`Qwen3.5-122B-A10B-OptiQ-2bit` is in the cache and is not loaded by Orbit: rung 3 serves
the verifier instead, at 6–9× the speed and no memory. Rung 1 is **slow, not dead** —
165 tok/s of streamed prefill, and a 30k review fits inside `request_timeout_s`
(`platform.md` §4.3). The snapshot is kept deliberately, because resolving one takes an
hour. `platform.md` §5 is the arithmetic for every candidate, and it is worth running
*before* a download rather than after.

---

## 6. Bringing up rung 1 on DeepSeek-V4-Flash

> [!WARNING]
> **`optiq serve` still does not work on this model, and the model itself does.** The engine
> loads it and then aborts during prompt processing on the first request, every time:
> `std::runtime_error: There is no Stream(gpu, 1) in current thread`. **Driving the same
> call chain single-threaded works** — `load_streaming` + `stream_generate` on the main
> thread has never failed, and that is where 2026-08-10's numbers come from. So the
> procedure below still does not produce a verifier over HTTP; `tools/ds4_probe.py` produces
> one in-process. **`platform.md` §4.7 is the record**: the crash mechanism, the measured
> prefill/decode/context figures, and why rung 3 still ships.

**Nothing here is scheduled** — `platform.md` §4.8 is the decision, and it was deferred on
cost before it became blocked on this: the model buys verdict independence at ~8× rung 3's
latency, and Gate B derives to ≤168 tok/s, failing on compute at every step size. The
container is on disk: 42 shards, 92.49 GB referenced, resolved snapshot path in
`platform.md` §4.6.

### 6.1 `orbit doctor` will wedge the machine here, and it looks like the next step

**`orbit doctor` builds tier 0 first**, to report its container hash — §2 is the general
rule and this is the case where it bites hardest.

> **`orbit bench tier1` no longer does, as of 2026-08-10.** It was
> `build_tier1(cfg, build_tier0(cfg))`, which is 23.0 GiB eagerly loaded for a rung that
> reaches its engine over a socket and never reads it. Tier 0 is now built only for the
> rungs that serve from it (3 and 2), and those carry no prefill instrument, so the
> command declines for them without loading anything — 0.06 s where it used to be 23.0 GiB.
> That removed one of the two blockers named below and let Gate B run for the first time
> (`platform.md` §4.1a). The memory arithmetic in this section is unchanged; what changed is
> that Gate B no longer pays it.

| Resident | GiB |
|---|---|
| Metal working set ceiling | **28.08** |
| tier 0 | 23.0 |
| left for anything else | **5.1** |
| DeepSeek-V4 at its *smallest* footprint (6.49 GB non-routed) | **6.04** |

**It does not fit beside tier 0 at any setting**, and the scales-resident configuration
below is 14.11 GiB rather than 6.04. That arithmetic is what `orbit doctor` still runs
into, and it is not a reason to relax a threshold.

It is no longer a reason Gate B cannot run: with `cmd_bench` no longer building tier 0, the
gate holds only the engine — 3.46 GB for the 122B — and it was run three times on
2026-08-10 against ~24 GB of headroom. **Against DeepSeek-V4 specifically it is still the
crash in §6, not the memory, that blocks it.**

### 6.2 Serve it

Pre-flight first: §3.1, then nothing on 11434, nothing already on 8081, no `orbit serve`
holding 8080.

```bash
S=~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-0731-OptiQ-2bit/snapshots/0edd7d3e70d562a0fc1d1574943ca4fe2b2c1e36
OPTIQ_STREAM_SCALES_BUDGET_GB=12 HF_HUB_OFFLINE=1 .venv-optiq/bin/optiq serve \
  --stream-experts --model "$S" --host 127.0.0.1 --port 8081 \
  --max-context 32768 --prefill-step-size 8192
```

Neither flag is an optimisation:

| Flag | Why |
|---|---|
| `--prefill-step-size 8192` | **A correctness knob.** At the 2048 default a 30k review is 15 sweeps — ~345 s of prefill plus ~47 s of decode against `request_timeout_s = 300`, so it times out. At 8192 it is four sweeps, ~270 s, with ~30 s of margin. It is `mlx_lm.server`'s flag, forwarded by optiq, which has none of its own (T13). |
| `OPTIQ_STREAM_SCALES_BUDGET_GB=12` | The engine prints its default budget as **3.9 GB** (`max(2 GB, 10 % RAM)`) against **9.3 GB** of scales and biases — the header sum for routed experts alone is 8.66 GB, and optiq counts the shared expert's meta too. Under the default the meta streams: 11% of the bytes for 23% of decode time, because 128 KiB at qd 6 runs at 2.50 GB/s against 6.04 for 2 MiB. Above 9.3 GB they stay resident, for ~33% of decode. **Derived from per-block rates and still never tested against the model, because it has never generated a token.** |

**Read the startup lines before letting it serve anything.** Both configurations are now
measured (`platform.md` §4.7):

| Budget | `moe_stream` line | resident | load peak |
|---|---|---|---|
| unset | `9.3 GB vs budget 3.9 GB -> STREAM` | 6.49 GB | 6.49 GB |
| `=12` | `9.3 GB vs budget 12.0 GB -> resident` | 15.21 GB | **23.80 GB** |

`swapped 129 expert projections` in both — 43 layers × 3, the whole main tower, which
`tools/ds4_headers.py` predicts. **A resident figure near 92 GB means the swap did not
happen: kill it before it wedges the machine.** And note the peak, not the resident
figure, is what has to fit: making the scales resident costs 8.7 GB of steady state and
raises the peak to **3.7×** it.

### 6.3 Pilot, then scale

One `curl` completion first — a trivial prompt is enough, since the failure arrives at
`Prompt processing progress: 0/N` before any token. Then ~500 tokens of input, then eight.
§6.2 says what to expect from the load, so a divergence there is a measurement bug or a
wedged machine rather than a finding.

**Run the 122B as a control before concluding anything about the host.** Same engine, same
flags, `Qwen3.5-122B-A10B-OptiQ-2bit`, resident 3.46 GB — it answers in ~15 s. That is what
separated "this model does not run" from "streaming is broken here", and it cost two
minutes.

### 6.4 What such a run would settle

| Question | Status |
|---|---|
| Does the load peak stay under the working set? | **measured** — 6.49 GB streamed, 23.80 GB with scales resident (`platform.md` §4.7) |
| Does it serve over HTTP? | **measured — no.** Aborts on the first request. `platform.md` §4.7 |
| Does it generate at all? | **measured — yes, single-threaded.** 48 tok/s prefill, 2.6 decode, peak 14.15 GB at 12.8k tokens. `platform.md` §4.7 §3 |
| What does `container_hash` cost over 92.49 GB? | still derived ~60 s at ~1.5 GB/s. `blake3`'s `max_threads` is the lever, **not** mmap (sec 8.4). Unreachable via `orbit doctor` here anyway — §6.1 |
| Does `OPTIQ_STREAM_SCALES_BUDGET_GB` pay? | arithmetic only, ~33% of decode, and untestable until it generates |
| **Do 2-bit routed experts produce usable verdicts at all?** | the only question about the model rather than the machine, and the crash keeps it closed |

**DSpark/MTP stays off regardless.** It is opt-in and experimental, buys nothing for a
verifier writing ≤128 tokens, and threatens the byte-identity G1 and G2 assert. Worth
recording as the one mechanism that could rescue *decode* if this model were ever asked to
generate: verifying K speculated tokens in one forward pass amortises the expert sweep
across all K, which is the prefill trick applied to decode.

**If Gate B ever does fail, chunk size is the first thing to move and it cannot be moved.**
`optiq serve` has no `--prefill-chunk` flag — checked against 0.4.18 on hardware, and
`--max-context`, `--prompt-concurrency` and `--stream-experts-cache` are not it. An engine
exposing no way to make it do a batch-union sweep is evidence it is not doing one. That
branch collapses into its own fallback: **ds4 (DwarfStar)**, purpose-built for this model,
with `--ssd-streaming-cache-experts 18GB` and the same gate. What ds4 would have to answer
for is schema-constrained output, which it does not document and which sec 5.2 makes an
invariant rather than a preference — `validate_or_raise` is the second line, not the
first. If both engines fail the gate, the in-house streaming loader is M-blocking rather
than optional. Worth reading before Phase 2 either way: ds4 independently implements exact
DSML replay maps for KV reuse, the same idea as sec 8.5.5 arrived at separately, and
survives restarts via an appended KTM section.

---

## 7. Bringing up rung 1 on Qwen3-Coder-Next

**This one works.** It is the only streamed model here that has served a request through
`optiq serve`, and the only configuration that has reported `meets_spec: true`. Numbers:
`platform.md` §4.8. Decision: `platform.md` §4.8 — and the decision is still rung 3.

### 7.1 It is cheap, and that is the point

| Resident | GiB |
|---|---|
| Metal working set ceiling | **28.08** |
| this engine, meta streamed | **1.27** |
| this engine, `OPTIQ_STREAM_SCALES_BUDGET_GB=6` | 5.79 (load peak 10.26) |

**Load peak equals steady state in the default configuration**, unlike DeepSeek, where
making the scales resident costs a 3.7× spike (§6.2). So §3's headroom precondition is
satisfied by almost any machine state — but §3.1 still comes first, because this is a
throughput measurement and a degraded host answers nothing in either direction.

It still does not co-reside with tier 0 for a *serving* deployment, and the reason is
unchanged: 23.0 GiB plus this is inside 28.08, but `cmd_bench` is the only command that
avoids building tier 0 at all, and rung 1 serving means a gateway that holds both.

### 7.2 Serve it

Pre-flight first: §3.1, then nothing on 11434, nothing on 8081, no `orbit serve` on 8080.

```bash
S=~/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5
HF_HUB_OFFLINE=1 .venv-optiq/bin/optiq serve --stream-experts --model "$S" \
  --host 127.0.0.1 --port 8081 --max-context 32768 --prefill-step-size 8192
```

Expect, and read before letting it serve anything:

```
[moe_stream] expert scales/biases 4.8 GB vs budget 3.9 GB -> STREAM (off SSD)
[moe_stream] swapped 144 expert projections; resident 1.36 GB (load peak 1.36 GB)
```

`swapped 144` is 48 layers × 3 projections, which `tools/qcn_headers.py` predicts without
downloading anything. **A resident figure near 45 GB means the swap did not happen: kill
it.** Adding `OPTIQ_STREAM_SCALES_BUDGET_GB=6` turns the first line into `-> resident` and
is worth 1.08× for 4.86 GB — cheap here, where the same lever on DeepSeek needs 9.3 GB.

### 7.3 Do not overrun the context, and count tokens to know that you have not

**A prompt past `--max-context` takes the process with it** — `[metal::malloc]` against
Metal's 21.06 GiB single-array limit, not an out-of-memory, not a refusal (T35,
`platform.md` §1). 21,008 input tokens is measured fine; 52,008 is measured fatal; nothing
between has been tried.

Size prompts by **tokens**, never characters or words: this tokenizer runs ~11.8 tokens per
generated identifier, which is how the fatal prompt was sent in the first place. Gate B's
own filler overshoots its frontier by ~36–39% (T31), so its 16,000 frontier delivers 21,008
— inside the limit, and not by much.

### 7.4 Gate B

```bash
.venv/bin/orbit --config orbit.qcn.toml.example bench tier1
```

That config is `orbit.toml` with `[tier1]` repointed and nothing else touched, so its
result compares row-for-row with the 122B's in `platform.md` §4.1a. It builds no tier 0.
Expect worst-of-run ~253–263 and `meets_spec: true`; a reading near 190 is the engine
falling back to 2048-token chunks, and **the configured 150.0 floor will not catch it** —
read `worst_tok_per_s`, not `pass`.

### 7.5 What a run would still settle

| Question | Status |
|---|---|
| Does it serve over HTTP? | **measured — yes** |
| Does it pass Gate B at spec? | **measured — yes.** 258.0 against 200 |
| Does the scales lever pay? | **measured — 1.08×**, against 1.36× on the sweep alone |
| What does `container_hash` cost over 44.84 GB? | unmeasured, derived ~30 s at ~1.5 GB/s |
| Where exactly does the buffer ceiling bite? | **unmeasured** — T35, and it is a bisection between 21k and 52k tokens |
| Does `reasoning_control` need to be set for this model? | **unmeasured.** Carried over from the 122B; Gate B reads prefill rate and never exercises it |
| **Do its verdicts decorrelate from tier 0's?** | **the only question that decides anything now** — T5, and `tools/rung3_agreement.py` is the shape of the answer |
