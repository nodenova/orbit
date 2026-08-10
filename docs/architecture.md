# Architecture

| | |
|---|---|
| **Purpose** | The shape of the system, and the decisions inside it that look simplifiable and are not. |
| **Answers** | "How does a request flow?" · "Where is the portability line?" · "Why is this written the strange way?" · "What has already gone wrong here?" |
| **Does not answer** | What the numbers are (`platform.md`), which module implements a spec section (`spec-map.md`), what is safe to start (`operations.md`). |
| **When the code and this file disagree** | The code is right. Fix this file in the same commit. |

## 1. The shape

**A local coding-agent runtime that optimises for merge quality.** Two model tiers on one
machine: a fast resident model carrying repo-derived LoRA adapters generates candidate
patches, and a large model reads them and picks or rejects — a **verifier, never a
generator**.

Three facts the whole design turns on:

| # | Fact | Consequence for the code |
|---|---|---|
| 1 | **Streamed models are ~40× cheaper per input token than per output token.** Decode streams top-k experts per token; prefill with a batch-union sweep reads each expert once per chunk. | Tier 1 is only ever asked input-dominated questions. |
| 2 | **The input-dominated tasks are exactly the ones that decide merge quality.** Reranking N candidates emits an integer; reviewing a diff emits a verdict. Both read 5–30k tokens and write 10–300. | Tier 1 is a verifier *structurally*: the interface has no `generate`. |
| 3 | **Merge quality, not benchmark score, is the binding constraint.** METR: ~half of SWE-bench-passing PRs would not be merged by maintainers. | A repo-derived adapter encodes this repository's conventions; a verifier pass enforces them. |

Both terms of fact 1 are measured — `platform.md` §4.

## 2. One request path

Fixed order, shared by all three wire protocols (`gateway/pipeline.py`):

```
compact → replay-aware render → prompt-cache probe → constrain
        → cascade (best-of-N, tier-1 rerank, T2 escalation)
        → repair → bounded retry → record replay → cache store
        → receipt + audit → context-scale reported usage
```

**Compaction is first** because everything downstream is measured against the prompt
actually sent. **Context scaling is last** because it is a *reporting* adjustment that must
never reach the model, the cache key or the audit record.

- `gateway/wire/*` are the only modules that know which harness spoke; everything
  downstream sees `types.py`.
- Tier 1 exposes `rerank`, `review` and `plan_critique`, all schema-constrained, each
  degrading to a failed `Verdict` rather than failing the turn.
- `[eval]` in `orbit.toml` is load-bearing: without it three of the merge eval's five
  metrics report *not measured*, `compare_arms` refuses the M3 gate, and T2 escalation
  stays dormant.

## 3. The hard line

**`backends/base.py::Backend` is the line the whole layout turns on.**

```
src/orbit/
  gateway/      wire protocols, compaction, caches, tool-call layer  (sec 8)
  router/       turn classification, best-of-N, escalation           (sec 7)
  tier1/        the verifier API and its schemas                     (sec 5)
  backends/     the hard line — Backend, the mock, the MLX engines   (sec 4, 5)
  adapters/     A0/A1/A2 extraction, routing profile, training       (sec 6)
  attest/       receipts, hash-chained audit log, provenance         (sec 9)
  eval/         merge eval, gates, regression, latency               (sec 10)
  thresholds.py the specification's gate figures, in one place
tools/          not installed: the two files that may touch a network
```

**Above the line** — gateway, router, tool-call layer, adapters, attestation, eval — is
pure Python that runs anywhere and is fully tested. `MockBackend` is what makes that half
testable: deterministic, adapter-sensitive, faultable.

**Below the line** both MLX backends run off-target, and the two halves are kept apart on
purpose:

- **`backends/mlx_tier1.py` imports no MLX.** Sec 5.4 puts mlx-optiq behind a *process
  boundary*, so the file is an httpx client and a `MockTransport` stands exactly where the
  socket does. Covered directly: the payload, the sec 5.1 clamp on the wire, the schema
  refusal, Gate B's arithmetic.
- **`backends/mlx_tier0.py` genuinely needs MLX**, so `tests/fake_mlx.py` models the
  surface it touches and the real `MLXTier0Backend` runs against it.

**A green run there proves the wiring, not the model.** Quantised deltas, adapter isolation
on Metal and every determinism claim are untouched by it. Read `tests/fake_mlx.py`'s
docstring before extending it: like `MockBackend`, it must never be easier to satisfy than
the real thing, and the one real-weights run so far found exactly that failure — a
`json_schema` the hardware ignored and the mock honoured, invisible to a green suite.

---
## 4. Invariants — things not to undo

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
- **A relaxed gate threshold never removes the spec figure** (`orbit.thresholds`). There is
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

## 5. Traps this repo has already fallen into

The first four were caught by CI or by running the thing, never by local unit tests. The
fifth was caught by nothing at all for months, which is the point of it.

| # | Trap | Lesson |
|---|---|---|
| 1 | **Unanchored `.gitignore` patterns.** `adapters/` matched `src/orbit/adapters/` and silently excluded the entire adapter pipeline from the first commit — on disk, importing fine, simply not in the repository. | Anchor new patterns with a leading `/`. |
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

