# Tandem — start here

**This file is the handoff.** It is committed on purpose: a session restarted from
a cold clone has no memory of how the code got this way, and everything below is
what would otherwise be lost. Read it before changing anything.

Verified against `main` at the commit that last touched this file —
`git log -1 --format='%h %ad' docs/HANDOFF.md`. Written as a lookup rather than a
pasted hash because a pasted one is stale the moment anything else lands, and a
handoff nobody trusts is a handoff nobody reads.

It lived at `specs/NEXT_STEPS.md` until `/specs/` became the ignored drop point for
the v1 specification, which is not ours to publish. The handoff had to move rather
than follow it into the ignore rule: a cold clone is the only reader this file has.

**If the code and this document disagree, the code is right and this needs fixing.**
Say so, and fix it in the same commit.

---

## 1. What this is, in one page

**A local coding-agent runtime that optimises for merge quality.** Two model tiers
on one machine: a fast resident model carrying repo-specific LoRA adapters, and a
large streamed model used as a **verifier rather than a generator**.

> A fast resident model, adapted to your repository from its own git history,
> generating candidate patches — with a large streamed model reading those
> candidates and picking or rejecting them, and a receipt proving which base and
> which adapter produced each change.

Three facts the whole design turns on:

1. **Streamed models are ~300× cheaper per input token than per output token.**
   Decode streams top-k experts per token (~4–10 tok/s, unusable). Prefill with a
   batch-union sweep reads each expert once per chunk (~650–1,300 tok/s). Every
   published number in the field is single-request decode, so the field concluded
   streamed models are too slow — correct *for generation*, wrong for anything
   where input dominates output.
2. **The input-dominated tasks are exactly the ones that decide merge quality.**
   Reranking N candidates emits an integer. Reviewing a diff emits a verdict. Both
   read 5–30k tokens and write 10–300.
3. **Merge quality, not benchmark score, is the binding constraint.** METR: ~half
   of SWE-bench-passing PRs would not be merged by maintainers. A local 35B at
   73.4% SWE-bench Verified is not short of capability — it is short of *this
   repository's* conventions, which is what a repo-derived adapter encodes.

Target hardware: MacBook Pro M4 Max, 64 GB unified memory, 1 TB SSD.

**The full technical specification (v1) is not in this repository.** Code
docstrings reference it by section (`sec 8.2`, `sec 6.3`, …). Without it those
references dangle — get a copy before doing design work, and drop it in `/specs/`,
which is gitignored for exactly that. `docs/STATUS.md` maps every section to the
code that implements it, which covers most day-to-day needs.

---

## 2. State

`main`, green CI (Python 3.11 and 3.14; jobs `tests` and `cli and gateway smoke`).
**515 tests**, passing on both `[dev,constrain]` and `[dev]`-only installs.

Working end to end against `MockBackend`: three wire protocols, harness compaction
(~41× measured), incremental streaming, prompt + disk KV caching across a restart,
the five-layer tool-call path, best-of-N with tier-1 rerank, T2 escalation through
a git worktree, the full merge eval, the regression detector, attestation with a
hash-chained audit log, and the A0/A1/A2 extractors.

All four rungs of the sec 5.5 fallback ladder are selectable by name. Rung 2's
residency policy is built and tested (the mutual exclusion, the lazy swap back, the
measured budget guard); its two `Occupant`s need the hardware. Rung 4 is complete,
including the gates that keep it from ever being reached by falling back.

**Both MLX backends now execute off-target**, which they did not until recently.
That claim needs its two halves kept apart:

* `backends/mlx_tier1.py` **imports no MLX**. It was on the Apple-Silicon-only list
  by association and did not belong there — sec 5.4 puts mlx-optiq behind a
  *process boundary*, so what is in that file is an httpx client and a
  `MockTransport` stands exactly where the socket does. It is now covered directly:
  the payload, the sec 5.1 clamp on the wire, the schema refusal, Gate B's
  arithmetic.
* `backends/mlx_tier0.py` genuinely needs MLX, so `tests/fake_mlx.py` models the
  surface it touches and the real `MLXTier0Backend` runs against it — mounting
  adapters, generating, and passing the sec 4.2 isolation gate for real rather than
  vacuously. **This proves the wiring and nothing else.** Real weights, quantised
  deltas, Metal determinism and every number are still untouched. Read the fake's
  docstring before trusting a green run for more than it says.

**The distinction that matters most:** every gate's *plumbing* is proven; none of
its *numbers* are. Nothing here has yet measured a 35B model.

```bash
pip install -e '.[dev,constrain]'
pytest -q            # 399 passed
tandem doctor        # runtime status, offline posture, configured tier-1 rung
```

---

## 3. What to do next, in order

The ordering is not arbitrary. Each step can invalidate the ones after it, so
running them out of order risks training an adapter against a premise already
falsified.

### 1. Gate A — is the interactive premise true? (half a day)

```bash
tandem bench latency --out var/gate-a.json
```

Needs tier-0 warm TTFT < 5 s at ≥ 30 tok/s, tool-call failures < 5%.

**Carries a kill condition.** If it fails badly the interactive premise is wrong
and the product becomes async review-assist. Cheap to learn on day one, expensive
in month two.

### 2. Gate B — is tier 1 three weeks or six? (half a day)

```bash
tandem bench tier1
```

Needs streamed prefill ≥ 200 tok/s at 4k/8k/16k. Below that, mlx-optiq is doing
per-token expert loading rather than the batch-union sweep the tier-1 thesis rests
on, and the in-house loader becomes M-blocking rather than optional.

A live risk, not a formality: `ds4` measures GLM-5.2 streaming prefill at 3–5 t/s,
~100× below the bandwidth bound.

**The filler this gate measures on was rebuilt.** It used to be one 23-character
line repeated to length, on the reasoning that only length matters. On a streamed
MoE that is false in the flattering direction: identical tokens route to identical
experts — by construction on layers using hash routing — so the chunk's expert union
collapses, the cache serves the sweep from RAM, and the gate reports a throughput no
real prompt reaches. `prefill_filler` is now deterministic and identifier-diverse.
A floor test must not fail in the direction that lets you pass.

**Running DeepSeek-V4-Flash-0731 here is a config change, not a port** — same rung,
same engine, same process boundary. `docs/DEEPSEEK_V4.md` is the analysis: role and
engine decided on numbers, the 64 GB budget, and the arithmetic showing Gate B turns
entirely on whether the engine amortises the expert sweep across a prefill chunk
(~330 tok/s derived at a 4k chunk, ~3.5 at per-position loading — a 60× fork on one
engine behaviour, not on the hardware).

### 3. Prove the adapter mounting

```bash
tandem gate isolation --adapters a0 a1-myrepo
```

Greedy output under adapter *i* with N mounted must be byte-identical to output
with only *i* mounted. **Until it passes on the hardware, treat every receipt
naming an adapter as unproven** — a failure means adapter deltas leak between
concurrent requests, which is a silent wrong-answer bug.

The *wiring* is no longer unexercised: `tests/test_mlx_tier0.py` runs this same
gate against the real `MLXTier0Backend` under the fake MLX, so `MultiAdapterLinear`
and the `ContextVar` binding are checked on every CI run, and the suite carries a
deliberately leaky wrapper to prove the gate can still fail. What that cannot reach
is real quantised deltas and Metal, which is most of what the gate is for. Cheap
now, still mandatory then.

### 4. Record a regression baseline *before* changing anything

```bash
tandem eval regression --baseline var/regression-baseline.json
```

It is a detector; it detects nothing without a reference point. Record it on the
untouched model, then again after each step below. A baseline taken after the first
change is a baseline of the wrong thing.

### 5. A0, then re-gate tool calls

Train the harness adapter, re-run `tandem gate toolcall --runs 100`. A0 should beat
base on tool-call validity — experiment E1. If A0 alone closes the gap, the repair
layer becomes droppable and the product simplifies.

### 6. A1 and the merge eval — the product thesis

```bash
tandem extract a1 --repo <real repo> --holdout 25 --out corpus/a1
tandem train sft --corpus corpus/a1/train.jsonl --out adapters/a1-x --name a1-x --repo <real repo>
tandem eval merge --repo <real repo> --a1 a1-x --out var/merge-eval.json
```

Needs a `[eval]` block in `tandem.toml` naming the repo's linters and test command,
or three of five metrics report "not measured" and `compare_arms` correctly refuses
to call it a pass.

**If A1 does not beat base on ≥3 of 5 metrics, stop and re-plan before building
tier 1.** That is the entire reason M3 sits at week 8 — before the expensive part.

### 7. Determinism, then the experiments

G1 (CPU vs Metal byte-identical) and G2 (expert cache at 0 vs max byte-identical).
G2 especially: if placement changes the output, every determinism claim in the
receipt is false — and that claim is what the regulated buyer is paying for. Then
E2 (int8 vs bf16 adapter deltas), E3 (expert coverage at 35B), E4 (candidate count).

**Before ordering hardware:** confirm the M4 Max bin (546 GB/s vs 410 GB/s — the
lower bin costs ~25% on tier-0 decode) and **SSD capacity, which is a performance
spec**. 1 TB is roughly 2× on the tier-1 path, because Apple SSD read bandwidth
scales with NAND package count.

---

## 4. Small things still open, no hardware needed

None of these blocks the sequence above.

- **`mlx_tier0.supports_state()` returns False.** The disk KV cache is wired end to
  end and tested; what is missing is serialising an `mlx_lm` prompt cache to bytes
  and back. The docstring states what an implementation must guarantee.
- **The routing-profile forward pass.** `tandem profile` builds and validates the
  sidecar from an activation dump; producing the dump needs the real model.

And one that does need hardware, listed here because the rest of it is done:

- **Rung 2's second occupant** (sec 5.5). The residency policy is built and tested
  against the mock — exclusivity, fairness, the lazy swap back, the measured budget
  guard, the failed-swap state. The tier-0 half is no longer just written:
  `MLXTier0Backend.load`/`.unload` run against `tests/fake_mlx.py`, and a round trip
  through them comes back with the same container and adapter identity. What is
  missing is the *other* occupant — a resident 80B verifier backend to put on the
  far side of the switch. It does not exist, and `build_tier1` says exactly that on
  `backend="mlx"` rather than building a swap with nothing to swap into.

---

## 5. Things not to undo

Each looks simplifiable and is not. Every one has a comment in the code saying so.

- **Adapters are never merged into the base** (sec 4.2). Merging costs ~20 GB per
  adapter instead of ~250 MB, kills multi-tenancy, and leaves the receipt unable to
  name what produced a change.
- **Adapter selection and restored KV state are per-request, never on the backend.**
  Backend-global state races under concurrency, and the failure is a silently wrong
  answer rather than an error.
- **A KV state carries the identity it belongs to** (`Backend.state_key`). Restoring
  one built under a different container or adapter gives fluent output from the
  wrong model and a receipt naming the adapter that did not produce it.
- **Tier 1 has no `generate` entrypoint** and clamps `max_tokens` per call type
  (sec 5.1). The model that reranks five candidates in 18 s takes six minutes to
  write one.
- **Tier 1 never reasons, and a reasoned verdict is refused rather than read.**
  Both halves are load-bearing and they are not redundant. The request-side
  `reasoning_control` guesses which dialect of "thinking off" the engine reads;
  `refuse_reasoned_answer` observes what it actually did, unconditionally, on every
  rung and every model. Dropping the second because the first "already handles it"
  restores the failure for exactly the models nobody thought about — and that
  failure is silent twice over: the `<think>` block spends the clamp before the
  verdict exists, and thinking mode discards `temperature` without erroring, so the
  receipt attests to a greedy judgement that was a sample.
- **`rerank_schema(n)` bounds the choice to the candidates on offer.** Without the
  `maximum`, a constrained decode can still name a candidate that does not exist.
- **Rung 3 strips the adapter.** An adapted model judging its own candidates is
  asking whether it agrees with itself.
- **The rung is selected, never descended.** No error path picks a different rung
  from the one the config names. A ladder that reached rung 4 in response to a
  timeout would turn an airgapped runtime into an exfiltrating one with nobody
  choosing it.
- **Rung 4 needs the consent sentence, and its HTTP lives outside the package.**
  `tier1.remote_consent` must read "tier 1 leaves this machine" verbatim — a rung
  name is one word copied from a README. And `tools/remote_tier1.py` sits outside
  `src/tandem/` for the same structural reason as the A2 exporter: an HTTP client
  inside the package would weaken the sec 8.6 claim for every deployment, including
  the ones that never enable the rung.
- **Rung 2 guards tier 0** (`Pipeline` wraps it in `SwapGuard`). Residency is
  exclusive; an unguarded tier 0 is asked to generate from weights that have been
  evicted, and on the mock that failure is invisible.
- **Rung 2's swap back is lazy**, and its budget guard reads a measured round trip
  rather than a configured estimate. Eager restore adds ~10 s to the tail of every
  verified turn for a model the answer does not need; declining on a guess disables
  verification for a cost nobody observed.
- **`Backend.renders_canonically()` is asked of the object, not the type.** A
  wrapper that delegates `render` changes none of the bytes; a `type(...).render`
  check reads it as a real chat template and silently drops replay-aware rendering
  (sec 8.5.5) — a cache-key bug whose only symptom is a lower hit rate.
- **Every backend is *handed* the replay renderer, including the ones with their own
  chat template.** `Backend.render` takes `render_tool_call` and `Pipeline` passes it
  down both branches. A backend left to find the map itself cannot reach it, and tier
  0 did exactly what that leads to: it dropped tool calls out of the prompt entirely.
- **Tool calls go into the message content, not into a structured `tool_calls`
  field** (`mlx_tier0.render`). Handed a parsed call, a chat template picks its own
  serialisation, and the template's bytes are not the bytes the model sampled — the
  prefix diverges at the first tool call and never recovers. The renderer returns the
  model's own block and a template passes content through verbatim.
- **The disk KV cache uses plain `read`/`write`, never mmap** (sec 8.4).
- **The tool-replay map is not optional** (sec 8.5.5).
- **`SourceKind` is a closed enum** with no member for another model's outputs.
- **The regression report has no score field** (sec 10.3) — asserted by a test.
- **`tools/` is outside the package** — `export_reviews.py` and `remote_tier1.py`
  both — so the offline claim is structural rather than a promise. A test pins the
  package's network surface to one file (`backends/mlx_tier1.py`, the loopback
  process boundary) and a second import there is a test failure, not a discovery.

Three documented deviations from the spec, each preserving its intent, are in
`docs/STATUS.md` — the largest being that merge commits are not skipped
unconditionally, because on a merge-commit repository that drops ~90% of usable
history (measured: 10 pairs vs 131 on `pallets/click`).

---

## 6. Five traps this repo has already fallen into

The first four were caught by CI or by running the thing, never by local unit tests.
The fifth was caught by nothing at all for months, which is the point of it.

1. **Unanchored `.gitignore` patterns.** `adapters/` matched `src/tandem/adapters/`
   and silently excluded the entire adapter pipeline from the first commit — on
   disk, importing fine, simply not in the repository. Anchor new patterns.
2. **The mock must never be easier to satisfy than the real thing.** It has failed
   this twice: ignoring `const`/`anyOf` (inventing tool names a constrained model
   could not), then ignoring `minimum`/`maximum` (picking candidate indices that did
   not exist). When extending `MockBackend`, ask whether the change makes it *more*
   permissive than a real backend.
3. **Run both install states.** `[dev,constrain]` and `[dev]` alone exercise
   genuinely different paths — prevention versus repair.
4. **Not every 403 is a rate limit.** The exporter retried policy refusals five
   times over ~60 s and then blamed throttling, when the answer was in the first
   response body.
5. **"Correct on inspection" is not a state a file can stay in.** `mlx_tier0.py` was
   416 lines nothing had ever imported, and inspection had signed off on a `render`
   that dropped every tool call and tool result on the floor — the model blind to
   what it had already called, and two conversations differing only in tool history
   hashed to one KV cache key. Both failures are silent. It survived because the
   file was unreachable off-target, so the question "does an import of this even
   work" had no answer. Where a real dependency is unavailable, model it and run the
   code; where it is not needed at all (`mlx_tier1.py` imports no MLX), notice that
   before filing the file under untestable.

---

## 7. Repo facts

- Remote: `github.com/nodenova/orbit`, **public**. `main` is the only branch and the
  default; commit and push directly to it. Do not open a PR unless asked.
- **A session cannot delete a branch or change repo settings.** The git proxy
  refuses `push --delete`, and the REST API answers *"Write access to this GitHub
  API path is not permitted through this proxy"*. Pushing commits works fine.
  Anything else — deleting a ref, changing the default branch — needs a local clone
  or the GitHub UI. Ask rather than burning a turn discovering it again.
- This file is committed so a cold clone carries it. It was gitignored at first,
  which defeated its entire purpose: an untracked file dies with the container it
  was written in. **`/specs/` is the ignored directory** — put a local copy of the
  v1 specification there and nothing else; it is not ours to publish and this
  repository is public. Anything a future session needs belongs here, in `docs/`,
  written so that publishing it is fine.
- **Keep this file true.** It is the only thing standing between a fresh session and
  re-deriving every decision below from scratch. When you change behaviour that
  contradicts something here, fix it in the same commit; a doc that is 80% right is
  worse than none, because the wrong 20% is indistinguishable from the rest.
