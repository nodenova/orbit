# Implementation status

| | |
|---|---|
| **Purpose** | Section-by-section map from the v1 technical specification to the code that implements it. |
| **Answers** | "What implements `sec 8.2`?" · "Is this proven or just written?" · "Where does the implementation deviate from the spec, and why?" |
| **Does not answer** | How to run any of it (`README.md`), what to do next (`HANDOFF.md`), what this host measures (`BASELINE.md`). |
| **Verified against** | `main` at the commit that last touched this file — `git log -1 --format='%h %ad' docs/STATUS.md`. |
| **When the code and this file disagree** | The code is right. Fix this file in the same commit. |

## Status vocabulary

Used in every table below and in `HANDOFF.md`. The distinction between the middle
two rows is the one that carries information.

| Term | Means | What it does not mean |
|---|---|---|
| **built** | Implemented, covered by tests against `MockBackend` or a real dependency. | That any number it produces has been observed on hardware. |
| **written** | Implemented, executes only against a stand-in (`tests/fake_mlx.py`, `MockTransport`). | That it has met a real model. A green run proves wiring. |
| **measured** | Run against real weights on real hardware, number recorded under `var/`. | That the number met its budget — see the verdict column. |
| **open** | Not started, or started and blocked on something named. | — |

`sec N.M` references point at the v1 specification, which is **not in this
repository** (`/specs/` is the gitignored drop point for a local copy). This file is
the substitute for most day-to-day needs. Do not invent a meaning for a section you
cannot resolve.

---

## 1. Coverage by specification section

### sec 2 — Hardware budget

Not code. The numbers appear where they bind:

| Number | Where | State |
|---|---|---|
| Expert cache 18 GB | `Tier1Config.expert_cache_bytes` | built (config); re-derived for 36 GB in `tandem.toml` |
| KV frontier 32k | `Tier0Config.max_kv_tokens` | built; this host runs 65536, see `BASELINE.md` §3 |
| Tier 1 unloaded to train | `train.preflight()` | built |
| SSD capacity recorded with every measurement | `eval/latency.Environment` | built — capacity *is* a performance spec (sec 2.3) |

### sec 3 — Architecture

| Component | Module | State |
|---|---|---|
| Gateway | `tandem/gateway/` | built |
| Router | `tandem/router/` | built |
| Tier 0 | `tandem/backends/mlx_tier0.py` | **measured** — `Qwen3.6-35B-A3B-OptiQ-4bit`, 23.0 GiB, on a 36 GB M4 Max |
| Tier 1 | `tandem/backends/mlx_tier1.py`, `tandem/tier1/` | **measured 2026-08-10** — Tandem itself drove the streamed engine, for Gate B and for schema-constrained reranks (`BASELINE.md` §4.1a) |
| Attestation | `tandem/attest/` | built |
| Adapter pipeline | `tandem/adapters/` | built |

### sec 4 — Tier 0

| Spec | Where | State |
|---|---|---|
| 4.1 model, MTP | `Tier0Config` | built (config) |
| 4.2 adapter mounting, no merging, ContextVar binding, mount-at-startup | `backends/mlx_tier0.py` | written |
| 4.2 isolation test (blocking) | `eval/gates.py::adapter_isolation_gate` | built; **never run on hardware** |
| 4.3 targeting and rank | `mlx_tier0._is_target`, `adapters/profile.py` | written / built |
| constrained decoding applied to logits | `mlx_tier0.build_logits_processor` | **measured** — sec 10.2 gate 0.81 → 1.00 |

`MultiAdapterLinear` holds every mounted adapter's `(A, B)` at once and selects per
request. The isolation gate is what proves it, and it runs two ways: vacuously in CI
against a backend with no adapters mounted, so the gate itself cannot rot, and for
real in `tests/test_mlx_tier0.py` against `MLXTier0Backend` with two overlapping
adapters under the fake MLX. The suite also drives a deliberately leaky wrapper past
it, because a gate nobody has seen fail is a gate nobody has tested.

**Still untested off-target: quantised deltas and Metal — which is most of what the
gate is for.** Until it passes on hardware, treat every receipt naming an adapter as
unproven.

### sec 5 — Tier 1

| Spec | Where | State |
|---|---|---|
| 5.1 three call types, schema-constrained, verifier-only | `tier1/schemas.py`, `tier1/verifier.py` | built |
| 5.1 output clamp per call type | `backends/tier1_call.CALL_BUDGETS` | built |
| 5.2 2-bit caveat mitigations | schema `additionalProperties: false`; verdict validation | built |
| 5.2 reasoning refusal (DeepSeek-V4 family) | `tier1_call.resolve_reasoning_control`, `refuse_reasoned_answer` | built |
| 5.3 prefill measurement | `mlx_tier1.measure_prefill`, `gate_b_report` | **measured 2026-08-10** — the instrument itself, six runs, 153.7 tok/s mean, sd 1.07 (`BASELINE.md` §4.1a) |
| 5.4 mlx-optiq behind a process boundary | `mlx_tier1.OptiqTier1Backend` | built |
| 5.5 rung 1 (streamed) | `build_tier1()`, `backends/mlx_tier1.py` | **measured** — Gate B ran six times, 153.7 tok/s mean; the engine also answered real reranks through this client, `BASELINE.md` §4.1a |
| 5.5 rung 2 (80B resident-swapped) | `backends/resident_swap.py` | built (policy) / open (MLX occupants) |
| 5.5 rung 3 (second opinion) | `backends/second_opinion.py` | built — **the rung this host serves** |
| 5.5 rung 4 (remote) | `backends/remote_tier1.py`, `tools/remote_tier1.py` | built |

**Output ceilings are enforced in code, not requested of the model.** The verifier API
has no `generate`. The clamp and the schema validation moved out of `mlx_tier1.py` once
there were two transports: a clamp that is right in one and drifts in the other is not
a clamp. The move gave that ceiling its first test, since it had lived only in a file
nothing had ever imported.

**Tier 1 never reasons.** DeepSeek-V4-Flash is the first candidate verifier that
reasons by default, and it breaks two invariants at once without reporting either: a
`<think>` block spends the sec 5.1 clamp before the verdict exists, and thinking mode
*silently ignores* `temperature`, so the greedy judgement the receipt attests to
(sec 9.3) becomes a sample. `tier1.reasoning_control` sends thinking off in every
spelling the engines read; `refuse_reasoned_answer` rejects a verdict that reasoned
anyway, on every rung and every model, ungated by config.

**Both halves are load-bearing and they are not redundant.** The request-side flag is a
guess about which spelling of "thinking off" a given engine reads — two authorities
document different keys and the engines disagree — so sending one is a coin flip. The
response-side check is an observation of what the model actually did, and it catches
the three cases the guess cannot: an engine that ignored the flag, a model the name
match missed, and an operator who configured thinking on at the engine. Dropping it
because the flag "already handles it" restores the failure for exactly the models
nobody thought about. **That second case was live until 2026-08-10** — the check read
`reasoning_content` and mlx-optiq emits `reasoning`, so it caught nothing on the one
engine this repo runs; `HANDOFF.md` §3.11. There is deliberately no config value that turns reasoning on —
that would be a knob for disabling the clamp.

**The rung is selected, never descended.** Nothing falls from one rung to the next on
an error, which matters most at rung 4. `Tier1Attestation.rung` records which rung
produced every verdict, so a base-model second opinion never reads as a streamed one.

| Rung | Note |
|---|---|
| 3 | Tier 0 with the adapter unmounted. Needs no second model; the rung available during M0–M3 and the one `tandem.toml` runs here. The adapter strip is the whole mechanism — an adapted model judging its own candidates is asking whether it agrees with itself. **Known weakness: verdicts are not independent.** A verifier sharing the generator's weights shares its blind spots. |
| 2 | Evicts tier 0 to admit an 80B. Residency is exclusive, so `ResidencySwitch` is real mutual exclusion and `Pipeline` puts tier 0 behind `SwapGuard`. Swap back is lazy; the backend declines once the *measured* round trip exceeds `tier1.swap_budget_s`. The evictable tier-0 occupant runs against the fake; **the resident 80B on the far side does not exist**, and `build_tier1` says so rather than building a swap with nothing to swap into. |
| 4 | Sends the repository's code to a third party, so the gates in front of it are the implementation: never reached by falling back, `tier1.remote_consent` must carry "tier 1 leaves this machine" verbatim, the transport is a file outside the package, and `OfflineReport.ok` is false whenever the rung is armed. A remote verdict has **no container attestation** — `container_hash()` is None by construction, and the rung in the receipt is what says the null is a property rather than a gap. |

### sec 6 — Adapter pipeline

| Spec | Where | State |
|---|---|---|
| 6.1 A0 synthetic harness traces | `adapters/extract_a0.py` | built |
| 6.2 A1 git-history extraction, filters, messages-JSONL, thin-corpus detection | `adapters/extract_a1.py`, `filters.py`, `gitwalk.py` | built |
| 6.3 A2 DPO pairs from review history + reverts | `adapters/extract_a2.py` | built |
| 6.4 routing profile, count- vs mass-ranking | `adapters/profile.py` | built |
| training driver (SFT + DPO), NEFTune, collapse detection | `adapters/train.py` | built |
| 6.4 routing-profile forward pass | — | **open** — needs an activation dump from the real model |

### sec 7 — Router

| Spec | Where | State |
|---|---|---|
| 7.1 turn classification | `router/classify.py` | built |
| 7.2 T1 best-of-N rerank | `router/cascade.py` | built |
| 7.2 T2 failure escalation, bounded to one per turn | `router/cascade.py`, `eval/worktree.py` | built |
| 7.3 latency contract + automatic pressure valve | `cascade._record`, `eval/latency.CONTRACT` | built |
| 7.3 contract thresholds, per host | `config.GatesConfig`, `tandem.thresholds` | built |

T2 needs a host that can run the repository's tests, so `Pipeline` builds a
`WorktreeRunner` from `[eval]` and hands it to `Cascade`. Off unless the config opts
in: turning it on runs the suite on every `code_change` turn. With it off, `Cascade`
reports the path dormant rather than answering "passed" to escalation checks it never
made.

### sec 8 — Gateway

| Spec | Where | State |
|---|---|---|
| 8.1 three wire protocols | `gateway/wire/{anthropic,openai_chat,openai_responses}.py` | built |
| 8.1 incremental SSE | `wire/*.StreamEncoder`, `Pipeline.stream`, `app._sse` | built |
| 8.2 compaction, versioned templates, stripping, `--no-compact`, diff view | `gateway/compaction.py` | built |
| 8.3 context scaling | `gateway/context_scale.py` | built |
| 8.4 prompt cache + disk KV | `gateway/cache/` | built |
| 8.4 tier-0 KV state serialisation | `backends/mlx_kv.py`, `mlx_tier0.{export_state,_warm_start}` | **built, measured inert** — the codec round-trips against real `mlx_lm` cache classes, and on this container a real follow-up restores 0 tokens (`HANDOFF.md` §3.9) |
| 8.5 prevent / train / repair / retry / replay | `gateway/toolcall/` | built |
| 8.6 offline posture + verification script | `tandem/offline.py`, `tandem doctor` | built |

### sec 9 — Attestation

| Spec | Where | State |
|---|---|---|
| 9.1 response metadata | `attest/receipt.py` | built |
| 9.2 append-only audit log | `attest/audit.py` | built |
| 9.3 G1 / G2 determinism gates | `eval/gates.py`, `tools/determinism_probe.py` | built; **the gates have still not run, and the question they ask is measured** — G1 is **red on hardware** (CPU vs Metal flips the first token) and T16's chunk mechanism is quantified, both via the probe. G2 reports *not measured* rather than the vacuous pass it used to. `BASELINE.md` §2.4, `HANDOFF.md` §3.12, T33/T34 |
| 9.4 provenance of training data | `attest/provenance.py` | built |

### sec 10 — Evaluation

| Spec | Where | State |
|---|---|---|
| 10.1 repo-held-out merge eval, four bars | `eval/merge_eval.py` | built |
| 10.1 test pass + convention conformance | `eval/worktree.py` | built |
| 10.1 review-comment proxy | `merge_eval.tier1_review_proxy`, `scored_review_proxy` | built |
| 10.2 tool-call validity gate (blocking) | `eval/gates.py::toolcall_gate` | **measured** — 1.00 on hardware, 100 runs |
| 10.3 regression suite | `eval/regression.py`, `eval/regression_items.py` | built |
| 10.4 latency suite | `eval/latency.py` | **measured** — `var/gate-a.json` |
| 10.5 measurement discipline | `eval/latency.Environment` | built |

All five merge-eval metrics are measurable, so `compare_arms` can return an M3
verdict — given an `[eval]` block naming the repo's linters and test command. Without
one it reports "not measured" and refuses the gate.

The regression suite is 90 fixed short-answer items across reasoning, exact-answer
maths and code localisation. Its output is a **diff against a recorded baseline, not
a score** — `RegressionReport` has no field holding a pass rate, and a first run can
only write a baseline and say so. A baseline recorded against a different container or
adapter is refused rather than compared, because reporting a deliberate model change
as a regression is how a detector gets ignored.

The review-comment proxy has two implementations, and the tier-1 one carries a caveat:
it scores the cascade arm with the same model that chose that arm's candidate. The
base-versus-A1 comparison that decides M3 is unaffected (neither arm touches tier 1),
but a cascade-arm win on that metric is partly the verifier agreeing with itself.

### sec 11 — Milestones and gates

| Gate | Command | Threshold source | State |
|---|---|---|---|
| M0 Gate A | `tandem bench latency` | `gates.gate_a_*` | **measured** — decode passes, TTFT fails at 32k |
| M0 Gate B | `tandem bench tier1` | `gates.gate_b_prefill_tok_per_s` | **measured** — 153.7 tok/s mean over 6 runs, passes the host floor, `meets_spec: false`. **The red is compute, not the architecture the 200 was written to detect — `HANDOFF.md` T32 before quoting it** |
| M2 tool-call | `tandem gate toolcall --runs 100` | `toolcall_gate` (0.99, not host-configurable) | **measured** — 1.00 |
| M2 isolation | `tandem gate isolation` | byte-identity, no threshold | built; never run on hardware |
| M3 merge eval | `tandem eval merge` | ≥3 of 5 metrics | open — needs a trained A1 |
| M6 audit | `tandem audit verify` | chain integrity | built |

**Gate thresholds are host-relative and the spec figure travels with every result.**
`tandem.thresholds` holds the sec 11 numbers; `[gates]` in `tandem.toml` sets what a
given host is judged against. Each report carries `budget` (effective), `spec_budget`,
`pass` and `meets_spec`, plus `relaxed_criteria` naming every row that is green only
because a floor was lowered. A pass against a relaxed floor means "this host cleared
its own floor", never "Gate A passed". See `BASELINE.md` §7 for the four
values this machine sets and why.

---

## 2. Deviations from the specification

Three places where the implementation does something other than what the spec says
literally. Each preserves the spec's intent.

### 2.1 Merge commits are not unconditionally skipped (sec 6.2)

| | |
|---|---|
| **Spec** | `skip if: merge commit` |
| **Implementation** | `ExtractionFilters.merge_policy` defaults to `"auto"` |
| **Restore the literal spec** | `merge_policy = "skip"` |

The spec's reason is sound — a merge's diff is the union of other people's work with
no single intent behind it. But that is only true against an arbitrary parent. Walked
first-parent, `git diff M^1 M` on a two-parent merge is exactly the branch's
contribution, and the subject line is the PR title.

It decides whether there is a corpus at all. A squash-merge repo puts merged work in
ordinary commits and skipping merges costs nothing; a merge-commit repo puts *all* of
it in merge commits. Measured on `pallets/click` over 150 commits: **10 usable pairs
with merges skipped, 131 with first-parent diffs.** Skipping would have reported a
thin corpus and sent the customer away from a repository with perfectly good history.

Octopus merges (>2 parents) are always skipped — there is no single branch whose
contribution the diff represents.

### 2.2 Compaction staleness is a ratio (sec 8.2)

| | |
|---|---|
| **Spec** | Version templates against harness versions; a stale fingerprint silently degrades. |
| **Implementation** | A template declares markers and `stale_below_ratio` (default 0.5). A match below that fraction is accepted but flagged. |

Requiring a full marker set would flag every real request as stale — harness prompts
legitimately vary by platform, flags and version. A signal that fires constantly is a
signal that gets ignored, which lands in exactly the silent degradation the spec warns
about, arrived at from the other direction.

### 2.3 A2's "first review comment" has a local approximation (sec 6.3)

| | |
|---|---|
| **Spec** | `rejected` = the diff as of the first review comment. |
| **Implementation** | With a forge export (`--reviews`), exactly that. Without one, the first branch commit — what the author proposed before review touched it. |

Review timestamps live in the forge, and reaching github.com during extraction would
break the airgap claim (sec 8.6) the offline verification script exists to prove. The
approximation is labelled in every record (`source: branch_review` vs `forge_review`)
so a corpus can be sliced by signal strength during ablation.

---

## 3. Known gaps

**The gap list is `HANDOFF.md` §5.** It was duplicated here and in `BASELINE.md` §9, three
partial copies of one list that had already drifted apart — two of them still described
rung 1 as "arithmetically dead" months after `BASELINE.md` §4.3 measured it at 165 tok/s
and withdrew that. One tracker, one home.

What belongs here instead is the one gap that is a property of this *map* rather than of
the project's state:

| # | Gap | Blocks | Note |
|---|---|---|---|
| 1 | **Streaming is not incremental for multi-candidate, `plan`, or tool-bearing turns** | Nothing — deliberate | Best-of-N cannot honestly stream: you cannot emit tokens from a candidate before the verifier has chosen it, and streaming candidate 0 then retracting would be worse than a pause. A `plan` turn's text is rewritten when the critique is appended; a tool-bearing turn needs the whole reply before repair can run. Those run to completion and emit one delta, with the reason in `/tandem/trace/last`. |

### What is no longer a gap

Kept because the previous version of this file asserted them, and a doc that quietly
drops a claim teaches nothing.

- ~~"No MLX backend has met a real model."~~ Tier 0 has: `Qwen3.6-35B-A3B-OptiQ-4bit`,
  23.0 GiB, on a 36 GB M4 Max. The claim before that was that both files were
  *unexecutable* off Apple Silicon, which was half wrong, and the wrong half hid two
  silent bugs for months: `mlx_tier1.py` imports no MLX at all, and `mlx_tier0.render`
  was dropping every tool call and tool result out of the prompt.
- ~~"No gate has been run against a real model."~~ Gate A and the sec 10.2 tool-call
  gate both have. Their numbers are in `var/gate-a.json` and
  `var/gate-toolcall-constrained.json`, and summarised in `BASELINE.md`.
- ~~"There is no queue of hardware-independent work."~~ Item 1 above is exactly that.
