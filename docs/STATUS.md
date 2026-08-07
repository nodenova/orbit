# Implementation status

Section-by-section map from the technical specification to the code, with what is
proven, what is written but unexercised, and where the implementation deviates.

Legend: **built** = implemented and covered by tests · **written** = implemented but
cannot run against real hardware off Apple Silicon · **open** = not started.

A **written** row now usually means "runs, against a stand-in". `mlx_tier1.py` needed
no stand-in at all — sec 5.4 puts mlx-optiq behind a process boundary, so that file is
an httpx client and `tests/test_mlx_tier1.py` drives it with a `MockTransport`.
`mlx_tier0.py` runs against `tests/fake_mlx.py`. Neither has met a real model: the
wiring is checked, the numbers are not, and no row below is promoted to **built** on
the strength of a fake.

---

## 2 — Hardware budget

Not code. The numbers appear where they bind: `Tier1Config.expert_cache_bytes`
(18 GB), `Tier0Config.max_kv_tokens` (32k), `train.preflight()` (tier 1 must be
unloaded to train a 35B adapter on 64 GB), and `eval/latency.py`, which records SSD
capacity with every published measurement because capacity *is* a performance spec
(sec 2.3).

## 3 — Architecture

| Component | Module | State |
|---|---|---|
| Gateway | `tandem/gateway/` | built |
| Router | `tandem/router/` | built |
| Tier 0 | `tandem/backends/mlx_tier0.py` | written |
| Tier 1 | `tandem/backends/mlx_tier1.py`, `tandem/tier1/` | written / built |
| Attestation | `tandem/attest/` | built |
| Adapter pipeline | `tandem/adapters/` | built |

## 4 — Tier 0

| Spec | Where | State |
|---|---|---|
| 4.1 model, MTP | `Tier0Config` | built (config) |
| 4.2 adapter mounting, no merging, ContextVar binding, mount-at-startup | `backends/mlx_tier0.py` | written |
| 4.2 isolation test (blocking) | `eval/gates.py::adapter_isolation_gate` | built |
| 4.3 targeting and rank | `mlx_tier0._is_target`, `adapters/profile.py` | written / built |

`MultiAdapterLinear` holds every mounted adapter's `(A, B)` at once and selects per
request. The isolation gate is what proves it, and it is now run two ways:
vacuously in CI against a backend with no adapters mounted (so the gate itself
cannot rot), and for real in `tests/test_mlx_tier0.py` against `MLXTier0Backend`
with two overlapping adapters mounted under the fake MLX. The suite also drives a
deliberately leaky wrapper past it, because a gate nobody has seen fail is a gate
nobody has tested. What is still untested off-target is what the gate exists for:
quantised deltas and Metal.

## 5 — Tier 1

| Spec | Where | State |
|---|---|---|
| 5.1 three call types, schema-constrained, verifier-only | `tier1/schemas.py`, `tier1/verifier.py` | built |
| 5.2 2-bit caveat mitigations | schema `additionalProperties: false`; verdict validation | built |
| 5.3 prefill measurement | `mlx_tier1.measure_prefill`, `gate_b_report` | built (instrument) / open (numbers) |
| 5.4 mlx-optiq behind a process boundary | `mlx_tier1.OptiqTier1Backend` | built |
| 5.5 rung 1 (streamed) | `build_tier1()`, `backends/mlx_tier1.py` | built (client) / open (engine) |
| 5.5 rung 2 (80B resident-swapped) | `backends/resident_swap.py` | built (policy) / open (MLX occupants) |
| 5.5 rung 3 (second opinion) | `backends/second_opinion.py` | built |
| 5.5 rung 4 (remote) | `backends/remote_tier1.py`, `tools/remote_tier1.py` | built |

Output ceilings are enforced in code (`CALL_BUDGETS` in `backends/tier1_call.py`),
not requested of the model. The verifier API has no `generate`. The clamp and the
schema validation moved out of `mlx_tier1.py` once there were two transports: a clamp
that is right in one and drifts in the other is not a clamp — and the move gave the
ceiling that keeps tier 1 a verifier its first test, since it had lived only in a file
nothing had ever imported. `tests/test_mlx_tier1.py` now checks the clamp again on the
wire, where a transport could still drop it between the helper and the request body.

`tier1.rung` selects which rung serves the verifier. Selected, never descended:
nothing falls from one rung to the next on an error, which matters most at rung 4.
`Tier1Attestation.rung` records which rung produced every verdict, so a base-model
second opinion never reads as a streamed-verifier one.

**Rung 3** — tier 0 with its adapter unmounted — needs no second model and is the rung
actually available during M0–M3. The adapter strip is the whole mechanism: an adapted
model judging its own candidates is asking whether it agrees with itself.

**Rung 2** evicts tier 0 to admit an 80B. Residency is exclusive, so `ResidencySwitch`
is real mutual exclusion and `Pipeline` puts tier 0 behind `SwapGuard` — an unguarded
tier 0 would be asked to generate from weights that are not in memory. The swap back is
lazy (whoever is resident stays until someone else needs the memory), the switch
measures every transition, and the backend declines once the *measured* round trip
exceeds `tier1.swap_budget_s`, degrading to a failed `Verdict`. The policy is built and
tested against the mock, which is where the concurrency bugs are. What needs the
hardware is one of the two `Occupant`s. The evictable MLX tier 0 is done and runs:
`MLXTier0Backend.load`/`.unload` are exercised against `tests/fake_mlx.py`, and a
round trip through them returns the same container and adapter identity. The
resident 80B verifier on the far side does not exist — `build_tier1` says so
precisely rather than building a swap with nothing to swap into.

**Rung 4** sends the repository's code to a third party, so the gates in front of it
are the implementation: the ladder never reaches it on its own, `tier1.remote_consent`
must carry "tier 1 leaves this machine" verbatim, the transport is a file outside the
package (`tools/remote_tier1.py`, loaded by path — the package still makes no outbound
call, and the test pinning that surface is unchanged), and `OfflineReport.ok` is false
whenever the rung is armed, whatever `lsof` caught. A remote verdict has **no container
attestation** — `container_hash()` is None by construction, because you cannot attest
to a model you do not hold, and the rung in the receipt is what says the null is a
property rather than a gap.

## 6 — Adapter pipeline

| Spec | Where | State |
|---|---|---|
| 6.1 A0 synthetic harness traces | `adapters/extract_a0.py` | built |
| 6.2 A1 git-history extraction, filters, messages-JSONL, thin-corpus detection | `adapters/extract_a1.py`, `filters.py`, `gitwalk.py` | built |
| 6.3 A2 DPO pairs from review history + reverts | `adapters/extract_a2.py` | built |
| 6.4 routing profile, count- vs mass-ranking | `adapters/profile.py` | built |
| training driver (SFT + DPO), NEFTune, collapse detection | `adapters/train.py` | built |

## 7 — Router

| Spec | Where | State |
|---|---|---|
| 7.1 turn classification | `router/classify.py` | built |
| 7.2 T1 best-of-N rerank | `router/cascade.py` | built |
| 7.2 T2 failure escalation, bounded to one per turn | `router/cascade.py`, `eval/worktree.py` | built |
| 7.3 latency contract + automatic pressure valve | `cascade._record`, `eval/latency.CONTRACT` | built |

T2 needs a host that can run the repository's tests, so `Pipeline` builds a
`WorktreeRunner` from `[eval]` and hands it to `Cascade`. Off unless the config opts
in: turning it on runs the suite on every `code_change` turn. With it off `Cascade`
reports the path dormant rather than answering "passed" to escalation checks it
never made.

## 8 — Gateway

| Spec | Where | State |
|---|---|---|
| 8.1 three wire protocols | `gateway/wire/{anthropic,openai_chat,openai_responses}.py` | built |
| 8.1 incremental SSE | `wire/*.StreamEncoder`, `Pipeline.stream`, `app._sse` | built |
| 8.2 compaction, versioned templates, stripping, `--no-compact`, diff view | `gateway/compaction.py` | built |
| 8.3 context scaling | `gateway/context_scale.py` | built |
| 8.4 prompt cache + disk KV | `gateway/cache/` | built |
| 8.5 prevent / train / repair / retry / replay | `gateway/toolcall/` | built |
| 8.6 offline posture + verification script | `tandem/offline.py`, `tandem doctor` | built |

## 9 — Attestation

| Spec | Where | State |
|---|---|---|
| 9.1 response metadata | `attest/receipt.py` | built |
| 9.2 append-only audit log | `attest/audit.py` | built |
| 9.3 G1 / G2 determinism gates | `eval/gates.py` | built |
| 9.4 provenance of training data | `attest/provenance.py` | built |

## 10 — Evaluation

| Spec | Where | State |
|---|---|---|
| 10.1 repo-held-out merge eval, four bars | `eval/merge_eval.py` | built |
| 10.1 test pass + convention conformance | `eval/worktree.py` | built |
| 10.1 review-comment proxy | `merge_eval.tier1_review_proxy`, `scored_review_proxy` | built |
| 10.2 tool-call validity gate (blocking) | `eval/gates.py::toolcall_gate` | built |
| 10.3 regression suite | `eval/regression.py`, `eval/regression_items.py` | built |
| 10.4 latency suite | `eval/latency.py` | built |
| 10.5 measurement discipline | `eval/latency.Environment` | built |

All five metrics are now measurable, so `compare_arms` can return an M3 verdict —
given a `[eval]` block naming the repo's linters and test command. Without one it
still reports "not measured" and refuses the gate, which is the same behaviour as
before and remains the correct one.

The regression suite is 90 fixed short-answer items across reasoning, exact-answer
maths and code localisation. Its output is a **diff against a recorded baseline**,
not a score — `RegressionReport` has no field holding a pass rate, and a first run
can only write a baseline and say so. That is the spec's "not a leaderboard number"
made structural: a field holding one is all it takes for a number to end up on a
slide. A baseline recorded against a different container or adapter is refused
rather than compared, because reporting a deliberate model change as a regression
is how a detector gets ignored.

The review-comment proxy has two implementations, and the tier-1 one carries a
caveat worth repeating: it scores the cascade arm with the same model that chose
that arm's candidate. The base-versus-A1 comparison that decides M3 is unaffected
(neither arm touches tier 1), but a cascade-arm win on that metric is partly the
verifier agreeing with itself. Real review history — `--review-proxy file:`, and A2
when it exists — is the stronger signal.

## 11 — Milestones

`tandem bench latency` reports M0 Gate A with its kill condition. `tandem bench
tier1` reports Gate B. `tandem eval merge` reports the M3 gate. `tandem gate
toolcall` and `tandem gate isolation` are the M2 gates. `tandem audit verify` is
part of M6.

---

# Deviations from the spec

Three places where the implementation does something other than what the spec says
literally. Each preserves the spec's intent.

### 1. Merge commits are not unconditionally skipped (sec 6.2)

**Spec:** `skip if: merge commit`.

**Implementation:** `ExtractionFilters.merge_policy` defaults to `"auto"`.

**Why.** The spec's reason is sound — a merge's diff is the union of other people's
work with no single intent behind it. But that is only true against an arbitrary
parent. Walked first-parent, `git diff M^1 M` on a two-parent merge is exactly the
branch's contribution, and the subject line is the PR title.

It matters because it decides whether there is a corpus at all. A squash-merge repo
puts merged work in ordinary commits and skipping merges costs nothing. A
merge-commit repo puts *all* of it in merge commits. Measured on `pallets/click`
over 150 commits: **10 usable pairs with merges skipped, 131 with first-parent
diffs.** Skipping would have reported a thin corpus and sent the customer away from
a repository with perfectly good history.

Octopus merges (>2 parents) are always skipped — there is no single branch whose
contribution the diff represents. `merge_policy="skip"` restores the literal spec.

### 2. Compaction staleness is a ratio, not "fewer than all markers" (sec 8.2)

**Spec:** version templates against harness versions; a stale fingerprint silently
degrades.

**Implementation:** a template declares markers and `stale_below_ratio` (default
0.5). A match below that fraction is accepted but flagged.

**Why.** Requiring a full marker set would flag every real request as stale —
harness prompts legitimately vary by platform, flags and version. A signal that
fires constantly is a signal that gets ignored, which lands in exactly the silent
degradation the spec warns about, arrived at from the other direction.

### 3. A2's "first review comment" has a local approximation (sec 6.3)

**Spec:** `rejected` = the diff as of the first review comment.

**Implementation:** with a forge export (`--reviews`), exactly that. Without one,
the first branch commit — what the author proposed before review touched it.

**Why.** Review timestamps live in the forge, and reaching out to github.com during
extraction would break the airgap claim (sec 8.6) that the offline verification
script exists to prove. The approximation is labelled in every record
(`source: branch_review` vs `forge_review`) so a corpus can be sliced by signal
strength during ablation.

---

# Known gaps

Everything below is either blocked on the M4 Max or is a deliberate limit. There is
no longer a queue of hardware-independent work.

- **No MLX backend has met a real model.** Both files now execute under CI —
  `mlx_tier0.py` against `tests/fake_mlx.py`, `mlx_tier1.py` against a
  `MockTransport` standing where the mlx-optiq socket does — so an import error, a
  signature drift or a broken forward is a test failure rather than a discovery on
  the hardware. What that does *not* cover is everything the gates exist for: real
  4-bit weights, int8 adapter deltas, Metal, and every number. `tandem gate
  isolation` and G1/G2 on the M4 Max are still what prove tier 0.

  The prior version of this line said the files were unexecutable off Apple
  Silicon. That was half wrong, and the wrong half hid two silent bugs for months:
  `mlx_tier1.py` imports no MLX at all, and `mlx_tier0.render` was dropping every
  tool call and tool result out of the prompt.
- **`mlx_tier0.supports_state()` returns False.** The disk KV cache is wired end to
  end and tested against the mock; what is missing is serialising an `mlx_lm`
  prompt cache to bytes and back. The method says so explicitly rather than
  inheriting the default, and documents what an implementation must guarantee.
  Until it lands the real backend prefills cold after a restart — slow, not wrong.
- **Rung 2's occupants are not implemented (sec 5.5).** The residency policy is built
  and tested; what needs the M4 Max is a resident 80B verifier backend to sit on the
  other side of the switch. `MLXTier0Backend.load`/`.unload` are exercised against the
  fake — an unload/load round trip must come back with the same adapters and the same
  hashes — but have never moved a real 20 GB of weights.
- **Streaming is not incremental for multi-candidate, `plan`, or tool-bearing
  turns.** Best-of-N cannot honestly stream — you cannot emit tokens from a
  candidate before the verifier has chosen it, and faking it by streaming candidate
  0 and retracting would be worse than a pause. A `plan` turn's text is rewritten
  when the tier-1 critique is appended, and a tool-bearing turn needs the whole
  reply before the tool-call layer can repair it. Those run to completion and emit
  one delta, with the reason recorded in `/tandem/trace/last`. Everything else
  streams token by token.
- **The routing-profile forward pass is not implemented.** `tandem profile` builds
  and validates the sidecar from an activation dump; producing that dump needs the
  real model and belongs with the tier-0 backend.
- **No gate has been run against a real model.** Every gate in `eval/` is built and
  tested against `MockBackend`, which means the *plumbing* is proven and none of
  the *numbers* are. Nothing in this repository has yet measured a 35B model.
- **Experiments E1–E4 are not run.** They need the target hardware.

---

# What to do next

The full sequence, with its ordering rationale, is in `docs/HANDOFF.md`. In short,
and in this order because each step can invalidate the ones after it:

1. **M0 Gate A** (`tandem bench latency`) — carries a kill condition. If tier 0
   cannot serve interactively, the product is async review-assist instead.
2. **M0 Gate B** (`tandem bench tier1`) — decides whether the in-house streaming
   loader is a three-week optional or M-blocking.
3. **`tandem gate isolation`** against real mounted adapters — until it passes,
   every receipt naming an adapter is unproven.
4. **`tandem eval regression`** once, to record a baseline *before* changing
   anything. A detector with no reference point detects nothing.
5. **Train A0**, re-run `tandem gate toolcall` (experiment E1).
6. **Train A1, run `tandem eval merge`** with an `[eval]` block naming the repo's
   linters and test command. This is the product thesis. If A1 does not beat base
   on ≥3 of 5 metrics, stop and re-plan before building tier 1 — which is the whole
   reason M3 sits at week 8, before tier 1 exists.
7. **G1/G2**, then E2–E4.
