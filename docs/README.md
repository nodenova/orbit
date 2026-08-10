# Orbit documentation

Five documents. **Each fact has exactly one home**, and the others link to it rather than
repeating it — that rule is the whole point of this directory, and it is the rule the
previous version broke. There were three copies of the gap list, two trackers with
overlapping IDs, and four places asserting `supports_state()` was False after it had been
True for a week. A fact stated twice is a fact that will disagree with itself.

| Document | Owns | Read it when you need to know |
|---|---|---|
| **[architecture.md](architecture.md)** | **The shape, the invariants, the traps** | How a request flows, where the portability line is, why a piece of code is written the strange way it is, and what has already gone wrong here. **Start here.** |
| **[platform.md](platform.md)** | **Every number** | What the reference machine measurably does — memory ceilings, throughput, and every budget and gate threshold derived from them. |
| **[spec-map.md](spec-map.md)** | **Specification section → code** | Which module implements `sec 8.2`, whether it is proven or merely written, and where the implementation deviates from the spec on purpose. |
| **[operations.md](operations.md)** | **What is safe to start** | What may already hold the GPU, which commands load 23 GiB, and the host-health gate that decides whether a measurement can be believed at all. |
| **[constrained-decoding.md](constrained-decoding.md)** | **Where the 2.4× goes** | The cost model, and the four changes that would remove it: F1 and F2 are built and measured, worth 11–15%; F3 is rejected on measurement; F4 is capped at ~1.11×. Committed because `src/orbit/backends/mlx_tier0.py` cites §5 of it directly. |

## Where to start

**New to the project** → [`../README.md`](../README.md) for what it is and how to run it,
then [architecture.md](architecture.md) for how it fits together.

**Changing code** → [spec-map.md](spec-map.md) to find the module,
[architecture.md](architecture.md) §4 for the invariants that look simplifiable and are
not, and §5 for the traps. [`../CLAUDE.md`](../CLAUDE.md) is the working conventions.

**Running anything against real weights** → [operations.md](operations.md) first, every
time. One process holds the GPU at a time and nothing enforces it.

**Quoting a number** → [platform.md](platform.md), which labels every figure *measured*,
*derived* or *published* and gives the command that reproduces it.

## What is deliberately not here

**State, the ordered plan and the tracker are not in `docs/`.** They live in
`specs/PROGRESS.md`, which is **gitignored** — it does not survive a clone, and that is the
trade. This directory is for what stays true; a tracker is stale by design. The same goes
for the full model write-ups (`specs/experiments/`) and the session diaries.

The rule that keeps this honest runs one way only:

> A file in `specs/` may cite anything. **A committed file may never cite `specs/`.**

A citation that does not survive a clone is not a citation — and **a reproduction command
is a citation**, so a measurement whose script died with the container it ran in is not
reproducible. This has been got wrong three times; it is trap 11 in
[architecture.md](architecture.md) §5.

## Conventions

**Status vocabulary** — used in every table across these files, defined in
[spec-map.md](spec-map.md#status-vocabulary):

| | |
|---|---|
| **built** | implemented, covered by tests |
| **written** | implemented, runs only against a stand-in — proves wiring, not the model |
| **measured** | run against real weights, number recorded under `var/` |
| **open** | not started, or blocked on something named |

**`sec N.M`** references point at the v1 technical specification, which is **not in this
repository**. [spec-map.md](spec-map.md) maps every section to the code that implements
it. Do not invent a meaning for a section you cannot resolve.

**The baseline platform** is a MacBook Pro M4 Max, 36 GB unified memory, 1 TB SSD. Every
budget in the repository derives from it rather than from a larger machine —
[platform.md](platform.md) is that derivation.

**If a document and the code disagree, the code is right** and the document needs fixing in
the same commit. A doc that is 80% right is worse than none, because the wrong 20% is
indistinguishable from the rest.
