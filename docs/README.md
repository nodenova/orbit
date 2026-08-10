# Orbit documentation

Eight documents. **Each fact has exactly one home**, and the others link to it rather than
repeating it — that rule is the whole point of this directory, and it is the rule the
previous version broke. There were three copies of the gap list, two trackers with
overlapping IDs, and four places asserting `supports_state()` was False after it had been
True for a week. A fact stated twice is a fact that will disagree with itself.

| Document | Owns | Read it when you need to know |
|---|---|---|
| **[HANDOFF.md](HANDOFF.md)** | **State, the plan, and the tracker** | What state the project is in, what to do next and in what order, what is open, and why a piece of code is written the strange way it is. **Start here.** |
| **[BASELINE.md](BASELINE.md)** | **Every number** | What the reference machine measurably does — memory ceilings, throughput, and every budget and gate threshold derived from them. |
| **[STATUS.md](STATUS.md)** | **Specification section → code** | Which module implements `sec 8.2`, whether it is proven or merely written, and where the implementation deviates from the spec on purpose. |
| **[PROCESSES.md](PROCESSES.md)** | **What is safe to start** | What may already hold the GPU, which commands load 23 GiB, and the host-health gate that decides whether a measurement can be believed at all. |
| **[A1_TRAINING.md](A1_TRAINING.md)** | **What step 3 needs before it can run** | The four inputs, the two silent defects in `build_sft_command` and the corpus layout, and the measured memory ceiling for a LoRA step on this host — **all-linear targeting peaks at 31.10 GB against a 30.15 GB ceiling.** Read before planning A1. |
| **[DEEPSEEK_V4.md](DEEPSEEK_V4.md)** | **What the streamed 92 GB model actually does here** | It generates (single-threaded, 2026-08-10), why `optiq serve` aborts on it, and the numbers that keep rung 3: 48 tok/s prefill against the 122B's 165. |
| **[QWEN3_CODER_NEXT.md](QWEN3_CODER_NEXT.md)** | **The streamed 45 GB model that serves, and passes** | Gate B at **258.0 tok/s with `meets_spec: true`** — the first on this host — at 1.36 GB resident, and why rung 3 still ships anyway. Also the `max_buffer_length` ceiling, which binds prefill and which `BASELINE.md` recorded as never having bitten. |
| **[CONSTRAINED_DECODE.md](CONSTRAINED_DECODE.md)** | **Where the 2.4× goes** | The cost model, and the four changes that would remove it: F1 and F2 are built **and measured — worth 11–15%**, F3 is rejected on measurement, F4 is open and capped. **§8 is the completed validation ladder and the one number to read: 22.15 ms/token of the penalty is unexplained (T26).** Committed because `BASELINE.md` used to point at it inside gitignored `/specs/`. |

## Where to start

**New to the project** → [`../README.md`](../README.md) for what it is and how to run it,
then [HANDOFF.md](HANDOFF.md) §1–2 for state.

**Changing code** → [STATUS.md](STATUS.md) to find the module, [HANDOFF.md](HANDOFF.md)
§6 for the invariants that look simplifiable and are not.
[`../CLAUDE.md`](../CLAUDE.md) is the working conventions.

**Picking the work up** → [HANDOFF.md](HANDOFF.md) §4 for the ordered plan and what each
step must *prove*, §5 for everything open.

**Running anything against real weights** → [PROCESSES.md](PROCESSES.md) first, every
time. One process holds the GPU at a time and nothing enforces it.

**Quoting a number** → [BASELINE.md](BASELINE.md), which labels every figure *measured*,
*derived* or *published* and gives the command that reproduces it.

## Conventions

**Status vocabulary** — used in every table across these files, defined in
[STATUS.md](STATUS.md#status-vocabulary):

| | |
|---|---|
| **built** | implemented, covered by tests |
| **written** | implemented, runs only against a stand-in — proves wiring, not the model |
| **measured** | run against real weights, number recorded under `var/` |
| **open** | not started, or blocked on something named |

**`sec N.M`** references point at the v1 technical specification, which is **not in this
repository**. [STATUS.md](STATUS.md) maps every section to the code that implements it. Do
not invent a meaning for a section you cannot resolve.

**The baseline platform** is a MacBook Pro M4 Max, 36 GB unified memory, 1 TB SSD. Every
budget in the repository derives from it rather than from a larger machine —
[BASELINE.md](BASELINE.md) is that derivation.

**`/specs/` is gitignored**, and holds a local copy of the v1 specification plus working
notes that are not ours to publish. **Nothing a future session needs may live there**, and
this has gone wrong twice: the handoff itself was `specs/NEXT_STEPS.md` until it moved
here, and the constrained-decode design was cited from a committed file while sitting in
`/specs/`. If a committed document points at it, it belongs here.

**If a document and the code disagree, the code is right** and the document needs fixing in
the same commit. A doc that is 80% right is worse than none, because the wrong 20% is
indistinguishable from the rest.
