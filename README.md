# Tandem

[![CI](https://github.com/nodenova/orbit/actions/workflows/ci.yml/badge.svg)](https://github.com/nodenova/orbit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-green)](#licence)

**A local coding-agent runtime that optimises for merge quality.**

Two model tiers on one machine. A fast resident model, adapted to your repository
from its own git history, generates candidate patches. A large streamed model reads
those candidates and picks or rejects them — a verifier, never a generator. A receipt
proves which base and which adapter produced each change.

> [!IMPORTANT]
> **Pre-hardware.** Everything above the backend interface is built and tested. The
> MLX backends beneath it now execute under CI against stand-ins, which proves they
> are wired correctly and nothing more: **no gate has yet been run against a real 35B
> model.** The plumbing is proven. None of the numbers are. See
> [What works today](#what-works-today) before you plan anything around it.

---

## Contents

- [Why this shape](#why-this-shape)
- [What works today](#what-works-today)
- [Requirements](#requirements)
- [Install](#install)
- [Run](#run)
- [Configure](#configure)
- [Build an adapter](#build-an-adapter)
- [Gates and evals](#gates-and-evals)
- [Design notes worth knowing before you change something](#design-notes-worth-knowing-before-you-change-something)
- [Repository map](#repository-map)
- [Testing and CI](#testing-and-ci)
- [Contributing](#contributing)
- [Licence](#licence)

---

## Why this shape

Three facts, one product.

1. **Streamed models are ~300× cheaper per input token than per output token.**
   Decode streams top-k experts per token (~4–10 tok/s — unusable). Prefill with
   batch-union reads each expert once per chunk (~650–1,300 tok/s). Every published
   number in the field is single-request decode, so the field concluded streamed
   models are too slow. That is correct *for generation* and wrong for any task
   where input dominates output.

2. **The tasks where input dominates output are exactly the tasks that determine
   merge quality.** Reranking N candidates emits an integer. Reviewing a diff emits
   a verdict. All read 5–30k tokens and write 10–300.

3. **Merge quality, not benchmark score, is the binding constraint.** METR's
   standing finding: ~half of SWE-bench-passing PRs would not be merged by
   maintainers. A local 35B at 73.4% SWE-bench Verified is not short of capability —
   it is short of *this repository's* conventions, which is what a repo-derived
   adapter encodes and a verifier pass enforces.

---

## What works today

The **hardware-independent core is built and tested**: gateway, router, tool-call
reliability, adapter pipeline, evaluation, attestation. 399 tests, all passing, on an
ordinary Linux box.

The **MLX backends run, against stand-ins.** `mlx_tier1.py` needs none — sec 5.4 puts
mlx-optiq behind a process boundary, so that file is an httpx client and a
`MockTransport` sits where the socket does. `mlx_tier0.py` runs against a small model
of the MLX surface (`tests/fake_mlx.py`), far enough to mount adapters, generate, and
pass the sec 4.2 isolation gate for real rather than vacuously.

Take that for exactly what it is: the wiring is checked, the model is not. No real
4-bit weight, int8 adapter delta or Metal kernel has been touched, and
`tandem gate isolation` plus the determinism gates on the target machine are still
what prove tier 0.

Everything above the backend interface runs against `MockBackend` — a deterministic,
adapter-sensitive, faultable stand-in. Good enough that the whole system is testable
without a MacBook, and honest enough that the tests mean something.

| Milestone | Component | State |
|---|---|---|
| M1 | Gateway — three wire protocols, compaction, caches, tool-call layer | built, tested |
| M4 | Router — classification, best-of-N, escalation, pressure valve | built, tested |
| M4 | Tier 1 — verifier API, schemas, the four-rung fallback ladder | built, tested; the streamed client and rung 2's occupants need the target machine |
| M3 | Adapter pipeline — A0/A1/A2 extraction, routing profile, training driver | built, tested |
| M6 | Attestation — hashes, receipts, hash-chained audit log, provenance | built, tested |
| M0/M2/M5 | Gates and evals — merge eval, tool-call gate, isolation, G1/G2, latency | built, tested |
| M2 | Tier 0 — MLX resident backend with multi-adapter mounting | written, **needs an M4 Max** |

The distinction that matters most: every gate's *plumbing* is proven; none of its
*numbers* are. [`docs/STATUS.md`](docs/STATUS.md) has the section-by-section map,
the three documented deviations, and the full list of known gaps.

---

## Requirements

Running the gateway, the extractors, the evals and the whole test suite needs only
**Python 3.11+**. That is the half of the system that is portable, and it is the half
you can develop against today.

Serving a real model needs the target machine the design is budgeted for: a
**MacBook Pro M4 Max, 64 GB unified memory, 1 TB SSD**. SSD capacity is a performance
spec rather than a storage one — Apple SSD read bandwidth scales with NAND package
count, and the 1 TB bin is roughly 2× the smaller one on the tier-1 path.

Python 3.14 is the current release and 3.11 the declared floor; CI runs both, so both
are guarded. 3.12 and 3.13 sit between two tested ends rather than being tested
themselves. The suite also passes on the free-threaded 3.14 build, which is a
data point rather than a supported configuration — nothing in CI holds it there.

## Install

```bash
pip install -e '.[dev]'                # core + tests, runs anywhere
pip install -e '.[dev,constrain]'      # + constrained decoding
pip install -e '.[dev,mlx]'            # + Apple Silicon backends
```

`[dev]` and `[dev,constrain]` exercise genuinely different paths — repair versus
prevention (sec 8.5) — and both are expected to pass.

Licence discipline for dependencies is Apache-2.0 / MIT / BSD only. Every dependency
is a security surface, not a convenience; the LiteLLM 1.82.7/1.82.8 supply-chain
compromise is the standing precedent.

## Run

```bash
tandem doctor                          # runtime status + offline posture
tandem serve --backend mock            # gateway on 127.0.0.1:8080
```

Point a harness at it:

```bash
source <(tandem offline-env)           # the airgap environment (sec 8.6)
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude --bare
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 opencode
```

Three wire protocols from one process, sharing one model, one prompt cache and one
router — because a product whose entire value depends on one closed client is one
product decision from zero:

| Endpoint | Serves |
|---|---|
| `POST /v1/messages` | Claude Code, OpenClaw |
| `POST /v1/chat/completions` | OpenCode, Crush, generic |
| `POST /v1/responses` | Codex |

Local introspection, no network: `/tandem/health`, `/tandem/stats`,
`/tandem/compaction/last` (the diff view), `/tandem/trace/last`,
`/tandem/audit/verify`.

Streaming is incremental where it can be: a `chat` turn carrying no tools emits
tokens as they are decoded, so TTFT is first-token latency and the sec 7.3 budget is
actually served. A `code_change` turn under best-of-N runs to completion first and
says so in `/tandem/trace/last` — there are no tokens to emit until the verifier has
chosen a candidate, and streaming candidate 0 and retracting it would be worse than a
pause.

## Configure

Copy [`tandem.toml.example`](tandem.toml.example) to `tandem.toml`. Every knob is
documented in place, and **unknown keys are a hard error** rather than a silent
no-op — a typo'd `expert_cache_bytes` that appears to take effect and does not is how
a wrong number ends up in a published measurement.

Two blocks are worth knowing about before anything else.

**`tier1.rung`** picks which rung of the sec 5.5 fallback ladder serves the verifier:

| Rung | `tier1.rung` | What it is |
|---|---|---|
| 1 | `streamed` | The 122B design target, experts streamed from NVMe. |
| 2 | `resident_swap` | An 80B swapped into residency, evicting tier 0 — ~10 s each way, and tier 0 cannot serve while it is in. |
| 3 | `second_opinion` | Tier 0 with its adapter unmounted. Free, weak, needs no second model, and still catches adapter overfit. The rung available before the 122B container exists. |
| 4 | `remote` | Somebody else's API. Breaks the offline claim, so it needs `tier1.remote_consent` written out in full. |

**`[eval]`** declares what the repository runs to check itself. Three of the merge
eval's five metrics and the whole of T2 failure escalation depend on it, and there is
no way to guess it:

```toml
[eval]
linters = [["ruff", "check"]]
test_command = ["pytest", "-q", "-x"]
```

Leave it out and those metrics report as *not measured*, which is the honest answer
rather than a passing one.

## Build an adapter

```bash
# A0 — harness adapter. Universal, cheapest, the sleeper win.
tandem extract a0 --n 4000 --out corpus/a0
tandem train sft --corpus corpus/a0/train.jsonl --out adapters/a0 \
    --name a0-harness --source-kind synthetic_harness

# A1 — repository adapter from your own git history.
tandem extract a1 --repo . --holdout 25 --out corpus/a1
tandem train sft --corpus corpus/a1/train.jsonl --out adapters/a1-myrepo \
    --name a1-myrepo --repo .

# A2 — reviewer adapter, DPO from review history. Starts from A1.
tandem extract a2 --repo . --out corpus/a2
tandem train dpo --corpus corpus/a2/train.jsonl --out adapters/a2-myrepo \
    --name a2-myrepo --repo . --mount-adapter adapters/a1-myrepo
```

`extract` exits **2** when the corpus is too thin (< 500 pairs for A1). That is not a
failure to work around — below that, A1 underfits, and the honest answer is to tell
the customer rather than ship a null adapter.

A2 is stronger with real review timestamps, which live in the forge rather than in
git. [`tools/export_reviews.py`](tools/export_reviews.py) fetches them; without it,
extraction falls back to the first branch commit and labels every pair with which
signal produced it.

## Gates and evals

```bash
tandem gate toolcall --runs 100    # sec 10.2 — blocking, ≥99% well-formed
tandem gate isolation              # sec 4.2  — blocking, adapter deltas must not leak
tandem eval merge --repo . --a1 a1-myrepo    # sec 10.1 — the four bars
tandem eval regression             # sec 10.3 — diff against a baseline, not a score
tandem bench latency               # sec 10.4 + M0 Gate A
tandem bench tier1                 # M0 Gate B — streamed prefill ≥200 tok/s
tandem audit verify                # sec 9.2 — hash-chained log
```

`tandem eval merge` is the one that matters. **If the adapter doesn't beat the base
model on the customer's own repo, there is no product** — and it is designed to say
so at M3, week 8, before tier 1 is built.

Each generated patch is applied in a detached git worktree at the held-out change's
own parent commit, and the `[eval]` commands run there — never in your checkout, and
never against the files as they sit on disk, which would score the repository rather
than the patch.

`tandem eval regression` reports a **diff against a recorded baseline, not a score**.
`RegressionReport` has no field holding a pass rate, and a first run can only write a
baseline and say so. A field holding one is all it takes for a number to end up on a
slide.

---

## Design notes worth knowing before you change something

Each of these looks simplifiable and is not, and each fails as a silently wrong
answer rather than an error. The full list with rationale is
[`docs/HANDOFF.md`](docs/HANDOFF.md) §5; every one also carries a comment in
the code saying why.

**Adapters are never merged into the base** (sec 4.2). Merging is the obvious
implementation and it destroys the product: ~20 GB per adapter instead of ~250 MB, no
multi-tenancy, and a receipt that cannot name what produced a change. The forward
stays `y = xW + s·(xA)B` with deltas resident and separate, selected per request via a
`ContextVar` — a module global would race under concurrency with a silent
wrong-answer failure mode.

**Tier 1 never generates a patch** (sec 5.1). The interface has no `generate`, and
`max_tokens` is clamped per call type. The same model that reranks five candidates in
18 s takes six minutes to write one. That asymmetry is the whole thesis, so it is
enforced structurally rather than by convention.

**The fallback ladder is selected, never descended** (sec 5.5). `tier1.rung` names
which rung serves the verifier, and no error path picks a different one. That matters
most at rung 4, a remote API: it sends the repository's code off the machine, so it
needs `tier1.remote_consent` written out in full, its HTTP lives outside the package,
and `tandem doctor` stops reporting a clean offline posture the moment it is armed. A
ladder that reached it in response to a timeout would turn an airgapped runtime into
an exfiltrating one with nobody choosing it.

**The KV cache is plain `read`/`write`, never mmap** (sec 8.4). A process already
mapping ~30 GB of weights should not add more VM mappings. This is the most likely
place for a well-meaning optimisation to regress the runtime.

**The tool-replay map is not optional** (sec 8.5.5). Clients hand tool calls back as
normalised JSON, not the model's sampled bytes. Re-rendering them differently breaks
the prompt's byte prefix, the cache misses, and the whole turn rebuilds — on every
tool-using turn, which is all of them.

**Compaction is fingerprinted with a staleness signal** (sec 8.2). A template pinned
to an exact prompt silently stops matching when the harness updates, and the customer
quietly loses the ~28× with no error anywhere. Detection is scored, and a drifted
prompt is reported rather than absorbed.

**Nothing under `src/tandem/` makes an outbound network call.** The offline posture
(sec 8.6) is a claim a customer can verify with `lsof`, so the two network-capable
files live in `tools/` and are not installed. A test pins the package's network
surface to one file — the loopback process boundary to the tier-1 engine — and a
second import there is a test failure rather than a discovery.

**Merge commits are not always skipped** (sec 6.2, deviation — see
[`docs/STATUS.md`](docs/STATUS.md)).

---

## Repository map

```
src/tandem/
  gateway/      wire protocols, compaction, caches, tool-call layer  (sec 8)
  router/       turn classification, best-of-N, escalation           (sec 7)
  tier1/        the verifier API and its schemas                     (sec 5)
  backends/     the hard line — Backend, the mock, the MLX engines    (sec 4, 5)
  adapters/     A0/A1/A2 extraction, routing profile, training        (sec 6)
  attest/       receipts, hash-chained audit log, provenance          (sec 9)
  eval/         merge eval, gates, regression, latency                (sec 10)
tools/          not installed: the two files that may touch a network
```

`backends/base.py::Backend` is the line the whole layout turns on. Above it,
everything is pure Python that runs anywhere and is fully tested. Below it,
`mlx_tier0.py` needs Apple Silicon to serve a real model and `mlx_tier1.py` needs an
mlx-optiq process to talk to; both are exercised off-target against stand-ins.

Further reading:

- [`docs/STATUS.md`](docs/STATUS.md) — spec section → code map, what is proven, what
  is written but unexercised, the three documented deviations, and the known gaps.
- [`docs/HANDOFF.md`](docs/HANDOFF.md) — where the project stands, what to do
  next and in what order, and the decisions that look simplifiable and are not.
- [`CLAUDE.md`](CLAUDE.md) — the working conventions, for coding agents and humans
  alike.

The `sec 8.2` / `sec 6.3` references throughout the code point at a technical
specification that is **not in this repository**. `docs/STATUS.md` maps every section
to the code that implements it, which covers most day-to-day needs without it. Please
don't invent a meaning for a section you cannot resolve.

## Testing and CI

```bash
pytest -q                                          # the whole suite, ~10 s
pytest tests/test_router.py::test_n_equals_one_disables_reranking -q
```

The git-facing tests build small real repositories rather than mocking `git`, because
every failure mode that matters lives in git's actual output format — `-z` numstat
framing, rename records, merge parentage. A mock would only agree with whatever the
code already believes.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs two jobs:

- **tests** — the full suite with `[constrain]` installed, so the prevention layer is
  exercised rather than skipped as unavailable. On Python 3.11 and 3.14, the two ends
  of `requires-python`: the floor was a claim nothing executed until it was tested.
- **cli and gateway smoke** — starts a real `tandem serve`, drives all three wire
  protocols against it, asserts the receipt carries an engine commit, and verifies
  the audit chain over the records those requests wrote. Then the blocking gates and
  both extractors. This covers the one thing unit tests cannot: an actual uvicorn
  process serving the product surface.

## Contributing

Issues and pull requests are welcome. Two things worth knowing before you open one:

- **Run both install states.** `[dev]` and `[dev,constrain]` exercise different code
  paths, and CI runs the second.
- **`tandem gate toolcall --runs 100` is blocking**, and CI runs it. If you touched
  the gateway, the tool-call layer, compaction or sampling, run it locally first.

Comments in this codebase explain *why*, and especially why a simpler implementation
was rejected. Please keep that when editing — several of them are the only record of
a bug that has already been paid for once.

No linter or formatter is configured, and adding one is its own conversation rather
than a drive-by.

## Licence

Apache-2.0. See [`pyproject.toml`](pyproject.toml) for the declaration and the
dependency licence discipline that goes with it.
