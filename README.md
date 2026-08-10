# Tandem

[![CI](https://github.com/nodenova/orbit/actions/workflows/ci.yml/badge.svg)](https://github.com/nodenova/orbit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-green)](#licence)

**A local coding-agent runtime that optimises for merge quality.**

Two model tiers on one machine. A fast resident model, adapted to your repository from
its own git history, generates candidate patches. A large model reads those candidates
and picks or rejects them — a verifier, never a generator. A receipt proves which base
and which adapter produced each change.

It speaks the Anthropic, OpenAI chat-completions and OpenAI responses APIs from one
process, so Claude Code, OpenCode, Crush and Codex all point at the same
`127.0.0.1:8080` and share one model, one prompt cache and one router.

```bash
pip install -e '.[dev,constrain]'
tandem serve --backend mock                       # gateway on 127.0.0.1:8080
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude   # point a harness at it
```

---

## Contents

- [Why this shape](#why-this-shape)
- [Status](#status)
- [Requirements](#requirements)
- [Install](#install)
- [Run](#run)
- [Configure](#configure)
- [Build an adapter](#build-an-adapter)
- [Gates and evals](#gates-and-evals)
- [Design notes worth knowing before you change something](#design-notes-worth-knowing-before-you-change-something)
- [Repository map](#repository-map)
- [Documentation](#documentation)
- [Testing and CI](#testing-and-ci)
- [Contributing](#contributing)
- [Licence](#licence)

---

## Why this shape

Three facts, one product.

1. **Streamed models are ~40× cheaper per input token than per output token.** Decode
   streams top-k experts per token (**4.05 tok/s measured** — unusable). Prefill with a
   batch-union sweep reads each expert once per chunk (**165 tok/s measured**). Every
   published number in the field is single-request decode, so the field concluded
   streamed models are too slow. That is correct *for generation* and wrong for any task
   where input dominates output. Both terms: `docs/BASELINE.md` §4.

2. **The tasks where input dominates output are exactly the tasks that determine merge
   quality.** Reranking N candidates emits an integer. Reviewing a diff emits a verdict.
   All read 5–30k tokens and write 10–300.

3. **Merge quality, not benchmark score, is the binding constraint.** METR's standing
   finding: ~half of SWE-bench-passing PRs would not be merged by maintainers. A local
   35B at 73.4% SWE-bench Verified is not short of capability — it is short of *this
   repository's* conventions, which is what a repo-derived adapter encodes and a
   verifier pass enforces.

---

## Status

Honest summary: **the system runs, one tier is real, and most of the numbers are not.**

| Layer | State |
|---|---|
| Gateway, router, tool-call reliability, adapters, eval, attestation | **built and tested** — 570 tests, runs on any Python 3.11+ box |
| Tier 0 (resident generator) | **run against real weights** — `Qwen3.6-35B-A3B-OptiQ-4bit`, 23.0 GiB, on an M4 Max |
| Tier 1 (verifier) | **rung 1 served, 2026-08-10** — a streamed 122B answered schema-constrained reranks over the process boundary. Still *deployed* as rung 3, tier 0 with its adapter stripped, which is 6–9× faster and costs no memory |
| Gate A (latency) | **run.** Decode passes at 69.6 tok/s; TTFT fails at 30.5 s @32k, and the failure is arithmetic |
| Tool-call gate (sec 10.2) | **run.** 1.00 well-formed over 100 runs, 100/100 first-attempt calls |
| Gate B (streamed prefill) | **run.** 153.7 tok/s mean over six runs (sd 1.1) — clears this host's floor, 1.3× under the spec's 200 |
| Determinism (sec 9.3) | **measured, and G1 is red.** CPU and Metal diverge 4.375 logits against a 1.625 greedy margin and flip the first token — MLX runs a different linear-attention algorithm on CPU, so byte-identity is not the platform's to give. Same configuration reproduces *bitwise*. Re-chunking one prompt diverges 2.031, which is the mechanism behind a restored cache changing an answer |
| Adapter isolation (sec 4.2) | **not run — needs a trained adapter.** No longer blocked by memory: the gate holds one arm at a time rather than two, so it fits under the 28.08 GiB ceiling |

The distinction that matters: every gate's *plumbing* is proven; most of its *numbers*
are not. Everything above the backend interface runs against `MockBackend` — a
deterministic, adapter-sensitive, faultable stand-in, good enough that the whole system
is testable without a Mac and honest enough that the tests mean something.

That honesty has been tested. The first real-weights run found that
`MLXTier0Backend` accepted a JSON schema and never applied it, so constrained decoding
was computed, carried down the whole pipeline and dropped — while the mock honoured it.
**The mock was stricter than the hardware**, which is why a fully green suite could not
see it. Fixing it moved the tool-call gate from 0.81 and failing to 1.00 and passing.

[`docs/STATUS.md`](docs/STATUS.md) has the section-by-section map, the three documented
deviations, and the full gap list.

---

## Requirements

**Running the gateway, the extractors, the evals and the whole test suite needs only
Python 3.11+.** That is the portable half of the system, and it is the half you can
develop against today on any OS.

**Serving a real model needs Apple Silicon.** The baseline platform is a **MacBook Pro
M4 Max, 36 GB unified memory, 1 TB SSD**, and every budget in the repository derives
from it — see [`docs/BASELINE.md`](docs/BASELINE.md). Two consequences worth knowing
before you plan around it:

- Metal's real ceiling is `max_recommended_working_set_size` = **28.08 GiB**, not
  36 GB. Tier 0 alone is 23.0 GiB of that.
- Tier 0 and a streamed tier 1 therefore **cannot co-reside**, so the shipped config
  serves the verifier from rung 3. A streamed verifier measures 165 tok/s of prefill by
  `curl` and 153.7 by Gate B, 1.3× under sec 11's floor — slow, not dead. Rung 3 ships
  because it is 6–9× faster and costs no memory.

Python 3.14 is the current release and 3.11 the declared floor; CI runs both, so both
are guarded. 3.12 and 3.13 sit between two tested ends rather than being tested
themselves.

## Install

```bash
pip install -e '.[dev]'                # core + tests, runs anywhere
pip install -e '.[dev,constrain]'      # + constrained decoding (what CI installs)
pip install -e '.[dev,mlx]'            # + Apple Silicon backends
```

`[dev]` and `[dev,constrain]` exercise genuinely different paths — repair versus
prevention (sec 8.5) — and both are expected to pass.

Dependency licence discipline is Apache-2.0 / MIT / BSD only. Every dependency is a
security surface, not a convenience; the LiteLLM 1.82.7/1.82.8 supply-chain compromise
is the standing precedent.

## Run

```bash
tandem doctor                          # runtime status + offline posture
tandem serve --backend mock            # gateway on 127.0.0.1:8080
```

Point a harness at it:

```bash
source <(tandem offline-env)           # the airgap environment (sec 8.6)
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
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
`/tandem/compaction/last` (the diff view), `/tandem/trace/last`, `/tandem/audit/verify`.

Streaming is incremental where it honestly can be: a `chat` turn carrying no tools emits
tokens as they are decoded. A `code_change` turn under best-of-N runs to completion
first and says so in `/tandem/trace/last` — there are no tokens to emit until the
verifier has chosen a candidate, and streaming candidate 0 then retracting it would be
worse than a pause.

> [!WARNING]
> On Apple Silicon, `tandem doctor`, `serve`, both gates, `bench`, `eval` and `train`
> each load 23.0 GiB eagerly, and only one process can hold the GPU.
> [`docs/PROCESSES.md`](docs/PROCESSES.md) is the pre-flight procedure — read it before
> starting a second serving process.

## Configure

Copy [`tandem.toml.example`](tandem.toml.example) to `tandem.toml`. Every knob is
documented in place, and **unknown keys are a hard error** rather than a silent no-op —
a typo'd `expert_cache_bytes` that appears to take effect and does not is how a wrong
number ends up in a published measurement.

**`tier1.rung`** picks which rung of the sec 5.5 fallback ladder serves the verifier.
The rung is *selected, never descended*: no error path silently picks a different one.

| Rung | `tier1.rung` | What it is |
|---|---|---|
| 1 | `streamed` | A large MoE with experts streamed from SSD. Off on the baseline platform: 153.7 tok/s of prefill by Gate B, 1.3× under its floor and 6–9× slower than rung 3. |
| 2 | `resident_swap` | An 80B swapped into residency, evicting tier 0 — ~10 s each way, and tier 0 cannot serve while it is in. |
| 3 | `second_opinion` | Tier 0 with its adapter unmounted. Free, weak, needs no second model, and still catches adapter overfit. **What ships.** |
| 4 | `remote` | Somebody else's API. Breaks the offline claim, so it needs `tier1.remote_consent` written out in full. |

**`[eval]`** declares what the repository runs to check itself. Three of the merge
eval's five metrics and the whole of T2 failure escalation depend on it, and there is no
way to guess it:

```toml
[eval]
linters = [["ruff", "check"]]
test_command = ["pytest", "-q", "-x"]
```

Leave it out and those metrics report as *not measured*, which is the honest answer
rather than a passing one.

**`[gates]`** sets what *this host* is judged against. Code defaults are the
specification's figures, so an absent block judges the spec. Lower one only when the
machine cannot meet it for a reason that is arithmetic rather than a bug — and note that
relaxing hides nothing: every gate reports `spec_budget` beside `budget`, `meets_spec`
beside `pass`, and `relaxed_criteria` naming each row that is green only because of the
block. A pass against a relaxed floor means "this host cleared its own floor", never
"Gate A passed".

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
failure to work around — below that, A1 underfits, and the honest answer is to tell the
customer rather than ship a null adapter.

A2 is stronger with real review timestamps, which live in the forge rather than in git.
[`tools/export_reviews.py`](tools/export_reviews.py) fetches them; without it,
extraction falls back to the first branch commit and labels every pair with which signal
produced it.

## Gates and evals

```bash
tandem gate toolcall --runs 100    # sec 10.2 — blocking, ≥99% well-formed
tandem gate isolation              # sec 4.2  — blocking, adapter deltas must not leak
tandem eval merge --repo . --a1 a1-myrepo    # sec 10.1 — the four bars
tandem eval regression             # sec 10.3 — diff against a baseline, not a score
tandem bench latency               # sec 10.4 + M0 Gate A
tandem bench tier1                 # M0 Gate B — streamed prefill
tandem audit verify                # sec 9.2 — hash-chained log
```

`tandem eval merge` is the one that matters. **If the adapter doesn't beat the base
model on the customer's own repo, there is no product** — and it is designed to say so
at M3, week 8, before tier 1 is built.

Each generated patch is applied in a detached git worktree at the held-out change's own
parent commit, and the `[eval]` commands run there — never in your checkout, and never
against the files as they sit on disk, which would score the repository rather than the
patch.

`tandem eval regression` reports a **diff against a recorded baseline, not a score**.
`RegressionReport` has no field holding a pass rate, and a first run can only write a
baseline and say so. A field holding one is all it takes for a number to end up on a
slide.

---

## Design notes worth knowing before you change something

Each of these looks simplifiable and is not, and each fails as a silently wrong answer
rather than an error. The full list with rationale is
[`docs/HANDOFF.md`](docs/HANDOFF.md) §6; every one also carries a comment in the code.

**Adapters are never merged into the base** (sec 4.2). Merging is the obvious
implementation and it destroys the product: ~20 GB per adapter instead of ~250 MB, no
multi-tenancy, and a receipt that cannot name what produced a change. The forward stays
`y = xW + s·(xA)B` with deltas resident and separate, selected per request via a
`ContextVar` — a module global would race under concurrency.

**Tier 1 never generates a patch** (sec 5.1). The interface has no `generate`, and
`max_tokens` is clamped per call type. The same model that reranks five candidates in
18 s takes six minutes to write one. That asymmetry is the whole thesis, so it is
enforced structurally rather than by convention.

**Tier 1 never reasons, and a reasoned verdict is refused rather than read.** A
`<think>` block spends the output clamp before the verdict exists, and thinking mode
silently ignores `temperature` — so the greedy judgement the receipt attests to becomes
a sample. Both the request-side flag and the response-side refusal are load-bearing:
the first is a guess about what the engine reads, the second an observation of what it
did.

**A backend handed `json_schema` must apply it to the logits.** Dropping it is silent,
and it already happened once — see [Status](#status).

**The KV cache is plain `read`/`write`, never mmap** (sec 8.4). A process already
mapping ~30 GB of weights should not add more VM mappings.

**The tool-replay map is not optional** (sec 8.5.5). Clients hand tool calls back as
normalised JSON, not the model's sampled bytes. Re-rendering them differently breaks the
prompt's byte prefix, the cache misses, and the whole turn rebuilds — on every
tool-using turn, which is all of them.

**Compaction is fingerprinted with a staleness signal** (sec 8.2). A template pinned to
an exact prompt silently stops matching when the harness updates, and the customer
quietly loses the win with no error anywhere. Detection is scored, and a drifted prompt
is reported rather than absorbed.

**Nothing under `src/tandem/` makes an outbound network call.** The offline posture
(sec 8.6) is a claim you can verify with `lsof`, so the two network-capable files live in
`tools/` and are not installed. A test pins the package's network surface to one file —
the loopback process boundary to the tier-1 engine — and a second import there is a test
failure rather than a discovery.

---

## Repository map

```
src/tandem/
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

`backends/base.py::Backend` is the line the whole layout turns on. Above it, everything
is pure Python that runs anywhere and is fully tested. Below it, `mlx_tier0.py` needs
Apple Silicon to serve a real model and `mlx_tier1.py` needs an engine process to talk
to; both are exercised off-target against stand-ins, which proves the wiring and nothing
more.

## Documentation

[`docs/`](docs/) — four documents, no overlap, each stating at the top what it does not
answer:

- [`docs/HANDOFF.md`](docs/HANDOFF.md) — project state, the ordered next steps, and the
  decisions that look simplifiable and are not. **Start here.**
- [`docs/STATUS.md`](docs/STATUS.md) — specification section → code, what is proven
  versus written, the deviations, the gap list.
- [`docs/BASELINE.md`](docs/BASELINE.md) — the reference machine, every measured number,
  and the budgets derived from them.
- [`docs/PROCESSES.md`](docs/PROCESSES.md) — what may hold the GPU, and the pre-flight
  checks before anything loads weights.

The `sec 8.2` / `sec 6.3` references throughout the code point at a technical
specification that is **not in this repository**. `docs/STATUS.md` maps every section to
the code, which covers most day-to-day needs. Please don't invent a meaning for a
section you cannot resolve.

## Testing and CI

```bash
pytest -q                                          # the whole suite, ~8 s
pytest tests/test_router.py::test_n_equals_one_disables_reranking -q
```

The git-facing tests build small real repositories rather than mocking `git`, because
every failure mode that matters lives in git's actual output format — `-z` numstat
framing, rename records, merge parentage. A mock would only agree with whatever the code
already believes.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs four jobs, all
blocking:

- **ruff (lint and format)** — `ruff check` and `ruff format --check`, as two steps so
  a defect and a reformat are two different failures.
- **mypy** — strict over `src/`, on Python 3.11 and 3.14. The config sets no
  `python_version` on purpose, so each leg checks against the semantics of the
  interpreter it runs under.
- **tests** — the full suite with `[constrain]` installed, so the prevention layer is
  exercised rather than skipped as unavailable. On Python 3.11 and 3.14, the two ends of
  `requires-python`.
- **cli and gateway smoke** — starts a real `tandem serve`, drives all three wire
  protocols against it, asserts the receipt carries an engine commit, and verifies the
  audit chain over the records those requests wrote. Then the blocking gates and both
  extractors. This covers the one thing unit tests cannot: an actual uvicorn process
  serving the product surface.

Lint and type-check configuration lives in `pyproject.toml`, and every selected,
ignored or relaxed rule carries a comment explaining the decision — including the ones
where ruff and mypy want opposite things.

## Contributing

Issues and pull requests are welcome. Four things worth knowing before you open one:

- **Run `ruff format`, `ruff check --fix` and `mypy` on what you touched.** All three
  are blocking in CI. Where a rule is wrong for this codebase specifically, a scoped
  `# noqa: RULE` or `# type: ignore[code]` with a reason is the right answer; a bare
  suppression is not, and mypy's `ignore-without-code` rejects it outright.
- **Absolute imports only** — `from tandem.types import GenRequest`, never
  `from ..types import ...` or `from .base import ...`. Enforced as `TID252` with
  `ban-relative-imports = "all"`.
- **Run both install states.** `[dev]` and `[dev,constrain]` exercise different code
  paths, and CI runs the second.
- **`tandem gate toolcall --runs 100` is blocking**, and CI runs it. If you touched the
  gateway, the tool-call layer, compaction or sampling, run it locally first.
- **`MockBackend` must never be easier to satisfy than a real backend.** It has failed
  this three times. When extending it, ask both whether the change makes it more
  permissive than a real backend, and whether a real backend actually does what the mock
  assumes.

Comments in this codebase explain *why*, and especially why a simpler implementation was
rejected. Please keep that when editing — several of them are the only record of a bug
that has already been paid for once.

## Licence

Apache-2.0. See [`pyproject.toml`](pyproject.toml) for the declaration and the
dependency licence discipline that goes with it.
