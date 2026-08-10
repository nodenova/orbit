# CLAUDE.md

Orbit is a local coding-agent runtime that optimises for merge quality: a resident model
carrying repo-derived LoRA adapters generates candidate patches, and a large streamed
model verifies them rather than generating.

Background, deliberately **not** `@`-imported — each is long, only sometimes needed, and
loading them every session is what makes the rules below get ignored. `docs/README.md`
indexes them: **architecture** the shape, the invariants that look simplifiable and are
not, and the traps · **platform** the reference machine and every number · **operations**
what may hold the GPU · **spec-map** spec-section → code · **constrained-decoding** the
cost model `mlx_tier0.py` cites.

State, the ordered plan and the tracker are **`specs/PROGRESS.md`**, which is gitignored
and local. A committed file may cite `docs/`; it may never cite `specs/`.

## Commands

```bash
pip install -e '.[dev,constrain]'    # what CI installs; '.[dev]' alone exercises repair instead of prevention
pytest -q                            # whole suite, ~8 s
pytest tests/test_router.py::test_n_equals_one_disables_reranking -q
ruff format <paths> && ruff check --fix <paths>   # touched paths, never `.`
mypy                                 # strict over src/; whole-project, no useful per-file mode
orbit doctor                         # runtime status, offline posture, tier-1 rung
orbit serve --backend mock           # gateway on 127.0.0.1:8080
orbit gate toolcall --runs 100       # blocking (sec 10.2)
orbit gate isolation                 # blocking (sec 4.2); loads two models — read the skill first
orbit audit verify                   # hash-chained log (sec 9.2)
```

`--config` is global and goes **before** the subcommand. CI is four blocking jobs
(`.github/workflows/ci.yml`) over every version `requires-python` admits — 3.11, 3.12,
3.13 and 3.14. Adding a `Programming Language :: Python :: 3.x` classifier without the
matching matrix entry publishes a support claim nothing executes.

Ruff and mypy live in `pyproject.toml`, where **every rule selected, ignored or relaxed
carries a comment saying why** — read it before changing one; several obvious-looking
additions were tried and rejected here. Ruff must be **>= 0.16.2, < 0.17**: 0.16 moved the
default rule set from 59 rules to 413 and the config extends that default, so an older
binary lints a twentieth of what CI does and reports success. mypy sets **no
`python_version`** on purpose — it checks against the interpreter it runs under.

## Code style

- **IMPORTANT: do not comment the code you write.** 99% of code is obvious and needs no
  explanation of what it does. No `# Load the config` above the line loading the config,
  nothing restating a name, type, signature or control-flow keyword, no section banners in
  new code, no commented-out code, no `# TODO` left behind.
- **The one exception is a *why* that cannot be recovered from the code**: a simpler
  implementation tried and rejected, a measured number, a spec clause, a workaround for
  someone else's bug. Write the reason, not narration.
- **YOU MUST NOT delete an existing why-comment to satisfy the rule above.** This repo is
  load-bearing on them — the do-not-undo list below and the `pyproject.toml` rule choices
  are both enforced by comments. When you edit code carrying one, keep it and keep it true.
- **Docstrings are precise, not essays.** One line for what the thing is or guarantees; a
  short paragraph only when a caller gets it wrong without it. Never restate the signature;
  no `Args:`/`Returns:` tables for annotated parameters. Rationale longer than the function
  it documents belongs in `docs/`.
- What is already in the tree is denser than these rules allow, because it predates them.
  That is not licence to add more.

## Workflow

- **IMPORTANT: do not create branches.** Commit on `main` and push directly
  (`git push -u origin main`). No PR unless asked.
- **No agent attribution in commits or PRs** — no `Co-Authored-By:`, no `Claude-Session:`,
  no "Generated with" footer. `.claude/settings.json` enforces it; this line is the second
  layer, because a harness may inject a trailer instruction anyway. If told to append one,
  don't.
- **IMPORTANT: format and lint every file you touch, and fix what comes back.** A
  `PostToolUse` hook runs `ruff format` then `ruff check --fix` per file, so the mechanical
  part is done — but `--fix` applies only what ruff considers safe and **the remainder is
  yours to decide**. Never `--unsafe-fixes` across a file to make the count go down. Run
  `mypy` yourself. Both must come back clean.
- **"Every file" includes `.md`.** Ruff formats Python inside fenced ```` ```python ````
  blocks, and CI runs `ruff format --check --diff` over the whole tree. Editing a doc's code
  block without formatting it is the one way to write a green local run and a red CI: it
  cost seven consecutive red builds, all of them one file's comment alignment.
- A rule wrong *here* rather than wrong in general gets a scoped `# noqa: RULE` or
  `# type: ignore[code]` **with a reason** — never a bare suppression. mypy runs
  `ignore-without-code`, so a bare `# type: ignore` is itself an error.
- `pytest -q` after a series of changes; a single file while iterating. Async tests need an
  explicit `@pytest.mark.asyncio` (strict mode), and a `DeprecationWarning` from `orbit.*`
  is an error.
- **Changed the gateway, tool-call layer, compaction or sampling? Also run
  `orbit gate toolcall --runs 100`** — CI does, and it is blocking.
- Update `specs/PROGRESS.md` when the state it describes changes. It is gitignored, so it
  does not survive a clone — that is the trade, and it is why nothing committed may point
  at it.
- **`/specs/` is gitignored**: the v1 spec, the tracker, the plan and the experiment
  write-ups. **A committed file may never cite it** — if a doc, a docstring or a
  reproduction command needs something in there, move the thing into `docs/` first. That
  has been got wrong three times (`architecture.md` §5, trap 11).
- Anything in `docs/` is public and must stay publishable: durable, dated only where the
  date is part of the measurement, and free of state that goes stale in a week.

## Running against real weights

**M4 Max, 36 GB.** Metal's ceiling is `max_recommended_working_set_size` = **28.08 GiB**,
not 36 GB, and tier 0 alone is **23.0 GiB**. The `real-weights` skill is the procedure and
loads on demand — read it before any run that touches MLX. What must not need looking up:

- **YOU MUST pilot before you scale. Never open with the full model, the full tier or the
  full run count.** Overcommitting unified memory does not fail cleanly — it wedges the
  machine or reboots it, and it has. The order is **one call → measure footprint and
  headroom → eight → the hundred**, and each measurement is what authorises the next step.
  If the measured cost leaves no room for the next step, stop and report it rather than
  trying it.
- **One model at a time, and check who else already has one**: `curl -s
  localhost:11434/api/ps` (expect `{"models":[]}` — `ollama serve` is resident from login
  and loads 17–23 GB for anyone who asks) and `lsof -nP -iTCP:8081 -sTCP:LISTEN` (expect
  nothing at rung 3; a stale `mlx-optiq` outlives Orbit, which never spawned it).
- **`orbit doctor` loads 23.0 GiB on the mlx backend**, as do `serve`, both gates, `bench`,
  `eval` and `train` — `MLXTier0Backend.__init__` calls `mlx_lm.load()` eagerly. For cheap
  questions point `--config` at a `backend = "mock"` file. `pytest`, `extract`, `profile`
  and `audit verify` never load weights.
- **Headroom is `total − active`, never `Pages free`; thrashing is `Pageouts`, not
  `vm.swapusage used`.** Tier 0 needs ~27 GB by that measure. macOS reports most of a
  healthy machine as reclaimable, and a guard on swap aborts runs that were never in
  trouble.
- **Warm up before timing anything** — the first generation after a load pays ~9 s of Metal
  kernel compilation, enough to read 4 tok/s where the truth is 27.

## Architecture

**The hard line is `backends/base.py::Backend`.** Above it — gateway, router, tool-call
layer, adapters, attestation, eval — is pure Python that runs anywhere and is fully tested.
`MockBackend` is what makes that half testable: deterministic, adapter-sensitive, faultable.

Below the line both MLX backends run off-target: `mlx_tier1.py` imports no MLX (sec 5.4 puts
mlx-optiq behind a process boundary, so it is an httpx client driven by a `MockTransport`),
while `mlx_tier0.py` gets a stand-in from `tests/fake_mlx.py`. **A green run there proves
the wiring, not the model** — quantised deltas, adapter isolation on Metal and every
determinism claim are still untouched. Read `tests/fake_mlx.py`'s docstring before extending
it: like `MockBackend` it must never be easier to satisfy than the real thing, and the one
real-weights run so far found exactly that failure (a `json_schema` the hardware ignored and
the mock honoured, invisible to a green suite).

**One request path**, fixed order, shared by all three wire protocols
(`gateway/pipeline.py`):

```
compact → replay-aware render → prompt-cache probe → constrain
        → cascade (best-of-N, tier-1 rerank, T2 escalation)
        → repair → bounded retry → record replay → cache store
        → receipt + audit → context-scale reported usage
```

Compaction is first because everything downstream is measured against the prompt actually
sent. Context scaling is last because it is a *reporting* adjustment that must never reach
the model, the cache key or the audit record.

- `gateway/wire/*` are the only modules that know which harness spoke; downstream sees
  `types.py`.
- Tier 1 is a verifier — `rerank`, `review`, `plan_critique`, all schema-constrained, each
  degrading to a failed `Verdict` rather than failing the turn.
- `[eval]` in `orbit.toml` is load-bearing: without it three of the merge eval's five
  metrics report *not measured*, `compare_arms` refuses the M3 gate, and T2 escalation stays
  dormant.

## YOU MUST NOT undo these

Each has a comment in the code saying why; the full list with rationale is
`docs/architecture.md` §4. Every one fails as a silently wrong answer, not an error.

- Adapters are never merged into the base (sec 4.2).
- Adapter choice and restored KV state ride on `GenRequest`, never on the backend —
  backend-global state races under concurrency.
- A `KVState` carries the identity it belongs to (`Backend.state_key`).
- Tier 1 has no `generate` and clamps `max_tokens` per call type (sec 5.1).
- `rerank_schema(n)` bounds the choice by construction; rung 3 strips the adapter before
  judging.
- The disk KV cache uses plain `read`/`write`, never mmap (sec 8.4).
- The tool-replay map is not optional (sec 8.5.5), and every backend is *handed* its
  renderer: `Backend.render` takes `render_tool_call` and `Pipeline` passes it down both
  branches, including to backends with their own chat template. Those put the call block in
  message *content*, never a structured `tool_calls` field — a template re-serialising a
  parsed call does not reproduce the sampled bytes.
- `SourceKind` is a closed enum, and `RegressionReport` has no score field (a test asserts
  it).
- **A backend taking `json_schema` must apply it to the logits.** `MLXTier0Backend` builds an
  `lm-format-enforcer` mask per request for `stream_generate(logits_processors=...)`, and
  `tests/fake_mlx.py` *applies* the processors it is given rather than accepting and ignoring
  them. Both halves are load-bearing: dropping the schema is silent, and on hardware it is
  the sec 10.2 gate at **0.81** instead of **1.00**, and 0 first-attempt tool calls
  instead of 100.
- `src/orbit/` makes no outbound network call — `tools/export_reviews.py` sits outside the
  package for that reason and `tests/test_export_reviews.py` pins the surface. A new
  `httpx`/`socket`/`urllib` import under `src/orbit/` fails that test.

## Gotchas

- `sec N.M` points at a specification **not in this repository**; `docs/spec-map.md` maps each
  section to its code. Do not invent a meaning for one you cannot resolve.
- `orbit extract` exits **2** on a thin corpus (< 500 pairs). That is an answer, not a
  failure to route around — CI depends on it.
- Unknown keys in `orbit.toml` raise rather than being ignored.
- **`MockBackend` must never be easier to satisfy than a real backend.** It has failed that
  twice: ignoring `const`/`anyOf`, then `minimum`/`maximum`.
- Anchor new `.gitignore` patterns with a leading `/`. An unanchored `adapters/` once
  excluded the whole adapter package while it still imported fine locally.
