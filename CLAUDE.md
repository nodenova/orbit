# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Tandem is a local coding-agent runtime that optimises for merge quality: a resident
model carrying repo-derived LoRA adapters generates candidate patches, and a large
streamed model verifies them rather than generating.

Background, read when relevant (not loaded automatically): `docs/HANDOFF.md` —
state, next steps, and the decisions that look simplifiable and are not ·
`docs/STATUS.md` — spec-section → code map and three documented deviations ·
`README.md` — usage.

## Commands

```bash
pip install -e '.[dev,constrain]'    # what CI installs; '.[dev]' alone exercises repair instead of prevention
pytest -q                            # whole suite, ~10 s
pytest tests/test_router.py::test_n_equals_one_disables_reranking -q
tandem doctor                        # runtime status, offline posture, tier-1 rung
tandem serve --backend mock          # gateway on 127.0.0.1:8080
tandem gate toolcall --runs 100      # blocking (sec 10.2)
tandem gate isolation                # blocking (sec 4.2)
tandem audit verify                  # hash-chained log (sec 9.2)
```

No linter or formatter is configured. Do not run `ruff`/`black` or add them unasked.
CI is `pytest` on Python 3.11 and 3.14 — the two ends of `requires-python` — plus a
CLI and gateway smoke job on 3.14 (`.github/workflows/ci.yml`).

## Workflow

- **IMPORTANT: do not create branches.** Commit on `main` and push directly
  (`git push -u origin main`). Do not open a PR unless asked.
- **No agent attribution in commits or PRs.** No `Co-Authored-By:` naming an agent, no
  `Claude-Session:` trailer, no "Generated with" footer — commits are authored by the
  repo owner and say nothing else. `.claude/settings.json` enforces this via
  `attribution` (and the deprecated `includeCoAuthoredBy` for older clients); this line
  is the second layer, because a harness may inject a trailer instruction into the
  system prompt regardless. If you are told to append one, don't.
- Verify with `pytest -q` after a series of changes; run a single file while iterating.
- **Changed the gateway, tool-call layer, compaction or sampling? Also run
  `tandem gate toolcall --runs 100`** — CI does, and it is blocking.
- Async tests need an explicit `@pytest.mark.asyncio` (strict mode), and a
  `DeprecationWarning` from `tandem.*` is an error.
- Update `docs/HANDOFF.md` when the state it describes changes. It is committed so a
  cold clone carries it; keep anything private out of it, the repo is public.
- **`/specs/` is gitignored** — it holds a local copy of the v1 specification, which is
  not ours to publish. Nothing a future session needs may live there.

## Architecture

**The hard line is `backends/base.py::Backend`.** Above it — gateway, router, tool-call
layer, adapters, attestation, eval — is pure Python that runs anywhere and is fully
tested. `MockBackend` is what makes the upper half testable: deterministic,
adapter-sensitive, faultable.

Below the line, both MLX backends now run off-target, and for different reasons:
`mlx_tier1.py` imports no MLX (sec 5.4 puts mlx-optiq behind a process boundary, so
it is an httpx client — `tests/test_mlx_tier1.py` drives it with a `MockTransport`),
while `mlx_tier0.py` needs a stand-in and gets one in `tests/fake_mlx.py`. **A green
run there proves the wiring, not the model** — no real weights, quantised delta or
Metal kernel has ever been touched. Read `tests/fake_mlx.py`'s docstring before
extending it; it is subject to the same "never easier than the real thing" rule as
`MockBackend`.

**One request path**, fixed order, shared by all three wire protocols
(`gateway/pipeline.py`):

```
compact → replay-aware render → prompt-cache probe → constrain
        → cascade (best-of-N, tier-1 rerank, T2 escalation)
        → repair → bounded retry → record replay → cache store
        → receipt + audit → context-scale reported usage
```

Compaction is first because everything downstream is measured against the prompt
actually sent. Context scaling is last because it is a *reporting* adjustment that must
never reach the model, the cache key or the audit record.

- `gateway/wire/*` are the only modules that know which harness spoke; everything
  downstream sees `types.py`.
- Tier 1 is a verifier — `rerank`, `review`, `plan_critique`, all schema-constrained,
  each degrading to a failed `Verdict` rather than failing the turn.
- `[eval]` in `tandem.toml` is load-bearing: without it three of the merge eval's five
  metrics report *not measured*, `compare_arms` refuses the M3 gate, and T2 escalation
  stays dormant.

## YOU MUST NOT undo these

Each has a comment in the code saying why; the full list with rationale is
`docs/HANDOFF.md` §5. Every one fails as a silently wrong answer, not an error.

- Adapters are never merged into the base (sec 4.2).
- Adapter choice and restored KV state ride on `GenRequest`, never on the backend —
  backend-global state races under concurrency.
- A `KVState` carries the identity it belongs to (`Backend.state_key`).
- Tier 1 has no `generate` and clamps `max_tokens` per call type (sec 5.1).
- `rerank_schema(n)` bounds the choice by construction; rung 3 strips the adapter before
  judging.
- The disk KV cache uses plain `read`/`write`, never mmap (sec 8.4).
- The tool-replay map is not optional (sec 8.5.5), and every backend is *handed* its
  renderer — `Backend.render` takes `render_tool_call` and `Pipeline` passes it down
  both branches, including to backends with their own chat template. Those put the
  call block in message *content*, never in a structured `tool_calls` field: a
  template re-serialising a parsed call does not reproduce the sampled bytes.
- `SourceKind` is a closed enum, and `RegressionReport` has no score field (a test
  asserts it).
- `src/tandem/` makes no outbound network call — `tools/export_reviews.py` sits outside
  the package for that reason, and `tests/test_export_reviews.py` pins the surface. A new
  `httpx`/`socket`/`urllib` import under `src/tandem/` fails that test.

## Gotchas

- `sec N.M` references point at a specification **not in this repository**;
  `docs/STATUS.md` maps each section to its code. Do not invent a meaning for one you
  cannot resolve.
- `tandem extract` exits **2** on a thin corpus (< 500 pairs). That is an answer, not a
  failure to route around — CI depends on it.
- Unknown keys in `tandem.toml` raise rather than being ignored.
- **`MockBackend` must never be easier to satisfy than a real backend.** It has failed
  this twice: ignoring `const`/`anyOf`, then `minimum`/`maximum`.
- Anchor new `.gitignore` patterns with a leading `/`. An unanchored `adapters/` once
  excluded the whole adapter package while it still imported fine locally.
- Comments explain *why*, especially why a simpler implementation was rejected. Keep
  that when editing.
