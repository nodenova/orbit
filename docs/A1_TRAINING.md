# Training A1 — what it needs, and what stops it today

**A1 is step 3, and step 3 is the thesis gate** (`HANDOFF.md` §4.7). It is the adapter
derived from a repository's own history: the thing that is supposed to make a local 35B
write patches in *this* codebase's idiom. If A1 does not beat base on ≥3 of the merge
eval's 5 metrics, steps 4 and 5 are not delayed — they are the wrong work.

This document owns **what a run needs and what a run costs**. `HANDOFF.md` §4.7 owns the
decision the result feeds; `BASELINE.md` owns every number quoted here.

> The pipeline has never been run end to end. Everything below was found by reading the
> code and by measuring the training step against real weights on 2026-08-10 — **three
> blockers, two of them in our own code and one in the machine.** None was visible to the
> test suite, because the suite runs `--dry-run` and never executes the trainer.

---

## 1. The four inputs

| Input | State | What it costs to get |
|---|---|---|
| **A repository with ≥500 usable pairs** | **missing, and it is the owner's call** | `tandem extract a1` exits **2** below 500 pairs (`extract_a1.py:99`, `min_usable_pairs`). This repo has **43 commits**, so it cannot be its own corpus — not close. A mature repo yields 1–5k. |
| **An `[eval]` block in `tandem.toml`** | **missing** | Names the repo's linters and test command (§3). Without it three of five merge-eval metrics report *not measured* and `compare_arms` refuses the M3 gate — which is the guard working, not a thing to route around. |
| **A trainer on PATH** | **present** | `trainer_available()` prefers `optiq` and falls back to `python3 -m mlx_lm lora`. `optiq` is **not** on PATH here, so the mlx-lm path is what runs. |
| **Memory for the training step** | **insufficient as configured** — §4 | The binding constraint, measured. Not fixable by waiting. |

## 2. Two defects in our own code, both silent

### 2.1 `--iters` is handed the epoch count, so a real corpus trains on three examples

`build_sft_command` (`adapters/train.py:165`) emits:

```
--iters  str(cfg.epochs)      # SFTConfig.epochs = 3
```

`mlx_lm lora --help` is unambiguous — *"`--iters ITERS`  Iterations to train for."* There
is **no `--epochs` flag**, and at `batch_size = 1` an iteration is one example. So a
1,000-pair corpus asked for 3 epochs trains on **3 pairs**, mlx-lm exits 0, and
`TrainResult.ok` is `rc == 0` → the run reports success. The adapter would be
indistinguishable from noise and the provenance record would attest it as trained on the
full corpus.

The correct value is `epochs × ceil(n_pairs / batch_size)` — for 1,000 pairs, 3 epochs,
batch 1: **3,000**, not 3. `train_sft` already counts `n_pairs` for the provenance record,
so the number is in hand at the call site.

**Why no test caught it:** the suite asserts on the command `--dry-run` prints, and it
prints exactly what the code intends. A dry-run test pins the *spelling* of a flag, never
its meaning to the callee — the same shape as trap 7 (`optiq serve --help` is not its
argument surface).

### 2.2 The corpus has no `valid.jsonl`, and mlx-lm treats that as an empty set rather than an error

`cli.py:203-207` writes `train.jsonl`, `manifest.jsonl` and, with `--holdout`,
`holdout.jsonl`. mlx-lm's `--data` wants a directory of `{train,valid,test}.jsonl`, and
`load_local_dataset`'s `load_subset` returns `[]` for a file that does not exist — no
warning. Validation then runs against nothing every `steps_per_eval` (200 by default),
which the current `--iters 3` never reaches. **Fixing 2.1 is what exposes 2.2**, which is
the worst possible order to find it in: a corrected iteration count would put the first
validation ~200 iterations into an ~88-minute run.

The holdout is *deliberately* not the validation set — it is the merge eval's held-out
commits (sec 10.1) and using it to select checkpoints would spend the thing being measured.
So a `valid.jsonl` has to be split off the training pairs, and `extract` is where that
belongs.

Minor, same function: the command begins with a bare `"python3"`, resolved against PATH at
run time rather than `sys.executable`. It works here only because the venv is active.

### 2.3 `mask_prompt` + a prompt longer than `max_seq_length` is a silent `nan`

Measured, not reasoned: 18 pairs of ~2,000-token prompts at `--max-seq-length 1024` with
`--mask-prompt` reported

```
Iter 2: Train loss nan, Tokens/sec 0.000, Trained Tokens 0, Peak mem 53.564 GB
```

and **exited 0, saving an adapter file.** The mechanism is two defaults meeting:
`trainer.py:164` truncates each sequence to `max_seq_length`, and `--mask-prompt` excludes
prompt tokens from the loss — so a pair whose *prompt alone* exceeds the cap contributes
**zero unmasked tokens**, the loss is taken over nothing, and `nan` propagates. Both are
`SFTConfig` defaults (`mask_prompt = True`, `max_seq_length = 4096`).

**A1's pairs are exactly this shape** — a long prompt (diff plus surrounding context) and a
shorter completion (the patch) — so any pair over 4,096 tokens is silently dead weight, and
one of them poisons its batch's loss. `extract_a1` does not bound prompt length, and nothing
in `train.py` checks it. Together with §2.1 this is three ways to finish a training run
with `ok=True` and an adapter that learned nothing.

## 3. The `[eval]` block

`EvalConfig` (`config.py:175`) is read by both the merge eval and T2 escalation. Empty
means *not measured*, everywhere:

```toml
[eval]
repo = "/path/to/the/repo"
linters = [["ruff", "check"], ["mypy"]]
test_command = ["pytest", "-q"]
setup_command = ["pip", "install", "-e", "."]
base_rev = "HEAD"
```

`setup_command` runs once per throwaway worktree; the suite never runs in the user's
checkout. `worktree_dir` defaults outside every repository on purpose — see its comment.

## 4. The measurement: the training step does not fit as configured

Real weights, `Qwen3.6-35B-A3B-OptiQ-4bit`, 2 iterations, `batch_size 1`,
`--grad-checkpoint`, `--mask-prompt`, mlx-lm 0.31.3. Peak is mlx-lm's own `Peak mem`, and
Metal's ceiling on this host is **28.08 GiB ≈ 30.15 GB**.

| Target layers | Pair length | Peak | Verdict |
|---|---|---|---|
| `-1` (all, what `build_sft_command` sends) | ~67 tokens | **31.10 GB** | **over the ceiling**, at the smallest sequence that exists |
| 16 (mlx-lm's default) | ~67 tokens | **26.44 GB** | fits, with ~3.7 GB to spare |
| 16 | ~67 tokens, cap 2048 | 26.44 GB | identical — the cap only truncates, so this row measures nothing new |
| 16 | ~2,000 tokens, cap 4096 (`SFTConfig` default) | — | **paged.** Killed at 10 min without finishing 2 iterations; pageouts +1,498, swap 433 MB → 2,250 MB |
| 16 | ~2,000 tokens, cap 1024 | **53.56 GB** | **1.8× physical memory.** Survived only by swapping, ~20 min for 2 iterations, and trained **0 tokens** (§2.3) |

Four things follow:

1. **`--num-layers -1` is the difference between fitting and not.** It costs **+4.66 GB**
   and **4.8×** the wall clock (0.569 against 2.753 it/s). It is a deliberate choice —
   `train.py:174` says *"all-linear targeting (sec 4.3), not attention-only"* — so cutting
   it is a change to what A1 learns, not a tuning knob. That is the trade to decide before
   a run, not during one.
2. **`max_seq_length` only truncates.** The 512-against-2048 comparison measured *identical*
   peaks (26.439 vs 26.435 GB) because the pilot pairs were 67 tokens. Activation cost
   follows the pairs' real length, so those two rows are a floor, not a budget — and real
   A1 pairs are a diff plus context.
3. **Real-length pairs page this machine at every cap tried.** Paging invalidates any timing
   the 4096 run would have produced (`real-weights` §7), so it has no number and needs none —
   the pageouts are the result. The 1024 run did report one, **53.56 GB**, which is ~1.8×
   physical memory and reached only because macOS swapped rather than failing.
4. **The host survived both**, which is worth recording rather than assuming: `mlxbench.py`
   read 323/344 GB/s afterwards, so a deep swap event did not reproduce T18's degradation.

What a run would cost if it fit: at 16 layers and short pairs, 2.753 it/s → a 1,000-pair
corpus at 3 epochs (3,000 iterations, once 2.1 is fixed) is **~18 minutes**. At all-linear's
0.569 it/s it is ~88 minutes. Neither figure survives contact with 2,000-token pairs, which
is the length that matters.

**One loose end worth checking before a real run.** Initial train loss differed sharply
between the two targeting settings on identical data and seed — 6.638 at `-1` against 0.652
at 16 — where LoRA's zero-initialised `B` should make them equal. All-linear targeting on a
MoE reaches the router as well as the experts, and adapting the gate changes which experts
run. Two iterations is not a loss measurement and this is not yet a finding, but it is
cheap to check and it bears on whether all-linear is even the right target here.

## 5. The order to do this in

1. **Fix 2.1, 2.2 and 2.3**, with a test that pins iterations against *pairs* rather than
   against the printed flag, and a corpus check that refuses a pair whose prompt exceeds
   `max_seq_length` instead of letting it become a `nan`.
2. **Decide the targeting/sequence trade** against §4 — this is the owner's call because it
   changes what the adapter is, and it is the reason step 3 is not simply "run the command".
3. **Choose the repository** and confirm `tandem extract a1 --repo … --holdout 25` clears
   500 pairs. Exit 2 is an answer.
4. **Add `[eval]`**, then re-run `tandem eval regression` so the pre-change reference point
   exists (`HANDOFF.md` §4.5 — already recorded at `baselines/regression-baseline.json`).
5. **Train, then `tandem eval merge`.** Exit criterion: A1 beats base on **≥3 of 5**. Below
   that, stop and re-plan before tier 1 (§4.7).

## 6. Reproduction

```bash
# a corpus this repo cannot supply: 43 commits against a 500-pair floor
tandem extract a1 --repo <real repo> --holdout 25 --out corpus/a1   # exit 2 if thin

# what would actually run, printed rather than executed
tandem train sft --corpus corpus/a1/train.jsonl --out adapters/a1-x --name a1-x \
    --repo <real repo> --dry-run

# the §4 measurement. Loads 23.0 GiB and trains: read PROCESSES.md §3.1 first, and do
# not raise --max-seq-length or drop --num-layers to -1 without watching Pageouts.
python3 -m mlx_lm lora --model "$TIER0_SNAPSHOT" --train --data <dir with train+valid> \
    --adapter-path /tmp/pilot --iters 2 --batch-size 1 --max-seq-length 512 \
    --num-layers 16 --grad-checkpoint --mask-prompt
```
