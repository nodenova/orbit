"""Training driver (spec sec 6.2, 6.3).

**Do not write a trainer.** This module builds argument vectors for `mlx_lm.lora` or
`optiq lora train`, runs them, and writes the provenance record. That is the whole
job. A hand-rolled trainer would be a second thing to keep correct against MLX
releases for no differentiating benefit — the differentiator is the *corpus*, not
the optimiser.

Defaults are the spec's, and the ones that are non-obvious are the ones that matter:

**SFT (A0/A1)** — 3 epochs, lr 2e-4, `mask_prompt` on, `max_seq_length` 4096,
`--grad-checkpoint`, rank per sec 4.3.

**DPO (A2)** — `--method dpo`, `--mount-adapter <A1>` so it starts from the SFT
weights, beta 0.1, lr 5e-5, and **1 epoch**: more invites preference collapse.
`--fused-dpo` above ~4k context, because DPO materialises the full `[seq, vocab]`
logits four times per step (policy and reference x chosen and rejected) and the
plain path OOMs early.

Memory: 4096-token context ~= 11.4 GB peak, 8192 ~= 19.9 GB, measured on an M3 Max for
a 4B model and scaled [V]. Fused cut-cross-entropy auto-engages above 4096 so the
`[seq, vocab]` logit tensor is never materialised — with a ~250k vocab that tensor is
multi-GB and is what OOMs first.

**Those two figures have not been re-measured against the baseline host** (36 GB M4
Max, 28.08 GiB Metal working set — `docs/BASELINE.md`), and the headroom there is
thin enough that the scaling is worth checking rather than trusting: tier 0 alone is
23.0 GiB. **Training a 35B adapter requires tier 1 unloaded** regardless, which
`preflight()` checks rather than discovering two hours in.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tandem.attest.hashing import hash_artefact
from tandem.attest.provenance import ProvenanceRecord, SourceKind, corpus_hash_for


@dataclass
class SFTConfig:
    """Supervised fine-tuning defaults (sec 6.2)."""

    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 1
    max_seq_length: int = 4096
    rank: int = 32
    alpha: int = 64
    mask_prompt: bool = True
    grad_checkpoint: bool = True
    # NEFTune, off by default (sec 6.2). Adds noise scaled by
    # alpha/sqrt(seq_len * embed_dim) to token embeddings during training and nothing
    # at inference; stops the adapter over-memorising a small dataset's surface form.
    neftune_alpha: float = 0.0
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DPOConfig:
    """Preference-tuning defaults (sec 6.3)."""

    # One epoch. More invites preference collapse — the spec is explicit and the
    # failure is not subtle when it happens.
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    batch_size: int = 1
    max_seq_length: int = 4096
    rank: int = 32
    alpha: int = 64
    grad_checkpoint: bool = True
    fused_dpo: bool = True
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    ok: bool
    adapter_path: str
    command: list[str] = field(default_factory=list)
    returncode: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    elapsed_s: float = 0.0
    provenance_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "adapter_path": self.adapter_path,
            "command": self.command,
            "returncode": self.returncode,
            "elapsed_s": round(self.elapsed_s, 1),
            "provenance": self.provenance_path,
            "warnings": self.warnings,
            "stdout_tail": self.stdout_tail[-2000:],
            "stderr_tail": self.stderr_tail[-2000:],
        }


def trainer_available() -> tuple[bool, str]:
    if shutil.which("optiq"):
        return True, "optiq"
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False, (
            "no trainer available: install mlx-lm (Apple Silicon) or the optiq CLI. "
            "Corpus extraction and evaluation run anywhere; training does not."
        )
    return True, "mlx_lm"


def preflight(tier1_loaded: bool, model_params_b: float = 35.0) -> list[str]:
    """Checks worth making before a multi-hour run (sec 6.2).

    Returns warnings rather than raising: the operator decides, but nobody should
    discover the memory ceiling two hours into a run.
    """
    warnings: list[str] = []
    if tier1_loaded and model_params_b >= 30:
        warnings.append(
            "Tier 1 is loaded. Training a 35B adapter requires tier 1 unloaded "
            "(sec 6.2) — stop the tier-1 process before training."
        )
    return warnings


def build_sft_command(
    *,
    model: str,
    corpus: Path,
    output: Path,
    cfg: SFTConfig,
    trainer: str,
    valid_corpus: Path | None = None,
) -> list[str]:
    base = (
        ["optiq", "lora", "train"]
        if trainer == "optiq"
        else ["python3", "-m", "mlx_lm", "lora"]
    )
    cmd = [
        *base,
        "--model",
        model,
        "--train",
        "--data",
        str(corpus.parent),
        "--adapter-path",
        str(output),
        "--iters",
        str(cfg.epochs),
        "--learning-rate",
        str(cfg.learning_rate),
        "--batch-size",
        str(cfg.batch_size),
        "--max-seq-length",
        str(cfg.max_seq_length),
        "--num-layers",
        "-1",  # all-linear targeting (sec 4.3), not attention-only
        "--seed",
        str(cfg.seed),
    ]
    if cfg.mask_prompt:
        cmd.append("--mask-prompt")
    if cfg.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if cfg.neftune_alpha > 0:
        cmd += ["--neftune-alpha", str(cfg.neftune_alpha)]
    if valid_corpus is not None:
        cmd += ["--val-batches", "25"]
    return cmd


def build_dpo_command(
    *,
    model: str,
    corpus: Path,
    output: Path,
    cfg: DPOConfig,
    trainer: str,
    mount_adapter: Path | None,
) -> list[str]:
    base = (
        ["optiq", "lora", "train"]
        if trainer == "optiq"
        else ["python3", "-m", "mlx_lm", "lora"]
    )
    cmd = [
        *base,
        "--model",
        model,
        "--train",
        "--method",
        "dpo",
        "--data",
        str(corpus.parent),
        "--adapter-path",
        str(output),
        "--iters",
        str(cfg.epochs),
        "--learning-rate",
        str(cfg.learning_rate),
        "--dpo-beta",
        str(cfg.beta),
        "--batch-size",
        str(cfg.batch_size),
        "--max-seq-length",
        str(cfg.max_seq_length),
        "--grad-checkpoint",
        "--seed",
        str(cfg.seed),
    ]
    if mount_adapter is not None:
        # Start from the SFT weights (sec 6.3), not from the base.
        cmd += ["--mount-adapter", str(mount_adapter)]
    if cfg.fused_dpo and cfg.max_seq_length >= 4096:
        cmd.append("--fused-dpo")
    return cmd


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, errors="replace", check=False
    )
    return proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - t0


def train_sft(
    *,
    model: str,
    corpus: Path,
    output: Path,
    adapter_name: str,
    source_kind: SourceKind,
    source_repo: str = "",
    commit_range: tuple[str, str] | None = None,
    extraction_filters: dict[str, Any] | None = None,
    n_pairs: int = 0,
    cfg: SFTConfig | None = None,
    licence: str = "Apache-2.0",
    tier1_loaded: bool = False,
    dry_run: bool = False,
) -> TrainResult:
    cfg = cfg or SFTConfig()
    ok, trainer = trainer_available()
    warnings = preflight(tier1_loaded)
    if not ok:
        return TrainResult(
            ok=False, adapter_path=str(output), stderr_tail=trainer, warnings=warnings
        )

    output.mkdir(parents=True, exist_ok=True)
    cmd = build_sft_command(
        model=model, corpus=corpus, output=output, cfg=cfg, trainer=trainer
    )
    if dry_run:
        return TrainResult(
            ok=True, adapter_path=str(output), command=cmd, warnings=warnings
        )

    rc, out, err, elapsed = _run(cmd)
    result = TrainResult(
        ok=rc == 0,
        adapter_path=str(output),
        command=cmd,
        returncode=rc,
        stdout_tail=out,
        stderr_tail=err,
        elapsed_s=elapsed,
        warnings=warnings,
    )
    if rc == 0:
        record = ProvenanceRecord(
            adapter_name=adapter_name,
            source_kind=source_kind,
            source_repo=source_repo,
            commit_range=commit_range,
            extraction_filters=extraction_filters or {},
            corpus_hash=corpus_hash_for(corpus),
            n_pairs=n_pairs,
            base_model_hash=hash_artefact(model) or "",
            base_model_name=model,
            training_config={"method": "sft", **cfg.as_dict()},
            licence=licence,
            created_ts=time.time(),
        )
        result.provenance_path = str(record.write(output / "provenance.json"))
    return result


def train_dpo(
    *,
    model: str,
    corpus: Path,
    output: Path,
    adapter_name: str,
    parent_adapter: Path | None,
    source_repo: str = "",
    commit_range: tuple[str, str] | None = None,
    extraction_filters: dict[str, Any] | None = None,
    n_pairs: int = 0,
    cfg: DPOConfig | None = None,
    licence: str = "Apache-2.0",
    tier1_loaded: bool = False,
    dry_run: bool = False,
) -> TrainResult:
    cfg = cfg or DPOConfig()
    ok, trainer = trainer_available()
    warnings = preflight(tier1_loaded)
    if not ok:
        return TrainResult(
            ok=False, adapter_path=str(output), stderr_tail=trainer, warnings=warnings
        )

    output.mkdir(parents=True, exist_ok=True)
    cmd = build_dpo_command(
        model=model,
        corpus=corpus,
        output=output,
        cfg=cfg,
        trainer=trainer,
        mount_adapter=parent_adapter,
    )
    if dry_run:
        return TrainResult(
            ok=True, adapter_path=str(output), command=cmd, warnings=warnings
        )

    rc, out, err, elapsed = _run(cmd)
    collapse = detect_collapse(out + err)
    if collapse:
        warnings.append(collapse)

    result = TrainResult(
        ok=rc == 0,
        adapter_path=str(output),
        command=cmd,
        returncode=rc,
        stdout_tail=out,
        stderr_tail=err,
        elapsed_s=elapsed,
        warnings=warnings,
    )
    if rc == 0:
        record = ProvenanceRecord(
            adapter_name=adapter_name,
            source_kind=SourceKind.CUSTOMER_REPO,
            source_repo=source_repo,
            commit_range=commit_range,
            extraction_filters=extraction_filters or {},
            corpus_hash=corpus_hash_for(corpus),
            n_pairs=n_pairs,
            base_model_hash=hash_artefact(model) or "",
            base_model_name=model,
            training_config={"method": "dpo", **cfg.as_dict()},
            parent_adapter_hash=hash_artefact(parent_adapter)
            if parent_adapter
            else None,
            licence=licence,
            created_ts=time.time(),
        )
        result.provenance_path = str(record.write(output / "provenance.json"))
    return result


def detect_collapse(log: str) -> str:
    """Watch for the preference-collapse signature (sec 6.3).

    Loss -> 0 with rewards drifting to large negatives means the pairs are not
    answering the same prompt. Catching it in the log is worth more than catching it
    in the eval a week later.
    """
    import re

    losses = [
        float(m)
        for m in re.findall(r"(?:train )?loss[ =:]+([0-9.]+)", log, re.IGNORECASE)
    ]
    rewards = [
        float(m)
        for m in re.findall(
            r"reward(?:s)?[a-z_/]*[ =:]+(-?[0-9.]+)", log, re.IGNORECASE
        )
    ]
    if len(losses) >= 3 and losses[-1] < 0.01 and losses[0] > losses[-1] * 10:
        if any(r < -5.0 for r in rewards[-5:]):
            return (
                "Preference collapse signature: loss -> 0 with rewards drifting to "
                "large negatives. The chosen/rejected pairs are not answering the "
                "same prompt (sec 6.3). Re-check extraction before shipping this adapter."
            )
        return (
            "Loss collapsed to ~0. Verify the DPO pairs share a prompt; a saturated "
            "margin means no signal was learned (sec 6.3)."
        )
    return ""
