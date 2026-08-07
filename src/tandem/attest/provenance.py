"""Training-data provenance records (spec sec 9.4).

Every adapter ships with one of these beside it. It answers, for an auditor: what
was this trained on, from which commits, under which filters, against which base.

The spec's hard rule — **never train on another model's outputs** — is enforced
here rather than left to discipline. `SourceKind` is a closed set; there is no
member for frontier-model traces, and `ProvenanceRecord` refuses to serialise
without a source kind. A corpus we cannot attest, we do not train on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .hashing import PROVENANCE_FILENAME, hash_file, hash_text

__all__ = [
    "PROVENANCE_FILENAME",
    "ProvenanceError",
    "ProvenanceRecord",
    "SourceKind",
    "corpus_hash_for",
    "file_hash",
    "redact_source_repo",
]


class SourceKind(str, Enum):
    """Permitted training-data origins. Closed by design.

    Adding a member is a commercial and legal decision, not a code change to make
    in passing: distillation from a frontier model conflicts with competing-model
    clauses and is unprovable to an auditor (sec 9.4).
    """

    CUSTOMER_REPO = "customer_repo"  # A1/A2 — the customer's own git history
    PERMISSIVE_CORPUS = "permissive_corpus"  # Apache-2.0 / MIT / BSD sources only
    SYNTHETIC_HARNESS = "synthetic_harness"  # A0 — traces we generate from a grammar


class ProvenanceError(ValueError):
    pass


def redact_source_repo(source_repo: str) -> str:
    """Make a local checkout path safe to ship inside an adapter.

    `provenance.json` travels with the adapter, so an absolute path publishes the
    operator's home directory, username and client name to everyone the adapter is
    shared with — `"/Users/alice/clients/acme-payments"` is a real shape. Keep the
    basename, because an auditor still needs to know *which* repo, and replace the
    rest with a short digest of the full path: stable run to run, so two records can
    still be compared for "same checkout", and not reversible into the path.

    Remote URLs and scp-style git addresses are left verbatim. They are already
    shareable, and they are what an auditor would use to re-derive the corpus —
    redacting them would cost the record its only reproducible pointer.

    Idempotent: the redacted form carries no separator, so re-reading a record and
    writing it back does not redact twice.
    """
    value = source_repo.strip()
    if not value:
        return value
    if "://" in value or (":" in value and "/" not in value.split(":", 1)[0]):
        return value  # https://…, ssh://…, git@host:org/repo
    if "/" not in value and os.sep not in value and not value.startswith("~"):
        return value  # a bare name leaks nothing
    name = Path(value.rstrip("/" + os.sep)).name or value
    return f"{name}#{hash_text(value)[:12]}"


@dataclass
class ProvenanceRecord:
    adapter_name: str
    source_kind: SourceKind
    # Where the data came from. For CUSTOMER_REPO: repo URL or local path.
    source_repo: str = ""
    # Inclusive commit range the extractor walked, as (first, last) shas.
    commit_range: tuple[str, str] | None = None
    # Every filter applied at extraction, verbatim (sec 6.2), so a second run is
    # reproducible from the record alone.
    extraction_filters: dict[str, Any] = field(default_factory=dict)
    corpus_hash: str = ""
    n_pairs: int = 0
    base_model_hash: str = ""
    base_model_name: str = ""
    training_config: dict[str, Any] = field(default_factory=dict)
    # For A2: the A1 adapter DPO started from (sec 6.3 --mount-adapter).
    parent_adapter_hash: str | None = None
    licence: str = ""
    created_ts: float = 0.0

    def __post_init__(self) -> None:
        # Redact once, at construction, rather than on the way out: there is then a
        # single representation of the field, and a serialiser added later cannot
        # reintroduce the leak by forgetting to call the redactor.
        self.source_repo = redact_source_repo(self.source_repo)

    def validate(self) -> None:
        if not self.adapter_name:
            raise ProvenanceError("adapter_name is required")
        if not isinstance(self.source_kind, SourceKind):
            raise ProvenanceError(
                f"source_kind must be a SourceKind, got {self.source_kind!r}. "
                "Training on another model's outputs is not a permitted source."
            )
        if not self.corpus_hash:
            raise ProvenanceError("corpus_hash is required — an unattested corpus is not trainable")
        if not self.base_model_hash:
            raise ProvenanceError("base_model_hash is required")
        if self.source_kind is SourceKind.CUSTOMER_REPO and not self.source_repo:
            raise ProvenanceError("customer_repo provenance requires source_repo")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "adapter_name": self.adapter_name,
            "source_kind": self.source_kind.value,
            "source_repo": self.source_repo,
            "commit_range": list(self.commit_range) if self.commit_range else None,
            "extraction_filters": self.extraction_filters,
            "corpus_hash": self.corpus_hash,
            "n_pairs": self.n_pairs,
            "base_model_hash": self.base_model_hash,
            "base_model_name": self.base_model_name,
            "training_config": self.training_config,
            "parent_adapter_hash": self.parent_adapter_hash,
            "licence": self.licence,
            "created_ts": self.created_ts,
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> ProvenanceRecord:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rng = raw.get("commit_range")
        return cls(
            adapter_name=raw["adapter_name"],
            source_kind=SourceKind(raw["source_kind"]),
            source_repo=raw.get("source_repo", ""),
            commit_range=(rng[0], rng[1]) if rng else None,
            extraction_filters=raw.get("extraction_filters", {}),
            corpus_hash=raw.get("corpus_hash", ""),
            n_pairs=raw.get("n_pairs", 0),
            base_model_hash=raw.get("base_model_hash", ""),
            base_model_name=raw.get("base_model_name", ""),
            training_config=raw.get("training_config", {}),
            parent_adapter_hash=raw.get("parent_adapter_hash"),
            licence=raw.get("licence", ""),
            created_ts=raw.get("created_ts", 0.0),
        )


def corpus_hash_for(path: str | Path) -> str:
    """Hash a JSONL corpus file.

    Line-order-independent: extraction walks git history and a topological tie can
    reorder two same-timestamp commits between runs, which should not invalidate an
    otherwise identical corpus.

    The decomposition is the reader's, not Python's convenience one. A JSONL reader
    — mlx-lm's, ours — takes one JSON document per ``\\n`` and nothing else, while
    `str.splitlines()` also breaks on U+0085, U+2028 and U+2029, which survive the
    extractors' ``ensure_ascii=False`` verbatim and occur in ordinary source (U+2028
    inside a JavaScript string literal is the common one). Hashing that
    decomposition hashes fragments no reader ever sees: after sorting, two different
    corpora can produce the same digest, and "same corpus_hash ⇒ same corpus" — the
    whole claim the provenance record makes — stops holding (M30). ``newline=""``
    for the same reason: universal-newline translation would rewrite a lone ``\\r``
    into a split the reader does not make.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    lines = sorted(line for line in text.split("\n") if line.strip())
    return hash_text("\n".join(lines))


def file_hash(path: str | Path) -> str:
    return hash_file(path)
