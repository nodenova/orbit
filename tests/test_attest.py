"""Attestation (spec sec 9): hashes, receipts, audit log, provenance."""

from __future__ import annotations

import json

import pytest

from tandem.attest.audit import GENESIS, AuditLog, AuditRecord, verify_chain
from tandem.attest.hashing import hash_artefact, hash_text, hash_tree
from tandem.attest.provenance import (
    ProvenanceError,
    ProvenanceRecord,
    SourceKind,
    corpus_hash_for,
)
from tandem.attest.receipt import REDUCTION_ORDER, Receipt, Tier0Attestation, Tier1Attestation
from tandem.types import Sampling


def _record(i: int, **kw) -> AuditRecord:
    base = dict(
        request_id=f"r{i}",
        ts=float(i),
        harness="claude_code",
        tier0_hash="t0",
        adapter_hash="a1",
        tier1_hash=None,
        prompt_sha256="p" * 64,
        output_sha256="o" * 64,
        tools_invoked=("read_file",),
        escalated=False,
    )
    base.update(kw)
    return AuditRecord(**base)


# --- hashing ----------------------------------------------------------------


def test_tree_hash_is_content_and_name_sensitive(tmp_path):
    root = tmp_path / "container"
    (root / "sub").mkdir(parents=True)
    (root / "a.safetensors").write_bytes(b"weights")
    (root / "sub" / "b.json").write_text("{}")
    first = hash_tree(root)

    (root / "sub" / "b.json").write_text('{"x": 1}')
    assert hash_tree(root) != first

    (root / "sub" / "b.json").write_text("{}")
    assert hash_tree(root) == first

    (root / "sub" / "b.json").rename(root / "sub" / "c.json")
    assert hash_tree(root) != first


def test_tree_hash_ignores_incidental_files(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "w.safetensors").write_bytes(b"w")
    before = hash_tree(root)
    (root / "README.md").write_text("hello")
    (root / ".DS_Store").write_bytes(b"junk")
    assert hash_tree(root) == before


def test_hash_artefact_returns_none_for_a_missing_path():
    """A receipt must be able to say 'not mounted' rather than invent a digest."""
    assert hash_artefact(None) is None
    assert hash_artefact("/nonexistent/adapter") is None


def test_hash_artefact_rehashes_after_an_edit(tmp_path):
    p = tmp_path / "adapter"
    p.mkdir()
    (p / "adapters.safetensors").write_bytes(b"v1")
    first = hash_artefact(p)
    (p / "adapters.safetensors").write_bytes(b"v2-longer")
    assert hash_artefact(p) != first


# --- audit log (sec 9.2) ----------------------------------------------------


def test_audit_log_appends_and_verifies(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.append(_record(i))
    ok, reason = verify_chain(tmp_path / "audit.jsonl")
    assert ok
    assert "5 records" in reason


def test_audit_log_records_the_named_tuple(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.append(_record(0))
    row = json.loads((tmp_path / "a.jsonl").read_text().strip())
    for field in (
        "request_id", "ts", "harness", "tier0_hash", "adapter_hash", "tier1_hash",
        "prompt_sha256", "output_sha256", "tools_invoked", "escalated",
    ):
        assert field in row


def test_audit_log_never_stores_the_prompt(tmp_path):
    """The log must be safe to hand to a compliance reviewer."""
    log = AuditLog(tmp_path / "a.jsonl")
    log.append(_record(0))
    body = (tmp_path / "a.jsonl").read_text()
    assert "p" * 64 in body  # the hash
    assert "secret source code" not in body


def test_tampering_with_a_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.append(_record(i))

    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["output_sha256"] = "f" * 64
    lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    ok, reason = verify_chain(path)
    assert not ok
    assert "does not match its link" in reason


def test_removing_a_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(4):
        log.append(_record(i))
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2], lines[3]]) + "\n")
    ok, reason = verify_chain(path)
    assert not ok
    assert "removed or reordered" in reason


def test_chain_resumes_across_a_restart(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append(_record(0))
    AuditLog(path).append(_record(1))  # fresh instance, as after a restart
    ok, _ = verify_chain(path)
    assert ok


def test_first_record_links_to_genesis(tmp_path):
    path = tmp_path / "a.jsonl"
    AuditLog(path).append(_record(0))
    assert json.loads(path.read_text().strip())["prev"] == GENESIS


# --- receipts (sec 9.1) -----------------------------------------------------


def test_receipt_has_the_spec_shape():
    receipt = Receipt(
        tier0=Tier0Attestation(container_blake3="c0", adapter_blake3="a0", profile_blake3="p0"),
        tier1=Tier1Attestation(container_blake3="c1", invoked=True, call="rerank"),
        compaction_template="cc-2026.08@v3",
        sampling=Sampling(temperature=0.2, top_p=1.0, seed=7),
        candidates_generated=3,
        candidate_selected=1,
    )
    d = receipt.as_dict()
    assert d["tier0"]["container_blake3"] == "c0"
    assert d["tier1"] == {
        "container_blake3": "c1",
        "invoked": True,
        "call": "rerank",
        "rung": None,
        "expert_cache_bytes": None,
    }
    assert d["compaction_template"] == "cc-2026.08@v3"
    assert d["seed"] == 7
    assert d["sampling"] == {"temperature": 0.2, "top_p": 1.0}
    assert d["reduction_order"] == REDUCTION_ORDER
    assert d["candidates_generated"] == 3
    assert d["candidate_selected"] == 1
    assert "engine_commit" in d


def test_receipt_uses_honest_nulls_not_absence():
    """A consumer diffing two receipts must not distinguish absent from unknown."""
    d = Receipt().as_dict()
    assert d["tier0"]["adapter_blake3"] is None
    assert d["tier1"]["invoked"] is False


# --- provenance (sec 9.4) ---------------------------------------------------


def test_provenance_round_trips(tmp_path):
    record = ProvenanceRecord(
        adapter_name="a1-myrepo",
        source_kind=SourceKind.CUSTOMER_REPO,
        source_repo="/srv/myrepo",
        commit_range=("abc123", "def456"),
        extraction_filters={"max_files": 15},
        corpus_hash="ch",
        n_pairs=1200,
        base_model_hash="bh",
        base_model_name="Qwen3.6-35B-A3B",
        training_config={"method": "sft", "epochs": 3},
        licence="Apache-2.0",
    )
    path = record.write(tmp_path / "provenance.json")
    loaded = ProvenanceRecord.load(path)
    assert loaded.adapter_name == "a1-myrepo"
    assert loaded.commit_range == ("abc123", "def456")
    assert loaded.source_kind is SourceKind.CUSTOMER_REPO


def test_source_kind_has_no_member_for_model_distillation():
    """Sec 9.4: never train on another model's outputs. Enforced, not remembered."""
    assert {k.value for k in SourceKind} == {
        "customer_repo", "permissive_corpus", "synthetic_harness"
    }
    with pytest.raises(ValueError):
        SourceKind("frontier_model_traces")


def test_provenance_refuses_an_unattested_corpus():
    record = ProvenanceRecord(
        adapter_name="x", source_kind=SourceKind.CUSTOMER_REPO,
        source_repo="/r", base_model_hash="b",
    )
    with pytest.raises(ProvenanceError, match="corpus_hash"):
        record.as_dict()


def test_provenance_refuses_a_string_source_kind():
    record = ProvenanceRecord(
        adapter_name="x", source_kind="frontier_traces",  # type: ignore[arg-type]
        corpus_hash="c", base_model_hash="b",
    )
    with pytest.raises(ProvenanceError, match="not a permitted source"):
        record.as_dict()


def test_corpus_hash_is_line_order_independent(tmp_path):
    """A topological tie in git history must not invalidate an identical corpus."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"x": 1}\n{"y": 2}\n')
    b.write_text('{"y": 2}\n{"x": 1}\n')
    assert corpus_hash_for(a) == corpus_hash_for(b)


def test_corpus_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text('{"x": 1}\n')
    first = corpus_hash_for(a)
    a.write_text('{"x": 2}\n')
    assert corpus_hash_for(a) != first


def test_hash_text_is_stable():
    assert hash_text("tandem") == hash_text("tandem")
    assert hash_text("tandem") != hash_text("Tandem")
