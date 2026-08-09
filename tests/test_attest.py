"""Attestation (spec sec 9): hashes, receipts, audit log, provenance."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from tandem.attest.audit import (
    GENESIS,
    AuditLog,
    AuditRecord,
    _link,
    receipt_fields,
    verify_chain,
)
from tandem.attest.hashing import hash_artefact, hash_text, hash_tree
from tandem.attest.provenance import (
    ProvenanceError,
    ProvenanceRecord,
    SourceKind,
    corpus_hash_for,
    redact_source_repo,
)
from tandem.attest.receipt import (
    REDUCTION_ORDER,
    Receipt,
    Tier0Attestation,
    Tier1Attestation,
    engine_commit,
)
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


def test_hash_artefact_is_not_fooled_by_a_same_length_swap_and_utime(tmp_path):
    """H13: size+mtime memoisation served a stale digest for swapped weights.

    The gateway is long-lived and `lru_cache` is per-process, so a hit here means
    every later receipt attests weights that are not on disk.
    """
    p = tmp_path / "adapter"
    p.mkdir()
    weights = p / "adapters.safetensors"
    weights.write_bytes(b"v1")
    st = weights.stat()
    first = hash_artefact(p)

    weights.write_bytes(b"v2")  # same length, same inode
    os.utime(weights, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert weights.stat().st_mtime_ns == st.st_mtime_ns  # the disguise worked
    assert weights.stat().st_size == st.st_size

    assert hash_artefact(p) != first
    assert hash_artefact(p) == hash_tree(p)


def test_tree_hash_follows_a_symlinked_directory(tmp_path):
    """M29: content-addressed weights are the normal layout, not an edge case."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "0001.safetensors").write_bytes(b"real weights")

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "weights").symlink_to(blobs, target_is_directory=True)

    first = hash_tree(adapter)
    (blobs / "0001.safetensors").write_bytes(b"other weights")
    assert hash_tree(adapter) != first


def test_tree_hash_sees_hidden_files(tmp_path):
    """A hidden-files-only directory used to hash identically to an empty one."""
    empty = tmp_path / "empty"
    empty.mkdir()
    hidden = tmp_path / "hidden"
    (hidden / ".cfg").mkdir(parents=True)
    (hidden / ".cfg" / ".weights").write_bytes(b"w")
    assert hash_tree(hidden) != hash_tree(empty)


def test_tree_hash_terminates_on_a_symlink_cycle(tmp_path):
    root = tmp_path / "adapter"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "w.safetensors").write_bytes(b"w")
    (root / "sub" / "loop").symlink_to(root, target_is_directory=True)
    assert hash_tree(root)  # terminates rather than recursing forever


def test_tree_hash_excludes_the_provenance_record(tmp_path):
    """M28: `created_ts` inside the adapter dir made adapter_blake3 a clock reading."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"w")
    before = hash_tree(adapter)

    (adapter / "provenance.json").write_text('{"created_ts": 1.0}')
    assert hash_tree(adapter) == before
    (adapter / "provenance.json").write_text('{"created_ts": 2.0}')
    assert hash_tree(adapter) == before


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


# --- C1: truncation, deletion, forgery, and the tip anchor ------------------


def test_trailing_truncation_verifies_green_without_an_anchor(tmp_path):
    """The reason a tip anchor exists: a shorter chain is a valid chain."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(5):
        log.append(_record(i))
    tip = log.head()
    assert tip.records == 5

    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:3]) + "\n")

    ok, reason = verify_chain(path)
    assert ok and "3 records" in reason  # green, and wrong

    ok, reason = verify_chain(path, expected_tip=tip.link)
    assert not ok
    assert "removed from the end" in reason


def test_truncate_then_forge_is_caught_by_the_anchor(tmp_path):
    """`AuditLog` is the forging tool: it resumes from whatever tail it finds."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(5):
        log.append(_record(i, escalated=True))
    tip = log.head()

    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:3]) + "\n")
    AuditLog(path).append(_record(3, escalated=False))  # the escalation, erased

    ok, _ = verify_chain(path)
    assert ok  # internally consistent — that is the whole problem

    ok, reason = verify_chain(path, expected_tip=tip.link)
    assert not ok
    assert "anchored tip" in reason


def test_a_matching_anchor_verifies(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    last = ""
    for i in range(3):
        last = log.append(_record(i))
    ok, reason = verify_chain(path, expected_tip=last)
    assert ok
    assert "tip anchored" in reason
    assert log.head().link == last


def test_a_missing_log_does_not_verify(tmp_path):
    """"No evidence" must not read as "verified" for a compliance artefact."""
    missing = tmp_path / "nothing.jsonl"
    ok, reason = verify_chain(missing)
    assert not ok
    assert "absent log is not an empty one" in reason

    ok, reason = verify_chain(missing, allow_empty=True)
    assert ok
    assert "nothing written yet" in reason

    ok, _ = verify_chain(missing, expected_tip="a" * 64)
    assert not ok


def test_a_zero_record_log_is_distinguishable_from_a_missing_one(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("")
    ok, reason = verify_chain(path)
    assert not ok
    assert "empty log is not evidence" in reason

    ok, reason = verify_chain(path, allow_empty=True)
    assert ok
    assert "as expected" in reason


def test_sequence_numbers_are_inside_the_hashed_payload(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.append(_record(i))

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["seq"] for r in rows] == [0, 1, 2]

    rows[1]["seq"] = 7  # renumbering is arithmetic, and caught as such
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n"
    )
    ok, reason = verify_chain(path)
    assert not ok
    assert "sequence number 7, expected 1" in reason

    # And the number is inside the hashed payload, not written beside it, so a
    # forger cannot renumber a record and leave its link alone.
    payload = {k: v for k, v in rows[0].items() if k not in ("prev", "link")}
    assert _link(payload, GENESIS) == rows[0]["link"]
    assert _link({**payload, "seq": 7}, GENESIS) != rows[0]["link"]


def test_a_log_without_sequence_numbers_is_named_not_accepted(tmp_path):
    """The record format changed; an old log must break loudly, not pass quietly."""
    path = tmp_path / "audit.jsonl"
    payload = _record(0).as_dict()
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    link = hashlib.sha256(f"{GENESIS}\n{body}".encode()).hexdigest()
    path.write_text(
        json.dumps({**payload, "prev": GENESIS, "link": link}, sort_keys=True,
                   separators=(",", ":")) + "\n"
    )
    ok, reason = verify_chain(path)
    assert not ok
    assert "no sequence number" in reason


# --- H12: two writers must not fork the chain -------------------------------


def test_two_log_instances_interleaved_do_not_fork_the_chain(tmp_path):
    """A cached tip made the second writer reuse the first's link."""
    path = tmp_path / "audit.jsonl"
    a = AuditLog(path)
    b = AuditLog(path)
    a.append(_record(0))
    b.append(_record(1))
    a.append(_record(2))
    b.append(_record(3))
    ok, reason = verify_chain(path)
    assert ok, reason
    assert "4 records" in reason


def test_two_processes_appending_concurrently_do_not_fork_the_chain(tmp_path):
    """flock, not the threading lock: the gateway and a CLI run share one log.

    A fork is indistinguishable from tampering after the fact, which is the worst
    failure mode a tamper-evidence tool has.
    """
    path = tmp_path / "audit.jsonl"
    writer = (
        "import sys;"
        "from tandem.attest.audit import AuditLog, AuditRecord, now;"
        "log = AuditLog(sys.argv[1]);"
        "[log.append(AuditRecord("
        "    request_id=f'{sys.argv[2]}-{i}', ts=now(), harness='h',"
        "    tier0_hash='t0', adapter_hash=None, tier1_hash=None,"
        "    prompt_sha256='p'*64, output_sha256='o'*64,"
        "    tools_invoked=(), escalated=False)) for i in range(25)]"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", writer, str(path), tag])
        for tag in ("a", "b")
    ]
    for proc in procs:
        assert proc.wait(timeout=60) == 0

    ok, reason = verify_chain(path)
    assert ok, reason
    assert "50 records" in reason


# --- M28: the receipt is bound by the chain ---------------------------------


def test_audit_record_carries_the_determinism_fields(tmp_path):
    """Sec 9.3 is stated in terms of these; a receipt outside the chain is a claim."""
    receipt = Receipt(
        tier1=Tier1Attestation(rung="rung-2"),
        compaction_template="cc-2026.08@v3",
        sampling=Sampling(temperature=0.2, top_p=0.9, seed=11),
        candidates_generated=4,
        candidate_selected=2,
    )
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append(_record(0, **receipt_fields(receipt)))

    row = json.loads(path.read_text().strip())
    assert row["seed"] == 11
    assert row["sampling"] == {"temperature": 0.2, "top_p": 0.9}
    assert row["compaction_template"] == "cc-2026.08@v3"
    assert row["engine_commit"] == receipt.as_dict()["engine_commit"]
    assert row["reduction_order"] == REDUCTION_ORDER
    assert row["candidates_generated"] == 4
    assert row["candidate_selected"] == 2
    assert row["tier1_rung"] == "rung-2"

    # And they are hashed, not merely written beside the hash.
    row["seed"] = 12
    path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    ok, reason = verify_chain(path)
    assert not ok
    assert "does not match its link" in reason


def test_the_determinism_fields_stay_optional(tmp_path):
    """A caller with only the spec tuple still writes a valid record."""
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append(_record(0))
    row = json.loads(path.read_text().strip())
    assert row["seed"] is None
    assert row["tier1_rung"] is None
    ok, _ = verify_chain(path)
    assert ok


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
        "expert_cache_configured_bytes": None,
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


def test_receipt_carries_the_correlation_fields():
    """M2/C1: the response, the receipt and the audit line must share a field."""
    d = Receipt(request_id="req-1", audit_tip="f" * 64).as_dict()
    assert d["request_id"] == "req-1"
    assert d["audit_tip"] == "f" * 64
    blank = Receipt().as_dict()
    assert blank["request_id"] == "" and blank["audit_tip"] == ""


def test_engine_commit_refuses_a_stamp_that_is_not_a_commit(monkeypatch):
    """M28: a tag or branch name in the env var was attested as the commit."""
    engine_commit.cache_clear()
    monkeypatch.setenv("TANDEM_ENGINE_COMMIT", "v2-hotfix")
    assert engine_commit() != "v2-hotfix"
    engine_commit.cache_clear()
    monkeypatch.setenv("TANDEM_ENGINE_COMMIT", "  DEADBEEF1234  ")
    assert engine_commit() == "deadbeef1234"
    engine_commit.cache_clear()


def test_engine_commit_never_attests_an_unrelated_repo(monkeypatch, tmp_path):
    """M28: from site-packages, `git -C <parents[3]>` names a stranger's HEAD."""
    from tandem.attest import receipt as receipt_mod

    (tmp_path / ".git").mkdir()  # an unrelated checkout of the user's
    installed = tmp_path / "site-packages" / "tandem" / "attest" / "receipt.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("")

    def _no_git(*a, **kw):  # pragma: no cover - asserted not to run
        raise AssertionError("git must not run outside this package's checkout")

    monkeypatch.setattr(receipt_mod, "__file__", str(installed))
    monkeypatch.setattr(receipt_mod.subprocess, "run", _no_git)
    monkeypatch.delenv("TANDEM_ENGINE_COMMIT", raising=False)
    engine_commit.cache_clear()
    assert engine_commit() == "unknown"
    engine_commit.cache_clear()


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


def test_corpus_hash_splits_the_way_the_reader_does(tmp_path):
    """M30: `splitlines()` broke on characters a JSONL reader passes straight through.

    U+2028 survives `ensure_ascii=False` and appears in real JavaScript. Under the
    old decomposition these two different corpora hash identically — "same
    corpus_hash ⇒ same corpus" simply stops holding.
    """
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    one.write_text('{"t": "a b"}\n', encoding="utf-8")
    two.write_text('{"t": "a\nb"}\n', encoding="utf-8")

    assert sorted('{"t": "a b"}'.splitlines()) == sorted('{"t": "a\nb"}'.splitlines())
    assert corpus_hash_for(one) != corpus_hash_for(two)


def test_corpus_hash_is_stable_across_a_form_feed(tmp_path):
    """A form feed in a C file must not turn one record into two."""
    p = tmp_path / "c.jsonl"
    p.write_text('{"t": "a\\fb"}\n{"t": "z"}\n', encoding="utf-8")
    first = corpus_hash_for(p)
    p.write_text('{"t": "z"}\n{"t": "a\\fb"}\n', encoding="utf-8")
    assert corpus_hash_for(p) == first


def test_source_repo_never_ships_an_absolute_local_path(tmp_path):
    """A shipped adapter carried "/Users/alice/clients/acme-payments"."""
    record = ProvenanceRecord(
        adapter_name="a1",
        source_kind=SourceKind.CUSTOMER_REPO,
        source_repo="/Users/alice/clients/acme-payments",
        corpus_hash="c",
        base_model_hash="b",
    )
    shipped = record.as_dict()["source_repo"]
    assert "/Users/alice" not in shipped
    assert shipped.startswith("acme-payments#")
    # Stable, so two records can still be compared for "same checkout"...
    assert redact_source_repo("/Users/alice/clients/acme-payments") == shipped
    # ...and reloading a record does not redact the redaction.
    path = record.write(tmp_path / "provenance.json")
    assert ProvenanceRecord.load(path).source_repo == shipped


def test_source_repo_keeps_a_remote_url_verbatim():
    """The URL is what an auditor re-derives the corpus from."""
    for url in ("https://github.com/org/repo", "git@github.com:org/repo.git"):
        assert redact_source_repo(url) == url


def test_hash_text_is_stable():
    assert hash_text("tandem") == hash_text("tandem")
    assert hash_text("tandem") != hash_text("Tandem")
