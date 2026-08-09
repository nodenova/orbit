"""Adapter pipeline (spec sec 6): extraction, filters, profile, training driver.

The git tests build small real repositories rather than mocking `git`, because the
failure modes that matter here are all in git's actual output format — `-z` numstat
framing, rename records, merge parentage. A mock would agree with whatever the code
already believes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tandem.adapters import extract_a0, extract_a1, extract_a2, profile
from tandem.adapters.filters import (
    ExtractionFilters,
    is_bot,
    is_lockfile,
    is_revert,
    is_vendored,
    keep_path,
)
from tandem.adapters.train import (
    DPOConfig,
    SFTConfig,
    build_dpo_command,
    build_sft_command,
    detect_collapse,
    preflight,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "dev@example.com")
    git(root, "config", "user.name", "A Developer")

    (root / "app.py").write_text("def handler():\n    return 1\n")
    (root / "util.py").write_text("def helper():\n    pass\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Initial commit with the handler and helper")

    (root / "app.py").write_text("def handler():\n    return 2\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Return 2 from the handler instead of 1")

    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n')
    (root / "util.py").write_text("def helper():\n    return None\n")
    git(root, "add", ".")
    git(
        root,
        "commit",
        "-q",
        "-m",
        "Make the helper return None and refresh the lockfile",
    )
    return root


# --- filters ----------------------------------------------------------------


@pytest.mark.parametrize(
    "path,vendored",
    [
        ("vendor/lib/x.go", True),
        ("node_modules/pkg/index.js", True),
        ("src/app.min.js", True),
        ("api/service_pb2.py", True),
        ("dist/bundle.js", True),
        ("assets/logo.png", True),
        ("src/click/core.py", False),
        ("tests/test_core.py", False),
    ],
)
def test_vendored_detection(path, vendored):
    assert is_vendored(path) is vendored


def test_lockfiles_are_filtered():
    assert is_lockfile("package-lock.json")
    assert is_lockfile("a/b/Cargo.lock")
    assert not is_lockfile("src/lock.py")


@pytest.mark.parametrize(
    "name,email,bot",
    [
        ("dependabot[bot]", "x@y", True),
        ("renovate", "renovate@x", True),
        ("A Developer", "dev@example.com", False),
        ("Someone", "noreply@github.com", True),
    ],
)
def test_bot_detection(name, email, bot):
    assert is_bot(name, email) is bot


def test_revert_detection():
    assert is_revert('Revert "Add the thing"')
    assert not is_revert("Reverse-engineer the protocol")


def test_keep_path_composes_the_filters():
    f = ExtractionFilters()
    assert keep_path("src/app.py", f)
    assert not keep_path("vendor/x.go", f)
    assert not keep_path("yarn.lock", f)


# --- A1 (sec 6.2) -----------------------------------------------------------


def test_a1_extracts_pairs_from_history(repo):
    train, _held, report = extract_a1.extract(repo)
    assert report.pairs > 0
    assert train[0].prompt and train[0].completion
    assert train[0].completion.startswith("diff --git")


def test_a1_context_comes_from_the_parent_commit(repo):
    """Showing the post-change file would make this a copying task, not a coding task."""
    train, _held, _report = extract_a1.extract(repo)
    pair = next(p for p in train if "Return 2" in p.prompt)
    # The parent had `return 1`; the completion introduces `return 2`.
    assert "return 1" in pair.prompt
    assert "+    return 2" in pair.completion


def test_a1_excludes_lockfiles_from_the_completion(repo):
    """A commit touching source and a lockfile must not teach lockfile churn."""
    train, _held, _report = extract_a1.extract(repo)
    pair = next(p for p in train if "helper" in p.prompt)
    assert "package-lock.json" not in pair.completion
    assert "util.py" in pair.completion


def test_a1_writes_messages_jsonl_never_bare_text(repo, tmp_path):
    """Sec 6.2: bare text cannot expose a prompt/response boundary, so prompt
    masking falls through to full-sequence loss."""
    train, _held, _report = extract_a1.extract(repo)
    path = extract_a1.write_jsonl(train, tmp_path / "train.jsonl")
    row = json.loads(path.read_text().splitlines()[0])
    assert "text" not in row
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]


def test_a1_reports_a_thin_corpus_honestly(repo):
    """Sec 13: below ~500 pairs, tell the customer rather than ship a null adapter."""
    _train, _held, report = extract_a1.extract(repo)
    assert report.thin
    assert "underfit" in report.as_dict()["advice"]


def test_a1_holdout_is_the_most_recent_work(repo):
    """The eval claims performance on unseen work; in a repo that means later work."""
    train, held, _report = extract_a1.extract(repo, holdout=1)
    assert len(held) == 1
    assert all(held[0].ts >= p.ts for p in train)
    assert held[0].sha not in {p.sha for p in train}


def test_a1_skips_reverts_and_bots(repo):
    git(repo, "config", "user.name", "dependabot[bot]")
    (repo / "app.py").write_text("def handler():\n    return 3\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "Bump a dependency to the latest version")
    _train, _held, report = extract_a1.extract(repo)
    assert report.skips.as_dict().get("bot_author") == 1


def test_merge_policy_auto_uses_first_parent_on_a_merge_heavy_branch(tmp_path):
    """Skipping merges on a merge-commit repo drops ~90% of usable history."""
    root = tmp_path / "merged"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "d@e.com")
    git(root, "config", "user.name", "Dev")
    (root / "a.py").write_text("x = 1\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Initial commit of the module")

    for i in range(3):
        git(root, "checkout", "-q", "-b", f"feature-{i}")
        (root / "a.py").write_text(f"x = {i + 2}\n")
        git(root, "add", ".")
        git(root, "commit", "-q", "-m", f"Change x to {i + 2} on the feature branch")
        git(root, "checkout", "-q", "main")
        git(
            root,
            "merge",
            "-q",
            "--no-ff",
            f"feature-{i}",
            "-m",
            f"Merge feature {i} into main",
        )

    _train, _held, report = extract_a1.extract(root)
    assert report.merge_policy_used == "first_parent"
    assert report.pairs >= 3

    filters = ExtractionFilters(merge_policy="skip")
    _t, _h, skipped_report = extract_a1.extract(root, filters=filters)
    assert skipped_report.pairs < report.pairs


def test_excerpt_centres_on_the_changed_hunks():
    content = "\n".join(f"line {i}" for i in range(1, 201))
    excerpt = extract_a1.excerpt_around(content, [(100, 102)], budget=10_000)
    assert "line 100" in excerpt
    # Distant lines are dropped: whole files blow the 4096-token training sequence.
    assert "line 5\n" not in excerpt
    assert "line 190" not in excerpt


def test_excerpt_marks_the_gap_between_distant_hunks():
    content = "\n".join(f"line {i}" for i in range(1, 201))
    excerpt = extract_a1.excerpt_around(content, [(10, 11), (150, 151)], budget=10_000)
    assert "line 10" in excerpt and "line 150" in excerpt
    assert "lines omitted" in excerpt


def test_clean_message_strips_trailers_and_issue_refs():
    from tandem.adapters.gitwalk import Commit

    commit = Commit(
        sha="x",
        parents=("p",),
        author_name="A",
        author_email="a@b",
        ts=0,
        subject="Fix the retry loop (#123)",
        body="Explains the change.\n\nSigned-off-by: A <a@b>\nFixes: #99\n",
    )
    msg = extract_a1.clean_message(commit)
    assert msg.startswith("Fix the retry loop")
    assert "#123" not in msg
    assert "Signed-off-by" not in msg
    assert "Fixes:" not in msg
    assert "Explains the change." in msg


def test_changed_line_ranges_parses_hunk_headers():
    unified = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -10,3 +10,4 @@ def f():\n-a\n+b\n"
        "@@ -50 +51 @@\n-c\n+d\n"
    )
    assert extract_a1.changed_line_ranges(unified, "x.py") == [(10, 12), (50, 50)]


# --- A2 (sec 6.3) -----------------------------------------------------------


def test_a2_builds_preference_pairs_from_branch_revisions(tmp_path):
    root = tmp_path / "reviewed"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "d@e.com")
    git(root, "config", "user.name", "Dev")
    (root / "a.py").write_text("def f():\n    return 1\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Initial commit of the module")

    git(root, "checkout", "-q", "-b", "feature")
    (root / "a.py").write_text("def f():\n    return 2  # first attempt\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "First attempt at the change")
    (root / "a.py").write_text("def f():\n    return 2\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Address review feedback and drop the comment")
    git(root, "checkout", "-q", "main")
    git(root, "merge", "-q", "--no-ff", "feature", "-m", "Merge the reviewed feature")

    train, _held, report = extract_a2.extract(root)
    assert report.pairs == 1
    pair = train[0]
    # rejected is what the author proposed; chosen is what survived review.
    assert "first attempt" in pair.rejected
    assert "first attempt" not in pair.chosen
    # Both must answer the same prompt, or the DPO margin saturates (sec 6.3).
    assert pair.prompt


def test_a2_divergence_is_linear_and_offset_insensitive():
    """Two diffs making the same edits at different offsets are the same proposal."""
    a = "@@ -1,3 +1,3 @@\n-old\n+new\n"
    b = "@@ -90,3 +90,3 @@\n-old\n+new\n"
    assert extract_a2.divergence(a, b) == 0.0
    c = "@@ -1,3 +1,3 @@\n-old\n+different\n"
    assert extract_a2.divergence(a, c) > 0.0


def test_a2_rejects_pairs_that_answer_different_tasks():
    """The collapse cause, filtered at source (sec 6.3)."""
    a = "@@ -1 +1 @@\n-a\n+b\n"
    b = "@@ -1 +1 @@\n-zzz\n+yyy\n"
    assert extract_a2.divergence(a, b) == 1.0


def test_a2_records_its_divergence_window(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "d@e.com")
    git(root, "config", "user.name", "Dev")
    (root / "a.py").write_text("x = 1\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "Initial commit for the window test")
    _t, _h, report = extract_a2.extract(root)
    assert report.as_dict()["divergence_window"] == [0.02, 0.85]


# --- A0 (sec 6.1) -----------------------------------------------------------


def test_a0_generates_deterministic_traces():
    a = extract_a0.generate(n=50, seed=7)
    b = extract_a0.generate(n=50, seed=7)
    assert [t.messages for t in a] == [t.messages for t in b]
    assert [t.messages for t in extract_a0.generate(n=50, seed=8)] != [
        t.messages for t in a
    ]


def test_a0_emits_the_one_canonical_call_shape():
    """A0, the repair layer and the constrainer must agree on one target shape."""
    from tandem.gateway.toolcall.repair import repair

    traces = extract_a0.generate(n=40, seed=1)
    for trace in traces:
        for msg in trace.messages:
            if msg["role"] != "assistant":
                continue
            out = repair(msg["content"], extract_a0.DEFAULT_TOOLS)
            assert out.ok, (
                f"A0 emitted a shape the repair layer rejects: {msg['content'][:80]}"
            )


def test_a0_includes_multi_step_traces():
    """The failure A0 targets is a loop, which needs a second turn to appear."""
    traces = extract_a0.generate(n=300, seed=3)
    rep = extract_a0.report(traces)
    assert 0.2 < rep["multi_step_fraction"] < 0.5
    assert rep["source_kind"] == "synthetic_harness"


# --- routing profile (sec 6.4) ----------------------------------------------


def test_profile_selects_the_hot_quarter():
    counts = [[100, 90, 80, 70, 5, 4, 3, 2] for _ in range(4)]
    p = profile.build(counts, model_name="Qwen3.6-35B-A3B")
    assert p.rank_by == "count"
    assert p.topk["25"] == [[0, 1] for _ in range(4)]
    assert p.cov_at_25 > 0.5


def test_profile_ranks_deepseek_by_mass_not_count():
    """Sec 6.4: a long low-weight tail inflates counts (J=0.646 vs 0.920)."""
    counts = [[10, 10, 10, 100]]
    mass = [[500.0, 1.0, 1.0, 10.0]]
    qwen = profile.build(counts, mass, model_name="Qwen3.6-35B")
    deepseek = profile.build(counts, mass, model_name="DeepSeek-V3")
    assert qwen.rank_by == "count" and qwen.topk["25"] == [[3]]
    assert deepseek.rank_by == "mass" and deepseek.topk["25"] == [[0]]


def test_profile_round_trips(tmp_path):
    p = profile.build([[5, 4, 3, 2]], model_name="Qwen")
    path = p.write(tmp_path / "profile.json")
    loaded = profile.RoutingProfile.load(path)
    assert loaded.topk == p.topk
    assert loaded.cov_at_25 == p.cov_at_25


def test_profile_compare_reports_jaccard():
    a = profile.build([[10, 9, 1, 1]], model_name="Qwen")
    b = profile.build([[10, 9, 2, 1]], model_name="Qwen")
    assert profile.compare(a, b)["mean_jaccard"] == 1.0
    c = profile.build([[1, 1, 10, 9]], model_name="Qwen")
    assert profile.compare(a, c)["mean_jaccard"] == 0.0


def test_profile_sanity_flags_a_uniform_profile():
    """A flat profile has no hot quarter; selecting one would be arbitrary."""
    flat = profile.build([[10] * 8 for _ in range(4)], model_name="Qwen")
    assert not profile.sanity(flat)["ok"]


def test_profile_rejects_ragged_layers():
    with pytest.raises(ValueError):
        profile.build([[1, 2, 3], [1, 2]])


# --- training driver (sec 6.2, 6.3) -----------------------------------------


def test_sft_command_carries_the_spec_defaults(tmp_path):
    cmd = build_sft_command(
        model="m",
        corpus=tmp_path / "d" / "train.jsonl",
        output=tmp_path / "out",
        cfg=SFTConfig(),
        trainer="mlx_lm",
    )
    assert "--mask-prompt" in cmd
    assert "--grad-checkpoint" in cmd
    assert cmd[cmd.index("--learning-rate") + 1] == "0.0002"
    assert cmd[cmd.index("--max-seq-length") + 1] == "4096"


def test_dpo_command_mounts_a1_and_uses_one_epoch(tmp_path):
    """Sec 6.3: start from the SFT weights; more than one epoch invites collapse."""
    cmd = build_dpo_command(
        model="m",
        corpus=tmp_path / "d" / "train.jsonl",
        output=tmp_path / "out",
        cfg=DPOConfig(),
        trainer="mlx_lm",
        mount_adapter=tmp_path / "a1",
    )
    assert "--method" in cmd and cmd[cmd.index("--method") + 1] == "dpo"
    assert "--mount-adapter" in cmd
    assert cmd[cmd.index("--iters") + 1] == "1"
    assert cmd[cmd.index("--dpo-beta") + 1] == "0.1"
    # DPO materialises [seq, vocab] logits four times per step; the plain path OOMs.
    assert "--fused-dpo" in cmd


def test_preflight_warns_when_tier1_is_resident():
    assert preflight(tier1_loaded=True, model_params_b=35.0)
    assert not preflight(tier1_loaded=False, model_params_b=35.0)


def test_collapse_signature_is_detected():
    log = "iter 1: train loss 2.5\niter 50: loss 0.20\niter 100: loss 0.001 rewards/chosen -12.4"
    assert "Preference collapse" in detect_collapse(log)
    healthy = "iter 1: train loss 2.5\niter 50: loss 1.10\niter 100: loss 0.85 rewards/chosen 0.4"
    assert detect_collapse(healthy) == ""
