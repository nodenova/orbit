"""A2 — reviewer adapter corpus from review history (spec sec 6.3).

**This is the piece that directly attacks the METR gap**, and it is the strongest
data asset a repository has. Code review produces free preference pairs:

    rejected = the diff as of the first review comment
    chosen   = the diff as merged
    prompt   = the same task description and parent-commit context

The hard requirement for DPO is that both completions answer the *same* prompt. If
`chosen` answers a different context than the prompt introduces, both reward terms
drift in lockstep and the margin signal saturates near zero [V, mlx-optiq].
Pre-review and post-review versions of one diff satisfy this exactly — same task,
same parent, two attempts.

Two sources, no network required for either:

**Branch shape.** For a two-parent merge M, the branch commits are the PR's history.
`diff(M^1, first_branch_commit)` is what the author proposed; `diff(M^1, M)` is what
survived review. When they differ, review changed something, and that difference is
the preference signal. This works on any repository with merge commits and needs no
forge API at all.

**Reverts.** A commit later reverted is a labelled failure with a known-good
counterfactual (sec 6.3). `rejected` is the reverted diff; `chosen` is whatever
replaced it.

An optional forge export (`--reviews`) refines source one: with real review
timestamps, `rejected` is the branch state at the *first review comment* rather than
at the first commit, which is what the spec asks for exactly. Absent that, the first
commit is the closest local approximation and is labelled as such in the record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .extract_a1 import build_prompt, clean_message
from .filters import ExtractionFilters, SkipTally, is_bot, is_revert, keep_path
from .gitwalk import Commit, CommitDiff, commit_diff, default_branch, is_repo, iter_commits, run_git


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    sha: str
    ts: int
    # branch_review | forge_review | revert — recorded so a corpus can be sliced by
    # signal strength during ablation.
    source: str = "branch_review"
    similarity: float = 0.0

    def as_record(self) -> dict[str, Any]:
        # mlx-lm's DPO reader takes prompt/chosen/rejected. Kept flat and explicit.
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}


@dataclass
class A2Report:
    repo: str = ""
    branch: str = ""
    merges_walked: int = 0
    pairs: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    skips: SkipTally = field(default_factory=SkipTally)
    first_sha: str = ""
    last_sha: str = ""
    min_usable_pairs: int = 200
    # Recorded because they select the corpus, so they belong in provenance (sec 9.4).
    min_divergence: float = 0.0
    max_divergence: float = 0.0

    @property
    def thin(self) -> bool:
        return self.pairs < self.min_usable_pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "branch": self.branch,
            "merges_walked": self.merges_walked,
            "pairs": self.pairs,
            "by_source": dict(sorted(self.by_source.items())),
            "skipped": self.skips.as_dict(),
            "commit_range": [self.first_sha, self.last_sha],
            "divergence_window": [self.min_divergence, self.max_divergence],
            "thin": self.thin,
            "advice": (
                f"Only {self.pairs} preference pairs (< {self.min_usable_pairs}). DPO on "
                "this little data will move the policy without a reliable margin "
                "signal. Prefer shipping A1 alone until the repository has more "
                "reviewed history."
                if self.thin
                else "corpus size is sufficient for A2"
            ),
        }


# Below this, the two diffs are near-identical and review changed nothing worth
# learning: the pair would be noise with a near-zero margin.
_MIN_DIVERGENCE = 0.02
# Above this, `chosen` and `rejected` are effectively different tasks — a rebase, a
# scope change, or a branch that was rewritten. Training on those is the failure the
# spec warns about, where both reward terms drift together.
_MAX_DIVERGENCE = 0.85


def _changed_lines(unified: str) -> set[str]:
    """The added/removed content lines of a unified diff, ignoring hunk framing.

    File headers and @@ markers are dropped: they shift whenever anything above them
    changes, which would report divergence for a pure line-number move.
    """
    out: set[str] = set()
    for line in unified.splitlines():
        if line.startswith(("+++", "---", "diff --git ", "index ", "@@")):
            continue
        if line and line[0] in "+-":
            body = line[1:].strip()
            if body:
                out.add(line[0] + body)
    return out


def divergence(a: str, b: str) -> float:
    """1 - Jaccard similarity over the two diffs' changed lines.

    Not `difflib.SequenceMatcher`: that is quadratic in the character length, and
    real PR diffs run to hundreds of kilobytes — extracting from a 150-commit branch
    took minutes and never finished on a repository of ordinary size. A line-set
    measure is linear, and it is also the better metric here, because two diffs that
    make the same edits at different offsets are the *same* proposal, which is
    exactly what a character-level ratio would score as divergent.
    """
    la, lb = _changed_lines(a), _changed_lines(b)
    if not la and not lb:
        return 0.0
    if not la or not lb:
        return 1.0
    return 1.0 - len(la & lb) / len(la | lb)


def branch_commits(repo: Path, merge: Commit) -> list[str]:
    """Commits contributed by the merged branch, oldest first."""
    if len(merge.parents) != 2:
        return []
    first, second = merge.parents
    out = run_git(
        repo, "rev-list", "--reverse", "--no-merges", f"{first}..{second}", check=False
    ).split()
    return [sha for sha in out if sha]


def _diff_between(repo: Path, base: str, head: str) -> str:
    return run_git(
        repo, "diff", "-U3", "-M", "--no-color", "--no-ext-diff", base, head, check=False
    )


def _load_reviews(path: str | Path | None) -> dict[str, str]:
    """Optional forge export: {merge_sha: first_review_iso_timestamp}.

    Deliberately a file, not an API call. The offline posture (sec 8.6) is a
    verifiable claim, and extraction reaching out to github.com would break it.
    """
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    out: dict[str, str] = {}
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("merge_sha") and item.get("first_review_at"):
            out[str(item["merge_sha"])] = str(item["first_review_at"])
    return out


def _commit_ts(repo: Path, sha: str) -> int:
    out = run_git(repo, "show", "-s", "--format=%at", sha, check=False).strip()
    return int(out) if out.isdigit() else 0


def _pre_review_head(repo: Path, shas: list[str], cutoff_iso: str | None) -> str | None:
    """The branch head at the moment of the first review comment.

    With a cutoff, that is the last branch commit authored before it — exactly the
    state a reviewer looked at. Without one, the first commit is the closest local
    approximation of "what the author proposed".
    """
    if not shas:
        return None
    if not cutoff_iso:
        return shas[0]
    try:
        from datetime import datetime

        cutoff = int(datetime.fromisoformat(cutoff_iso.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return shas[0]
    before = [sha for sha in shas if _commit_ts(repo, sha) <= cutoff]
    return before[-1] if before else shas[0]


def extract(
    repo_path: str | Path,
    *,
    filters: ExtractionFilters | None = None,
    branch: str = "",
    limit: int | None = None,
    reviews_path: str | Path | None = None,
    holdout: int = 0,
    min_divergence: float = _MIN_DIVERGENCE,
    max_divergence: float = _MAX_DIVERGENCE,
) -> tuple[list[PreferencePair], list[PreferencePair], A2Report]:
    repo = Path(repo_path).resolve()
    if not is_repo(repo):
        raise ValueError(f"{repo} is not a git repository")
    filters = filters or ExtractionFilters()
    branch = branch or default_branch(repo)
    reviews = _load_reviews(reviews_path)

    report = A2Report(
        repo=str(repo),
        branch=branch,
        min_divergence=min_divergence,
        max_divergence=max_divergence,
    )
    pairs: list[PreferencePair] = []
    commits = list(iter_commits(repo, branch, limit=limit))

    for commit in commits:
        if not report.last_sha:
            report.last_sha = commit.sha
        report.first_sha = commit.sha

        if filters.skip_bots and is_bot(commit.author_name, commit.author_email):
            report.skips.bump("bot_author")
            continue

        if is_revert(commit.message):
            pair = _revert_pair(repo, commit, filters)
            if pair is not None:
                pairs.append(pair)
                report.by_source[pair.source] = report.by_source.get(pair.source, 0) + 1
            else:
                report.skips.bump("revert_without_target")
            continue

        if len(commit.parents) != 2:
            continue
        report.merges_walked += 1

        shas = branch_commits(repo, commit)
        if len(shas) < 2:
            # A single-commit branch went through review unchanged, or was squashed
            # before merge. Either way there is no before/after to compare.
            report.skips.bump("no_intermediate_revisions")
            continue

        pre_head = _pre_review_head(repo, shas, reviews.get(commit.sha))
        if pre_head is None or pre_head == commit.sha:
            report.skips.bump("no_pre_review_state")
            continue

        base = commit.parents[0]
        rejected = _diff_between(repo, base, pre_head)
        chosen = _diff_between(repo, base, commit.sha)
        if not rejected.strip() or not chosen.strip():
            report.skips.bump("empty_diff")
            continue

        div = divergence(chosen, rejected)
        if div < min_divergence:
            report.skips.bump("review_changed_nothing")
            continue
        if div > max_divergence:
            # Both reward terms would drift in lockstep and the margin saturates
            # near zero (sec 6.3). This is the collapse cause, filtered at source.
            report.skips.bump("diverged_too_far_not_same_task")
            continue

        diff = commit_diff(repo, commit)
        diff.files = [f for f in diff.files if keep_path(f.path, filters)]
        if not diff.files or len(diff.files) > filters.max_files:
            report.skips.bump("path_or_size_filtered")
            continue

        prompt = build_prompt(repo, diff, filters)
        if prompt is None:
            report.skips.bump("no_usable_prompt")
            continue

        source = "forge_review" if commit.sha in reviews else "branch_review"
        pairs.append(
            PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                sha=commit.sha,
                ts=commit.ts,
                source=source,
                similarity=round(1 - div, 3),
            )
        )
        report.by_source[source] = report.by_source.get(source, 0) + 1

    held = pairs[:holdout] if holdout else []
    train = pairs[holdout:] if holdout else pairs
    report.pairs = len(train)
    return train, held, report


def _revert_pair(repo: Path, revert: Commit, filters: ExtractionFilters) -> PreferencePair | None:
    """A reverted commit is a labelled failure with a known-good counterfactual.

    `rejected` is the change that was reverted; `chosen` is the revert itself — the
    repository's own verdict that the original should not have landed.
    """
    target = _reverted_sha(repo, revert)
    if not target:
        return None
    parent = run_git(repo, "rev-parse", f"{target}^", check=False).strip()
    if not parent:
        return None

    rejected = _diff_between(repo, parent, target)
    revert_diff = commit_diff(repo, revert)
    revert_diff.files = [f for f in revert_diff.files if keep_path(f.path, filters)]
    if not rejected.strip() or not revert_diff.files:
        return None
    if len(revert_diff.files) > filters.max_files:
        return None

    prompt = build_prompt(repo, revert_diff, filters)
    if prompt is None:
        return None
    return PreferencePair(
        prompt=prompt,
        chosen=revert_diff.unified,
        rejected=rejected,
        sha=revert.sha,
        ts=revert.ts,
        source="revert",
        similarity=round(1 - divergence(revert_diff.unified, rejected), 3),
    )


def _reverted_sha(repo: Path, revert: Commit) -> str | None:
    """Pull the reverted sha out of git's own revert message body."""
    import re

    m = re.search(r"[Tt]his reverts commit ([0-9a-f]{7,40})", revert.message)
    if not m:
        return None
    sha = run_git(repo, "rev-parse", "--verify", "--quiet", m.group(1), check=False).strip()
    return sha or None


def write_jsonl(pairs: list[PreferencePair], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.as_record(), ensure_ascii=False) + "\n")
    return p
