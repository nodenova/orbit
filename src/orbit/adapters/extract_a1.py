"""A1 — repository adapter corpus from git history (spec sec 6.2).

Git history is a free, labelled, in-domain SFT corpus that every customer already
has. Each merged commit is one training pair:

    prompt     = commit message (or linked issue/PR title + body)
               + the files touched, at the *parent* commit, truncated to a per-file
                 budget around the changed hunks
    completion = the unified diff of the commit

Two details that are easy to get wrong and expensive to get wrong:

**Context comes from the parent commit, not the commit.** Showing the model the
post-change file and asking for the diff that produces it is a copying task, not a
coding task. The adapter trained that way scores well on held-out diffs it has
effectively already seen and does nothing useful at inference, when the future does
not exist yet.

**The format is `messages`, never bare `text`.** A bare-text format cannot expose a
prompt/response boundary, so prompt masking falls through to full-sequence loss and
degrades quality on tasks the base model already handles [V, mlx-optiq]. The chat
template is applied at train time; nothing here is pre-templated.

Yield on a mature repo is typically 1-5k usable pairs. Below ~500 the adapter will
underfit, and `ExtractionReport.thin` says so rather than shipping a null adapter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbit.adapters.filters import (
    ExtractionFilters,
    SkipTally,
    is_bot,
    is_revert,
    keep_path,
)
from orbit.adapters.gitwalk import (
    Commit,
    CommitDiff,
    commit_diff,
    default_branch,
    file_at,
    is_repo,
    iter_commits,
)

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
# Trailers a task description should not carry into the prompt: they leak the answer
# (the reviewer's name, the merge commit) or are pure noise.
_TRAILER = re.compile(
    r"^(?:Signed-off-by|Co-authored-by|Reviewed-by|Acked-by|Tested-by|Cc|Change-Id|"
    r"Fixes|Closes|Refs|PR-URL|Reviewed-on|Auto-Submit|Commit-Queue):.*$",
    re.IGNORECASE | re.MULTILINE,
)
_ISSUE_REF = re.compile(r"\(#(\d+)\)\s*$")


@dataclass
class Pair:
    """One SFT training pair, plus provenance for the eval split."""

    prompt: str
    completion: str
    sha: str
    ts: int
    n_files: int
    n_lines: int

    def as_messages(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.completion},
            ]
        }


@dataclass
class ExtractionReport:
    repo: str = ""
    branch: str = ""
    commits_walked: int = 0
    pairs: int = 0
    skips: SkipTally = field(default_factory=SkipTally)
    # Commits that were kept but modified (e.g. some paths filtered out). Counted
    # separately: folding them into "skipped" makes the report claim work was
    # dropped that in fact contributed a pair.
    adjustments: SkipTally = field(default_factory=SkipTally)
    first_sha: str = ""
    last_sha: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    min_usable_pairs: int = 500
    merge_policy_used: str = ""

    @property
    def thin(self) -> bool:
        return self.pairs < self.min_usable_pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "branch": self.branch,
            "commits_walked": self.commits_walked,
            "pairs": self.pairs,
            "skipped": self.skips.as_dict(),
            "skipped_total": self.skips.total(),
            "adjustments": self.adjustments.as_dict(),
            "merge_policy_used": self.merge_policy_used,
            "commit_range": [self.first_sha, self.last_sha],
            "filters": self.filters,
            "thin": self.thin,
            "advice": (
                f"Only {self.pairs} usable pairs (< {self.min_usable_pairs}). A1 will "
                "underfit on this history. Widen the commit range, include more "
                "branches, or pool sibling repositories before training — shipping "
                "an adapter trained on this would be shipping a null adapter."
                if self.thin
                else "corpus size is sufficient for A1"
            ),
        }


def resolve_merge_policy(commits: list[Commit], filters: ExtractionFilters) -> str:
    """Decide how to treat merge commits on this branch (see `ExtractionFilters`).

    "auto" inspects the branch: if most first-parent commits are two-parent merges,
    this is a merge-commit repository and its merged work lives *in* those merges,
    so skipping them would drop the corpus. Otherwise merges carry nothing an
    ordinary commit does not and are skipped as the spec says.
    """
    policy = filters.merge_policy
    if policy in ("skip", "first_parent"):
        return policy
    if not commits:
        return "skip"
    two_parent = sum(1 for c in commits if len(c.parents) == 2)
    ratio = two_parent / len(commits)
    return "first_parent" if ratio >= filters.merge_heavy_threshold else "skip"


def clean_message(commit: Commit) -> str:
    """Commit message as a task description.

    Trailers are stripped: `Reviewed-by:` is noise, and `Fixes: #123` is a hint the
    model cannot have at inference time.
    """
    subject = _ISSUE_REF.sub("", commit.subject).strip()
    body = _TRAILER.sub("", commit.body or "").strip()
    # Drop the auto-generated "* commit ..." lists some merge tools append.
    body = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("* commit ")
    ).strip()
    return f"{subject}\n\n{body}".strip() if body else subject


def changed_line_ranges(unified: str, path: str) -> list[tuple[int, int]]:
    """Pre-image line ranges touched in `path`, from the unified diff."""
    ranges: list[tuple[int, int]] = []
    in_file = False
    for line in unified.splitlines():
        if line.startswith("diff --git "):
            in_file = line.endswith(f" b/{path}")
            continue
        if not in_file:
            continue
        m = _HUNK_HEADER.match(line)
        if m:
            start = int(m.group(1))
            length = int(m.group(2) or 1)
            ranges.append((start, start + max(0, length - 1)))
    return ranges


def excerpt_around(content: str, ranges: list[tuple[int, int]], budget: int) -> str:
    """A per-file excerpt centred on the changed hunks (sec 6.2).

    Whole files blow the context budget on a 4096-token training sequence, and the
    lines that matter are the ones near the change. With no ranges (a new file), the
    head of the file is the honest fallback.
    """
    lines = content.splitlines()
    if not lines:
        return ""
    if not ranges:
        return _truncate("\n".join(lines), budget)

    keep: set[int] = set()
    pad = 25
    for start, end in ranges:
        for i in range(max(1, start - pad), min(len(lines), end + pad) + 1):
            keep.add(i)

    out: list[str] = []
    last = 0
    for i in sorted(keep):
        if last and i > last + 1:
            out.append(f"... ({i - last - 1} lines omitted)")
        out.append(f"{i:>5}  {lines[i - 1]}")
        last = i
    return _truncate("\n".join(out), budget)


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    head = text[: int(budget * 0.7)]
    tail = text[-int(budget * 0.25) :]
    return f"{head}\n... (truncated) ...\n{tail}"


def build_prompt(
    repo: Path, diff: CommitDiff, filters: ExtractionFilters
) -> str | None:
    task = clean_message(diff.commit)
    if len(task) < filters.min_message_chars:
        return None

    parent = diff.commit.parent
    if parent is None:
        return None

    parts = [
        "<task>\n" + task + "\n</task>\n",
        "\n<files>\n",
    ]
    any_context = False
    for fc in diff.files:
        source_path = fc.old_path or fc.path
        content = file_at(repo, parent, source_path)
        if content is None:
            # New file: no pre-image. Name it so the model knows it is creating one.
            parts.append(f"\n--- {fc.path} (new file) ---\n")
            any_context = True
            continue
        excerpt = excerpt_around(
            content,
            changed_line_ranges(diff.unified, fc.path),
            filters.per_file_context_chars,
        )
        if not excerpt:
            continue
        parts.append(f"\n--- {source_path} ---\n{excerpt}\n")
        any_context = True

    if not any_context:
        return None
    parts.append("\n</files>\n")
    return "".join(parts)


def extract(
    repo_path: str | Path,
    *,
    filters: ExtractionFilters | None = None,
    branch: str = "",
    limit: int | None = None,
    since: str = "",
    holdout: int = 0,
) -> tuple[list[Pair], list[Pair], ExtractionReport]:
    """Walk history and build (train, holdout, report).

    `holdout` reserves the most recent K commits for the merge eval (sec 10.1) and
    removes them from training. The split is by recency rather than at random
    because that is what the eval claims to measure: performance on work the adapter
    has not seen, which in a repository means *later* work, not a random sample of
    the same period.
    """
    repo = Path(repo_path).resolve()
    if not is_repo(repo):
        raise ValueError(f"{repo} is not a git repository")
    filters = filters or ExtractionFilters()
    branch = branch or default_branch(repo)

    commits = list(iter_commits(repo, branch, limit=limit, since=since))
    policy = resolve_merge_policy(commits, filters)

    report = ExtractionReport(
        repo=str(repo),
        branch=branch,
        filters=filters.as_dict(),
        min_usable_pairs=filters.min_usable_pairs,
        merge_policy_used=policy,
    )
    pairs: list[Pair] = []

    for commit in commits:
        report.commits_walked += 1
        if not report.last_sha:
            report.last_sha = commit.sha
        report.first_sha = commit.sha

        if len(commit.parents) > 2:
            # Octopus: no single branch whose contribution the diff represents.
            report.skips.bump("octopus_merge")
            continue
        if commit.is_merge and policy == "skip":
            report.skips.bump("merge_commit")
            continue
        if filters.skip_reverts and is_revert(commit.message):
            report.skips.bump("revert")
            continue
        if filters.skip_bots and is_bot(commit.author_name, commit.author_email):
            report.skips.bump("bot_author")
            continue

        diff = commit_diff(repo, commit)
        if not diff.files:
            report.skips.bump("empty_diff")
            continue

        kept = [f for f in diff.files if keep_path(f.path, filters)]
        if not kept:
            report.skips.bump("all_paths_filtered")
            continue
        if len(kept) != len(diff.files):
            report.adjustments.bump("paths_filtered_from_kept_commit")
        diff.files = kept

        if len(diff.files) > filters.max_files:
            report.skips.bump("too_many_files")
            continue
        if diff.total_lines > filters.max_lines:
            report.skips.bump("too_many_lines")
            continue

        prompt = build_prompt(repo, diff, filters)
        if prompt is None:
            report.skips.bump("no_usable_prompt")
            continue
        completion = _filtered_unified(diff)
        if not completion.strip():
            report.skips.bump("empty_completion")
            continue

        pairs.append(
            Pair(
                prompt=prompt,
                completion=completion,
                sha=commit.sha,
                ts=commit.ts,
                n_files=len(diff.files),
                n_lines=diff.total_lines,
            )
        )

    # iter_commits walks newest-first; the most recent K are the holdout.
    held = pairs[:holdout] if holdout else []
    train = pairs[holdout:] if holdout else pairs
    report.pairs = len(train)
    return train, held, report


def _filtered_unified(diff: CommitDiff) -> str:
    """The unified diff, restricted to the file set that survived filtering.

    A commit that touched both source and a lockfile must not train the model to
    emit lockfile churn alongside its patch.
    """
    keep = {f.path for f in diff.files} | {f.old_path for f in diff.files if f.old_path}
    out: list[str] = []
    emit = False
    for line in diff.unified.splitlines(keepends=True):
        if line.startswith("diff --git "):
            parts = line.split(" b/", 1)
            emit = len(parts) == 2 and parts[1].strip() in keep
        if emit:
            out.append(line)
    return "".join(out)


def write_jsonl(pairs: list[Pair], path: str | Path) -> Path:
    """Write the corpus as `messages` JSONL (sec 6.2). Never bare text."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.writelines(
            json.dumps(pair.as_messages(), ensure_ascii=False) + "\n" for pair in pairs
        )
    return p


def write_manifest(pairs: list[Pair], path: str | Path) -> Path:
    """Sidecar mapping each corpus line to its commit, for the held-out eval."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.writelines(
            json.dumps(
                {
                    "line": i,
                    "sha": pair.sha,
                    "ts": pair.ts,
                    "n_files": pair.n_files,
                    "n_lines": pair.n_lines,
                }
            )
            + "\n"
            for i, pair in enumerate(pairs)
        )
    return p


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
