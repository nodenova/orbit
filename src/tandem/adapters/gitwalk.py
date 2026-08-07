"""Git plumbing for corpus extraction.

Shells out to `git` rather than taking a libgit2 dependency: every customer already
has git, the commands used here are stable across a decade of versions, and it keeps
the dependency list — which is a security surface, not a convenience (sec 8.6) —
short.

All commands are run with `-z` and explicit `--no-color`/`--no-textconv` so parsing
does not depend on the user's `~/.gitconfig`. A customer with `diff.external` set
would otherwise silently produce a corpus of nonsense.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Neutralise user config that would change diff output under us.
_GIT_BASE = [
    "git",
    "-c", "core.quotepath=false",
    "-c", "diff.noprefix=false",
    "-c", "diff.external=",
    "-c", "diff.renames=true",
    "--no-pager",
]

_SEP = "\x1e"  # record separator
_FIELD = "\x1f"  # field separator


class GitError(RuntimeError):
    pass


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        [*_GIT_BASE, "-C", str(repo), *args],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args[:3])} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


@dataclass
class Commit:
    sha: str
    parents: tuple[str, ...]
    author_name: str
    author_email: str
    ts: int
    subject: str
    body: str

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1

    @property
    def message(self) -> str:
        return f"{self.subject}\n\n{self.body}".strip()

    @property
    def parent(self) -> str | None:
        return self.parents[0] if self.parents else None


@dataclass
class FileChange:
    path: str
    old_path: str | None
    status: str  # A M D R C
    added: int = 0
    deleted: int = 0


@dataclass
class CommitDiff:
    commit: Commit
    files: list[FileChange] = field(default_factory=list)
    unified: str = ""

    @property
    def total_lines(self) -> int:
        return sum(f.added + f.deleted for f in self.files)


def default_branch(repo: Path) -> str:
    """The repo's default branch.

    Tries the remote HEAD, then the current branch. Extraction walks the default
    branch because that is where *merged* work lives (sec 6.2) — a feature branch's
    history includes work that was rejected.
    """
    out = run_git(repo, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD", check=False).strip()
    if out:
        return out.rsplit("/", 1)[-1]
    for candidate in ("main", "master"):
        if run_git(repo, "rev-parse", "--verify", "--quiet", candidate, check=False).strip():
            return candidate
    return run_git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() or "HEAD"


def iter_commits(
    repo: Path, branch: str = "", *, limit: int | None = None, since: str = ""
) -> Iterator[Commit]:
    """Walk first-parent history of `branch`, newest first."""
    fmt = _FIELD.join(["%H", "%P", "%an", "%ae", "%at", "%s", "%b"]) + _SEP
    args = ["log", "--first-parent", f"--format={fmt}"]
    if limit:
        args.append(f"-n{limit}")
    if since:
        args.append(f"--since={since}")
    args.append(branch or default_branch(repo))

    raw = run_git(repo, *args)
    for record in raw.split(_SEP):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split(_FIELD)
        if len(parts) < 7:
            continue
        sha, parents, an, ae, ts, subject, body = parts[:7]
        yield Commit(
            sha=sha.strip(),
            parents=tuple(p for p in parents.split() if p),
            author_name=an,
            author_email=ae,
            ts=int(ts) if ts.isdigit() else 0,
            subject=subject,
            body=body,
        )


def commit_diff(repo: Path, commit: Commit, *, context_lines: int = 3) -> CommitDiff:
    """Numstat plus unified diff for one commit against its first parent."""
    if commit.parent is None:
        return CommitDiff(commit=commit)

    numstat = run_git(
        repo, "diff", "--numstat", "-z", "-M", commit.parent, commit.sha, check=False
    )
    files = _parse_numstat_z(numstat)

    unified = run_git(
        repo,
        "diff",
        f"-U{context_lines}",
        "-M",
        "--no-color",
        "--no-ext-diff",
        commit.parent,
        commit.sha,
        check=False,
    )
    return CommitDiff(commit=commit, files=files, unified=unified)


def _parse_numstat_z(raw: str) -> list[FileChange]:
    """Parse `git diff --numstat -z`.

    The -z format is awkward: a rename emits three NUL-separated fields
    (`added\\tdeleted\\t`, old, new) while an ordinary change emits one
    (`added\\tdeleted\\tpath`). Getting this wrong silently misattributes renames,
    so it is parsed as a small state machine rather than a split.
    """
    fields = [f for f in raw.split("\0") if f != ""]
    out: list[FileChange] = []
    i = 0
    while i < len(fields):
        head = fields[i]
        parts = head.split("\t")
        if len(parts) < 3:
            i += 1
            continue
        added_s, deleted_s, path = parts[0], parts[1], parts[2]
        added = int(added_s) if added_s.isdigit() else 0
        deleted = int(deleted_s) if deleted_s.isdigit() else 0
        if path == "" and i + 2 < len(fields):
            old_path, new_path = fields[i + 1], fields[i + 2]
            out.append(
                FileChange(path=new_path, old_path=old_path, status="R", added=added, deleted=deleted)
            )
            i += 3
            continue
        out.append(FileChange(path=path, old_path=None, status="M", added=added, deleted=deleted))
        i += 1
    return out


def file_at(repo: Path, rev: str, path: str) -> str | None:
    """File contents at a revision, or None if absent or binary."""
    proc = subprocess.run(
        [*_GIT_BASE, "-C", str(repo), "show", f"{rev}:{path}"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    data = proc.stdout
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def is_repo(path: Path) -> bool:
    return (
        subprocess.run(
            [*_GIT_BASE, "-C", str(path), "rev-parse", "--git-dir"],
            capture_output=True,
        ).returncode
        == 0
    )
