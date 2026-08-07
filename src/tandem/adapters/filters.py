"""Extraction filters (spec sec 6.2).

    skip if: merge commit, revert, bot author, > 15 files, > 1000 lines,
             vendored/generated paths, lockfiles

Every filter is recorded verbatim in the provenance record (sec 9.4), so this module
is also the schema for "what did we train on". Changing a threshold changes the
corpus hash and therefore the adapter's identity, which is the correct coupling.

The filters are not fussiness. A merge commit's diff is the union of other people's
work with no single intent behind it; a revert teaches the model to undo; a bot
commit teaches it to write like a bot; a lockfile is 40,000 lines of noise that will
dominate a 2,000-pair corpus. Each one, left in, actively degrades the adapter.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Path patterns that are vendored, generated, or otherwise not written by a human
# on this team. Matched against POSIX-style relative paths.
VENDORED_PATTERNS: tuple[str, ...] = (
    r"(^|/)vendor/",
    r"(^|/)node_modules/",
    r"(^|/)third_party/",
    r"(^|/)3rdparty/",
    r"(^|/)dist/",
    r"(^|/)build/",
    r"(^|/)\.venv/",
    r"(^|/)site-packages/",
    r"(^|/)target/(debug|release)/",
    r"(^|/)__pycache__/",
    r"(^|/)migrations?/\d+_",
    r"\.min\.(js|css)$",
    r"\.(pb|pb2)\.(go|py|js|ts)$",
    r"_pb2\.pyi?$",
    r"\.generated\.",
    r"(^|/)gen/",
    r"\.(snap|golden)$",
    r"\.(png|jpg|jpeg|gif|ico|pdf|zip|tar|gz|woff2?|ttf|eot|mp4|wasm|so|dylib|dll|bin)$",
)

LOCKFILE_NAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "uv.lock",
        "Cargo.lock",
        "go.sum",
        "composer.lock",
        "Gemfile.lock",
        "mix.lock",
        "flake.lock",
        "packages.lock.json",
        "gradle.lockfile",
    }
)

BOT_PATTERNS: tuple[str, ...] = (
    r"\[bot\]",
    r"^dependabot",
    r"^renovate",
    r"^greenkeeper",
    r"^snyk-bot",
    r"^github-actions",
    r"^semantic-release",
    r"^allcontributors",
    r"^pre-commit-ci",
    r"^copybara",
    r"^imgbot",
    r"noreply@github\.com$",
)

REVERT_PATTERN = re.compile(r"^\s*(revert|reverts|reverting)\b", re.IGNORECASE)

_VENDORED_RE = re.compile("|".join(VENDORED_PATTERNS))
_BOT_RE = re.compile("|".join(BOT_PATTERNS), re.IGNORECASE)


@dataclass
class ExtractionFilters:
    max_files: int = 15
    max_lines: int = 1000
    # Merge-commit policy. The spec says "skip merge commits" (sec 6.2), and the
    # reason is sound: a merge's diff is the union of other people's work with no
    # single intent behind it. But that is only true against an arbitrary parent.
    # Walked first-parent, `git diff M^1 M` on a two-parent merge is exactly the
    # branch's contribution and the subject is the PR title — the cleanest pair in
    # the repository.
    #
    # It matters because it decides whether there is a corpus at all. A squash-merge
    # repo puts merged work in ordinary commits and skipping merges costs nothing;
    # a merge-commit repo puts *all* of it in merge commits, and skipping them drops
    # ~90% of usable history. Measured on pallets/click: 129 of 150 commits.
    #
    #   "skip"          - the spec as literally written
    #   "first_parent"  - use the first-parent diff of two-parent merges
    #   "auto"          - first_parent when the branch is merge-heavy, else skip
    #
    # Octopus merges (>2 parents) are always skipped: there is no single branch
    # whose contribution the diff represents.
    merge_policy: str = "auto"
    # Fraction of merge commits on the branch above which "auto" switches to
    # first_parent.
    merge_heavy_threshold: float = 0.5
    skip_reverts: bool = True
    skip_bots: bool = True
    skip_vendored: bool = True
    skip_lockfiles: bool = True
    # A commit whose message is "fix" or "wip" carries no task description, so the
    # prompt would be empty and the pair teaches nothing.
    min_message_chars: int = 12
    # Per-file context budget around the changed hunks (sec 6.2).
    per_file_context_chars: int = 6_000
    # Corpus size below which A1 will underfit and the customer must be told
    # honestly rather than shipped a null adapter (sec 13).
    min_usable_pairs: int = 500

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkipTally:
    """Why commits were dropped. Reported at extraction; never silent."""

    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1

    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items(), key=lambda kv: -kv[1]))


def is_vendored(path: str) -> bool:
    return bool(_VENDORED_RE.search(path))


def is_lockfile(path: str) -> bool:
    return path.rsplit("/", 1)[-1] in LOCKFILE_NAMES


def is_bot(author_name: str, author_email: str) -> bool:
    return bool(_BOT_RE.search(author_name or "") or _BOT_RE.search(author_email or ""))


def is_revert(message: str) -> bool:
    first_line = (message or "").strip().splitlines()[0] if message else ""
    return bool(REVERT_PATTERN.match(first_line))


def keep_path(path: str, filters: ExtractionFilters) -> bool:
    if filters.skip_vendored and is_vendored(path):
        return False
    if filters.skip_lockfiles and is_lockfile(path):
        return False
    return True
