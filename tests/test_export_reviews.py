"""The A2 forge exporter's pure logic (spec sec 6.3).

Loaded by path, because `tools/` is deliberately not part of the installed package:
nothing under `src/tandem/` makes an outbound network call, and keeping the exporter
out of the package is what makes the sec 8.6 offline claim structural rather than a
promise.

No test here touches the network. The HTTP layer is injected, so what is under test
is the decision — which timestamp counts as "the first review" — which is the part
that changes what A2 learns.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "export_reviews.py"
_spec = importlib.util.spec_from_file_location("export_reviews", _PATH)
assert _spec and _spec.loader
export_reviews = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_reviews)


def review(login: str, when: str, state: str = "COMMENTED") -> dict:
    return {"user": {"login": login}, "submitted_at": when, "state": state}


def comment(login: str, when: str) -> dict:
    return {"user": {"login": login}, "created_at": when}


# --- which timestamp is "the first review" ----------------------------------


def test_earliest_of_reviews_and_inline_comments():
    """A maintainer leaving line comments without submitting a review has reviewed.

    Taking only formal reviews would miss it entirely on repos that work that way.
    """
    got = export_reviews.earliest_review_at(
        [review("maintainer", "2026-03-02T10:00:00Z")],
        [comment("maintainer", "2026-03-01T09:00:00Z")],
        author_login="author",
    )
    assert got == "2026-03-01T09:00:00Z"


def test_author_self_review_is_not_review():
    """Cutting the branch where the author commented on their own diff would put
    `rejected` at a point nobody reviewed."""
    got = export_reviews.earliest_review_at(
        [review("author", "2026-03-01T08:00:00Z")],
        [comment("author", "2026-03-01T08:30:00Z")],
        author_login="author",
    )
    assert got is None


def test_author_check_is_case_insensitive():
    got = export_reviews.earliest_review_at(
        [review("Author", "2026-03-01T08:00:00Z")], [], author_login="author"
    )
    assert got is None


def test_author_comments_do_not_mask_a_real_review():
    got = export_reviews.earliest_review_at(
        [review("author", "2026-03-01T08:00:00Z"), review("maintainer", "2026-03-03T12:00:00Z")],
        [],
        author_login="author",
    )
    assert got == "2026-03-03T12:00:00Z"


def test_pending_reviews_are_ignored():
    """A pending review has not been shown to the author and is not yet feedback."""
    got = export_reviews.earliest_review_at(
        [review("maintainer", "2026-03-01T08:00:00Z", state="PENDING")], [], "author"
    )
    assert got is None


def test_no_engagement_at_all():
    assert export_reviews.earliest_review_at([], [], "author") is None


def test_missing_author_login_keeps_every_review():
    """Unknown author is not a reason to discard the signal."""
    got = export_reviews.earliest_review_at(
        [review("someone", "2026-03-01T08:00:00Z")], [], author_login=None
    )
    assert got == "2026-03-01T08:00:00Z"


# --- record building --------------------------------------------------------


def _fake_fetch(by_path: dict[str, list]) -> object:
    def fetch(path: str):
        return by_path.get(path, [])

    return fetch


def test_build_records_emits_the_shape_extract_a2_accepts():
    pulls = [
        {
            "number": 7,
            "merged_at": "2026-03-05T00:00:00Z",
            "merge_commit_sha": "abc123",
            "user": {"login": "author"},
        }
    ]
    fetch = _fake_fetch(
        {
            "/pulls/7/reviews": [review("maintainer", "2026-03-02T10:00:00Z")],
            "/pulls/7/comments": [],
        }
    )
    records = export_reviews.build_records(pulls, fetch)
    assert records == [
        {
            "merge_sha": "abc123",
            "first_review_at": "2026-03-02T10:00:00Z",
            "pr": 7,
            "merged_at": "2026-03-05T00:00:00Z",
        }
    ]


def test_unmerged_and_unreviewed_pulls_are_omitted():
    """A PR merged without review has no 'before review' state to point at, so a
    record would claim a signal that does not exist."""
    pulls = [
        {"number": 1, "merged_at": None, "merge_commit_sha": "x", "user": {"login": "a"}},
        {"number": 2, "merged_at": "2026-01-01T00:00:00Z", "merge_commit_sha": None,
         "user": {"login": "a"}},
        {"number": 3, "merged_at": "2026-01-01T00:00:00Z", "merge_commit_sha": "y",
         "user": {"login": "a"}},
    ]
    records = export_reviews.build_records(pulls, _fake_fetch({}))
    assert records == []


def test_output_is_loadable_by_extract_a2(tmp_path):
    """The contract that actually matters: what this writes, extraction reads."""
    from tandem.adapters.extract_a2 import _load_reviews

    records = [
        {"merge_sha": "abc123", "first_review_at": "2026-03-02T10:00:00Z", "pr": 7},
        {"merge_sha": "def456", "first_review_at": "2026-04-02T10:00:00Z", "pr": 9},
    ]
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    loaded = _load_reviews(path)
    assert loaded == {
        "abc123": "2026-03-02T10:00:00Z",
        "def456": "2026-04-02T10:00:00Z",
    }


# --- pagination -------------------------------------------------------------


@pytest.mark.parametrize(
    "header,expected",
    [
        ('<https://api.github.com/x?page=2>; rel="next", <https://api.github.com/x?page=9>; rel="last"',
         "https://api.github.com/x?page=2"),
        ('<https://api.github.com/x?page=1>; rel="prev"', None),
        ("", None),
    ],
)
def test_next_link_parsing(header, expected):
    assert export_reviews._next_link(header) == expected


# --- throttle versus refusal ------------------------------------------------


class _Headers(dict):
    """dict with .get, standing in for an HTTPMessage."""


@pytest.mark.parametrize(
    "headers,body,expected",
    [
        (_Headers({"Retry-After": "30"}), "", True),
        (_Headers({"X-RateLimit-Remaining": "0"}), "", True),
        (_Headers({}), "You have exceeded a secondary rate limit", True),
        (_Headers({"X-RateLimit-Remaining": "4980"}), "", False),
        # The refusal that cost a minute of backoff and reported the wrong cause.
        (_Headers({}), '{"message":"GitHub access to this repository is not enabled"}', False),
        (_Headers({}), '{"message":"Bad credentials"}', False),
        (None, "", False),
    ],
)
def test_rate_limit_is_distinguished_from_refusal(headers, body, expected):
    """A policy 403 will never succeed; waiting on it wastes a minute and then
    blames throttling for an answer that was in the first response body."""
    assert export_reviews._is_rate_limit(headers, body) is expected


def test_the_exporter_is_not_part_of_the_package():
    import tandem

    pkg_root = Path(tandem.__file__).resolve().parent
    assert _PATH.parent.name == "tools"
    assert pkg_root not in _PATH.parents


def test_the_package_s_network_surface_stays_one_known_file():
    """Sec 8.6, stated accurately.

    The package is *not* free of HTTP: `backends/mlx_tier1.py` holds an httpx client
    for the tier-1 process boundary (sec 5.4), pointed at a configured endpoint that
    defaults to loopback. That is the entire network surface, and pinning it to one
    file is what makes a second one a test failure rather than a discovery.

    Extraction in particular must stay clean, because A2's fallback to
    first-branch-commit pairs is what lets a corpus be built with the network off.
    """
    import tandem

    pkg_root = Path(tandem.__file__).resolve().parent
    allowed = {"backends/mlx_tier1.py"}
    # Import- and call-shaped, not bare names: `offline.py` lists "httpx" as a
    # string in its dependency allow-list, which is the opposite of a network call.
    markers = (
        "import httpx",
        "httpx.",
        "urllib.request",
        "requests.get",
        "socket.create_connection",
        "import aiohttp",
    )

    found: dict[str, list[str]] = {}
    for py in sorted(pkg_root.rglob("*.py")):
        rel = py.relative_to(pkg_root).as_posix()
        text = py.read_text(encoding="utf-8")
        hits = [m for m in markers if m in text]
        if hits:
            found[rel] = hits

    assert set(found) == allowed, f"network surface changed: {found}"

    for py in (pkg_root / "adapters").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not any(m in text for m in markers), f"{py.name} reaches the network"
