#!/usr/bin/env python3
"""Export first-review timestamps for merged PRs, for A2 extraction (spec sec 6.3).

    python tools/export_reviews.py --owner pallets --repo click --out reviews.json
    tandem extract a2 --repo ./click --reviews reviews.json

**Deliberately outside the package.** `src/tandem/` makes no outbound network call
anywhere, and the offline posture (sec 8.6) is a claim the verification script has
to be able to prove by running `lsof` during a session and seeing nothing but
loopback. A network-capable module inside the package would make that claim rest on
"we do not call it" rather than "it is not there". So this lives in `tools/`, is not
installed, and imports nothing from tandem.

Stdlib only, for the same reason: every dependency is a security surface, not a
convenience, and this one would be a dependency the runtime does not otherwise need.

**What it is for.** A2's preference pairs are (`rejected` = the diff as of the first
review comment, `chosen` = the diff as merged). Without this file, extraction falls
back to the first branch commit — the closest local approximation of "what the
author proposed" — and labels those pairs `source: branch_review` so a corpus can be
sliced by signal strength. With it, `rejected` is the state a reviewer actually
looked at, which is what the spec asks for.

Self-reviews are ignored. A PR author commenting on their own diff is not the review
signal, and treating it as one would cut the branch at a point nobody reviewed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterable

API = "https://api.github.com"
USER_AGENT = "tandem-export-reviews/1.0"

Fetch = Callable[[str], Any]


# --- pure logic (tested; no network) ----------------------------------------


def earliest_review_at(
    reviews: Iterable[dict[str, Any]],
    review_comments: Iterable[dict[str, Any]],
    author_login: str | None,
) -> str | None:
    """Earliest timestamp at which someone other than the author reviewed.

    Considers both formal reviews (`submitted_at`) and inline review comments
    (`created_at`) — a maintainer who leaves three line comments without submitting
    a review has still reviewed, and taking only formal reviews would miss it
    entirely on repositories that work that way.

    Returns an ISO-8601 string, or None when nobody but the author engaged.
    """
    stamps: list[str] = []
    for item in reviews:
        if _is_author(item, author_login):
            continue
        # A PENDING review has not been shown to the author and is not yet feedback.
        if str(item.get("state", "")).upper() == "PENDING":
            continue
        when = item.get("submitted_at") or item.get("created_at")
        if when:
            stamps.append(str(when))
    for item in review_comments:
        if _is_author(item, author_login):
            continue
        when = item.get("created_at")
        if when:
            stamps.append(str(when))
    # ISO-8601 UTC strings from GitHub sort lexicographically in time order.
    return min(stamps) if stamps else None


def _is_author(item: dict[str, Any], author_login: str | None) -> bool:
    if not author_login:
        return False
    user = item.get("user") or {}
    return str(user.get("login", "")).lower() == author_login.lower()


def build_records(pulls: Iterable[dict[str, Any]], fetch: Fetch) -> list[dict[str, Any]]:
    """One record per merged PR that has a merge commit and a real review."""
    out: list[dict[str, Any]] = []
    for pr in pulls:
        if not pr.get("merged_at"):
            continue
        merge_sha = pr.get("merge_commit_sha")
        if not merge_sha:
            continue
        number = pr.get("number")
        author = ((pr.get("user") or {}).get("login")) or None

        reviews = fetch(f"/pulls/{number}/reviews") or []
        comments = fetch(f"/pulls/{number}/comments") or []
        first = earliest_review_at(reviews, comments, author)
        if first is None:
            # Merged without review. There is no "before review" state to point at,
            # so emitting a record would claim a signal that does not exist.
            continue
        out.append(
            {
                "merge_sha": merge_sha,
                "first_review_at": first,
                "pr": number,
                "merged_at": pr.get("merged_at"),
            }
        )
    return out


# --- network ----------------------------------------------------------------


class GitHub:
    def __init__(self, owner: str, repo: str, token: str | None, *, base: str = API):
        self.prefix = f"{base}/repos/{owner}/{repo}"
        self.token = token

    def _request(self, url: str) -> tuple[Any, dict[str, str]]:
        req = urllib.request.Request(url, headers=self._headers())
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8")), dict(resp.headers)
            except urllib.error.HTTPError as exc:
                body = _read_error(exc)
                # Not every 403 is a rate limit, and the difference matters: a
                # permission or policy 403 will never succeed, so retrying it five
                # times with backoff wastes a minute and then reports throttling as
                # the cause when the real answer was in the first response body.
                if exc.code == 429 or (exc.code == 403 and _is_rate_limit(exc.headers, body)):
                    wait = _retry_after(exc.headers, attempt)
                    print(f"  rate limited, waiting {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                if exc.code == 404:
                    return None, {}
                raise RuntimeError(
                    f"GitHub returned {exc.code} for {url}\n  {body.strip()[:400]}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError(f"gave up on {url} after repeated rate limiting")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get(self, path: str) -> Any:
        """A single paginated resource, concatenated."""
        url = f"{self.prefix}{path}"
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}per_page=100"
        items: list[Any] = []
        while url:
            body, headers = self._request(url)
            if body is None:
                break
            if isinstance(body, list):
                items.extend(body)
            else:
                return body
            url = _next_link(headers.get("Link", ""))
        return items


def _next_link(link_header: str) -> str | None:
    for part in link_header.split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        if 'rel="next"' in section[1].replace(" ", ""):
            return section[0].strip().strip("<>")
    return None


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - the body is diagnostics, never load-bearing
        return ""


def _is_rate_limit(headers: Any, body: str) -> bool:
    """Distinguish a throttle from a refusal.

    Primary limits answer 403 with `X-RateLimit-Remaining: 0`; secondary ones send
    `Retry-After`. Everything else with a 403 — a policy block, a missing scope, a
    proxy that will not forward the request — is a refusal, and waiting changes
    nothing about it.
    """
    try:
        if headers is not None:
            if headers.get("Retry-After"):
                return True
            remaining = headers.get("X-RateLimit-Remaining")
            if remaining is not None and int(remaining) <= 0:
                return True
    except (TypeError, ValueError):
        pass
    low = body.lower()
    return "rate limit" in low or "secondary rate" in low or "abuse detection" in low


def _retry_after(headers: Any, attempt: int) -> int:
    try:
        if headers and headers.get("Retry-After"):
            return max(1, int(headers["Retry-After"]))
        reset = headers.get("X-RateLimit-Reset") if headers else None
        if reset:
            return max(1, int(reset) - int(time.time()) + 1)
    except (TypeError, ValueError):
        pass
    return min(60, 2 ** (attempt + 1))


# --- entry point ------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--owner", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--out", default="reviews.json")
    p.add_argument("--limit", type=int, default=0, help="stop after N merged PRs (0 = all)")
    p.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
        help="defaults to $GITHUB_TOKEN or $GH_TOKEN; anonymous is 60 req/hour",
    )
    args = p.parse_args(argv)

    if not args.token:
        print(
            "warning: no token; GitHub allows 60 anonymous requests/hour and this "
            "makes two per pull request. Set GITHUB_TOKEN.",
            file=sys.stderr,
        )

    gh = GitHub(args.owner, args.repo, args.token)
    print(f"listing merged pull requests for {args.owner}/{args.repo}…", file=sys.stderr)
    pulls = gh.get("/pulls?state=closed&sort=updated&direction=desc")
    merged = [pr for pr in pulls if pr.get("merged_at")]
    if args.limit:
        merged = merged[: args.limit]
    print(f"  {len(merged)} merged; fetching reviews", file=sys.stderr)

    records = build_records(merged, gh.get)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, sort_keys=True)
        fh.write("\n")

    reviewed = len(records)
    print(
        f"wrote {reviewed} records to {args.out} "
        f"({len(merged) - reviewed} merged without review, omitted)",
        file=sys.stderr,
    )
    if reviewed == 0:
        print(
            "No reviewed PRs found. A2 will fall back to first-branch-commit pairs, "
            "labelled source=branch_review (sec 6.3).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
