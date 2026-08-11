#!/usr/bin/env python3
"""Export first-review timestamps for merged PRs, for A2 extraction (spec sec 6.3).

    python tools/corpus/export_reviews.py --owner pallets --repo click --out reviews.json
    orbit extract a2 --repo ./click --reviews reviews.json

**Deliberately outside the package.** `src/orbit/` makes no outbound network call
anywhere, and the offline posture (sec 8.6) is a claim the verification script has
to be able to prove by running `lsof` during a session and seeing nothing but
loopback. A network-capable module inside the package would make that claim rest on
"we do not call it" rather than "it is not there". So this lives in `tools/`, is not
installed, and imports nothing from orbit.

Stdlib only, for the same reason: every dependency is a security surface, not a
convenience, and this one would be a dependency the runtime does not otherwise need.

**The token comes from `$GITHUB_TOKEN` (or `$GH_TOKEN`) and from nowhere else** — not
from a flag, which would put it in shell history and in `/proc/*/cmdline`. It rides on
every request, so the transport is pinned as well: TLS is required, and neither a
redirect nor a `Link: rel="next"` may take the request off the host it was aimed at.
Both of those are chosen by the server, and `urllib` re-sends `Authorization` across
redirects — including cross-host and https to http.

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
from collections.abc import Callable, Iterable
from typing import Any

API = "https://api.github.com"
USER_AGENT = "orbit-export-reviews/1.0"

Fetch = Callable[[str], Any]


# --- transport safety --------------------------------------------------------
#
# Two properties this file has to keep, because the token travels on every request
# and the thing on the other end controls both the redirects and the pagination
# links: the connection is TLS, and it stays on the one host it was aimed at.


def origin_of(url: str) -> tuple[str, str]:
    """(scheme, netloc) — what "the same place" means for a redirect."""
    parts = urllib.parse.urlsplit(url)
    return parts.scheme.lower(), parts.netloc.lower()


def _is_loopback(netloc: str) -> bool:
    host = netloc.rsplit("@", 1)[-1]
    host = (
        host[1 : host.find("]")]
        if host.startswith("[")
        else host.rsplit(":", 1)[0]
        if ":" in host
        else host
    )
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


def require_tls(url: str) -> None:
    """Refuse a cleartext endpoint.

    Without this an `http://` base ships the bearer token, and the repository and
    pull-request metadata it fetches, to anyone on the path. The single opt-out is
    loopback — a test double or a local proxy, where there is no path to be on.
    """
    scheme, netloc = origin_of(url)
    if scheme == "https":
        return
    if scheme == "http" and _is_loopback(netloc):
        return
    raise ValueError(
        f"refusing to send an API token over {scheme or 'no'}:// to {netloc or url!r}. "
        "Use https:// (http:// is accepted only for a loopback host)."
    )


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that leaves the origin the request was aimed at.

    `urllib`'s default handler re-sends the headers it was given on the redirected
    request — `Authorization` included — and follows `http://` as readily as
    `https://` (verified against CPython 3.11.15's `HTTPRedirectHandler`). A 302 to
    an attacker's host therefore hands over the token, in cleartext if they ask for
    it, and nothing about the transaction looks wrong from here.

    Refusing rather than stripping the header: this exporter talks to exactly one
    host, so there is no legitimate cross-origin redirect to preserve, and a
    stripped-header retry would quietly return an anonymous, rate-limited answer
    instead of saying what happened.
    """

    def __init__(self, origin: tuple[str, str]):
        self.origin = origin

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if origin_of(newurl) != self.origin:
            raise urllib.error.HTTPError(
                newurl,
                code,
                f"refusing a redirect off {self.origin[1]} — the Authorization "
                "header would travel with it",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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


def build_records(
    pulls: Iterable[dict[str, Any]], fetch: Fetch
) -> list[dict[str, Any]]:
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
        require_tls(base)
        self.prefix = f"{base}/repos/{owner}/{repo}"
        self.token = token
        self.origin = origin_of(base)
        # A dedicated opener rather than the module-level one: the redirect policy
        # is a property of *this* client and its token, not of the process.
        self._opener = urllib.request.build_opener(_SameOriginRedirect(self.origin))

    def _request(self, url: str) -> tuple[Any, dict[str, str]]:
        req = urllib.request.Request(url, headers=self._headers())
        for attempt in range(5):
            try:
                with self._opener.open(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8")), dict(resp.headers)
            except urllib.error.HTTPError as exc:
                body = _read_error(exc)
                # Not every 403 is a rate limit, and the difference matters: a
                # permission or policy 403 will never succeed, so retrying it five
                # times with backoff wastes a minute and then reports throttling as
                # the cause when the real answer was in the first response body.
                if exc.code == 429 or (
                    exc.code == 403 and _is_rate_limit(exc.headers, body)
                ):
                    wait = _retry_after(exc.headers, attempt)
                    print(f"  rate limited, waiting {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                if exc.code == 404:
                    return None, {}
                # `reason` carries the refusal from `_SameOriginRedirect` on a 3xx,
                # where the body is the redirect's own (empty) one and says nothing.
                raise RuntimeError(
                    f"GitHub returned {exc.code} for {url}\n"
                    f"  {exc.reason}\n  {body.strip()[:400]}"
                ) from exc
            except urllib.error.URLError:
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
        first = f"{self.prefix}{path}"
        sep = "&" if "?" in first else "?"
        # Optional because `_next_page` returns None at the last page, which is
        # what ends the loop below.
        url: str | None = f"{first}{sep}per_page=100"
        items: list[Any] = []
        while url:
            body, headers = self._request(url)
            if body is None:
                break
            if isinstance(body, list):
                items.extend(body)
            else:
                return body
            url = self._next_page(headers.get("Link", ""))
        return items

    def _next_page(self, link_header: str) -> str | None:
        """The `rel="next"` URL, once it has been shown to be the same server.

        The Link header is chosen by whatever answered the last request, and the
        token goes on whatever it names. GitHub paginates within one origin (the
        `next` URL may be the `/repositories/{id}/…` form rather than the
        `/repos/{owner}/{repo}/…` one, so the check is the origin, not the prefix);
        anything else is a redirect wearing a different hat.
        """
        nxt = _next_link(link_header)
        if nxt is None:
            return None
        if origin_of(nxt) != self.origin:
            raise RuntimeError(
                f"refusing to follow a pagination link to {origin_of(nxt)[1]!r}: the "
                f"Link header is server-controlled and this client's token would go "
                f"with it (expected {self.origin[1]})"
            )
        return nxt


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
    return min(60, 1 << (attempt + 1))


# --- entry point ------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--owner", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--out", default="reviews.json")
    p.add_argument(
        "--limit", type=int, default=0, help="stop after N merged PRs (0 = all)"
    )

    # The token is read from the environment and from nowhere else. It used to be
    # accepted on argv too, which puts it in shell history and in `/proc/*/cmdline`
    # for every user on the box, for the whole run. Rejecting the flag by name
    # rather than letting argparse call it unrecognised, because the fix is
    # specific and the person hitting this is one export away from pasting it in
    # again.
    supplied = sys.argv[1:] if argv is None else argv
    if any(a == "--token" or a.startswith("--token=") for a in supplied):
        print(
            "--token is no longer accepted: a token on the command line is in the "
            "shell history and in /proc/*/cmdline for every user on the machine. "
            "Set GITHUB_TOKEN (or GH_TOKEN) instead.",
            file=sys.stderr,
        )
        return 2
    args = p.parse_args(argv)
    args.token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if not args.token:
        print(
            "warning: no token; GitHub allows 60 anonymous requests/hour and this "
            "makes two per pull request. Set GITHUB_TOKEN.",
            file=sys.stderr,
        )

    gh = GitHub(args.owner, args.repo, args.token)
    print(
        f"listing merged pull requests for {args.owner}/{args.repo}…", file=sys.stderr
    )
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
