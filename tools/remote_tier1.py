#!/usr/bin/env python3
"""The transport for fallback ladder rung 4 — a remote tier-1 verifier (spec sec 5.5).

    python tools/remote_tier1.py --endpoint https://api.example.com/v1 --model m --ping

    # tandem.toml
    [tier1]
    enabled = true
    rung = "remote"
    remote_endpoint = "https://api.example.com/v1"
    remote_model = "some-large-model"
    remote_transport = "tools/remote_tier1.py"
    remote_consent = "tier 1 leaves this machine"

**Deliberately outside the package**, for the same reason as `export_reviews.py`:
`src/tandem/` makes no outbound network call anywhere, and the offline posture
(sec 8.6) is a claim the verification script proves by running `lsof` during a
session and seeing nothing but loopback. A network-capable module inside the package
would make that claim rest on "we do not call it" rather than "it is not there" — and
it would weaken it for every deployment, including the ones that never enable this
rung. So the socket lives here, `tandem.backends.remote_tier1` holds none, and
`build_tier1` loads this file by path only when the config names the rung *and*
carries the consent string.

Stdlib only, same discipline: every dependency is a security surface, and this would
be one the runtime does not otherwise need.

**The key is never in the config file.** `remote_api_key_env` names an environment
variable; this reads it at startup. A config file is committed, diffed, pasted into
issues and shipped in support bundles, and a key in one is a key that has leaked.

**TLS is required**, with loopback the only exception. The payload is the candidate
patch and the code it was generated from, and the header is a bearer key; `http://`
puts all three on the wire in the clear, which is not what the consent sentence
consented to.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable

USER_AGENT = "tandem-remote-tier1/1.0"


def _is_loopback(netloc: str) -> bool:
    host = netloc.rsplit("@", 1)[-1]
    if host.startswith("["):
        host = host[1 : host.find("]")]
    elif ":" in host:
        host = host.rsplit(":", 1)[0]
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


def require_tls(url: str) -> None:
    """Refuse a cleartext endpoint (sec 5.5 rung 4).

    What goes over this socket is the candidate patch, the context it was generated
    from, and a bearer key — the whole reason rung 4 needs a consent sentence. Over
    `http://` all three are readable by anyone on the path, and the operator who
    typed the consent sentence consented to *one* third party seeing them, not to
    everyone between here and there.

    The one opt-out is a loopback host: an on-machine proxy or a test double, where
    there is no path to be on. It is deliberately not a flag — a flag named
    `--insecure` gets set once during setup and never unset.
    """
    scheme, _, rest = url.partition("://")
    scheme = scheme.lower()
    netloc = rest.split("/", 1)[0]
    if scheme == "https":
        return
    if scheme == "http" and _is_loopback(netloc):
        return
    raise ValueError(
        f"tier1.remote_endpoint={url!r} is not TLS. Rung 4 sends the candidate "
        "patch, the code it was generated from and the API key over this socket; "
        "use https:// (http:// is accepted only for a loopback host)."
    )


def build_transport(
    *,
    endpoint: str,
    api_key: str = "",
    timeout_s: float = 180.0,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Return the async callable `RemoteTier1Backend` posts through.

    The contract is narrow on purpose: a chat-completions payload in, the parsed
    response body out. Everything the runtime cares about — the output clamp, the
    schema, the greedy sampling — is already in the payload by the time it gets here
    (`backends/tier1_call.py`), so this file cannot weaken any of it. It moves bytes.
    """
    require_tls(endpoint)
    url = endpoint.rstrip("/") + "/chat/completions"

    def _post(payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", USER_AGENT)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:200]
            # No retry loop. A tier-1 call that fails degrades to a failed verdict
            # and the turn continues on tier 0 alone (sec 5.5) — retrying here would
            # spend the user's latency budget on a rung that is already the last one.
            raise RuntimeError(f"remote tier 1 returned {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"remote tier 1 unreachable at {url}: {exc}") from exc

    async def transport(payload: dict[str, Any]) -> dict[str, Any]:
        # urllib blocks; a blocking call on the event loop would stall every other
        # in-flight request in the gateway for the length of the verdict.
        return await asyncio.to_thread(_post, payload)

    return transport


def transport_from_config(
    *,
    endpoint: str,
    model: str = "",
    api_key_env: str = "",
    timeout_s: float = 180.0,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Entry point `tandem.backends.build_tier1` calls after loading this file.

    Named separately from `build_transport` so the config-facing signature can grow
    without changing the one a test or a script calls directly. `model` is accepted
    and unused: it is already in every payload, and taking it here keeps the caller
    from having to know which arguments this particular transport happens to need.
    """
    if not endpoint:
        raise ValueError("tier1.remote_endpoint is not set")
    key = os.environ.get(api_key_env, "") if api_key_env else ""
    if api_key_env and not key:
        raise ValueError(
            f"{api_key_env} is not set. The key is read from the environment rather "
            "than the config file, which is committed and shared."
        )
    return build_transport(endpoint=endpoint, api_key=key, timeout_s=timeout_s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="TANDEM_REMOTE_API_KEY")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument(
        "--ping",
        action="store_true",
        help="send a one-token request, so a misconfigured endpoint is found here "
        "rather than in the middle of a turn",
    )
    args = p.parse_args(argv)

    if not args.ping:
        print("nothing to do; pass --ping to check the endpoint", file=sys.stderr)
        return 0

    transport = transport_from_config(
        endpoint=args.endpoint,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout_s=args.timeout,
    )
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        body = asyncio.run(transport(payload))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(body, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
