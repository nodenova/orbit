"""Loopback shim that terminates optiq's Anthropic SSE stream, so Claude Code can drive it.

optiq 0.4.18 emits a complete and correct `/v1/messages` event sequence ending in
`message_stop`, then never closes the connection. The answer has already arrived, so every
client blocks until its own timeout and reports failure on a response it fully received —
Claude Code shows "API Error: The operation timed out." after `CLAUDE_STREAM_IDLE_TIMEOUT_MS`
with `output_tokens` already non-zero. Its `/v1/chat/completions` path closes correctly, so
the defect is in the Anthropic writer alone.

This forwards verbatim to the upstream engine and closes once `message_stop` has been
relayed. Everything else passes through untouched. See docs/operations.md §7.6.
"""

import argparse
import contextlib
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

UPSTREAM_HOST = "127.0.0.1"
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length"}
SAMPLED_PATHS = ("/v1/messages", "/v1/chat/completions", "/v1/responses")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_port = 8081
    # None means relay the client's own value, or none at all — in which case
    # `optiq serve` applies its model-recommended sampler (temperature 1.0,
    # top_p 0.95, top_k 40). Claude Code exposes no sampling flags whatsoever, so
    # this proxy is the only place in the path where they can be pinned, and
    # without pinning a local-model measurement is sampled where every other
    # recorded measurement of this model is greedy.
    sampling: ClassVar[dict[str, Any]] = {}
    hoist_system = False
    announced = False

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[proxy] {self.address_string()} {fmt % args}")

    def _hoist_system(self, payload: dict[str, Any]) -> bool:
        """Merge `system`-role entries in `messages` into the top-level `system`.

        Claude Code sends a system-role message at index 1 of `messages`, after the
        first user turn, as well as the top-level `system` field. Anthropic's schema
        has no system role inside `messages` at all, and a chat template that takes
        it literally can refuse the request outright: `Agents-A1-4B` raises
        "System message must be at the beginning" from its Jinja and the backend
        answers 500 before the model runs. Merging preserves the text and produces a
        request the templates agree on.
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return False
        stray = [
            m for m in messages if isinstance(m, dict) and m.get("role") == "system"
        ]
        if not stray:
            return False

        blocks: list[Any] = []
        top = payload.get("system")
        if isinstance(top, str):
            blocks.append({"type": "text", "text": top})
        elif isinstance(top, list):
            blocks.extend(top)
        for m in stray:
            content = m.get("content")
            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                blocks.extend(b for b in content if isinstance(b, dict))

        payload["system"] = blocks
        payload["messages"] = [
            m
            for m in messages
            if not (isinstance(m, dict) and m.get("role") == "system")
        ]
        return True

    def _rewrite(self, body: bytes) -> bytes:
        wants = self.sampling or self.hoist_system
        if not wants or not any(p in self.path for p in SAMPLED_PATHS):
            return body
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return body
        if not isinstance(payload, dict):
            return body

        before = {k: payload.get(k) for k in self.sampling}
        payload.update(self.sampling)
        hoisted = self._hoist_system(payload) if self.hoist_system else False

        if not Handler.announced:
            Handler.announced = True
            if self.sampling:
                print(
                    f"[proxy] pinning sampling {self.sampling} (client sent {before})"
                )
            if self.hoist_system:
                print(f"[proxy] hoisting stray system messages: {hoisted}")
        return json.dumps(payload).encode()

    def _relay(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        # Content-Length is hop-by-hop here and http.client recomputes it from the
        # body, so rewriting the body below cannot desynchronise the two.
        body = self._rewrite(body)

        conn = http.client.HTTPConnection(
            UPSTREAM_HOST, self.upstream_port, timeout=3600
        )
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Host"] = f"{UPSTREAM_HOST}:{self.upstream_port}"
        try:
            conn.request(method, self.path, body=body, headers=headers)
            upstream = conn.getresponse()
        except OSError as exc:
            self.send_error(502, f"upstream unreachable: {exc}")
            return

        ctype = upstream.getheader("Content-Type", "")
        if "text/event-stream" in ctype:
            self._relay_sse(upstream)
        else:
            payload = upstream.read()
            self.send_response(upstream.status)
            for k, v in upstream.getheaders():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        conn.close()

    def _relay_sse(self, upstream: http.client.HTTPResponse) -> None:
        self.send_response(upstream.status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # No Content-Length and an explicit close: the client reads to EOF, which is the
        # termination upstream omits.
        self.send_header("Connection", "close")
        self.end_headers()

        saw_stop = False
        trailing = 0
        while True:
            line = upstream.readline()
            if not line:
                break
            try:
                self.wfile.write(line)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            if line.startswith(b"event: message_stop"):
                saw_stop = True
            elif saw_stop:
                trailing += 1
                if trailing >= 2:
                    break
        self.close_connection = True

    def do_POST(self) -> None:
        self._relay("POST")

    def do_GET(self) -> None:
        self._relay("GET")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-port", type=int, default=8082)
    ap.add_argument("--upstream-port", type=int, default=8081)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument(
        "--hoist-system",
        action="store_true",
        help="merge stray system-role messages into the top-level system field; "
        "required by chat templates that refuse a system message mid-conversation",
    )
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="shorthand for --temperature 0 --top-p 1 --seed 0, matching "
        "eval.regression.run and tools/quality/qcn_quality.py",
    )
    args = ap.parse_args()

    if args.greedy:
        args.temperature = 0.0 if args.temperature is None else args.temperature
        args.top_p = 1.0 if args.top_p is None else args.top_p
        args.seed = 0 if args.seed is None else args.seed
    Handler.sampling = {
        k: v
        for k, v in (
            ("temperature", args.temperature),
            ("top_p", args.top_p),
            ("top_k", args.top_k),
            ("seed", args.seed),
        )
        if v is not None
    }

    Handler.hoist_system = args.hoist_system
    Handler.upstream_port = args.upstream_port
    server = ThreadingHTTPServer(("127.0.0.1", args.listen_port), Handler)
    print(
        f"[proxy] 127.0.0.1:{args.listen_port} -> 127.0.0.1:{args.upstream_port}\n"
        f"[proxy] ANTHROPIC_BASE_URL=http://127.0.0.1:{args.listen_port}\n"
        f"[proxy] sampling: {Handler.sampling or 'passthrough (engine default 1.0/0.95/40)'}"
    )
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
