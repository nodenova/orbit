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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

UPSTREAM_HOST = "127.0.0.1"
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_port = 8081

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[proxy] {self.address_string()} {fmt % args}")

    def _relay(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

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
    args = ap.parse_args()

    Handler.upstream_port = args.upstream_port
    server = ThreadingHTTPServer(("127.0.0.1", args.listen_port), Handler)
    print(
        f"[proxy] 127.0.0.1:{args.listen_port} -> 127.0.0.1:{args.upstream_port}\n"
        f"[proxy] ANTHROPIC_BASE_URL=http://127.0.0.1:{args.listen_port}"
    )
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
