"""Serve a model whose generation takes mlx-lm's *sequential* path, on this engine.

`optiq serve` inherits mlx-lm's server, which runs generation on a worker thread and
picks one of two paths per request. The batchable path (`BatchGenerator`) is explicitly
put on `mx.default_stream`, and works. The sequential path — taken when any cache the
model builds lacks `merge`, which is every DeepSeek-V4-Flash request — runs inside
`mlx_lm.generate`'s module-level

    generation_stream = mx.new_thread_local_stream(mx.default_device())

created at *import*, on the main thread. Entering it from the worker thread and then
forcing an eval raises, from C++, through the server's thread and into `libc++abi`,
which aborts the process:

    std::runtime_error: There is no Stream(gpu, 1) in current thread

Measured 2026-08-10 on mlx-lm 0.31.3 / mlx 0.32.0 / optiq 0.4.18. The abort lands at
`optiq/runtime/moe_stream.py` `__call__`, because a streamed expert projection is the
first thing that needs routing indices on the host and so is the first forced eval —
which is why this reads as a streaming bug and is not one. A resident 122B on the same
engine and flags is unaffected: it is batchable, so it never enters that context.

Rebinding the module global to the default stream is what the server already does for
the other path, so both paths end up on the stream the synchronization messages use.
It costs the cross-request overlap a thread-local stream would buy, which a
single-user local deployment was not using.

Use it exactly as `optiq serve`; every argument is forwarded:

    tools/ds4_serve.py --stream-experts --model "$SNAPSHOT" --port 8081
"""

from __future__ import annotations

import sys


def main() -> int:
    import importlib

    import mlx.core as mx

    # `import mlx_lm.generate` binds the *function* of that name from the package
    # namespace, not the module — so the attribute below would not exist.
    generate_mod = importlib.import_module("mlx_lm.generate")

    before = generate_mod.generation_stream
    # Assigned at import, so mypy does not see it as a module attribute.
    generate_mod.generation_stream = mx.default_stream(mx.default_device())  # type: ignore[attr-defined]
    print(
        f"[ds4_serve] generation_stream {before} -> "
        f"{generate_mod.generation_stream} (see this file's docstring)",
        flush=True,
    )

    from optiq.cli import cli

    sys.argv = ["optiq", "serve", *sys.argv[1:]]
    cli()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
