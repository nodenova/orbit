"""Where the constrained-decode 2.4x actually goes. No model weights.

`BASELINE.md` §2.3 recorded ~21 ms/token of overhead and attributed most of it to the
synchronisation `tokens.tolist()` forces. These five measurements withdraw that
attribution: the sync alone is free, and the cost is host work serialised against an
idle GPU. See `docs/CONSTRAINED_DECODE.md` for the design that follows.

Every subcommand runs without weights. `loop` builds a ~4 GB synthetic decoder; the
rest need only the tokenizer, and `filter`/`identity`/`components` pay LMFE's ~1.2 s
vocabulary build.

    python tools/constrained_decode_bench.py loop --layers 48
    python tools/constrained_decode_bench.py filter
    python tools/constrained_decode_bench.py identity

**Check host health first.** `tools/mlxbench.py` must report ~247 GB/s; at the
23 GB/s of `HANDOFF.md` T18 the GPU is so slow that host work vanishes into
noise and `loop` reports every variant as equal.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VOCAB_FALLBACK = 248_077
HIDDEN = 4096

TOOLS_JSON: list[dict[str, Any]] = [
    {
        "name": "edit_file",
        "description": "Replace an exact string in a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
    }
]

TARGET_CALL = {
    "name": "edit_file",
    "arguments": {
        "path": "src/tandem/gateway/pipeline.py",
        "old": "compact(req)",
        "new": "compact(req, budget=budget)",
    },
}

# The same call as the model actually emitted it under constraint, newlines and all.
# Every string argument a coding agent writes has them; TARGET_CALL has none, and that
# one difference is a 9x in host cost (`escape`, CONSTRAINED_DECODE.md §8.4).
ESCAPED_CALL = {
    "name": "edit_file",
    "arguments": {
        "path": "src/tandem/gateway/pipeline.py",
        "old": "compact(req)\n",
        "new": "compact(req, budget=budget)\n",
    },
}


def default_model() -> Path:
    root = Path.home() / ".cache/huggingface/hub"
    for name in ("models--mlx-community--Qwen3.6-35B-A3B-OptiQ-4bit",):
        snaps = root / name / "snapshots"
        if snaps.is_dir():
            return next(snaps.iterdir())
    raise SystemExit("no tier-0 snapshot found; pass --model")


def load_filter(model: Path) -> tuple[Any, Any, list[int]]:
    """LMFE token filter for the `edit_file` schema, plus the tokenizer and target ids."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from tandem.gateway.toolcall.constrain import Constrainer, tool_call_schema
    from tandem.types import ToolDef

    tok = load_tokenizer(model)
    inner = tok._tokenizer
    constrainer = Constrainer()
    t0 = time.perf_counter()
    vocabulary = constrainer.vocabulary(tok)
    print(f"vocabulary build {time.perf_counter() - t0:.2f} s, vocab {len(inner)}")

    tools = [
        ToolDef(
            name=t["name"], description=t["description"], parameters=t["parameters"]
        )
        for t in TOOLS_JSON
    ]
    token_filter = constrainer.token_filter(tool_call_schema(tools), vocabulary)
    if token_filter is None:
        raise SystemExit(
            "lm-format-enforcer unavailable; pip install 'tandem[constrain]'"
        )
    ids = list(inner.encode(json.dumps(TARGET_CALL), add_special_tokens=False))
    return token_filter, inner, ids


def bench(fn: Any, n: int = 20) -> float:
    fn()
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(xs)


def cmd_loop(args: argparse.Namespace) -> None:
    """Synthetic decoder reproducing mlx_lm's loop shape (generate.py:453-470)."""
    import mlx.core as mx

    def build(n_layers: int) -> Any:
        mx.random.seed(0)
        embed = mx.random.normal((VOCAB_FALLBACK, 512)).astype(mx.float16)
        up = mx.random.normal((512, HIDDEN)).astype(mx.float16)
        ws = [
            mx.random.normal((HIDDEN, HIDDEN)).astype(mx.float16)
            for _ in range(n_layers)
        ]
        out = mx.random.normal((HIDDEN, VOCAB_FALLBACK)).astype(mx.float16)
        mx.eval(embed, up, ws, out)

        def call(y: Any) -> Any:
            x = embed[y].reshape(1, -1) @ up
            for w in ws:
                x = x @ w
            return x @ out

        return call

    def run_mlxlm(model_call: Any, processor: Any, steps: int) -> float:
        """`tokens` is concatenated *lazily*, so a processor that reads it forces
        an eval mid-graph and collapses generate_step's one-step lookahead."""

        def gen() -> Any:
            y = mx.array([1], dtype=mx.int32)
            tokens = y
            mx.async_eval(y)
            n = 0
            while True:
                logits = model_call(y)
                if processor is not None:
                    tokens = mx.concat([tokens, y])
                    logits = processor(tokens, logits)
                next_y = mx.argmax(logits, axis=-1)
                mx.async_eval(next_y)
                if n == steps:
                    break
                yield y.item()
                y = next_y
                n += 1

        it = gen()
        for _ in range(8):
            next(it)
        t0 = time.perf_counter()
        k = 0
        for _ in it:
            k += 1
        return (time.perf_counter() - t0) / k * 1000

    def run_restructured(model_call: Any, host_work: Any, steps: int) -> float:
        """Dispatch the forward pass, then do host work while the GPU runs it."""

        def gen() -> Any:
            y = mx.array([1], dtype=mx.int32)
            mx.async_eval(y)
            hist: list[int] = []
            n = 0
            while True:
                logits = model_call(y)
                mx.async_eval(logits)
                tok = y.item()
                hist.append(tok)
                mask = host_work(hist)
                if mask is not None:
                    logits = logits + mask
                y = mx.argmax(logits, axis=-1)
                mx.async_eval(y)
                if n == steps:
                    break
                yield tok
                n += 1

        it = gen()
        for _ in range(8):
            next(it)
        t0 = time.perf_counter()
        k = 0
        for _ in it:
            k += 1
        return (time.perf_counter() - t0) / k * 1000

    model_call = build(args.layers)
    ids = mx.array(list(range(0, args.allowed * 3, 3)), dtype=mx.int32)
    mx.eval(ids)
    spin = args.host_ms / 1000.0

    def busy() -> None:
        t = time.perf_counter()
        while time.perf_counter() - t < spin:
            pass

    def mask_for(dtype: Any) -> Any:
        m = mx.full((VOCAB_FALLBACK,), -float("inf"), dtype=dtype)
        m[ids] = 0.0
        return m

    def p_sync(tokens: Any, logits: Any) -> Any:
        tokens.tolist()
        return logits

    def p_mask(_tokens: Any, logits: Any) -> Any:
        return logits + mask_for(logits.dtype)

    def p_sync_mask(tokens: Any, logits: Any) -> Any:
        tokens.tolist()
        return logits + mask_for(logits.dtype)

    def p_full(tokens: Any, logits: Any) -> Any:
        tokens.tolist()
        busy()
        return logits + mask_for(logits.dtype)

    print(
        f"\nsynthetic decode, layers={args.layers}, vocab={VOCAB_FALLBACK}, "
        f"allowed={args.allowed}, host_ms={args.host_ms}"
    )
    print(f"{'variant':38s} {'ms/tok':>9s} {'tok/s':>8s} {'vs A':>7s}")

    base = None
    for label, proc in [
        ("A unconstrained (no processor)", None),
        ("B tolist() only, no mask", p_sync),
        ("C mask only, no tolist()", p_mask),
        ("D tolist() + mask", p_sync_mask),
        ("E tolist() + mask + LMFE spin", p_full),
    ]:
        ms = statistics.median(
            [run_mlxlm(model_call, proc, args.steps) for _ in range(3)]
        )
        base = base if base is not None else ms
        print(f"{label:38s} {ms:9.2f} {1000 / ms:8.1f} {ms / base:6.2f}x")

    def hw_none(_hist: list[int]) -> Any:
        return None

    def hw_full(_hist: list[int]) -> Any:
        busy()
        return mask_for(mx.float16)

    assert base is not None
    print()
    for label, hw in [
        ("F restructured, no host work", hw_none),
        ("G restructured, LMFE spin + mask", hw_full),
    ]:
        ms = statistics.median(
            [run_restructured(model_call, hw, args.steps) for _ in range(3)]
        )
        print(f"{label:38s} {ms:9.2f} {1000 / ms:8.1f} {ms / base:6.2f}x")

    print(f"\npeak mem {mx.get_peak_memory() / 2**30:.2f} GiB")
    print(
        "\nB ~ A is the finding: the sync is free. E - A ~ the spin, and G ~ A "
        "with the same spin, is the fix."
    )


def state_class(n_allowed: int, vocab: int) -> str:
    """Name the parser state from the size of its allowed set.

    The four sizes are far apart and stable — 3–269 skeleton, 649–658 string-open,
    5,795–5,798 post-backslash, 246,908 interior — so any threshold inside those
    gaps names the same states. Sizes, not decoded text, because the escape state
    is invisible in the text: it depends on where the tokenizer split (§8.4).
    """
    if n_allowed >= vocab // 2:
        return "string interior"
    if n_allowed >= 2_000:
        return "post-backslash escape"
    if n_allowed >= 300:
        return "string-open transition"
    return "JSON skeleton"


def cmd_filter(args: argparse.Namespace) -> None:
    """Per-token host cost across the parser states a real tool call walks.

    Runs both fixtures. `TARGET_CALL` alone is what made §3.2 and the §4 cost model
    5x low — it never enters the escape state, which is 97% of the real cost (T28).
    Both the pre-F1 shape and the landed one are measured, so §4's "landed" column
    is a measurement rather than a subtraction.
    """
    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from tandem.gateway.toolcall.constrain import Constrainer, tool_call_schema
    from tandem.types import ToolDef

    tok = load_tokenizer(args.model)
    inner = tok._tokenizer
    vocab_n = len(inner)
    constrainer = Constrainer()
    t0 = time.perf_counter()
    vocabulary = constrainer.vocabulary(tok)
    print(f"vocabulary build {time.perf_counter() - t0:.2f} s, vocab {vocab_n}")
    schema = tool_call_schema(
        [
            ToolDef(
                name=t["name"], description=t["description"], parameters=t["parameters"]
            )
            for t in TOOLS_JSON
        ]
    )

    totals = {}
    for label, call in (("TARGET_CALL", TARGET_CALL), ("ESCAPED_CALL", ESCAPED_CALL)):
        token_filter = constrainer.token_filter(schema, vocabulary)
        if token_filter is None:
            raise SystemExit("lm-format-enforcer unavailable")
        ids = list(inner.encode(json.dumps(call), add_special_tokens=False))
        rows = []
        prev_ids: Any = None
        for i in range(len(ids)):
            t0 = time.perf_counter()
            allowed = token_filter(ids[:i])
            t_lmfe = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            [int(t) for t in allowed if 0 <= int(t) < vocab_n]
            t_pre_f1 = (time.perf_counter() - t0) * 1000

            # F2 as it landed: content equality against the previous call's own
            # list, which is the only comparison that measures what production does.
            t0 = time.perf_counter()
            hit = (
                prev_ids is not None and prev_ids is not allowed and prev_ids == allowed
            )
            t_key = (time.perf_counter() - t0) * 1000

            # Timed on every step, then charged only where each variant pays it:
            # pre-F1 rebuilt the mask per token, F2 skips it on a hit.
            t0 = time.perf_counter()
            mask = mx.full((vocab_n,), -float("inf"), dtype=mx.float16)
            mask[mx.array(list(allowed), dtype=mx.int32)] = 0.0
            mx.eval(mask)
            t_mask = (time.perf_counter() - t0) * 1000

            prev_ids = allowed
            rows.append((i, len(allowed), t_lmfe, t_pre_f1, t_key, t_mask, hit))

        if args.per_token:
            print(
                f"\n### {label} per token\n{'step':>4} {'|allowed|':>10} "
                f"{'LMFE ms':>9} {'preF1 ms':>9} {'key ms':>7} {'mask ms':>8}  decoded"
            )
            for i, n, a, b, k, d, _ in rows:
                print(
                    f"{i:>4} {n:>10} {a:>9.2f} {b:>9.2f} {k:>7.2f} {d:>8.2f}  "
                    f"{inner.decode([ids[i]])!r:.18}"
                )

        print(f"\n### {label} — {len(rows)} tokens, by parser state")
        print(
            f"{'state':24s}{'tokens':>7}{'|allowed|':>20}{'LMFE ms':>10}"
            f"{'mask ms':>9}{'pre-F1':>9}{'landed':>9}{'share':>7}"
        )
        landed = [r[2] + r[4] + (0.0 if r[6] else r[5]) for r in rows]
        by_class: dict[
            str, list[tuple[int, int, float, float, float, float, bool]]
        ] = {}
        for r in rows:
            by_class.setdefault(state_class(r[1], vocab_n), []).append(r)
        for name in (
            "JSON skeleton",
            "string-open transition",
            "post-backslash escape",
            "string interior",
        ):
            rs = by_class.get(name)
            if not rs:
                continue
            sizes = sorted({r[1] for r in rs})
            span = f"{sizes[0]:,}" if len(sizes) == 1 else f"{sizes[0]:,}-{sizes[-1]:,}"
            cls_ms = sum(r[2] + r[4] + (0.0 if r[6] else r[5]) for r in rs)
            cls_pre = sum(r[2] + r[3] + r[5] for r in rs)
            print(
                f"{name:24s}{len(rs):>7}{span:>20}"
                f"{statistics.median([r[2] for r in rs]):>10.2f}"
                f"{statistics.median([r[5] for r in rs]):>9.2f}"
                f"{cls_pre:>9.1f}{cls_ms:>9.1f}{cls_ms / sum(landed):>6.0%}"
            )

        pre_f1 = sum(r[2] + r[3] + r[5] for r in rows)
        hits = sum(1 for r in rows if r[6])
        lmfe = sum(r[2] for r in rows)
        totals[label] = (len(rows), pre_f1, sum(landed))
        f1_ms = sum(r[3] for r in rows)
        f2_ms = sum(r[5] for r in rows if r[6])
        print(
            f"\n  LMFE alone   {lmfe:8.1f} ms   {lmfe / len(rows):6.2f} ms/token\n"
            f"  F1 removes   {f1_ms:8.1f} ms   {f1_ms / len(rows):6.2f} ms/token\n"
            f"  F2 removes   {f2_ms:8.1f} ms   {f2_ms / len(rows):6.2f} ms/token\n"
            f"  pre-F1 total {pre_f1:8.1f} ms   {pre_f1 / len(rows):6.2f} ms/token\n"
            f"  landed total {sum(landed):8.1f} ms   {sum(landed) / len(rows):6.2f} "
            f"ms/token   F2 hit {hits}/{len(rows)} ({hits / len(rows):.0%})"
        )

    (n_t, _, land_t), (n_e, _, land_e) = totals.values()
    print(
        f"\n{land_e / land_t:.1f}x, and per token {(land_e / n_e) / (land_t / n_t):.1f}x. "
        "The fixture is the whole difference — CONSTRAINED_DECODE.md §3.2, §4, T28."
    )


def cmd_escape(args: argparse.Namespace) -> None:
    """What one backslash inside a JSON string argument costs. Answers T26.

    `filter` walks TARGET_CALL, which contains no escape, and reports ~2.5 ms/token.
    Real weights reported 20.4. The whole difference is here and needs no model: the
    parser state entered on `\\` inside a string has ~5,800 allowed tokens and LMFE
    spends ~435 ms deciding which, with no cache to hold the answer (§6.1).
    """
    rows = []
    for label, call in (
        ("no escape (TARGET_CALL)", TARGET_CALL),
        ("two \\n", ESCAPED_CALL),
    ):
        token_filter, inner, _ = load_filter(args.model)
        ids = list(inner.encode(json.dumps(call), add_special_tokens=False))
        total, hot = 0.0, []
        for i in range(len(ids)):
            t0 = time.perf_counter()
            allowed = token_filter(ids[:i])
            dt = (time.perf_counter() - t0) * 1000
            total += dt
            if dt > 100:
                hot.append(
                    (i, len(allowed), dt, inner.decode(ids[max(0, i - 2) : i + 1]))
                )
        rows.append((label, len(ids), total, hot))
        print(
            f"\n{label:24s} {len(ids):>3} tokens  {total:8.1f} ms  "
            f"{total / len(ids):6.2f} ms/token"
        )
        for i, n, dt, ctx in hot:
            print(f"    step {i:>3}  |allowed|={n:>7}  {dt:8.2f} ms  context {ctx!r}")
        if not hot:
            print("    no call over 100 ms")

    (_, _, a, _), (_, _, b, hot_b) = rows
    print(
        f"\n{b / a:.1f}x for two escaped newlines. {sum(h[2] for h in hot_b):.0f} of the "
        f"{b:.0f} ms is {len(hot_b)} calls.\n"
        "A coding agent's arguments are file contents, so this is the common case and "
        "TARGET_CALL is the outlier — CONSTRAINED_DECODE.md §8.4."
    )

    # It is the *split* that costs, not the escape. `)\` + `n` leaves the parser
    # resting between the backslash and its escape character; `\n` as one token
    # never rests there and is free. Which happens is decided by the byte before
    # the newline, so both bodies below carry n newlines and only one pays.
    print(
        f"\n{'newlines':>9} {'split':>6} {'tokens':>7} {'total ms':>9} {'ms/split':>9}"
    )
    for label, body in (("f(x)\\n", "{}()"), ("word\\n", "line{}")):
        for n in (0, 1, 2, 4):
            call = {
                "name": "edit_file",
                "arguments": {
                    "path": "p.py",
                    "old": "x",
                    "new": "\n".join(body.format(i) for i in range(n + 1)),
                },
            }
            token_filter, inner, _ = load_filter(args.model)
            ids = list(inner.encode(json.dumps(call), add_special_tokens=False))
            splits = sum(
                1
                for i in range(len(ids))
                if inner.decode([ids[i]]).endswith("\\")
                and not inner.decode([ids[i]]).endswith("\\\\")
            )
            t0 = time.perf_counter()
            for i in range(len(ids)):
                token_filter(ids[:i])
            total = (time.perf_counter() - t0) * 1000
            print(
                f"{label:>9} {splits:>6} {len(ids):>7} {total:>9.1f} "
                f"{(total / splits if splits else 0):>9.1f}"
            )
    print(
        "\n~425 ms per split escape, and zero for the same newline that tokenizes "
        "whole. So the cost of a tool call is set by the bytes preceding its "
        "newlines, which is why TARGET_CALL measured nothing and the model's own "
        "output measured 9x."
    )


def cmd_statekey(args: argparse.Namespace) -> None:
    """Is the post-backslash state one state, and what would a cache key have to hold?

    `escape` shows the state costs ~425 ms and recurs. Whether a cache collapses it
    depends on something no measurement had: the two occurrences returned 5,798 and
    5,795 allowed tokens, so they are near-identical rather than provably identical.
    This compares the sets themselves, then counts how much a sound key would win.
    """
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from tandem.gateway.toolcall.constrain import Constrainer, tool_call_schema
    from tandem.types import ToolDef

    tok = load_tokenizer(args.model)
    inner = tok._tokenizer
    constrainer = Constrainer()
    t0 = time.perf_counter()
    vocabulary = constrainer.vocabulary(tok)
    print(f"vocabulary build {time.perf_counter() - t0:.2f} s, vocab {len(inner)}")
    schema = tool_call_schema(
        [
            ToolDef(
                name=t["name"], description=t["description"], parameters=t["parameters"]
            )
            for t in TOOLS_JSON
        ]
    )

    def walk(call: dict[str, Any]) -> list[tuple[int, frozenset[int], float]]:
        token_filter = constrainer.token_filter(schema, vocabulary)
        if token_filter is None:
            raise SystemExit("lm-format-enforcer unavailable")
        ids = list(inner.encode(json.dumps(call), add_special_tokens=False))
        hot = []
        for i in range(len(ids)):
            t = time.perf_counter()
            allowed = token_filter(ids[:i])
            dt = (time.perf_counter() - t) * 1000
            if dt > 100:
                hot.append((i, frozenset(allowed), dt))
        return hot

    def names(ids: Iterable[int]) -> list[str]:
        return [inner.decode([t]) for t in sorted(ids)]

    print("\n=== the two occurrences in one request ===")
    hot = walk(ESCAPED_CALL)
    for i, s, dt in hot:
        print(f"  step {i:>3}  |allowed|={len(s):>6}  {dt:8.1f} ms")
    a, b = hot[0][1], hot[1][1]
    print(f"\n  identical: {a == b}   |a & b| = {len(a & b)}")
    print(f"  only in the first  ({len(a - b)}): {names(a - b)!r}")
    print(f"  only in the second ({len(b - a)}): {names(b - a)!r}")

    # The escape sits in "old" the first time and "new" the second. If the difference
    # tracked the string content the two would differ arbitrarily; if it tracks the
    # object below, only the tokens that close the string and continue can differ.
    print("\n=== the same escape under three different stacks ===")
    variants: list[tuple[str, dict[str, Any]]] = [
        (
            "middle property",
            {
                "name": "edit_file",
                "arguments": {"path": "p.py", "old": "f()\n", "new": "y"},
            },
        ),
        (
            "last property, short",
            {
                "name": "edit_file",
                "arguments": {"path": "p.py", "old": "x", "new": "f()\n"},
            },
        ),
        (
            "last property, long",
            {
                "name": "edit_file",
                "arguments": {
                    "path": "p.py",
                    "old": "x",
                    "new": "f()" + "y" * 40 + "()\n",
                },
            },
        ),
    ]
    sets = {}
    for label, call in variants:
        h = walk(call)
        sets[label] = h[0][1]
        print(f"  {label:22s} |allowed|={len(h[0][1]):>6}  {h[0][2]:8.1f} ms")
    labels = list(sets)
    print()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            x, y = sets[labels[i]], sets[labels[j]]
            print(
                f"  {labels[i]:22s} vs {labels[j]:22s} same={x == y!s:5s} "
                f"differ by {len(x ^ y)} {names(x ^ y)!r}"
            )

    # Four split escapes inside one property: the twenty-line-edit shape, and the
    # only case where a cache could fire more than once.
    print("\n=== how much a sound key would collapse ===")
    hot = walk(
        {
            "name": "edit_file",
            "arguments": {
                "path": "p.py",
                "old": "x",
                "new": "\n".join(f"{i}()" for i in range(5)),
            },
        }
    )
    distinct: dict[frozenset[int], list[float]] = {}
    for _, s, dt in hot:
        distinct.setdefault(s, []).append(dt)
    for n, (s, occ) in enumerate(distinct.items(), 1):
        print(
            f"  set {n}: |{len(s)}| seen {len(occ)}x  "
            + " ".join(f"{v:.0f}" for v in occ)
        )
    paid = sum(dt for _, _, dt in hot)
    first = sum(occ[0] for occ in distinct.values())
    print(
        f"\n  {len(distinct)} distinct set(s) over {len(hot)} occurrences: {paid:.0f} ms paid, "
        f"{first:.0f} ms irreducible, {paid - first:.0f} ms ({(paid - first) / paid:.0%}) "
        "a stack-wide key would collapse.\n"
        "  A key on 'after a backslash' alone is unsound by the tokens above — they "
        "close the string and emit a comma, which is legal only while a property is "
        "still required. CONSTRAINED_DECODE.md §8.5."
    )


def cmd_components(args: argparse.Namespace) -> None:
    """Is each suspected cost real, and how big?"""
    import mlx.core as mx

    token_filter, inner, ids = load_filter(args.model)
    vocab_n = len(inner)

    big = None
    for i in range(len(ids)):
        a = token_filter(ids[:i])
        if len(a) > 200_000:
            big = list(a)
            break
    if big is None:
        raise SystemExit("no large-allowed-set state reached; is the schema right?")

    print("\n=== what does LMFE return? ===")
    print(f"type(allowed)      {type(big).__name__}")
    print(f"type(element)      {type(big[0]).__name__}")
    print(f"all already int    {all(type(x) is int for x in big[:5000])}")
    print(
        f"[int(t) for t ...] {bench(lambda: [int(t) for t in big if 0 <= int(t) < vocab_n]):6.2f} ms"
        f"   (|allowed|={len(big)})"
    )

    print("\n=== mask construction ===")
    arr = mx.array(big, dtype=mx.int32)
    mx.eval(arr)
    forbidden = mx.array(sorted(set(range(vocab_n)) - set(big)), dtype=mx.int32)
    mx.eval(forbidden)
    print(
        f"|forbidden| = {forbidden.size} ({forbidden.size / vocab_n * 100:.2f}% of vocab)"
    )

    def current() -> None:
        m = mx.full((vocab_n,), -float("inf"), dtype=mx.float16)
        m[mx.array(big, dtype=mx.int32)] = 0.0
        mx.eval(m)

    def prebuilt() -> None:
        m = mx.full((vocab_n,), -float("inf"), dtype=mx.float16)
        m[arr] = 0.0
        mx.eval(m)

    def complement() -> None:
        m = mx.zeros((vocab_n,), dtype=mx.float16)
        m[forbidden] = -float("inf")
        mx.eval(m)

    print(f"current (mx.array each token) {bench(current):6.2f} ms")
    print(f"prebuilt id array             {bench(prebuilt):6.2f} ms")
    print(f"complement scatter            {bench(complement):6.2f} ms")

    print("\n=== are the expensive LMFE calls per-token or one-time? ===")
    for i in range(len(ids)):
        a = token_filter(ids[:i])
        if 100 < len(a) < 1000:
            t = bench(lambda i=i: token_filter(ids[:i]), n=5)
            print(f"  step {i:>3} |allowed|={len(a):>5} repeat {t:6.2f} ms")
    print(
        "\nA cheap repeat means the expensive first call was a one-time parser-state "
        "build, so it is a per-request cost, not a per-token one."
    )


def cmd_identity(args: argparse.Namespace) -> None:
    """Can a mask cache key on id()? `BASELINE.md` §2.3's rejected cache assumed yes."""
    token_filter, _inner, ids = load_filter(args.model)

    prev: Any = None
    same_obj = same_content = n = 0
    print(f"\n{'step':>4} {'|allowed|':>10} {'same object':>12} {'same content':>13}")
    for i in range(len(ids)):
        a = token_filter(ids[:i])
        if prev is not None:
            n += 1
            same_obj += a is prev
            same_content += a == prev
            if i < 26:
                print(f"{i:>4} {len(a):>10} {a is prev!s:>12} {a == prev!s:>13}")
        prev = a

    print(f"\nover {n} consecutive pairs:")
    print(f"  identical object   {same_obj:>3}/{n}  ({same_obj / n * 100:.0f}%)")
    print(
        f"  identical content  {same_content:>3}/{n}  ({same_content / n * 100:.0f}%)"
    )
    print(f"\nid()-keyed cache: {'WORKS' if same_obj else 'IMPOSSIBLE'}")


def cmd_key(args: argparse.Namespace) -> None:
    """What a *sound* cache key costs, given id() is unavailable."""
    import mlx.core as mx

    token_filter, inner, ids = load_filter(args.model)
    vocab_n = len(inner)
    width = args.width or vocab_n

    # Two *consecutive* LMFE outputs, not a copy. `list(allowed)` shares element
    # objects, so `==` short-circuits on identity per element and reads ~7x fast —
    # a number the real cache would never see, since its two lists come from
    # separate LMFE calls.
    allowed = same = None
    for i in range(len(ids) - 1):
        a, b = token_filter(ids[:i]), token_filter(ids[: i + 1])
        if len(a) > 200_000 and a == b:
            allowed, same = list(a), list(b)
            break
    if allowed is None or same is None:
        raise SystemExit("no consecutive large-allowed-set pair reached")

    diff = list(same)
    diff[-1] = 0

    print(f"\n=== per-token costs, |allowed| = {len(allowed)} ===")
    conv = bench(lambda: [int(t) for t in allowed if 0 <= int(t) < width])
    print(f"  [int(t) for t in a ...]        {conv:6.2f} ms   <- removable")
    print(
        f"  mx.array(a)                    "
        f"{bench(lambda: mx.eval(mx.array(allowed, dtype=mx.int32))):6.2f} ms"
    )

    print("\n=== sound cache keys ===")
    print(
        f"  len(a) == len(prev)            {bench(lambda: len(allowed) == len(same)):6.4f} ms"
    )
    hit = bench(lambda: allowed == same)
    miss = bench(lambda: allowed == diff)
    print(f"  a == prev  (hit)               {hit:6.2f} ms")
    print(f"  a == prev  (miss)              {miss:6.2f} ms")

    def rebuild() -> None:
        m = mx.full((width,), -float("inf"), dtype=mx.float16)
        m[mx.array(allowed, dtype=mx.int32)] = 0.0
        mx.eval(m)

    build_ms = bench(rebuild)
    print("\n=== per string-interior token ===")
    print(f"  current                        {conv + build_ms:6.2f} ms")
    print(f"  fixed, cache miss              {miss + build_ms:6.2f} ms")
    print(f"  fixed, cache hit               {hit:6.2f} ms")


def cmd_reuse(args: argparse.Namespace) -> None:
    """What a second request re-pays, and whether sharing anything recovers it.

    F3 proposed sharing the schema-derived parser across requests to recover
    ~78 ms/request of "one-time parser-state builds". This measures the three
    candidates separately, because the design named a part that turns out not to be
    where the cost is.
    """
    import lmformatenforcer
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from tandem.gateway.toolcall.constrain import Constrainer, tool_call_schema
    from tandem.types import ToolDef

    tok = load_tokenizer(args.model)
    inner = tok._tokenizer
    vocabulary = Constrainer().vocabulary(tok)
    schema = tool_call_schema(
        [
            ToolDef(
                name=t["name"], description=t["description"], parameters=t["parameters"]
            )
            for t in TOOLS_JSON
        ]
    )
    parser_ms = bench(lambda: lmformatenforcer.JsonSchemaParser(schema))
    root = lmformatenforcer.JsonSchemaParser(schema)
    enforcer_ms = bench(lambda: lmformatenforcer.TokenEnforcer(vocabulary, root))
    print("\n=== construction, the part F3 proposed sharing ===")
    print(f"  JsonSchemaParser(schema)       {parser_ms:8.3f} ms")
    print(f"  TokenEnforcer(vocab, parser)   {enforcer_ms:8.3f} ms")
    print(f"  root parser cache_key()        {root.cache_key()}")

    # Both fixtures, because the original measurement used only the first and the
    # first contains no escape sequence — the state that turns out to cost 425 ms
    # (§8.4). A conclusion about reuse drawn from TARGET_CALL alone is a conclusion
    # about a workload a coding agent does not produce.
    for call_label, call in (
        ("TARGET_CALL, no escape", TARGET_CALL),
        ("ESCAPED_CALL", ESCAPED_CALL),
    ):
        ids = list(inner.encode(json.dumps(call), add_special_tokens=False))

        def one_request(
            enforcer: Any, prefix: list[int], ids: list[int] = ids
        ) -> list[float]:
            out = []
            for i in range(len(ids)):
                t = time.perf_counter()
                enforcer.get_allowed_tokens(prefix + ids[:i])
                out.append((time.perf_counter() - t) * 1e3)
            return out

        print(f"\n### {call_label} — {len(ids)} tokens")
        print("  a fresh enforcer per request, as constrain.py builds one:")
        for label, prefix in (("request 1", [1, 2, 3]), ("request 2", [4, 5, 6])):
            e = lmformatenforcer.TokenEnforcer(
                vocabulary, lmformatenforcer.JsonSchemaParser(schema)
            )
            ms = one_request(e, prefix)
            top = " ".join(f"{v:.1f}" for v in sorted(ms, reverse=True)[:4])
            print(
                f"    {label}  total {sum(ms):7.1f} ms   top4 {top}   "
                f"allowed_token_cache {len(e.allowed_token_cache)}"
            )

        shared = lmformatenforcer.TokenEnforcer(
            vocabulary, lmformatenforcer.JsonSchemaParser(schema)
        )
        first = sum(one_request(shared, [7, 8, 9]))
        second = sum(one_request(shared, [10, 11, 12]))
        print("  one enforcer shared across both, which F3 forbids and this tests:")
        print(
            f"    {first:7.1f} -> {second:7.1f} ms      "
            f"prefix_states {len(shared.prefix_states)} (was {len(ids)} after one)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("loop", help="synthetic decode loop, no tokenizer")
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--allowed", type=int, default=4000)
    p.add_argument("--host-ms", type=float, default=6.0)
    p.set_defaults(fn=cmd_loop)

    for name, fn, helptext in [
        ("filter", cmd_filter, "per-token cost across real parser states"),
        ("escape", cmd_escape, "what one backslash in a string argument costs"),
        ("statekey", cmd_statekey, "is the post-backslash state one state"),
        ("components", cmd_components, "is each suspected cost real"),
        ("identity", cmd_identity, "can a mask cache key on id()"),
        ("key", cmd_key, "what a sound cache key costs"),
        ("reuse", cmd_reuse, "what a second request re-pays"),
    ]:
        p = sub.add_parser(name, help=helptext)
        if name == "filter":
            p.add_argument(
                "--per-token",
                action="store_true",
                help="dump every token's cost, not just the per-state summary",
            )
        if name == "key":
            p.add_argument(
                "--width",
                type=int,
                default=None,
                help="model logit width; defaults to the tokenizer vocab",
            )
        p.set_defaults(fn=fn)

    args = ap.parse_args()
    if args.model is None:
        args.model = default_model()
    args.fn(args)


if __name__ == "__main__":
    main()
