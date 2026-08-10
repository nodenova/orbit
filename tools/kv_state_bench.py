"""Does restoring a tier-0 KV state actually save anything? (HANDOFF 3.8, HANDOFF step 1.)

`supports_state()` is True and the whole loop is green off-target. That proves the
wiring. It says nothing about the two numbers that decide whether the feature is worth
having, and both need the real container:

  * **Does a real follow-up turn restore at all?** The state carries token ids and the
    backend refuses unless they are a prefix of the next prompt. Whether they are is a
    property of the *chat template*, not of this code.
  * **What does a restore save when it fires?** Prefill is the whole of TTFT at long
    context, so the prize is bounded by prefill time minus the cost of reading the blob
    back.

Three scenarios, because the first two alone are uninterpretable:

  * **continuation** -- the product scenario. Turn 1, then turn 2 carrying turn 1's
    reply. This is what a coding harness does on every keystroke.
  * **best case** -- a prompt whose ids genuinely begin with the state's, built by
    rendering the continuation the way a template with no generation-prompt tail would.
    Isolates the cache from the template question and measures the prize.
  * **cold** -- the same prompt with no state. The baseline both are read against.

Reports prefill separately from decode: a single tok/s over a short generation is mostly
prefill and would hide the entire effect.

Lives in `tools/` rather than the package: it is a measurement, not a shipped feature.
It loads tier 0 (23.0 GiB on the mlx backend) -- read `docs/PROCESSES.md` first, and
climb the rungs. `--rung 1` loads and reports footprint and stops.

    python tools/kv_state_bench.py --rung 1
    python tools/kv_state_bench.py --rung 3 --out var/kv-state-bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tandem.backends import build_tier0
from tandem.config import Config
from tandem.types import GenRequest, GenResult, Message, Role, Sampling

# A body with the shape of the thing being reviewed. Fixed, because a corpus that
# drifts between runs measures the corpus.
_UNIT = '''
def paginate(items, limit=None, offset=0):
    """Return a page of `items`, clamped to the configured maximum."""
    if limit is None:
        limit = DEFAULT_PAGE
    limit = min(max(1, int(limit)), MAX_PAGE)
    offset = max(0, int(offset))
    window = items[offset : offset + limit]
    return Page(items=window, total=len(items), offset=offset, limit=limit)

'''

GIB = 1 << 30


def body(target_tokens: int, backend: Any) -> str:
    """A prompt body of roughly `target_tokens`, measured rather than assumed."""
    per_unit = int(backend.count_tokens(_UNIT))
    return _UNIT * max(1, target_tokens // max(1, per_unit))


def memory(mx: Any) -> dict[str, float]:
    info = mx.device_info()
    return {
        "active_gib": mx.get_active_memory() / GIB,
        "peak_gib": mx.get_peak_memory() / GIB,
        "ceiling_gib": info["max_recommended_working_set_size"] / GIB,
    }


def pageouts() -> int:
    """`Pageouts`, which is what thrashing looks like. Not `vm.swapusage used`."""
    import subprocess

    out = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, check=False
    ).stdout
    for line in out.splitlines():
        if line.startswith("Pageouts:"):
            return int(line.split(":")[1].strip().rstrip("."))
    return -1


async def run_once(backend: Any, req: GenRequest, state: Any = None) -> dict[str, Any]:
    """One generation, with prefill and decode separated."""
    if state is not None:
        req = req.with_(warm_state=state)
    started = time.perf_counter()
    ttft = 0.0
    parts: list[str] = []
    result = None
    async for delta in backend.stream(req):
        if delta.text:
            if not parts:
                ttft = time.perf_counter() - started
            parts.append(delta.text)
        if delta.done:
            result = delta.result
    total = time.perf_counter() - started
    assert result is not None
    result.text = "".join(parts)
    decode_s = max(1e-9, total - ttft)
    out = {
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "prompt_tokens": result.usage.input_tokens,
        "cached_tokens": result.usage.cached_input_tokens,
        "output_tokens": result.usage.output_tokens,
        "prefill_tok_per_s": round(
            (result.usage.input_tokens - result.usage.cached_input_tokens)
            / max(1e-9, ttft),
            1,
        ),
        "decode_tok_per_s": round(result.usage.output_tokens / decode_s, 1),
        "text": result.text,
        "handle": result.kv_handle,
    }
    return out


def _shared_chars(a: str, b: str) -> int:
    n = 0
    for left, right in zip(a, b):
        if left != right:
            break
        n += 1
    return n


def conversation(opening: str, reply: str, follow_up: str) -> GenRequest:
    return GenRequest(
        messages=[
            Message(role=Role.USER, content=opening),
            Message(role=Role.ASSISTANT, content=reply),
            Message(role=Role.USER, content=follow_up),
        ],
        sampling=Sampling(temperature=0.0, top_p=1.0, seed=0, max_tokens=24),
    )


async def scenario(backend: Any, target_tokens: int) -> dict[str, Any]:
    """Turn 1, then the same turn 2 three ways: continuation, best case, cold."""
    opening = f"Review this module and name the single worst bug.\n\n{body(target_tokens, backend)}"
    first = GenRequest(
        messages=[Message(role=Role.USER, content=opening)],
        sampling=Sampling(temperature=0.0, top_p=1.0, seed=0, max_tokens=24),
    )

    turn1 = await run_once(backend, first)
    rendered1 = backend.render(first)
    state = backend.export_state(
        first, rendered1, GenResult(kv_handle=turn1.pop("handle"))
    )

    follow_up = conversation(
        opening, turn1["text"], "Now write the test that catches it."
    )
    rendered2 = backend.render(follow_up)

    cold = await run_once(backend, follow_up)
    cold.pop("handle")
    warm = await run_once(backend, follow_up, state=state)
    warm.pop("handle")

    # What the template did to the shared prefix, which is what decides the above.
    ids1, ids2 = backend._encode(rendered1), backend._encode(rendered2)
    shared = 0
    for a, b in zip(ids1, ids2):
        if a != b:
            break
        shared += 1

    out = {
        "target_tokens": target_tokens,
        "turn1": {k: v for k, v in turn1.items() if k != "text"},
        "state_bytes": len(state.blob) if state else 0,
        "state_tokens": state.n_tokens if state else 0,
        "template": {
            "turn1_prompt_tokens": len(ids1),
            "turn2_prompt_tokens": len(ids2),
            "shared_prefix_tokens": shared,
            "turn1_is_a_prefix_of_turn2": shared == len(ids1),
            # Sliced off the rendered strings rather than decoded back from the
            # ids: it is the template's own bytes that are in question here, and a
            # detokenized approximation of them would be the wrong evidence.
            "turn1_tail_not_shared": rendered1[_shared_chars(rendered1, rendered2) :],
        },
        "continuation": {k: v for k, v in warm.items() if k != "text"},
        "cold": {k: v for k, v in cold.items() if k != "text"},
        "identical_answer": warm["text"] == cold["text"],
    }

    # The prize, isolated from the template: a prompt whose ids really do begin with
    # the state's. Rendered the way a template with no generation-prompt tail would.
    if state is not None:
        best = await _best_case(backend, follow_up, state, rendered1, turn1)
        if best is not None:
            out["best_case"] = best
    return out


async def _best_case(
    backend: Any,
    follow_up: GenRequest,
    state: Any,
    rendered1: str,
    turn1: dict[str, Any],
) -> dict[str, Any] | None:
    """Time a restore that genuinely fires, by rendering around the template.

    The continuation number above is a property of the chat template. This one is a
    property of the cache, and it is the ceiling the template question is worth
    arguing about.
    """
    tail = "<|im_end|>\n<|im_start|>user\nNow write the test.<|im_end|>\n<|im_start|>assistant\n"
    rendered = rendered1 + turn1["text"] + tail
    ids = backend._encode(rendered)
    if list(state.token_ids) != ids[: state.n_tokens]:
        return {
            "fired": False,
            "why": "state ids are not a prefix of the hand-rendered prompt",
        }

    original = backend.render
    backend.render = lambda req, render_tool_call=None: rendered
    try:
        warm = await run_once(backend, follow_up, state=state)
        cold = await run_once(backend, follow_up)
        cold_repeat = await run_once(backend, follow_up)
    finally:
        backend.render = original
    warm.pop("handle")
    cold.pop("handle")
    cold_repeat.pop("handle")
    saved = cold["ttft_s"] - warm["ttft_s"]
    return {
        "fired": warm["cached_tokens"] > 0,
        # The null for `identical_answer`. A restore is supposed to reproduce the
        # cold answer, but "warm differed from cold" only means something if cold
        # equals cold — greedy decode over a long prompt is not obviously
        # reproducible, and blaming the cache for the model's own variance is the
        # same mistake as quoting a rate without it.
        "cold_is_reproducible": cold_repeat["text"] == cold["text"],
        "cold_repeat_ttft_s": cold_repeat["ttft_s"],
        "warm": {k: v for k, v in warm.items() if k != "text"},
        "cold": {k: v for k, v in cold.items() if k != "text"},
        "ttft_saved_s": round(saved, 3),
        "ttft_speedup": round(cold["ttft_s"] / max(1e-9, warm["ttft_s"]), 2),
        "identical_answer": warm["text"] == cold["text"],
    }


def _legacy_probe(backend: Any) -> dict[str, Any]:
    """Decode rate through the call shape tier 0 used *before* KV state landed.

    A string prompt and no `prompt_cache`, which is what `stream_generate` was handed
    when BASELINE recorded ~65 tok/s. If this and `decode_probe` agree, a slow decode
    is the model on this host; if they disagree, the KV state work is a regression and
    that is the finding.
    """
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = backend.render(
        GenRequest(
            messages=[Message(role=Role.USER, content="Count from one to forty.")]
        )
    )
    started = time.perf_counter()
    ttft = 0.0
    n = 0
    for step in stream_generate(
        backend.model,
        backend.tokenizer,
        prompt=prompt,
        max_tokens=96,
        sampler=make_sampler(temp=0.0, top_p=1.0),
    ):
        if not n:
            ttft = time.perf_counter() - started
        n += 1
    total = time.perf_counter() - started
    return {
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "output_tokens": n,
        "decode_tok_per_s": round(n / max(1e-9, total - ttft), 1),
    }


async def amain(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config) if args.config else Config()
    before = pageouts()
    t0 = time.perf_counter()
    backend: Any = build_tier0(cfg)
    load_s = time.perf_counter() - t0
    mx = backend.mx

    report: dict[str, Any] = {
        "container": cfg.tier0.container_path or cfg.tier0.model,
        "load_s": round(load_s, 1),
        "after_load": {k: round(v, 2) for k, v in memory(mx).items()},
    }

    # Rung 1: one warm-up (Metal kernel compilation, ~9 s) then one real generation.
    warm_req = GenRequest(
        messages=[Message(role=Role.USER, content="Say ok.")],
        sampling=Sampling(temperature=0.0, max_tokens=8),
    )
    first = await run_once(backend, warm_req)
    first.pop("handle")
    second = await run_once(backend, warm_req)
    second.pop("handle")
    report["warmup"] = {"cold_call": first, "warm_call": second}
    report["after_generate"] = {k: round(v, 2) for k, v in memory(mx).items()}

    headroom = (
        report["after_generate"]["ceiling_gib"] - report["after_generate"]["peak_gib"]
    )
    report["headroom_gib"] = round(headroom, 2)
    if args.rung == 1 or headroom < args.min_headroom_gib:
        report["stopped_at_rung"] = 1
        report["reason"] = (
            "rung 1 requested"
            if args.rung == 1
            else f"headroom {headroom:.2f} GiB below the {args.min_headroom_gib} GiB floor"
        )
        report["pageouts_delta"] = pageouts() - before
        _emit(report, args)
        return 0

    # A decode number over 8 tokens is mostly the detokenizer's first flush, not the
    # model. BASELINE quotes ~65 tok/s unconstrained; anything far off that is a
    # measurement bug before it is a hardware finding, so it gets its own window.
    probe = await run_once(
        backend,
        GenRequest(
            messages=[Message(role=Role.USER, content="Count from one to forty.")],
            sampling=Sampling(temperature=0.0, max_tokens=96),
        ),
    )
    probe.pop("handle")
    probe.pop("text")
    report["decode_probe"] = probe
    report["decode_probe_legacy"] = _legacy_probe(backend)

    sizes = args.sizes or ([512, 2048] if args.rung == 2 else [512, 2048, 8192])
    report["scenarios"] = []
    for size in sizes:
        report["scenarios"].append(await scenario(backend, size))
        report["after_scenario"] = {k: round(v, 2) for k, v in memory(mx).items()}
        if (
            report["after_scenario"]["peak_gib"]
            > report["after_generate"]["ceiling_gib"] - 1.0
        ):
            report["aborted"] = f"peak within 1 GiB of the ceiling after {size} tokens"
            break

    report["stopped_at_rung"] = args.rung
    report["pageouts_delta"] = pageouts() - before
    _emit(report, args)
    return 0


def _emit(report: dict[str, Any], args: argparse.Namespace) -> None:
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="tandem.toml")
    ap.add_argument("--rung", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--min-headroom-gib", type=float, default=2.0)
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
