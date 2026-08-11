"""What is Qwen3-Coder-Next actually worth as a generator? (platform.md §4.8.)

Everything recorded about this model so far measures it as a *verifier*: Gate B reads
prefill rate, and `qcn_streambench.py` reads the SSD. Neither looks at an answer. This
asks the three questions that decide whether it can drive an agent loop instead --
does it answer correctly, does it emit tool calls, and how fast does it generate.

Three modes, because they fail independently and a single number hides which one broke:

  * **suite** -- the sec 10.3 regression items, graded by `eval.regression.grade`, run
    against the engine over HTTP. The set is fixed and a tier-0 baseline already exists
    at `baselines/regression-baseline.json`, so the interesting output is not a score
    but a per-item *diff against tier 0*: same items, same grader, same greedy sampling.
  * **toolcall** -- `eval.gates.SCENARIO_TOOLS` and `SCENARIO_STEPS`, the sec 10.2
    scenario, so the rate compares row-for-row with tier 0's measured 100/100.
  * **throughput** -- TTFT, prefill and decode measured separately off a streamed
    response at several context sizes. A single tok/s over a short generation is mostly
    prefill and says nothing about either.

**The suite is a regression detector, not a leaderboard.** `eval.regression` refuses to
hold a score field for that reason, and this tool prints a rate only beside the tier-0
rate it is a comparison against. Do not quote either alone.

Two engine behaviours this had to be written around, both of which silently corrupt a
measurement rather than failing:

  * `optiq serve` applies a **model-recommended sampler** (temperature 1.0, top_p 0.95,
    top_k 40) to any request that does not pin one. The tier-0 baseline is greedy, so
    every request here pins temperature 0.0 / top_p 1.0 / seed 0 to match
    `eval.regression.run`. Unpinned, this measures the sampler.
  * The model loads on the **first request**, not at startup, and the first generation
    after a load also pays Metal kernel compilation. Every mode discards a warm-up call.

Lives in `tools/` rather than the package: it is a measurement, and it makes outbound
HTTP calls, which nothing under `src/orbit/` may do.

    python tools/quality/qcn_quality.py suite --out var/qcn-suite.json
    python tools/quality/qcn_quality.py toolcall --runs 100 --out var/qcn-toolcall.json
    python tools/quality/qcn_quality.py throughput --out var/qcn-throughput.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orbit.eval.gates import SCENARIO_STEPS, SCENARIO_TOOLS
from orbit.eval.regression import grade
from orbit.eval.regression_items import SUITE, Item

DEFAULT_ENDPOINT = "http://127.0.0.1:8081/v1"
# The engine serves `--model` whatever this says; it is a label, not a selector.
DEFAULT_MODEL = "qwen3-coder-next"

# Matches `eval.regression.run` exactly. Greedy, because a sampled answer is not a
# stable baseline and the tier-0 side of the comparison was recorded this way.
SUITE_SAMPLING: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "seed": 0}


def _openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in SCENARIO_TOOLS
    ]


@dataclass(frozen=True, slots=True)
class Call:
    """One completed request."""

    body: dict[str, Any]
    wall_s: float

    @property
    def text(self) -> str:
        choices = self.body.get("choices") or [{}]
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    @property
    def reasoning(self) -> str:
        """Whatever the engine called a thinking block, under either spelling.

        Read for the same reason `tier1_call.refuse_reasoned_answer` reads it (§3.11):
        an engine that answers in a reasoning field and leaves `content` empty grades as
        a wrong answer on every item, which looks like a bad model rather than a
        misread wire format.
        """
        choices = self.body.get("choices") or [{}]
        message = choices[0].get("message") or {}
        return str(message.get("reasoning") or message.get("reasoning_content") or "")

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        choices = self.body.get("choices") or [{}]
        message = choices[0].get("message") or {}
        calls = message.get("tool_calls") or []
        return [c for c in calls if isinstance(c, dict)]

    @property
    def prompt_tokens(self) -> int:
        return int((self.body.get("usage") or {}).get("prompt_tokens") or 0)

    @property
    def completion_tokens(self) -> int:
        return int((self.body.get("usage") or {}).get("completion_tokens") or 0)


def call(
    client: httpx.Client,
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    sampling: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> Call:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **(sampling or SUITE_SAMPLING),
    }
    if tools:
        payload["tools"] = tools
    started = time.perf_counter()
    response = client.post(f"{endpoint}/chat/completions", json=payload)
    response.raise_for_status()
    return Call(response.json(), time.perf_counter() - started)


def warm_up(client: httpx.Client, endpoint: str, model: str) -> float:
    """One discarded call, so the load and the Metal kernel compile are not in a number."""
    started = time.perf_counter()
    call(client, endpoint, model, [{"role": "user", "content": "hi"}], max_tokens=8)
    return time.perf_counter() - started


# --- suite ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ItemRecord:
    id: str
    category: str
    passed: bool
    response: str
    reasoning_chars: int
    wall_s: float
    prompt_tokens: int
    completion_tokens: int


def run_suite(
    client: httpx.Client,
    endpoint: str,
    model: str,
    items: tuple[Item, ...],
    *,
    max_tokens: int,
) -> list[ItemRecord]:
    out: list[ItemRecord] = []
    for n, item in enumerate(items, 1):
        try:
            result = call(
                client,
                endpoint,
                model,
                [{"role": "user", "content": item.prompt}],
                max_tokens=max_tokens,
            )
        except (httpx.HTTPError, ValueError) as exc:
            print(f"  [{n}/{len(items)}] {item.id}: request failed: {exc}", flush=True)
            out.append(ItemRecord(item.id, item.category, False, "", 0, 0.0, 0, 0))
            continue
        passed = grade(item, result.text)
        out.append(
            ItemRecord(
                item.id,
                item.category,
                passed,
                result.text,
                len(result.reasoning),
                result.wall_s,
                result.prompt_tokens,
                result.completion_tokens,
            )
        )
        print(
            f"  [{n}/{len(items)}] {item.id} {'pass' if passed else 'FAIL'} "
            f"{result.wall_s:.2f}s",
            flush=True,
        )
    return out


def load_baseline(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    items = data.get("items") or {}
    return {str(k): bool(v) for k, v in items.items()}


def summarise_suite(
    records: list[ItemRecord], baseline: dict[str, bool]
) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    for r in records:
        bucket = by_category.setdefault(r.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(r.passed)

    both_ran = [r for r in records if r.id in baseline]
    agree = sum(1 for r in both_ran if r.passed == baseline[r.id])
    only_qcn = sorted(r.id for r in both_ran if r.passed and not baseline[r.id])
    only_tier0 = sorted(r.id for r in both_ran if not r.passed and baseline[r.id])

    passed = sum(1 for r in records if r.passed)
    return {
        "items": len(records),
        "passed": passed,
        "rate": passed / len(records) if records else 0.0,
        "by_category": {
            k: {**v, "rate": v["passed"] / v["total"] if v["total"] else 0.0}
            for k, v in sorted(by_category.items())
        },
        "tier0_baseline": {
            "compared": len(both_ran),
            "tier0_passed": sum(1 for r in both_ran if baseline[r.id]),
            "tier0_rate": (
                sum(1 for r in both_ran if baseline[r.id]) / len(both_ran)
                if both_ran
                else 0.0
            ),
            "agreement": agree / len(both_ran) if both_ran else 0.0,
            "passed_only_by_qcn": only_qcn,
            "passed_only_by_tier0": only_tier0,
        },
        "latency_s": {
            "median": statistics.median([r.wall_s for r in records])
            if records
            else 0.0,
            "max": max((r.wall_s for r in records), default=0.0),
        },
        "answered_in_reasoning_field": sum(
            1 for r in records if r.reasoning_chars > 0 and not r.response.strip()
        ),
    }


# --- toolcall ---------------------------------------------------------------


@dataclass
class ToolCallReport:
    runs: int = 0
    wellformed: int = 0
    no_call_attempted: int = 0
    bad_json: int = 0
    unknown_name: int = 0
    missing_required: int = 0
    request_failed: int = 0
    names: dict[str, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.wellformed / self.runs if self.runs else 0.0


def _validate(calls: list[dict[str, Any]], report: ToolCallReport) -> bool:
    known = {t.name: t for t in SCENARIO_TOOLS}
    if not calls:
        report.no_call_attempted += 1
        return False
    for c in calls:
        fn = c.get("function") or {}
        name = str(fn.get("name") or "")
        report.names[name] = report.names.get(name, 0) + 1
        if name not in known:
            report.unknown_name += 1
            return False
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            report.bad_json += 1
            return False
        if not isinstance(args, dict):
            report.bad_json += 1
            return False
        required = (known[name].parameters or {}).get("required") or []
        if any(key not in args for key in required):
            report.missing_required += 1
            return False
    return True


def run_toolcall(
    client: httpx.Client, endpoint: str, model: str, *, runs: int, max_tokens: int
) -> ToolCallReport:
    report = ToolCallReport()
    tools = _openai_tools()
    for i in range(runs):
        step = SCENARIO_STEPS[i % len(SCENARIO_STEPS)]
        report.runs += 1
        try:
            result = call(
                client,
                endpoint,
                model,
                [{"role": "user", "content": step}],
                max_tokens=max_tokens,
                # The gate varies the seed so it samples behaviour rather than
                # measuring one generation `runs` times.
                sampling={"temperature": 0.2, "top_p": 1.0, "seed": i},
                tools=tools,
            )
        except (httpx.HTTPError, ValueError) as exc:
            report.request_failed += 1
            print(f"  [{i + 1}/{runs}] request failed: {exc}", flush=True)
            continue
        ok = _validate(result.tool_calls, report)
        report.wellformed += int(ok)
        print(
            f"  [{i + 1}/{runs}] {'ok' if ok else 'BAD'} "
            f"{len(result.tool_calls)} call(s) {result.wall_s:.2f}s",
            flush=True,
        )
    return report


# --- throughput -------------------------------------------------------------

_FILLER = (
    "def process_record_{n}(payload, *, strict=False):\n"
    "    result = validate_schema(payload, strict=strict)\n"
    "    return normalise_identifier(result, index={n})\n\n"
)


def _prompt_of_tokens(target: int) -> str:
    """Filler sized in tokens rather than words.

    §5 of the write-up: this tokenizer runs ~11.8 tokens per generated identifier, and
    a probe that sized prompts in words sent 52,008 tokens at a 32,768 cap and took the
    engine down with it. Approximated at ~4 chars/token and verified against the
    engine's own `prompt_tokens` in the result.
    """
    body = "".join(_FILLER.format(n=i) for i in range(target // 6 + 8))
    return body[: target * 4]


@dataclass(frozen=True, slots=True)
class ThroughputRow:
    label: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float
    prefill_tok_per_s: float
    decode_tok_per_s: float


def measure_throughput(
    client: httpx.Client,
    endpoint: str,
    model: str,
    *,
    target_tokens: int,
    max_tokens: int,
    label: str,
) -> ThroughputRow:
    """TTFT off the stream, so prefill and decode are separated rather than averaged."""
    prompt = (
        _prompt_of_tokens(target_tokens) if target_tokens else "Count from 1 to 40."
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"{prompt}\n\nSummarise what the code above does, in one paragraph."
                if target_tokens
                else prompt
            ),
        }
    ]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **SUITE_SAMPLING,
    }
    started = time.perf_counter()
    ttft = 0.0
    completion = 0
    prompt_tokens = 0
    with client.stream("POST", f"{endpoint}/chat/completions", json=payload) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            chunk = line[6:].strip()
            if chunk == "[DONE]":
                break
            try:
                event = json.loads(chunk)
            except ValueError:
                continue
            usage = event.get("usage") or {}
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                completion = int(usage.get("completion_tokens") or completion)
            choices = event.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                if not ttft:
                    ttft = time.perf_counter() - started
                completion += 1
    total = time.perf_counter() - started
    decode_window = max(total - ttft, 1e-9)
    return ThroughputRow(
        label=label,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion,
        ttft_s=ttft,
        total_s=total,
        prefill_tok_per_s=prompt_tokens / ttft if ttft else 0.0,
        decode_tok_per_s=completion / decode_window,
    )


# --- cli --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("suite", "toolcall", "throughput"))
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="suite: first N items only")
    ap.add_argument("--runs", type=int, default=20, help="toolcall: how many runs")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument(
        "--baseline", type=Path, default=Path("baselines/regression-baseline.json")
    )
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    client = httpx.Client(timeout=args.timeout)
    payload: dict[str, Any]
    with client:
        warm_s = warm_up(client, args.endpoint, args.model)
        print(f"warm-up (discarded): {warm_s:.2f}s", flush=True)

        if args.mode == "suite":
            items = tuple(SUITE[: args.limit]) if args.limit else tuple(SUITE)
            records = run_suite(
                client,
                args.endpoint,
                args.model,
                items,
                max_tokens=args.max_tokens or 96,
            )
            summary = summarise_suite(records, load_baseline(args.baseline))
            payload = {
                "mode": "suite",
                "model": args.model,
                "summary": summary,
                "items": [asdict(r) for r in records],
            }
            print(json.dumps(summary, indent=2))

        elif args.mode == "toolcall":
            report = run_toolcall(
                client,
                args.endpoint,
                args.model,
                runs=args.runs,
                max_tokens=args.max_tokens or 256,
            )
            payload = {
                "mode": "toolcall",
                "model": args.model,
                "summary": {**asdict(report), "rate": report.rate},
            }
            print(json.dumps(payload["summary"], indent=2))

        else:
            rows = [
                measure_throughput(
                    client,
                    args.endpoint,
                    args.model,
                    target_tokens=n,
                    max_tokens=args.max_tokens or 128,
                    label=label,
                )
                for label, n in (
                    ("short", 0),
                    ("2k", 2000),
                    ("8k", 8000),
                    ("16k", 16000),
                )
            ]
            payload = {
                "mode": "throughput",
                "model": args.model,
                "rows": [asdict(r) for r in rows],
            }
            for r in rows:
                print(
                    f"  {r.label:>6}: {r.prompt_tokens:>6} in  "
                    f"ttft {r.ttft_s:6.2f}s  "
                    f"prefill {r.prefill_tok_per_s:7.1f} tok/s  "
                    f"decode {r.decode_tok_per_s:5.2f} tok/s",
                    flush=True,
                )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
