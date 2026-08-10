"""Does rung 3 ever disagree with tier 0's own top candidate? (HANDOFF 4.2a, T5.)

`Cascade._code_change_turn` falls back to candidate 0 when the rerank fails, because
that is "what a no-tier-1 install would have produced anyway". So the value of the whole
tier-1 apparatus, on any rung, is bounded by how often reranking moves off candidate 0.
Near-100% agreement means rung 3 is measuring nothing and an independent verifier is
urgent; low agreement means a 92.83 GB download buys a decorrelation nobody needs yet.

Three quantities, because the headline one is uninterpretable alone:

  * **disagreement** -- how often `choice != 0`. The number T5 asks for.
  * **candidate diversity** -- how many of the N candidates are even distinct. Best-of-N
    over five identical patches is a no-op whatever the verifier says, and that failure
    looks exactly like agreement.
  * **positional stability** -- the same candidates reranked in a rotated order. A
    verifier tracking content picks the same *candidate*; one tracking position picks
    the same *index*. Without this control a high agreement rate cannot be told apart
    from a reranker that always answers 0.

On a host with no `adapters/`, `SecondOpinionBackend._strip` is a no-op -- `req.adapter`
is already None -- so rung 3 is tier 0 judging its own samples with identical weights.
That is the degenerate case, and its number is the floor for every other rung.

Lives in `tools/` rather than the package: it is a measurement, not a shipped feature.
It loads tier 0 (23.0 GiB on the mlx backend) -- read `docs/PROCESSES.md` first.

    python tools/rung3_agreement.py --config tandem.toml --prompts 12 --candidates 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tandem.backends import build_tier0, build_tier1
from tandem.config import Config
from tandem.router.cascade import _context_of
from tandem.tier1.verifier import Candidate, Tier1Verifier
from tandem.types import GenRequest, Message, Role, Sampling

# Fixed on purpose, for the same reason `regression_items` is: a corpus that drifts
# between runs measures the corpus. Each asks for a small, self-contained edit with more
# than one defensible answer -- the shape best-of-N is supposed to help with. A prompt
# with exactly one right answer would report agreement that is a property of the task.
TASKS: tuple[str, ...] = (
    (
        "Write a Python function `dedupe(items)` that removes duplicates from a list "
        "while preserving first-seen order. Return only the function."
    ),
    (
        "Write a Python function `chunk(seq, n)` splitting a sequence into lists of at "
        "most n items. Decide yourself what an n < 1 should do. Return only the function."
    ),
    (
        "Write a Python function `parse_duration(s)` accepting strings like '1h30m' or "
        "'45s' and returning whole seconds. Return only the function."
    ),
    (
        "Write a Python decorator `retry(times)` that re-calls the wrapped function on "
        "any exception, re-raising the last one. Return only the decorator."
    ),
    (
        "Write a Python function `flatten(nested)` that flattens arbitrarily nested "
        "lists into one flat list. Return only the function."
    ),
    (
        "Write a Python context manager `timed(label)` printing the elapsed wall time on "
        "exit. Return only the context manager."
    ),
    (
        "Write a Python function `merge_dicts(a, b)` doing a recursive merge where b "
        "wins on conflicts. Return only the function."
    ),
    (
        "Write a Python function `truncate(text, limit)` cutting text to limit characters "
        "on a word boundary, adding an ellipsis. Return only the function."
    ),
    (
        "Write a Python function `group_by(items, key)` returning a dict of key to list. "
        "Return only the function."
    ),
    (
        "Write a Python function `safe_get(d, path, default=None)` reading a nested dict "
        "by a dotted path. Return only the function."
    ),
    (
        "Write a Python function `moving_average(xs, window)` returning the simple moving "
        "average. Decide yourself how to handle the first window-1 points. Return only the function."
    ),
    (
        "Write a Python function `natural_key(s)` usable as a `sorted` key so that 'x10' "
        "sorts after 'x9'. Return only the function."
    ),
)


@dataclass
class PromptResult:
    task: int
    distinct_candidates: int
    choice: int | None
    choice_rotated_raw: int | None
    choice_rotated_mapped: int | None
    rerank_ok: bool
    rerank_latency_s: float
    error: str = ""


@dataclass
class Report:
    prompts: int
    candidates: int
    temperature: float
    rung: str
    container_hash: str | None
    adapter_mounted: bool
    results: list[PromptResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        ok = [r for r in self.results if r.rerank_ok and r.choice is not None]
        # Only prompts where the verifier had a real choice to make. A prompt whose
        # candidates are all identical reports agreement no matter what the verifier
        # does, and folding it in would flatter the agreement rate.
        real = [r for r in ok if r.distinct_candidates > 1]
        moved = [r for r in real if r.choice != 0]
        rotated = [r for r in real if r.choice_rotated_mapped is not None]
        stable = [r for r in rotated if r.choice_rotated_mapped == r.choice]
        # Answered index 0 whatever was in slot 0. The signature of a reranker reading
        # position rather than content, and it is indistinguishable from perfect
        # agreement in the headline number.
        positional = [r for r in rotated if r.choice == 0 and r.choice_rotated_raw == 0]
        lat = [r.rerank_latency_s for r in ok]
        return {
            "prompts_run": len(self.results),
            "rerank_ok": len(ok),
            "prompts_with_distinct_candidates": len(real),
            "mean_distinct_candidates": (
                round(statistics.mean([r.distinct_candidates for r in self.results]), 2)
                if self.results
                else None
            ),
            "disagreement_rate": (round(len(moved) / len(real), 3) if real else None),
            "agreement_rate": (round(1 - len(moved) / len(real), 3) if real else None),
            "choice_histogram": {
                str(i): sum(1 for r in real if r.choice == i)
                for i in range(self.candidates)
            },
            "positional_stability": (
                round(len(stable) / len(rotated), 3) if rotated else None
            ),
            "rotated_compared": len(rotated),
            "answered_0_regardless_of_content": len(positional),
            "rerank_latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        }


def _norm(text: str) -> str:
    return "\n".join(
        line.rstrip() for line in text.strip().splitlines() if line.strip()
    )


async def run(cfg: Config, n_prompts: int, n_candidates: int) -> Report:
    tier0 = build_tier0(cfg)
    tier1 = build_tier1(cfg, tier0)
    if tier1 is None:
        sys.exit("tier1 is disabled; this measurement needs a verifier")
    verifier = Tier1Verifier(tier1, timeout_s=cfg.tier1.request_timeout_s)

    report = Report(
        prompts=n_prompts,
        candidates=n_candidates,
        temperature=cfg.router.candidate_temperature,
        rung=cfg.tier1.rung,
        container_hash=tier0.container_hash(),
        adapter_mounted=bool(tier0.mounted_adapters()),
    )

    for t, task in enumerate(TASKS[:n_prompts]):
        req = GenRequest(
            messages=[Message(role=Role.USER, content=task)],
            system="You are a careful Python engineer.",
            sampling=Sampling(temperature=0.0, max_tokens=320, seed=1000 + t),
        )
        # Exactly what `Cascade._generate_candidates` does: candidate i is the same
        # request at the router's temperature with seed base+i. Reproducing it rather
        # than calling it keeps this honest about which code path the number describes.
        reqs = [
            req.with_(
                sampling=Sampling(
                    temperature=cfg.router.candidate_temperature,
                    top_p=req.sampling.top_p,
                    seed=req.sampling.seed + i,
                    max_tokens=req.sampling.max_tokens,
                    stop=req.sampling.stop,
                )
            )
            for i in range(n_candidates)
        ]
        gens = [await tier0.generate(r) for r in reqs]
        texts = [g.text for g in gens]
        distinct = len({_norm(x) for x in texts})
        context = _context_of(req)

        t0 = time.perf_counter()
        verdict = await verifier.rerank(
            [Candidate(index=i, text=x) for i, x in enumerate(texts)],
            context,
            seed=req.sampling.seed,
        )
        latency = time.perf_counter() - t0

        choice = int(verdict.data["choice"]) if verdict.ok else None

        # Rotate by one and map the answer back to the original index. Content-tracking
        # verifiers land on the same original candidate; position-tracking ones land on
        # the same index.
        raw: int | None = None
        mapped: int | None = None
        if verdict.ok:
            order = [(i + 1) % n_candidates for i in range(n_candidates)]
            rotated = await verifier.rerank(
                [Candidate(index=j, text=texts[o]) for j, o in enumerate(order)],
                context,
                seed=req.sampling.seed,
            )
            if rotated.ok:
                raw = int(rotated.data["choice"])
                mapped = order[raw]

        report.results.append(
            PromptResult(
                task=t,
                distinct_candidates=distinct,
                choice=choice,
                choice_rotated_raw=raw,
                choice_rotated_mapped=mapped,
                rerank_ok=verdict.ok,
                rerank_latency_s=round(latency, 2),
                error=verdict.error or "",
            )
        )
        print(
            f"  task {t:>2d}  distinct {distinct}/{n_candidates}  "
            f"choice {choice}  rotated->{mapped}  {latency:5.1f}s"
            f"{'  ' + (verdict.error or '') if not verdict.ok else ''}",
            flush=True,
        )

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="tandem.toml")
    ap.add_argument("--prompts", type=int, default=len(TASKS))
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    report = asyncio.run(run(cfg, min(args.prompts, len(TASKS)), args.candidates))
    payload = {
        "config": {k: v for k, v in asdict(report).items() if k != "results"},
        "summary": report.summary(),
        "results": [asdict(r) for r in report.results],
    }
    print(json.dumps(payload["summary"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
