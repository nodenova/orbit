"""Regression suite (spec sec 10.3).

A small curated capability set, run after any kernel, quantization,
prompt-rendering or KV change.

> **Not a leaderboard number** — a regression detector. Do not publish it as a
> benchmark score.

That instruction is enforced rather than requested. The suite's output is a **diff
against a baseline**, not a score: `RegressionReport` reports which items changed
state and in which direction, and it has no field holding a bare pass rate. A first
run cannot report anything at all — it writes a baseline and says so. There is
deliberately no way to ask this module "what did the model score", because that
number is the thing the spec says not to produce, and a field holding it is all it
takes for one to end up on a slide.

Per-item pass/fail against a fixed set is also the more useful signal. "84%" tells
you nothing after a kernel change; "these three code-localisation items broke and
nothing else moved" tells you where to look.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..backends.base import Backend
from ..types import GenRequest, Message, Role, Sampling
from .regression_items import SUITE, Item

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_WORD_SPLIT = re.compile(r"[^a-z0-9_.@\-]+")


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _tokens(text: str) -> set[str]:
    """Whole tokens, each contributed both raw and edge-stripped.

    `.` and `-` stay inside the token charset because real answers contain them —
    `__init__.py`, `--first-parent`, `pytest.mark.parametrize`. That means trailing
    sentence punctuation sticks: "Carol." tokenises as `carol.`, which would never
    match the expected `carol`. Adding the stripped form too costs nothing and
    keeps both cases working, where choosing one charset or the other breaks one.
    """
    out: set[str] = set()
    for raw in _WORD_SPLIT.split(_normalise(text)):
        if not raw:
            continue
        out.add(raw)
        stripped = raw.strip(".-")
        if stripped:
            out.add(stripped)
    return out


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(text.replace(",", ""))


def _num_eq(a: str, b: str) -> bool:
    try:
        # Tolerance covers the recurring decimals in the suite (2/3 + 1/6), not
        # arithmetic slop: an item whose answer is an integer must match exactly.
        return abs(float(a) - float(b)) < 1e-3
    except (TypeError, ValueError):
        return False


def grade(item: Item, response: str) -> bool:
    """Did the model answer this item correctly?

    Lenient about *form*, strict about *value*. A model that says "The answer is 12."
    has answered 12; one that says "about 12" has too. One that says 13 has not.
    """
    if not response or not response.strip():
        return False
    if item.mode == "exact":
        return _normalise(response) in {_normalise(e) for e in item.expected}
    if item.mode == "last_number":
        found = _numbers(response)
        if not found:
            return False
        return any(_num_eq(found[-1], e) for e in item.expected)
    # contains: the answer appears as a whole token, so "silent" is not matched by
    # "silently" and "5" is not matched by "15".
    present = _tokens(response)
    for expected in item.expected:
        want = _normalise(expected)
        if " " in want:
            if want in _normalise(response):
                return True
        elif want in present:
            return True
    return False


@dataclass(frozen=True, slots=True)
class ItemResult:
    id: str
    category: str
    passed: bool
    response: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "category": self.category, "passed": self.passed}


@dataclass
class Baseline:
    """A recorded pass/fail state, per item. The thing the next run diffs against."""

    # Everything that could legitimately change the answers. A baseline taken under
    # a different container or adapter is not a baseline for this one, and comparing
    # across them would report the model change as a regression.
    container_hash: str | None = None
    adapter: str | None = None
    engine_commit: str = ""
    items: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "container_hash": self.container_hash,
            "adapter": self.adapter,
            "engine_commit": self.engine_commit,
            "items": self.items,
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Baseline | None:
        p = Path(path)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return cls(
            container_hash=raw.get("container_hash"),
            adapter=raw.get("adapter"),
            engine_commit=raw.get("engine_commit", ""),
            items={str(k): bool(v) for k, v in (raw.get("items") or {}).items()},
        )

    @classmethod
    def from_results(cls, results: Sequence[ItemResult], **meta: Any) -> Baseline:
        return cls(items={r.id: r.passed for r in results}, **meta)


@dataclass
class RegressionReport:
    """What changed. Deliberately not what was scored."""

    regressions: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    unchanged: int = 0
    new_items: list[str] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    baseline_written: bool = False
    comparable: bool = True
    note: str = ""

    @property
    def clean(self) -> bool:
        """No item that used to pass now fails."""
        return not self.regressions

    def as_dict(self) -> dict[str, Any]:
        # There is no aggregate score here on purpose (sec 10.3). If you find
        # yourself adding one, that is the leaderboard number the spec says not to
        # produce — the per-item diff is what a kernel change needs anyway.
        return {
            "clean": self.clean,
            "regressions": self.regressions,
            "fixes": self.fixes,
            "unchanged": self.unchanged,
            "new_items": self.new_items,
            "missing_items": self.missing_items,
            "by_category": self.by_category,
            "baseline_written": self.baseline_written,
            "comparable": self.comparable,
            "note": self.note,
            "disclaimer": (
                "Regression detector, not a benchmark. These counts are only "
                "meaningful as a diff against the recorded baseline on the same "
                "container and adapter; do not publish them as a score (sec 10.3)."
            ),
        }

    def summary(self) -> str:
        if self.baseline_written:
            return self.note
        if self.clean and not self.fixes:
            return f"no change across {self.unchanged} items"
        parts = []
        if self.regressions:
            parts.append(f"{len(self.regressions)} regressed: {', '.join(self.regressions[:8])}")
        if self.fixes:
            parts.append(f"{len(self.fixes)} fixed")
        return "; ".join(parts)


async def run(
    backend: Backend,
    items: Iterable[Item] = SUITE,
    *,
    adapter: str | None = None,
    max_tokens: int = 96,
    concurrency: int = 4,
) -> list[ItemResult]:
    """Run the suite. Greedy, because a sampled answer is not a stable baseline."""
    items = tuple(items)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(item: Item) -> ItemResult:
        async with sem:
            req = GenRequest(
                messages=[Message(role=Role.USER, content=item.prompt)],
                adapter=adapter,
                sampling=Sampling(temperature=0.0, top_p=1.0, seed=0, max_tokens=max_tokens),
            )
            try:
                result = await backend.generate(req)
            except Exception:  # noqa: BLE001 - one bad item must not void the run
                return ItemResult(item.id, item.category, False, "")
            return ItemResult(item.id, item.category, grade(item, result.text), result.text)

    return list(await asyncio.gather(*(one(item) for item in items)))


def compare(results: Sequence[ItemResult], baseline: Baseline | None) -> RegressionReport:
    """Diff this run against the baseline.

    With no baseline there is nothing to report — that is not a failure, it is the
    first run, and the honest output is "baseline written".
    """
    report = RegressionReport()
    current = {r.id: r.passed for r in results}
    categories = {r.id: r.category for r in results}

    if baseline is None or not baseline.items:
        report.baseline_written = True
        report.note = (
            f"No baseline: recorded {len(current)} items as the reference. "
            "Re-run after a change to see what moved."
        )
        return report

    for item_id, passed in sorted(current.items()):
        was = baseline.items.get(item_id)
        cat = categories.get(item_id, "unknown")
        bucket = report.by_category.setdefault(
            cat, {"regressed": 0, "fixed": 0, "unchanged": 0}
        )
        if was is None:
            report.new_items.append(item_id)
            continue
        if was and not passed:
            report.regressions.append(item_id)
            bucket["regressed"] += 1
        elif passed and not was:
            report.fixes.append(item_id)
            bucket["fixed"] += 1
        else:
            report.unchanged += 1
            bucket["unchanged"] += 1

    report.missing_items = sorted(set(baseline.items) - set(current))
    return report


def check_comparable(baseline: Baseline | None, container_hash: str | None, adapter: str | None) -> str:
    """Is this baseline about the same model?

    A baseline from a different container or adapter is not a baseline for this one.
    Comparing across them reports a deliberate model change as a regression, which
    trains everyone to ignore the suite — the worst outcome for a detector.
    """
    if baseline is None:
        return ""
    if baseline.container_hash and baseline.container_hash != container_hash:
        return (
            "Baseline was recorded against a different container. Re-baseline "
            "before reading these results as regressions (sec 10.3)."
        )
    if baseline.adapter != adapter:
        return (
            f"Baseline was recorded with adapter {baseline.adapter!r}, this run used "
            f"{adapter!r}. Differences are the adapter, not a regression."
        )
    return ""
