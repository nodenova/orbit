"""Regression suite (spec sec 10.3).

The spec's constraint on this component is unusual: it says what the output must
*not* be. "Not a leaderboard number — a regression detector. Do not publish it as a
benchmark score." So the first tests here assert an absence, which is the only way
that instruction survives contact with a future contributor who wants a number.
"""

from __future__ import annotations

import json

import pytest

from tandem.backends.mock import MockBackend
from tandem.eval.regression import (
    Baseline,
    ItemResult,
    check_comparable,
    compare,
    grade,
    run,
)
from tandem.eval.regression_items import SUITE, Item, by_category
from tandem.types import GenResult


# --- the shape of the suite -------------------------------------------------


def test_suite_is_about_ninety_items_across_three_categories():
    assert 85 <= len(SUITE) <= 95
    counts = {c: len(by_category(c)) for c in ("reasoning", "maths", "code_localisation")}
    assert all(n >= 25 for n in counts.values()), counts
    assert sum(counts.values()) == len(SUITE)


def test_item_ids_are_unique_and_stable():
    """A diff against a baseline is keyed on the id; a duplicate silently drops one."""
    ids = [i.id for i in SUITE]
    assert len(ids) == len(set(ids))


def test_every_item_has_an_expected_answer():
    for item in SUITE:
        assert item.expected, item.id
        assert all(e.strip() for e in item.expected), item.id


def test_the_maths_answers_are_arithmetically_correct():
    """An item that can never pass is noise in a detector, and it is noise that
    looks exactly like a persistent model weakness."""
    import math
    from fractions import Fraction

    def fib(n: int) -> int:
        a, b = 1, 1
        for _ in range(n - 2):
            a, b = b, a + b
        return b

    expected = {
        "maths-01": 47 * 23, "maths-02": 1024 / 16, "maths-03": 0.17 * 250,
        "maths-04": 2**12, "maths-05": (31 - 7) / 3, "maths-06": sum(range(1, 101)),
        "maths-07": 987 - 654, "maths-08": 14 * 9, "maths-09": math.gcd(84, 126),
        "maths-10": math.factorial(15) / math.factorial(13), "maths-11": 0xFF,
        "maths-12": 0b101101, "maths-13": 144 % 13, "maths-14": 80 * 0.65,
        "maths-15": math.lcm(12, 18), "maths-16": 240 / 3, "maths-17": fib(10),
        "maths-18": math.isqrt(1369), "maths-19": 3 * 3600 + 25 * 60, "maths-20": 3 / 8,
        "maths-21": 20 / 2, "maths-22": sum([2, 3, 5, 7, 11, 13, 17, 19]),
        "maths-23": 7 * 2, "maths-24": math.factorial(6), "maths-25": 1000 - 37 * 12,
        "maths-26": 4 * 1024 * 8, "maths-27": sorted([3, 9, 4, 1, 7])[2],
        "maths-28": 5, "maths-29": float(Fraction(2, 3) + Fraction(1, 6)),
        "maths-30": 4 * math.isqrt(81),
    }
    by_id = {i.id: i for i in by_category("maths")}
    assert set(expected) == set(by_id), "every maths item must be verified here"

    wrong = [
        (item_id, by_id[item_id].expected[0], value)
        for item_id, value in expected.items()
        if abs(float(by_id[item_id].expected[0]) - float(value)) > 1e-3
    ]
    assert not wrong, wrong


# --- the absence the spec asks for ------------------------------------------


def test_the_report_has_no_score_field():
    """Sec 10.3, enforced rather than requested.

    A field holding a pass rate is all it takes for one to end up on a slide.
    """
    report = compare(
        [ItemResult("maths-01", "maths", True), ItemResult("maths-02", "maths", False)],
        Baseline(items={"maths-01": True, "maths-02": True}),
    )
    keys = set(report.as_dict())
    forbidden = {k for k in keys if any(w in k.lower() for w in ("score", "rate", "percent", "accuracy"))}
    assert not forbidden, forbidden


def test_the_report_carries_its_own_disclaimer():
    report = compare([ItemResult("m", "maths", True)], Baseline(items={"m": True}))
    assert "not a benchmark" in report.as_dict()["disclaimer"].lower()


# --- baseline and diff ------------------------------------------------------


def test_a_first_run_reports_nothing_but_writes_a_baseline():
    """There is nothing to report on a first run, and saying so is the honest output."""
    report = compare([ItemResult("a", "maths", True)], None)
    assert report.baseline_written
    assert report.clean
    assert "No baseline" in report.note


def test_an_empty_baseline_is_treated_as_no_baseline():
    report = compare([ItemResult("a", "maths", True)], Baseline(items={}))
    assert report.baseline_written


def test_a_regression_is_detected_and_named():
    results = [
        ItemResult("maths-01", "maths", False),
        ItemResult("reason-01", "reasoning", True),
        ItemResult("loc-01", "code_localisation", True),
    ]
    baseline = Baseline(items={"maths-01": True, "reason-01": True, "loc-01": True})
    report = compare(results, baseline)
    assert not report.clean
    assert report.regressions == ["maths-01"]
    assert report.unchanged == 2
    assert report.by_category["maths"]["regressed"] == 1


def test_a_fix_is_reported_but_is_not_a_regression():
    report = compare(
        [ItemResult("a", "maths", True)], Baseline(items={"a": False})
    )
    assert report.clean
    assert report.fixes == ["a"]


def test_new_and_missing_items_are_reported_separately():
    """Neither is a regression: the item set changed, not the model."""
    report = compare(
        [ItemResult("a", "maths", True), ItemResult("b", "maths", True)],
        Baseline(items={"a": True, "c": True}),
    )
    assert report.clean
    assert report.new_items == ["b"]
    assert report.missing_items == ["c"]


def test_baseline_round_trips(tmp_path):
    path = tmp_path / "baseline.json"
    Baseline(
        container_hash="abc", adapter="a1", engine_commit="deadbeef",
        items={"maths-01": True, "maths-02": False},
    ).write(path)
    loaded = Baseline.load(path)
    assert loaded is not None
    assert loaded.container_hash == "abc"
    assert loaded.adapter == "a1"
    assert loaded.items == {"maths-01": True, "maths-02": False}


def test_a_corrupt_baseline_is_treated_as_absent(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("not json", encoding="utf-8")
    assert Baseline.load(path) is None


# --- comparability ----------------------------------------------------------


def test_a_baseline_from_another_container_is_not_comparable():
    """Comparing across models reports a deliberate change as a regression, which
    trains everyone to ignore the suite — the worst outcome for a detector."""
    baseline = Baseline(container_hash="old", adapter=None)
    assert "different container" in check_comparable(baseline, "new", None)


def test_a_baseline_from_another_adapter_is_not_comparable():
    baseline = Baseline(container_hash="same", adapter="a1")
    warning = check_comparable(baseline, "same", "a2")
    assert "adapter" in warning


def test_a_matching_baseline_is_comparable():
    baseline = Baseline(container_hash="same", adapter="a1")
    assert check_comparable(baseline, "same", "a1") == ""


# --- grading ----------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,expected,response,ok",
    [
        ("last_number", ("42",), "Let me work through it. 6 * 7 = 42", True),
        ("last_number", ("42",), "I think 41", False),
        ("last_number", ("42",), "no digits here", False),
        # Shown working: the answer is the last number, not the first.
        ("last_number", ("8",), "3x + 7 = 31, so 3x = 24, x = 8", True),
        ("last_number", ("0.8333",), "That is 0.83333", True),
        ("contains", ("carol",), "The shortest is Carol.", True),
        # Whole-token match: "silently" must not satisfy "silent".
        ("contains", ("silent",), "The word is silently reversed", False),
        # …nor "15" satisfy "5".
        ("contains", ("5",), "there are 15 of them", False),
        ("contains", ("yes",), "Yes, it follows.", True),
        ("contains", ("summarise",), "The bug is in `summarise`.", True),
        ("contains", ("--first-parent", "first-parent"), "use --first-parent", True),
        ("exact", ("no",), "no", True),
        ("exact", ("no",), "no, it does not", False),
        ("contains", ("x",), "", False),
    ],
)
def test_grading(mode, expected, response, ok):
    item = Item("t", "maths", "q", expected, mode)
    assert grade(item, response) is ok


# --- running --------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_grades_every_item():
    items = SUITE[:6]
    results = await run(MockBackend(use_tools=False), items)
    assert len(results) == len(items)
    assert {r.id for r in results} == {i.id for i in items}


@pytest.mark.asyncio
async def test_run_is_greedy_so_a_baseline_is_stable():
    """A sampled answer is not a reference point."""
    backend = MockBackend(use_tools=False)
    await run(backend, SUITE[:3])
    assert all(c.sampling.temperature == 0.0 for c in backend.calls)
    assert all(c.sampling.seed == 0 for c in backend.calls)


@pytest.mark.asyncio
async def test_a_failing_item_does_not_void_the_run():
    class Boom(MockBackend):
        async def generate(self, req):
            if "47 * 23" in req.messages[0].content:
                raise RuntimeError("backend hiccup")
            return await MockBackend.generate(self, req)

    results = await run(Boom(use_tools=False), SUITE[:4])
    assert len(results) == 4
    assert any(not r.passed for r in results)


@pytest.mark.asyncio
async def test_end_to_end_detects_a_real_regression():
    """A model that answers correctly, then stops, must show up as regressions."""
    items = by_category("maths")[:5]
    answers = {i.prompt: i.expected[0] for i in items}

    good = MockBackend(use_tools=False, responder=lambda req: GenResult(
        text=answers.get(req.messages[0].content, "")
    ))
    before = await run(good, items)
    assert all(r.passed for r in before), [r.id for r in before if not r.passed]

    baseline = Baseline.from_results(before, container_hash="c", adapter=None)

    broken = MockBackend(use_tools=False, responder=lambda req: GenResult(text="0"))
    after = await run(broken, items)
    report = compare(after, baseline)

    assert not report.clean
    assert len(report.regressions) == len(items)
    assert "regressed" in report.summary()
