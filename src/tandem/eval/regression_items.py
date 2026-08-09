"""The curated regression items (spec sec 10.3).

~90 items across reasoning, exact-answer maths and code localisation. Fixed on
purpose: a set that drifts between runs measures nothing, because every difference
could be the set rather than the model.

Every item is short-answer and self-contained. Short because this runs after any
kernel, quantization, prompt-rendering or KV change, and a suite that costs an hour
gets skipped exactly when it matters. Self-contained because an item that depends on
a repository checkout stops being comparable the moment that repository changes.

`CONTAINS` accepts the answer appearing as a whole token anywhere in the reply, so
an item is not failed for the model saying "The answer is 12." rather than "12".
`LAST_NUMBER` takes the final number in the reply, which is where a model that
shows its working puts the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["exact", "contains", "last_number"]
Category = Literal["reasoning", "maths", "code_localisation"]


@dataclass(frozen=True, slots=True)
class Item:
    id: str
    category: Category
    prompt: str
    expected: tuple[str, ...]
    mode: Mode = "contains"


def _m(n: int, prompt: str, answer: str) -> Item:
    return Item(f"maths-{n:02d}", "maths", prompt, (answer,), "last_number")


def _r(n: int, prompt: str, *answers: str) -> Item:
    return Item(f"reason-{n:02d}", "reasoning", prompt, answers, "contains")


def _c(n: int, prompt: str, *answers: str) -> Item:
    return Item(f"loc-{n:02d}", "code_localisation", prompt, answers, "contains")


_SUFFIX = "\nAnswer with the final value only."
_NAME_SUFFIX = "\nAnswer with the identifier only."

MATHS: tuple[Item, ...] = (
    _m(1, "What is 47 * 23?" + _SUFFIX, "1081"),
    _m(2, "What is 1024 / 16?" + _SUFFIX, "64"),
    _m(3, "What is 17% of 250?" + _SUFFIX, "42.5"),
    _m(4, "What is 2^12?" + _SUFFIX, "4096"),
    _m(5, "Solve for x: 3x + 7 = 31." + _SUFFIX, "8"),
    _m(6, "What is the sum of the integers from 1 to 100?" + _SUFFIX, "5050"),
    _m(7, "What is 987 - 654?" + _SUFFIX, "333"),
    _m(8, "A rectangle is 14 by 9. What is its area?" + _SUFFIX, "126"),
    _m(9, "What is the greatest common divisor of 84 and 126?" + _SUFFIX, "42"),
    _m(10, "What is 15 factorial divided by 13 factorial?" + _SUFFIX, "210"),
    _m(11, "Convert 0xFF to decimal." + _SUFFIX, "255"),
    _m(12, "Convert binary 101101 to decimal." + _SUFFIX, "45"),
    _m(13, "What is 144 mod 13?" + _SUFFIX, "1"),
    _m(
        14,
        "A shirt costs 80 and is discounted 35%. What is the new price?" + _SUFFIX,
        "52",
    ),
    _m(15, "What is the least common multiple of 12 and 18?" + _SUFFIX, "36"),
    _m(
        16,
        "If a train travels 240 km in 3 hours, what is its speed in km/h?" + _SUFFIX,
        "80",
    ),
    _m(
        17, "What is the 10th Fibonacci number, with F(1)=1 and F(2)=1?" + _SUFFIX, "55"
    ),
    _m(18, "What is the square root of 1369?" + _SUFFIX, "37"),
    _m(19, "How many seconds are in 3 hours and 25 minutes?" + _SUFFIX, "12300"),
    _m(20, "What is 3/8 as a decimal?" + _SUFFIX, "0.375"),
    _m(21, "Solve for y: 5y - 12 = 3y + 8." + _SUFFIX, "10"),
    _m(22, "What is the sum of the first 8 prime numbers?" + _SUFFIX, "77"),
    _m(23, "A circle has radius 7. What is its diameter?" + _SUFFIX, "14"),
    _m(24, "What is 6! (six factorial)?" + _SUFFIX, "720"),
    _m(25, "What is 1000 - 37 * 12?" + _SUFFIX, "556"),
    _m(26, "How many bits are in 4 kibibytes?" + _SUFFIX, "32768"),
    _m(27, "What is the median of 3, 9, 4, 1, 7?" + _SUFFIX, "4"),
    _m(
        28,
        "If 5 machines take 5 minutes to make 5 widgets, how many minutes do "
        "100 machines take to make 100 widgets?" + _SUFFIX,
        "5",
    ),
    _m(29, "What is 2/3 + 1/6, as a decimal?" + _SUFFIX, "0.8333"),
    _m(30, "What is the perimeter of a square with area 81?" + _SUFFIX, "36"),
)

REASONING: tuple[Item, ...] = (
    _r(
        1,
        "Alice is taller than Bob. Bob is taller than Carol. Who is shortest?"
        + _NAME_SUFFIX,
        "carol",
    ),
    _r(
        2,
        "All roses are flowers. Some flowers fade quickly. Does it follow that some "
        "roses fade quickly? Answer yes or no.",
        "no",
    ),
    _r(
        3,
        "A bat and a ball cost 1.10 together. The bat costs 1.00 more than the ball. "
        "How much does the ball cost, in cents?" + _SUFFIX,
        "5",
    ),
    _r(
        4,
        "If today is Wednesday, what day is it 100 days from now?" + _NAME_SUFFIX,
        "friday",
    ),
    _r(
        5,
        "I have two coins totalling 30 cents. One is not a nickel. What is the other "
        "coin, in cents?" + _SUFFIX,
        "5",
    ),
    _r(6, "A farmer has 17 sheep. All but 9 run away. How many remain?" + _SUFFIX, "9"),
    _r(
        7,
        "Which is heavier: a kilogram of feathers or a kilogram of lead? "
        "Answer 'same' if equal.",
        "same",
        "equal",
        "neither",
    ),
    _r(
        8,
        "In a race you overtake the person in second place. What position are you in "
        "now?" + _SUFFIX,
        "second",
        "2",
    ),
    _r(
        9,
        "Some cats are black. All black things absorb light. Do some cats absorb "
        "light? Answer yes or no.",
        "yes",
    ),
    _r(
        10,
        "A rope ladder hangs over a ship's side with rungs 30 cm apart, the bottom "
        "rung at the water. The tide rises 1 m. How many rungs are underwater?"
        + _SUFFIX,
        "0",
        "none",
        "zero",
    ),
    _r(
        11,
        "If it takes 8 minutes to boil 1 egg, how many minutes to boil 3 eggs in the "
        "same pot at once?" + _SUFFIX,
        "8",
    ),
    _r(
        12,
        "X is north of Y. Z is south of Y. Which is furthest north?" + _NAME_SUFFIX,
        "x",
    ),
    _r(13, "Every P is Q. No Q is R. Can any P be R? Answer yes or no.", "no"),
    _r(
        14,
        "A clock shows 3:15. Is the hour hand exactly on the 3? Answer yes or no.",
        "no",
    ),
    _r(
        15,
        "You have 3 boxes: apples, oranges, mixed. All are labelled wrongly. What is "
        "the minimum number of fruits you must draw to relabel them all?" + _SUFFIX,
        "1",
    ),
    _r(
        16,
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops "
        "Lazzies? Answer yes or no.",
        "yes",
    ),
    _r(
        17,
        "Sarah's mother has four children: April, May, June, and ___. What is the "
        "fourth child's name?" + _NAME_SUFFIX,
        "sarah",
    ),
    _r(
        18,
        "A doctor gives you 3 pills, one every half hour. How many minutes until the "
        "last one is taken?" + _SUFFIX,
        "60",
    ),
    _r(
        19, "Which month has 28 days? Answer 'all' if every month does.", "all", "every"
    ),
    _r(
        20,
        "If you rearrange the letters of 'LISTEN' you get a word meaning quiet. What "
        "is it?" + _NAME_SUFFIX,
        "silent",
    ),
    _r(
        21,
        "Two fathers and two sons went fishing and caught 3 fish, one each. How is "
        "that possible? Answer with the smallest number of people present." + _SUFFIX,
        "3",
    ),
    _r(22, "A is twice B. B is twice C. How many times C is A?" + _SUFFIX, "4"),
    _r(
        23,
        "If some A are B, and all B are C, must some A be C? Answer yes or no.",
        "yes",
    ),
    _r(
        24,
        "A snail climbs 3 m by day and slips 2 m each night in a 10 m well. On which "
        "day does it reach the top?" + _SUFFIX,
        "8",
    ),
    _r(
        25,
        "You are in a room with 3 switches and a bulb in another room. You may enter "
        "the other room once. What is the minimum number of switches you must flip "
        "before entering?" + _SUFFIX,
        "2",
    ),
    _r(
        26,
        "P implies Q. Q is false. What can you conclude about P? Answer 'true' or "
        "'false'.",
        "false",
    ),
    _r(
        27,
        "A car travels 60 km at 60 km/h then 60 km at 30 km/h. What is the average "
        "speed in km/h?" + _SUFFIX,
        "40",
    ),
    _r(28, "If yesterday was Sunday, what day is tomorrow?" + _NAME_SUFFIX, "tuesday"),
    _r(
        29,
        "Three people check into a room... ignore that. Simply: what is the next "
        "number in 2, 6, 12, 20, 30?" + _SUFFIX,
        "42",
    ),
    _r(
        30, "What is the next letter in the sequence A, C, F, J, O?" + _NAME_SUFFIX, "u"
    ),
)

_SNIPPET_1 = """```python
def load(path):
    with open(path) as fh:
        return fh.read()

def parse(text):
    return [line.split(",") for line in text.splitlines()]

def summarise(rows):
    total = 0
    for row in rows:
        total += int(row[1])
    return total / len(rows)
```"""

_SNIPPET_2 = """```python
class Cache:
    def __init__(self, limit):
        self.limit = limit
        self.items = {}

    def put(self, key, value):
        self.items[key] = value

    def get(self, key):
        return self.items[key]
```"""

_SNIPPET_3 = """```python
 1  def retry(fn, attempts=3):
 2      last = None
 3      for i in range(attempts):
 4          try:
 5              return fn()
 6          except Exception as exc:
 7              last = exc
 8      raise last
```"""

CODE_LOCALISATION: tuple[Item, ...] = (
    _c(
        1,
        _SNIPPET_1
        + "\nWhich function divides by a possibly-zero length?"
        + _NAME_SUFFIX,
        "summarise",
    ),
    _c(2, _SNIPPET_1 + "\nWhich function opens a file?" + _NAME_SUFFIX, "load"),
    _c(
        3,
        _SNIPPET_1
        + "\nWhich function would raise IndexError on a one-column row?"
        + _NAME_SUFFIX,
        "summarise",
    ),
    _c(
        4,
        _SNIPPET_1 + "\nWhich function never raises on empty input?" + _NAME_SUFFIX,
        "parse",
    ),
    _c(
        5,
        _SNIPPET_2 + "\nWhich method ignores the configured limit?" + _NAME_SUFFIX,
        "put",
    ),
    _c(
        6,
        _SNIPPET_2 + "\nWhich method raises KeyError on a missing key?" + _NAME_SUFFIX,
        "get",
    ),
    _c(
        7,
        _SNIPPET_2 + "\nWhich attribute is set but never read?" + _NAME_SUFFIX,
        "limit",
    ),
    _c(
        8,
        _SNIPPET_3 + "\nOn which line number is the exception re-raised?" + _SUFFIX,
        "8",
    ),
    _c(
        9,
        _SNIPPET_3 + "\nOn which line number is the return value produced?" + _SUFFIX,
        "5",
    ),
    _c(
        10,
        _SNIPPET_3 + "\nWhat happens if attempts is 0? Answer with the exception "
        "type raised." + _NAME_SUFFIX,
        "typeerror",
        "attributeerror",
    ),
    _c(
        11,
        "In a Python package, which file makes a directory importable as a module?"
        + _NAME_SUFFIX,
        "__init__.py",
    ),
    _c(
        12,
        "Which Python builtin returns the number of items in a list?" + _NAME_SUFFIX,
        "len",
    ),
    _c(
        13,
        "In `for i, x in enumerate(xs, 1)`, what is the value of i on the first "
        "iteration?" + _SUFFIX,
        "1",
    ),
    _c(
        14,
        "Which dict method returns a default instead of raising on a missing key?"
        + _NAME_SUFFIX,
        "get",
    ),
    _c(
        15,
        "What does `x[::-1]` do to a Python list? Answer in one word." + _NAME_SUFFIX,
        "reverses",
        "reverse",
        "reversed",
    ),
    _c(
        16,
        "Which keyword makes a Python function return a generator?" + _NAME_SUFFIX,
        "yield",
    ),
    _c(
        17,
        "Which statement ensures a file is closed even if an exception is raised?"
        + _NAME_SUFFIX,
        "with",
    ),
    _c(
        18,
        "In git, which command shows the commit that last modified each line of a "
        "file?" + _NAME_SUFFIX,
        "blame",
    ),
    _c(
        19,
        "In git, which flag on `log` follows only the first parent of merges?"
        + _NAME_SUFFIX,
        "--first-parent",
        "first-parent",
    ),
    _c(
        20,
        "In a unified diff, what character begins a line that was removed?"
        + _NAME_SUFFIX,
        "-",
        "minus",
    ),
    _c(
        21,
        "In a unified diff, what marks the start of a hunk header?" + _NAME_SUFFIX,
        "@@",
    ),
    _c(
        22,
        "Which HTTP status code means the request was understood but refused?"
        + _SUFFIX,
        "403",
    ),
    _c(23, "Which HTTP status code means the resource was not found?" + _SUFFIX, "404"),
    _c(
        24,
        "In JSON Schema, which keyword pins a property to one exact value?"
        + _NAME_SUFFIX,
        "const",
    ),
    _c(
        25,
        "In JSON Schema, which keyword forbids properties not listed?" + _NAME_SUFFIX,
        "additionalproperties",
    ),
    _c(
        26,
        "In Python, which exception does `int('abc')` raise?" + _NAME_SUFFIX,
        "valueerror",
    ),
    _c(
        27,
        "In Python, which exception does `{}['k']` raise?" + _NAME_SUFFIX,
        "keyerror",
    ),
    _c(28, "Which Python module provides `dataclass`?" + _NAME_SUFFIX, "dataclasses"),
    _c(
        29,
        "In pytest, which decorator parameterises a test?" + _NAME_SUFFIX,
        "parametrize",
        "pytest.mark.parametrize",
    ),
    _c(
        30,
        "In Python, what does `__slots__` on a class reduce? Answer in one word."
        + _NAME_SUFFIX,
        "memory",
    ),
)

SUITE: tuple[Item, ...] = MATHS + REASONING + CODE_LOCALISATION


def by_category(category: str) -> tuple[Item, ...]:
    return tuple(item for item in SUITE if item.category == category)
