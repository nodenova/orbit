"""Tool-call repair (spec sec 8.5.3).

This is what actually breaks. Local models emit XML/JSON hybrids, fenced blocks,
bare objects, Python-ish call syntax, trailing commas, smart quotes and truncated
objects. The harness sees no valid call, re-prompts, the model garbles identically,
and the session enters an infinite "let me do that" loop.

The strategies below are ordered cheapest-and-most-certain first, and each one is
tried against the whole text before the next. Four rules run through all of them:

* **Names are never invented.** A parsed call whose name is not in the request's
  tool list is rejected, not passed through. A model that hallucinates
  `run_arbitrary_command` must get a rejection, not an execution.
* **A mangled name is inferred from the argument keys**, not guessed from string
  similarity alone — if the keys uniquely identify one tool's required parameters,
  that is strong evidence; if two tools match equally well, we refuse rather than
  pick.
* **A call is never fabricated out of prose.** Inference needs a payload that was
  *shaped* like a call — one carrying a name key or an arguments wrapper. In an
  agentic loop the model's context is full of untrusted material: file contents,
  tool output, web pages. A JSON blob the model echoed out of a file it just read,
  or a config snippet it is asking permission about, must never become a shell
  execution because its keys happen to overlap a tool's parameters.
* **A recovered call satisfies the tool's schema.** Its required parameters are
  present and its values carry the declared types; anything else is a rejection,
  because a call that is accepted here is counted as well-formed by the sec 10.2
  gate and is then actually executed by the harness.

Everything here is linear in the length of the model's output. Degenerate
repetition — ten thousand unclosed `<tool_call>` tags — is the canonical 2-bit
failure this module exists to serve, and `repair()` runs synchronously on the
gateway's event loop, so being merely *correct* on that input is not enough.

Repair is the third line of defence, behind constrained decoding (sec 8.5.1) and the
A0 harness adapter (sec 6.1). Experiment E1 asks whether A0 makes this layer
droppable; until that answers yes, it stays.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tandem.types import ToolCall, ToolDef

# --- text normalisation -----------------------------------------------------

# Quote characters are the only *structural* ones: a model that emitted “name”
# instead of "name" broke the JSON parse, and rewriting the quote fixes it. The
# rest of the table is cosmetic to JSON, so it is only applied between tokens —
# see `desmarten_structural`.
_SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    "‘": "'",
    "’": "'",
    "′": "'",
    "«": '"',
    "»": '"',
}
_SMART_BETWEEN_TOKENS = {
    # Non-ASCII dashes turn up inside mangled tool names, and a non-breaking space
    # between two JSON tokens is a parse error where an ordinary space is not.
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
}
_SMART = {**_SMART_QUOTES, **_SMART_BETWEEN_TOKENS}
_SMART_RE = re.compile("|".join(re.escape(k) for k in _SMART))

# Which characters can *close* a JSON string that this character opened. Only the
# double-quote family opens one: `{'a': 1}` is not JSON however it is rewritten,
# so treating a lone ’ as a delimiter would only let a prose apostrophe swallow
# the rest of the text.
_QUOTE_CLOSERS = {
    '"': ('"',),
    "“": ("”", "“", '"'),
    "”": ("”", '"'),
    "„": ("”", "“", '"'),
    "″": ("″", '"'),
    "«": ("»", '"'),
}

# Quote pairs a *value* may be wrapped in, for `_unquote`. Wider than the set that
# opens a JSON string because a function-syntax or XML argument is not JSON.
_QUOTE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("„", "”"),
    ("«", "»"),
)

_FENCE_RE = re.compile(
    r"```(?:json|tool_call|python|xml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def desmarten(text: str) -> str:
    """Rewrite every smart character. For *names*, where every byte is punctuation.

    Not for text that contains argument values — see `desmarten_structural`.
    """
    return _SMART_RE.sub(lambda m: _SMART[m.group(0)], text)


def desmarten_structural(text: str) -> str:
    """Desmarten JSON's structural punctuation, leaving string values byte-intact.

    `desmarten` rewrites every character it knows about wherever it appears. That
    is right for a tool *name* and wrong for an argument *value*: it turned
    `git log --grep=—fix` into `--grep=-fix` (a different command runs), `don’t.py`
    into `don't.py` (a different file is read) and a non-breaking space inside a
    filename into a space. Only the quote characters break the parse, so only they
    are rewritten inside a string — and only where they close it. Dashes and
    non-breaking spaces are rewritten between tokens, where JSON has no values to
    corrupt.

    `lenient_loads` tries the raw text first, so valid model output is never
    rewritten at all. This confines the damage on the repair path too, which is
    where a wrong answer is hardest to notice.
    """
    out: list[str] = []
    opener: str | None = None
    esc = False
    for ch in text:
        if esc:
            out.append(ch)
            esc = False
            continue
        if opener is not None:
            if ch == "\\":
                out.append(ch)
                esc = True
                continue
            if ch in _QUOTE_CLOSERS[opener]:
                out.append('"')
                opener = None
                continue
            out.append(ch)  # inside a value: never rewritten
            continue
        if ch in _QUOTE_CLOSERS:
            out.append('"')
            opener = ch
            continue
        out.append(_SMART_BETWEEN_TOKENS.get(ch, ch))
    return "".join(out)


def _unquote(s: str) -> str | None:
    """Strip one matching pair of value delimiters, or None if there is no pair."""
    for open_q, close_q in _QUOTE_PAIRS:
        if (
            len(s) >= len(open_q) + len(close_q)
            and s.startswith(open_q)
            and s.endswith(close_q)
        ):
            return s[len(open_q) : -len(close_q)]
    return None


@dataclass(frozen=True, slots=True)
class _Fence:
    start: int  # of the whole ```…``` span, which is what the model sampled
    end: int
    body: str
    body_start: int  # of the stripped body, for spans inside it
    body_end: int


def fence_spans(text: str) -> list[_Fence]:
    """Every fenced block: the span the model sampled, and the body that parses.

    Both are needed. The body is what `lenient_loads` sees; the span is what goes
    into the replay map, because recording the body dropped the fence markers and
    the newline layout from the next turn's prompt — the byte divergence the
    replay map exists to prevent (sec 8.5.5).
    """
    out: list[_Fence] = []
    for m in _FENCE_RE.finditer(text):
        raw = m.group(1)
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        body = raw.strip()
        if not body:
            continue
        out.append(
            _Fence(
                start=m.start(),
                end=m.end(),
                body=body,
                body_start=m.start(1) + lead,
                body_end=m.end(1) - trail,
            )
        )
    return out


def strip_fences(text: str) -> list[str]:
    """Bodies of every fenced block."""
    return [f.body for f in fence_spans(text)]


def _balance(text: str) -> str:
    """Close unterminated braces/brackets on a truncated object.

    A max_tokens cut mid-object is common and entirely recoverable when the missing
    part is only closers. Strings are tracked so a brace inside a string literal
    does not corrupt the count.
    """
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif (
            ch in "}]"
            and stack
            and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "["))
        ):
            stack.pop()
    out = text
    if in_str:
        out += '"'
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def lenient_loads(text: str) -> Any | None:
    """json.loads with the malformations local models actually produce fixed up.

    Candidate order is load-bearing: the raw text is tried first, so output that
    was already valid is never rewritten, and each repair is only reached once the
    cheaper reading has failed.
    """
    candidates = [text]
    de = desmarten_structural(text)
    if de != text:
        candidates.append(de)
    for base in list(candidates):
        no_comma = _TRAILING_COMMA_RE.sub(r"\1", base)
        if no_comma != base:
            candidates.append(no_comma)
    for base in list(candidates):
        balanced = _balance(base)
        if balanced != base:
            candidates.append(balanced)
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def find_json_object_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every balanced top-level {...} span, in order.

    Brace-counting with string awareness rather than a regex: a regex cannot match
    nested objects, and tool arguments are routinely nested.
    """
    out: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append((start, i + 1))
                    start = -1
    if depth > 0 and start >= 0:
        # Truncated tail — hand it over anyway; _balance may rescue it.
        out.append((start, len(text)))
    return out


def find_json_objects(text: str) -> list[str]:
    """Every balanced top-level {...} span in the text, in order."""
    return [text[s:e] for s, e in find_json_object_spans(text)]


# --- extraction shapes ------------------------------------------------------
#
# Opening and closing tags are matched separately and paired by `_paired`, never
# with one `<tag>(.*?)</tag>` pattern. See that function for why.

_XML_OPEN_RE = re.compile(r"<(?:tool_call|function_call|invoke)\b[^>]*>", re.IGNORECASE)
_XML_CLOSE_RE = re.compile(r"</(?:tool_call|function_call|invoke)\s*>", re.IGNORECASE)
_XML_NAME_ATTR_RE = re.compile(
    r"""<(?:invoke|tool_call|function_call)\b[^>]*\bname\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)
_XML_NAME_OPEN_RE = re.compile(r"<(?:tool_name|function_name|name)\s*>", re.IGNORECASE)
_XML_NAME_CLOSE_RE = re.compile(
    r"</(?:tool_name|function_name|name)\s*>", re.IGNORECASE
)
_XML_PARAM_OPEN_RE = re.compile(
    r"""<parameter\b[^>]*\bname\s*=\s*["']([^"']+)["']\s*>""", re.IGNORECASE
)
_XML_PARAM_CLOSE_RE = re.compile(r"</parameter\s*>", re.IGNORECASE)
_XML_ARGS_OPEN_RE = re.compile(r"<(?:arguments|args|parameters)\s*>", re.IGNORECASE)
_XML_ARGS_CLOSE_RE = re.compile(r"</(?:arguments|args|parameters)\s*>", re.IGNORECASE)
_XML_TAG_OPEN_RE = re.compile(r"<([A-Za-z_][\w.-]*)\s*>")
_XML_TAG_CLOSE_RE = re.compile(r"</([A-Za-z_][\w.-]*)\s*>")

_FUNC_SYNTAX_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*\((.*)\)\s*(;?)\s*$", re.DOTALL)

_XML_STRUCTURAL_TAGS = frozenset(
    {"tool_name", "function_name", "name", "arguments", "args", "parameters"}
)


def _paired(
    text: str,
    open_re: re.Pattern[str],
    close_re: re.Pattern[str],
    *,
    by_name: bool = False,
) -> list[tuple[re.Match[str], re.Match[str]]]:
    """Non-overlapping (open, close) tag pairs, in document order.

    Two `finditer` passes and a merge, which is linear. The obvious
    `<tag>(.*?)</tag>` with re.DOTALL is quadratic: every *unclosed* opening tag
    becomes a start position whose lazy group rescans to end-of-text before
    failing. Measured on repeated `<tool_call>`: 24 KB took 0.28 s and 384 KB took
    70.83 s — and `repair()` is called synchronously from the gateway's event
    loop, so that stalls every concurrent stream, not one request. Degenerate
    repetition is precisely the failure this module exists to serve, so it has to
    be fast, not merely correct.

    `by_name` pairs on the tag name (the old `</\\1>` backreference); otherwise any
    closer matches any opener, because models mix `<invoke>` with `</tool_call>`.
    """
    opens = list(open_re.finditer(text))
    if not opens:
        return []
    closes = list(close_re.finditer(text))
    if not closes:
        return []
    buckets: dict[str | None, list[re.Match[str]]] = {}
    for cm in closes:
        buckets.setdefault(cm.group(1) if by_name else None, []).append(cm)
    # One cursor per bucket, never rewound, so the whole merge is linear.
    cursor: dict[str | None, int] = {}
    pairs: list[tuple[re.Match[str], re.Match[str]]] = []
    last_end = 0
    for om in opens:
        if om.start() < last_end:
            continue  # nested inside a span already taken
        key = om.group(1) if by_name else None
        bucket = buckets.get(key)
        if not bucket:
            continue
        i = cursor.get(key, 0)
        while i < len(bucket) and bucket[i].start() < om.end():
            i += 1
        if i >= len(bucket):
            cursor[key] = i
            continue
        cursor[key] = i + 1
        pairs.append((om, bucket[i]))
        last_end = bucket[i].end()
    return pairs


def _declared_type(schema: dict[str, Any] | None) -> str | None:
    if not isinstance(schema, dict):
        return None
    declared = schema.get("type")
    if isinstance(declared, list):
        # ["string", "null"] — the nullable spelling. A scalar has to satisfy the
        # non-null branch.
        declared = next(
            (t for t in declared if isinstance(t, str) and t != "null"), None
        )
    return declared if isinstance(declared, str) else None


def _cast(text: str, declared: str | None) -> Any:
    """Coerce to the declared JSON-Schema type, or leave the text alone."""
    s = text.strip()
    if declared == "integer":
        try:
            return int(s, 10)
        except ValueError:
            return text
    if declared == "number":
        try:
            return float(s)
        except ValueError:
            return text
    if declared == "boolean":
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        return text
    if declared in ("object", "array"):
        parsed = lenient_loads(s)
        if isinstance(parsed, (dict, list)):
            return parsed
        return text
    return text


def _coerce_scalar(raw: str, schema: dict[str, Any] | None = None) -> Any:
    r"""One textual argument value, typed by the tool's own declaration.

    Schema-aware because guessing corrupts real arguments: for a parameter the
    tool declares as a string, `path=0644` is a file mode and not the integer 644,
    and `command=007` is a command name and not 7. Nothing downstream validates
    types, so a guess here is what the harness executes.

    A JSON string literal is *decoded*, never quote-stripped. The old code parsed
    `"line1\nline2"` correctly and then threw the parse away in favour of removing
    the outer quotes, so the escape survived as the two characters `\` and `n` —
    silently rewriting file contents and shell commands.
    """
    s = raw.strip()
    declared = _declared_type(schema)

    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            decoded = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            decoded = None
        if isinstance(decoded, str):
            return decoded if declared in (None, "string") else _cast(decoded, declared)

    unquoted = _unquote(s)
    if unquoted is not None:
        # Delimiters JSON does not have ('…', “…”): the inside is the value
        # verbatim, since no escape syntax is defined for it.
        return unquoted if declared in (None, "string") else _cast(unquoted, declared)

    if declared is not None:
        return _cast(s, declared)

    # Undeclared parameter: the heuristics are all there is. They stay for exactly
    # that case and are no longer reached for a parameter the tool described.
    parsed = lenient_loads(s)
    if parsed is not None and not isinstance(parsed, str):
        return parsed
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "none"):
        return None
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d*\.\d+", s):
        return float(s)
    return s


def _schemas_for(
    raw_name: str | None, tools: tuple[ToolDef, ...]
) -> dict[str, dict[str, Any]]:
    """Declared parameter schemas to coerce this call's arguments against.

    Coercion happens before name resolution, so a mangled name has no tool yet.
    Rather than fall back to guessing, use the types the tools *agree* on: a
    parameter every tool declares as a string is a string whichever tool this
    turns out to be. Disagreement yields nothing, which lands on the untyped
    heuristics — no worse than before, and never a confident wrong type.
    """
    by_name = {t.name: t for t in tools}
    tool = by_name.get(raw_name or "")
    if tool is None and raw_name:
        norm = _normalise_name(raw_name)
        matches = [t for t in tools if _normalise_name(t.name) == norm]
        tool = matches[0] if len(matches) == 1 else None
    if tool is not None:
        props = tool.parameters.get("properties")
        return (
            {k: v for k, v in props.items() if isinstance(v, dict)}
            if isinstance(props, dict)
            else {}
        )

    merged: dict[str, dict[str, Any]] = {}
    conflicting: set[str] = set()
    for candidate in tools:
        props = candidate.parameters.get("properties")
        if not isinstance(props, dict):
            continue
        for key, schema in props.items():
            if key in conflicting or not isinstance(schema, dict):
                continue
            if key in merged and _declared_type(merged[key]) != _declared_type(schema):
                conflicting.add(key)
                merged.pop(key, None)
                continue
            merged[key] = schema
    return merged


def _args_from_xml(block: str, schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for om, cm in _paired(block, _XML_PARAM_OPEN_RE, _XML_PARAM_CLOSE_RE):
        name = om.group(1)
        args[name] = _coerce_scalar(block[om.end() : cm.start()], schemas.get(name))
    if args:
        return args
    inner = block
    args_block = _paired(block, _XML_ARGS_OPEN_RE, _XML_ARGS_CLOSE_RE)
    if args_block:
        om, cm = args_block[0]
        inner = block[om.end() : cm.start()]
        # <arguments>{"a": 1}</arguments> is common enough to be worth trying first.
        parsed = lenient_loads(inner.strip())
        if isinstance(parsed, dict):
            return parsed
    for om, cm in _paired(inner, _XML_TAG_OPEN_RE, _XML_TAG_CLOSE_RE, by_name=True):
        name = om.group(1)
        if name.lower() in _XML_STRUCTURAL_TAGS:
            continue
        args[name] = _coerce_scalar(inner[om.end() : cm.start()], schemas.get(name))
    return args


def _split_kwargs(
    body: str, schemas: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Split `a=1, b="x,y"` respecting quotes, brackets and escapes."""
    schemas = schemas or {}
    args: dict[str, Any] = {}
    depth = 0
    in_str: str | None = None
    esc = False
    token = []
    parts: list[str] = []
    for ch in body:
        if esc:
            token.append(ch)
            esc = False
            continue
        if ch == "\\":
            token.append(ch)
            esc = True
            continue
        if in_str:
            token.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in "\"'":
            in_str = ch
            token.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(token))
            token = []
            continue
        token.append(ch)
    if token:
        parts.append("".join(token))

    for part in parts:
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        if key:
            args[key] = _coerce_scalar(value, schemas.get(key))
    return args


# --- name resolution --------------------------------------------------------

_NAME_CLEAN_RE = re.compile(r"[^a-z0-9_]+")


_CLEAN_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


def _normalise_name(name: str) -> str:
    return _NAME_CLEAN_RE.sub("_", desmarten(name).strip().lower()).strip("_")


def _is_clean_identifier(name: str) -> bool:
    """Does this look like a name the model meant, rather than a garbled one?

    `definitely_not_a_real_tool` is clean — an invention. `tool-call-???` is not —
    a mangling. The two get opposite treatment.
    """
    return bool(_CLEAN_IDENT_RE.match(desmarten(name).strip()))


@dataclass(frozen=True, slots=True)
class NameResolution:
    name: str | None
    how: str  # exact | normalised | inferred | rejected
    reason: str = ""


def _checked(
    tool: ToolDef, args: dict[str, Any], how: str, reason: str = ""
) -> NameResolution:
    """Accept only a call the tool would actually accept.

    The required-parameter check used to run on the inference path alone, so
    `{"name": "read_file"}` — no `path` — came back as a well-formed call, was
    counted by `tool_call_rate()` towards the blocking sec 10.2 gate, and reached
    the harness as an unexecutable call. A rejection here becomes a retry, which
    is the outcome that can still recover.
    """
    missing = sorted(p for p in tool.required_names() if p not in args)
    if missing:
        return NameResolution(
            None,
            "rejected",
            f"{tool.name!r} is missing required parameter(s) {', '.join(missing)}",
        )
    return NameResolution(tool.name, how, reason)


def resolve_name(
    raw_name: str | None,
    args: dict[str, Any],
    tools: Iterable[ToolDef],
    *,
    may_infer: bool = True,
    strict_keys: bool = False,
) -> NameResolution:
    """Map a possibly-mangled name onto a real tool, or reject.

    Never returns a name that is not in `tools` — the spec's rule that the model
    cannot invent a tool is enforced structurally rather than checked later.

    `may_infer=False` says the payload was not shaped like a call attempt (no name
    key, no arguments wrapper), so its keys are not evidence of anything.
    `strict_keys` says the arguments were read off a bare object's top level, where
    a key the tool does not declare is the tell that the object was never a call:
    a pasted CI config with `cwd` next to `command` is not a request to run it.
    Both default to the permissive reading so a direct caller gets the old
    behaviour; `repair` sets them per shape.
    """
    by_name = {t.name: t for t in tools}
    if not by_name:
        return NameResolution(None, "rejected", "no tools available on this request")

    if raw_name and raw_name in by_name:
        return _checked(by_name[raw_name], args, "exact")

    if raw_name:
        norm = _normalise_name(raw_name)
        matches = [n for n in by_name if _normalise_name(n) == norm]
        if len(matches) == 1:
            return _checked(
                by_name[matches[0]],
                args,
                "normalised",
                f"{raw_name!r} -> {matches[0]!r}",
            )

        # A *clean* unknown name is an invention, not a mangling: the model believes
        # a tool exists that does not. Inferring a substitute from the argument keys
        # would silently run a different tool than the one it asked for, which is
        # worse than failing. Reject (sec 8.5.3). Only garbled names fall through to
        # inference, which is what "infer when the name is mangled" means.
        if _is_clean_identifier(raw_name):
            return NameResolution(
                None,
                "rejected",
                f"unknown tool {raw_name!r}; the model may not invent a tool name",
            )

    if not may_infer:
        # Nothing here claimed to be a call. Argument-key overlap alone turned an
        # illustrative JSON blob in prose into an execution, and in an agent loop
        # that blob is as likely to have come out of a file the model read as out
        # of the model's own intent.
        return NameResolution(
            None,
            "rejected",
            "payload carries no tool name and no arguments wrapper; not a call attempt",
        )

    # Infer from argument keys (sec 8.5.3). Score = does the tool's required set fit
    # inside the observed keys, then overlap fraction. Ties refuse rather than guess.
    keys = set(args)
    if not keys:
        return NameResolution(
            None,
            "rejected",
            f"unknown tool {raw_name!r} and no arguments to infer from",
        )
    scored: list[tuple[float, str]] = []
    for name, tool in by_name.items():
        params = set(tool.param_names())
        required = set(tool.required_names())
        if required and not required.issubset(keys):
            continue
        if not params:
            continue
        if strict_keys and not keys.issubset(params):
            continue
        overlap = len(keys & params)
        if overlap == 0:
            continue
        union = len(keys | params)
        scored.append((overlap / union, name))
    if not scored:
        return NameResolution(
            None,
            "rejected",
            f"unknown tool {raw_name!r}; no tool matches its arguments",
        )
    scored.sort(reverse=True)
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-9:
        return NameResolution(
            None,
            "rejected",
            f"arguments match {scored[0][1]!r} and {scored[1][1]!r} equally; refusing to guess",
        )
    return _checked(
        by_name[scored[0][1]],
        args,
        "inferred",
        f"inferred from argument keys {sorted(keys)}",
    )


# --- the repair pass --------------------------------------------------------


@dataclass
class RepairOutcome:
    calls: tuple[ToolCall, ...] = ()
    # Text with the recovered call spans removed, so prose around a call survives.
    residual_text: str = ""
    repaired: bool = False
    strategy: str = ""
    rejected: list[str] = field(default_factory=list)
    # call id -> the exact bytes the model sampled for that call. Byte-exact for
    # every strategy: `raw_blocks[id] == text[slice(*call_spans[id])]`.
    raw_blocks: dict[str, str] = field(default_factory=dict)
    # call id -> (start, end) of that block in the sampled text. The model's own
    # ordering, which `residual_text` alone cannot express: a call followed by
    # prose comes back from the harness as content-then-call and is re-rendered
    # in the wrong order (sec 8.5.5).
    call_spans: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.calls)


def _call_id(name: str, args: dict[str, Any], salt: str = "") -> str:
    material = f"{name}\x00{json.dumps(args, sort_keys=True, default=str)}\x00{salt}"
    return "call_" + hashlib.sha256(material.encode()).hexdigest()[:16]


def _record(
    text: str,
    span: tuple[int, int],
    res: NameResolution,
    args: dict[str, Any],
    out: RepairOutcome,
    strategy: str,
) -> None:
    call = ToolCall(
        id=_call_id(res.name or "", args, str(len(out.calls))),
        name=res.name or "",
        arguments=args,
    )
    out.calls = (*out.calls, call)
    # Sliced out of the sampled text rather than passed in, so the recorded block
    # cannot drift from the span that produced it.
    out.raw_blocks[call.id] = text[span[0] : span[1]]
    out.call_spans[call.id] = span
    out.repaired = True
    out.strategy = strategy if not out.strategy else f"{out.strategy}+{strategy}"


def _accept(
    text: str,
    span: tuple[int, int],
    raw_name: str | None,
    args: dict[str, Any],
    tools: Iterable[ToolDef],
    out: RepairOutcome,
    strategy: str,
    *,
    may_infer: bool = True,
    strict_keys: bool = False,
) -> bool:
    res = resolve_name(
        raw_name, args, tools, may_infer=may_infer, strict_keys=strict_keys
    )
    if res.name is None:
        out.rejected.append(res.reason)
        return False
    _record(text, span, res, args, out, strategy)
    return True


def repair(text: str, tools: Iterable[ToolDef]) -> RepairOutcome:
    """Recover tool calls from malformed model output.

    Returns an outcome with zero calls when nothing is recoverable — that is a
    normal result for a plain prose turn, not an error.
    """
    tools = tuple(tools)
    out = RepairOutcome(residual_text=text)
    if not text or not tools:
        return out

    # 1. XML / XML-in-JSON hybrids.
    consumed: list[tuple[int, int]] = []
    for om, cm in _paired(text, _XML_OPEN_RE, _XML_CLOSE_RE):
        span = (om.start(), cm.end())
        block = text[span[0] : span[1]]
        inner = text[om.end() : cm.start()]
        name = None
        attr = _XML_NAME_ATTR_RE.search(block)
        if attr:
            name = attr.group(1)
        else:
            tag = _paired(inner, _XML_NAME_OPEN_RE, _XML_NAME_CLOSE_RE)
            if tag:
                name = inner[tag[0][0].end() : tag[0][1].start()].strip()
        args = _args_from_xml(inner, _schemas_for(name, tools))
        # A `<tool_call>` block is unambiguously a call attempt, so a missing name
        # may still be inferred from the argument keys.
        if _accept(text, span, name, args, tools, out, "xml"):
            consumed.append(span)
    if out.calls:
        out.residual_text = _remove_spans(text, consumed)
        return out

    # 2. Fenced blocks.
    consumed = []
    for fence in fence_spans(text):
        if _try_json_payload(
            text,
            fence.body,
            fence.body_start,
            tools,
            out,
            "fenced",
            span=(fence.start, fence.end),
        ):
            consumed.append((fence.start, fence.end))
    if out.calls:
        out.residual_text = _remove_spans(text, consumed)
        return out

    # 3. Embedded JSON objects, in document order, name-carrying payloads first.
    #
    #    Not longest-first. That sort's rationale — "a wrapper object should be
    #    tried before the argument object nested inside it" — was never true:
    #    `find_json_object_spans` returns only depth-0 spans, so a nested arguments
    #    object is never a candidate. All it ever did was reorder siblings, and it
    #    systematically preferred a long illustrative blob to the short real call
    #    beside it, dropping the call the model actually asked for.
    candidates: list[tuple[int, int, int, int, Any]] = []
    for order, (start, end) in enumerate(find_json_object_spans(text)):
        obj = lenient_loads(text[start:end])
        if obj is None:
            continue
        candidates.append((0 if _carries_name(obj) else 1, order, start, end, obj))
    candidates.sort(key=lambda c: (c[0], c[1]))
    for _rank, _order, start, end, obj in candidates:
        if _try_json_payload(
            text, text[start:end], start, tools, out, "json", span=(start, end), obj=obj
        ):
            out.residual_text = _remove_spans(text, [(start, end)])
            return out

    # 4. Function-call syntax: name(a=1, b="x").
    fm = _FUNC_SYNTAX_RE.match(text)
    if fm:
        # The span is the call itself, not the whole text: the whitespace around it
        # was sampled too and belongs in the residual (sec 8.5.5).
        span = (fm.start(1), fm.end(3) if fm.group(3) else fm.end(2) + 1)
        args = _split_kwargs(fm.group(2), _schemas_for(fm.group(1), tools))
        if _accept(text, span, fm.group(1), args, tools, out, "function_syntax"):
            out.residual_text = _remove_spans(text, [span])
            return out

    return out


def _try_json_payload(
    text: str,
    payload: str,
    payload_start: int,
    tools: tuple[ToolDef, ...],
    out: RepairOutcome,
    strategy: str,
    *,
    span: tuple[int, int],
    obj: Any | None = None,
) -> bool:
    """Parse `payload` (already sliced out of `text` at `payload_start`) as a call.

    `span` is the region of `text` the model sampled for it — the whole fence for a
    fenced call, the object span for a bare one.
    """
    if obj is None:
        obj = lenient_loads(payload)
    if obj is None:
        return False
    if isinstance(obj, list):
        items = [item for item in obj if isinstance(item, dict)]
        spans = find_json_object_spans(payload)
        got = False
        for i, item in enumerate(items):
            # Per-item spans when they line up, which they do for a plain array of
            # objects; the whole payload otherwise. Each call needs its own block
            # because `render_message` frames them one at a time.
            item_span = (
                (payload_start + spans[i][0], payload_start + spans[i][1])
                if len(spans) == len(items)
                else span
            )
            got |= _try_dict(text, item, tools, out, strategy, span=item_span)
        return got
    if isinstance(obj, dict):
        return _try_dict(text, obj, tools, out, strategy, span=span)
    return False


_NAME_KEYS = (
    "name",
    "tool",
    "tool_name",
    "function",
    "function_name",
    "recipient_name",
)
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input", "tool_input")
# Keys that are call envelope rather than argument — unless the tool declares a
# parameter of that name, see `_bare_args`.
_ENVELOPE_KEYS = frozenset(_NAME_KEYS) | {"type"}


def _carries_name(obj: Any) -> bool:
    """Does this payload name a tool, rather than merely have overlapping keys?"""
    if isinstance(obj, list):
        return any(_carries_name(item) for item in obj)
    if not isinstance(obj, dict):
        return False
    inner = obj.get("function")
    if isinstance(inner, dict) and _carries_name(inner):
        return True
    return any(isinstance(obj.get(key), str) and obj.get(key) for key in _NAME_KEYS)


def _bare_args(
    obj: dict[str, Any], name_key: str | None, declared: Iterable[str]
) -> dict[str, Any]:
    """Arguments read off a bare object's top level.

    Envelope keys are dropped only when the tool does not declare a parameter of
    that name. Dropping `type`, `name` and `input` unconditionally deleted real
    parameters — the call then went out missing an argument the tool required and
    was still counted as well-formed. The key the name was *read from* always
    goes: its value was consumed as the tool name.
    """
    keep = set(declared)
    return {
        k: v
        for k, v in obj.items()
        if k != name_key and (k not in _ENVELOPE_KEYS or k in keep)
    }


def _params_of(name: str, tools: tuple[ToolDef, ...]) -> tuple[str, ...]:
    for tool in tools:
        if tool.name == name:
            return tool.param_names()
    return ()


def _try_dict(
    text: str,
    obj: dict[str, Any],
    tools: tuple[ToolDef, ...],
    out: RepairOutcome,
    strategy: str,
    *,
    span: tuple[int, int],
) -> bool:
    # OpenAI-style envelope: {"type": "function", "function": {"name":..,"arguments":..}}
    inner = obj.get("function")
    if isinstance(inner, dict) and ("name" in inner or "arguments" in inner):
        obj = {**inner}

    name: str | None = None
    name_key: str | None = None
    for key in _NAME_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value:
            name, name_key = value, key
            break

    wrapped: dict[str, Any] | None = None
    arg_key: str | None = None
    for key in _ARG_KEYS:
        value = obj.get(key)
        if isinstance(value, dict):
            wrapped, arg_key = value, key
            break
        if isinstance(value, str):
            # Arguments as a JSON *string* is the OpenAI wire shape.
            parsed = lenient_loads(value)
            if isinstance(parsed, dict):
                wrapped, arg_key = parsed, key
                break

    # Shaped like a call attempt? Only then may the tool be inferred from the keys.
    shaped = name_key is not None or arg_key is not None

    attempts: list[tuple[dict[str, Any], bool]] = []
    if wrapped is not None:
        attempts.append((wrapped, False))
    # The bare reading is tried even when a wrapper was found, because `input` and
    # `arguments` are legitimate *parameter* names too — unwrapping one of those
    # deleted the parameter and left a call the tool cannot execute.
    attempts.append((_bare_args(obj, name_key, ()), True))

    reason = ""
    for args, bare in attempts:
        if not args and not name:
            continue
        res = resolve_name(name, args, tools, may_infer=shaped, strict_keys=bare)
        if res.name is not None and bare:
            # The tool is known now: put back any envelope-looking key it actually
            # declares, and resolve again against the arguments that will be sent.
            args = _bare_args(obj, name_key, _params_of(res.name, tools))
            res = resolve_name(name, args, tools, may_infer=shaped, strict_keys=bare)
        if res.name is not None:
            _record(text, span, res, args, out, strategy)
            return True
        reason = reason or res.reason
    if reason:
        out.rejected.append(reason)
    return False


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """The sampled text with the recovered call spans cut out, byte for byte.

    Deliberately not stripped. What is left is the prose the model sampled around
    the call, and the whitespace between the two was sampled as well: turn N+1's
    prompt has to reproduce turn N's bytes or the prompt cache misses on every
    tool-using turn (sec 8.5.5).
    """
    if not spans:
        return text
    out = []
    last = 0
    for start, end in sorted(spans):
        if start < last:
            continue
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out)


def looks_like_tool_intent(text: str) -> bool:
    """Did the model *try* to call a tool and fail?

    Gates the bounded retry (sec 8.5.4): re-prompting a turn that was legitimately
    prose wastes a whole generation and makes the model chattier, not more correct.
    """
    if not text:
        return False
    # Desmarten first: a model that emitted “name” instead of "name" was very much
    # trying to call a tool, and missing that is exactly the case worth retrying.
    low = desmarten(text).lower()
    markers = (
        "<tool_call",
        "<function_call",
        "<invoke",
        '"name"',
        '"arguments"',
        "tool_name",
        "function_call",
        "```json",
        "```tool",
    )
    if any(m in low for m in markers):
        return True
    # Bare function-call syntax: `some_tool(arg="x")` on its own.
    return bool(_FUNC_SYNTAX_RE.match(text.strip()) and "=" in text)
