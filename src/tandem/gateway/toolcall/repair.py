"""Tool-call repair (spec sec 8.5.3).

This is what actually breaks. Local models emit XML/JSON hybrids, fenced blocks,
bare objects, Python-ish call syntax, trailing commas, smart quotes and truncated
objects. The harness sees no valid call, re-prompts, the model garbles identically,
and the session enters an infinite "let me do that" loop.

The strategies below are ordered cheapest-and-most-certain first, and each one is
tried against the whole text before the next. Two rules run through all of them:

* **Names are never invented.** A parsed call whose name is not in the request's
  tool list is rejected, not passed through. A model that hallucinates
  `run_arbitrary_command` must get a rejection, not an execution.
* **A mangled name is inferred from the argument keys**, not guessed from string
  similarity alone — if the keys uniquely identify one tool's required parameters,
  that is strong evidence; if two tools match equally well, we refuse rather than
  pick.

Repair is the third line of defence, behind constrained decoding (sec 8.5.1) and the
A0 harness adapter (sec 6.1). Experiment E1 asks whether A0 makes this layer
droppable; until that answers yes, it stays.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ...types import ToolCall, ToolDef

# --- text normalisation -----------------------------------------------------

_SMART = {
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    "‘": "'",
    "’": "'",
    "′": "'",
    "«": '"',
    "»": '"',
    # Non-ASCII dashes turn up inside mangled tool names.
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
}
_SMART_RE = re.compile("|".join(re.escape(k) for k in _SMART))

_FENCE_RE = re.compile(r"```(?:json|tool_call|python|xml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def desmarten(text: str) -> str:
    return _SMART_RE.sub(lambda m: _SMART[m.group(0)], text)


def strip_fences(text: str) -> list[str]:
    """Bodies of every fenced block, plus the text with fences removed."""
    bodies = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    return [b for b in bodies if b]


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
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()
    out = text
    if in_str:
        out += '"'
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def lenient_loads(text: str) -> Any | None:
    """json.loads with the malformations local models actually produce fixed up."""
    candidates = [text]
    de = desmarten(text)
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


def find_json_objects(text: str) -> list[str]:
    """Every balanced top-level {...} span in the text, in order.

    Brace-counting with string awareness rather than a regex: a regex cannot match
    nested objects, and tool arguments are routinely nested.
    """
    out: list[str] = []
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
                    out.append(text[start : i + 1])
                    start = -1
    if depth > 0 and start >= 0:
        # Truncated tail — hand it over anyway; _balance may rescue it.
        out.append(text[start:])
    return out


# --- extraction shapes ------------------------------------------------------

_XML_CALL_RE = re.compile(
    r"<(?:tool_call|function_call|invoke)\b[^>]*>(.*?)</(?:tool_call|function_call|invoke)>",
    re.DOTALL | re.IGNORECASE,
)
_XML_NAME_ATTR_RE = re.compile(r"""<(?:invoke|tool_call|function_call)\b[^>]*\bname\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_XML_NAME_TAG_RE = re.compile(
    r"<(?:tool_name|function_name|name)>\s*(.*?)\s*</(?:tool_name|function_name|name)>",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAM_ATTR_RE = re.compile(
    r"""<parameter\b[^>]*\bname\s*=\s*["']([^"']+)["']\s*>(.*?)</parameter>""",
    re.DOTALL | re.IGNORECASE,
)
_XML_ARGS_BLOCK_RE = re.compile(
    r"<(?:arguments|args|parameters)>(.*?)</(?:arguments|args|parameters)>", re.DOTALL | re.IGNORECASE
)
_XML_GENERIC_TAG_RE = re.compile(r"<([A-Za-z_][\w.-]*)>(.*?)</\1>", re.DOTALL)

_FUNC_SYNTAX_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*\((.*)\)\s*;?\s*$", re.DOTALL)


def _coerce_scalar(raw: str) -> Any:
    s = raw.strip()
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
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _args_from_xml(block: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for name, value in _XML_PARAM_ATTR_RE.findall(block):
        args[name] = _coerce_scalar(value)
    if args:
        return args
    inner = block
    m = _XML_ARGS_BLOCK_RE.search(block)
    if m:
        inner = m.group(1)
        # <arguments>{"a": 1}</arguments> is common enough to be worth trying first.
        parsed = lenient_loads(inner.strip())
        if isinstance(parsed, dict):
            return parsed
    for name, value in _XML_GENERIC_TAG_RE.findall(inner):
        if name.lower() in ("tool_name", "function_name", "name", "arguments", "args", "parameters"):
            continue
        args[name] = _coerce_scalar(value)
    return args


def _split_kwargs(body: str) -> dict[str, Any]:
    """Split `a=1, b="x,y"` respecting quotes, brackets and escapes."""
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
            args[key] = _coerce_scalar(value)
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


def resolve_name(
    raw_name: str | None, args: dict[str, Any], tools: Iterable[ToolDef]
) -> NameResolution:
    """Map a possibly-mangled name onto a real tool, or reject.

    Never returns a name that is not in `tools` — the spec's rule that the model
    cannot invent a tool is enforced structurally rather than checked later.
    """
    by_name = {t.name: t for t in tools}
    if not by_name:
        return NameResolution(None, "rejected", "no tools available on this request")

    if raw_name and raw_name in by_name:
        return NameResolution(raw_name, "exact")

    if raw_name:
        norm = _normalise_name(raw_name)
        matches = [n for n in by_name if _normalise_name(n) == norm]
        if len(matches) == 1:
            return NameResolution(matches[0], "normalised", f"{raw_name!r} -> {matches[0]!r}")

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

    # Infer from argument keys (sec 8.5.3). Score = does the tool's required set fit
    # inside the observed keys, then overlap fraction. Ties refuse rather than guess.
    keys = set(args)
    if not keys:
        return NameResolution(
            None, "rejected", f"unknown tool {raw_name!r} and no arguments to infer from"
        )
    scored: list[tuple[float, str]] = []
    for name, tool in by_name.items():
        params = set(tool.param_names())
        required = set(tool.required_names())
        if required and not required.issubset(keys):
            continue
        if not params:
            continue
        overlap = len(keys & params)
        if overlap == 0:
            continue
        union = len(keys | params)
        scored.append((overlap / union, name))
    if not scored:
        return NameResolution(None, "rejected", f"unknown tool {raw_name!r}; no tool matches its arguments")
    scored.sort(reverse=True)
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-9:
        return NameResolution(
            None,
            "rejected",
            f"arguments match {scored[0][1]!r} and {scored[1][1]!r} equally; refusing to guess",
        )
    return NameResolution(scored[0][1], "inferred", f"inferred from argument keys {sorted(keys)}")


# --- the repair pass --------------------------------------------------------


@dataclass
class RepairOutcome:
    calls: tuple[ToolCall, ...] = ()
    # Text with the recovered call spans removed, so prose around a call survives.
    residual_text: str = ""
    repaired: bool = False
    strategy: str = ""
    rejected: list[str] = field(default_factory=list)
    raw_blocks: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.calls)


def _call_id(name: str, args: dict[str, Any], salt: str = "") -> str:
    material = f"{name}\x00{json.dumps(args, sort_keys=True, default=str)}\x00{salt}"
    return "call_" + hashlib.sha256(material.encode()).hexdigest()[:16]


def _accept(
    raw_name: str | None,
    args: dict[str, Any],
    tools: Iterable[ToolDef],
    raw_text: str,
    out: RepairOutcome,
    strategy: str,
) -> bool:
    res = resolve_name(raw_name, args, tools)
    if res.name is None:
        out.rejected.append(res.reason)
        return False
    call = ToolCall(id=_call_id(res.name, args, str(len(out.calls))), name=res.name, arguments=args)
    out.calls = out.calls + (call,)
    out.raw_blocks[call.id] = raw_text
    out.repaired = True
    out.strategy = strategy if not out.strategy else f"{out.strategy}+{strategy}"
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
    for m in _XML_CALL_RE.finditer(text):
        block = m.group(0)
        inner = m.group(1)
        name = None
        attr = _XML_NAME_ATTR_RE.search(block)
        if attr:
            name = attr.group(1)
        else:
            tag = _XML_NAME_TAG_RE.search(inner)
            if tag:
                name = tag.group(1)
        args = _args_from_xml(inner)
        if _accept(name, args, tools, block, out, "xml"):
            consumed.append(m.span())
    if out.calls:
        out.residual_text = _remove_spans(text, consumed)
        return out

    # 2. Fenced blocks.
    for body in strip_fences(text):
        if _try_json_payload(body, tools, out, "fenced"):
            pass
    if out.calls:
        out.residual_text = _FENCE_RE.sub("", text).strip()
        return out

    # 3. Embedded JSON objects, longest first — a wrapper object should be tried
    #    before the argument object nested inside it.
    objects = sorted(find_json_objects(text), key=len, reverse=True)
    for raw in objects:
        if _try_json_payload(raw, tools, out, "json"):
            break
    if out.calls:
        for raw in objects:
            if raw in text:
                text = text.replace(raw, "", 1)
        out.residual_text = text.strip()
        return out

    # 4. Function-call syntax: name(a=1, b="x").
    fm = _FUNC_SYNTAX_RE.match(text.strip())
    if fm:
        args = _split_kwargs(fm.group(2))
        if _accept(fm.group(1), args, tools, text.strip(), out, "function_syntax"):
            out.residual_text = ""
            return out

    return out


def _try_json_payload(raw: str, tools: tuple[ToolDef, ...], out: RepairOutcome, strategy: str) -> bool:
    obj = lenient_loads(raw)
    if obj is None:
        return False
    if isinstance(obj, list):
        got = False
        for item in obj:
            if isinstance(item, dict):
                got |= _try_dict(item, raw, tools, out, strategy)
        return got
    if isinstance(obj, dict):
        return _try_dict(obj, raw, tools, out, strategy)
    return False


_NAME_KEYS = ("name", "tool", "tool_name", "function", "function_name", "recipient_name")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input", "tool_input")


def _try_dict(obj: dict[str, Any], raw: str, tools: tuple[ToolDef, ...], out: RepairOutcome, strategy: str) -> bool:
    # OpenAI-style envelope: {"type": "function", "function": {"name":..,"arguments":..}}
    inner = obj.get("function")
    if isinstance(inner, dict) and ("name" in inner or "arguments" in inner):
        obj = {**inner}

    name: str | None = None
    for key in _NAME_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value:
            name = value
            break

    args: dict[str, Any] | None = None
    for key in _ARG_KEYS:
        value = obj.get(key)
        if isinstance(value, dict):
            args = value
            break
        if isinstance(value, str):
            # Arguments as a JSON *string* is the OpenAI wire shape.
            parsed = lenient_loads(value)
            if isinstance(parsed, dict):
                args = parsed
                break

    if args is None:
        # Bare object: everything that is not a name key is an argument.
        args = {k: v for k, v in obj.items() if k not in _NAME_KEYS and k != "type"}

    if not args and not name:
        return False
    return _accept(name, args, tools, raw, out, strategy)


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    out = []
    last = 0
    for start, end in sorted(spans):
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out).strip()


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
