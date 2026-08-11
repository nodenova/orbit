"""Tool-call reliability (spec sec 8.5).

The malformed shapes here are the ones the spec names, plus the ones that turn up
alongside them. Each is a real failure mode, not a hypothetical: a model that emits
any of these puts the harness into the infinite "let me do that" loop that A0 and
this layer exist to break.
"""

from __future__ import annotations

import pytest

from orbit.gateway.toolcall.constrain import tool_call_schema
from orbit.gateway.toolcall.repair import (
    _balance,
    find_json_objects,
    lenient_loads,
    looks_like_tool_intent,
    repair,
    resolve_name,
)
from orbit.gateway.toolcall.replay import ReplayMap, coverage, render_call
from orbit.types import Message, Role, ToolCall, ToolDef

READ = ToolDef(
    name="read_file",
    description="Read a file",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"],
    },
)
BASH = ToolDef(
    name="run_bash",
    description="Run a command",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
        "required": ["command"],
    },
)
TOOLS = (READ, BASH)


@pytest.mark.parametrize(
    "label,text,expect_name,expect_args",
    [
        (
            "xml_hybrid",
            (
                "<tool_call>\n<tool_name>read_file</tool_name>\n"
                "<arguments><path>src/a.py</path><limit>10</limit></arguments>\n</tool_call>"
            ),
            "read_file",
            {"path": "src/a.py", "limit": 10},
        ),
        (
            "xml_invoke_attrs",
            '<invoke name="run_bash"><parameter name="command">pytest -q</parameter></invoke>',
            "run_bash",
            {"command": "pytest -q"},
        ),
        (
            "xml_json_args",
            (
                "<tool_call><tool_name>read_file</tool_name>"
                '<arguments>{"path": "x.py"}</arguments></tool_call>'
            ),
            "read_file",
            {"path": "x.py"},
        ),
        (
            "fenced",
            'Sure.\n```json\n{"name": "read_file", "arguments": {"path": "x.py"}}\n```',
            "read_file",
            {"path": "x.py"},
        ),
        (
            "bare_object",
            '{"name": "read_file", "path": "y.py"}',
            "read_file",
            {"path": "y.py"},
        ),
        (
            "trailing_comma",
            '{"name": "read_file", "arguments": {"path": "z.py",}}',
            "read_file",
            {"path": "z.py"},
        ),
        (
            "smart_quotes",
            "{“name”: “read_file”, “arguments”: {“path”: “q.py”}}",
            "read_file",
            {"path": "q.py"},
        ),
        (
            "function_syntax",
            'run_bash(command="ls -la", timeout=30)',
            "run_bash",
            {"command": "ls -la", "timeout": 30},
        ),
        (
            "truncated_json",
            '{"name": "read_file", "arguments": {"path": "t.py"',
            "read_file",
            {"path": "t.py"},
        ),
        (
            "openai_envelope",
            (
                '{"type":"function","function":{"name":"run_bash",'
                '"arguments":"{\\"command\\": \\"echo hi\\"}"}}'
            ),
            "run_bash",
            {"command": "echo hi"},
        ),
        (
            "tool_input_key",
            '{"tool": "read_file", "tool_input": {"path": "w.py"}}',
            "read_file",
            {"path": "w.py"},
        ),
    ],
)
def test_repair_recovers_malformed_shapes(label, text, expect_name, expect_args):
    out = repair(text, TOOLS)
    assert out.ok, f"{label}: nothing recovered (rejected={out.rejected})"
    assert out.calls[0].name == expect_name
    assert out.calls[0].arguments == expect_args


def test_mangled_name_is_inferred_from_argument_keys():
    """Sec 8.5.3: infer tool names from parameter keys when the name is mangled."""
    out = repair(
        '{"name": "tool‑call‑???", "arguments": {"command": "make test"}}', TOOLS
    )
    assert out.ok
    assert out.calls[0].name == "run_bash"


def test_unknown_tool_is_rejected_not_invented():
    """The model must not be able to invent a tool (sec 8.5.3)."""
    out = repair('{"name": "exfiltrate", "arguments": {"url": "http://evil"}}', TOOLS)
    assert not out.ok
    assert out.rejected


def test_ambiguous_arguments_refuse_rather_than_guess():
    a = ToolDef(
        name="alpha",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    b = ToolDef(
        name="beta",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    res = resolve_name(None, {"x": "1"}, (a, b))
    assert res.name is None
    assert "refusing to guess" in res.reason


def test_prose_is_not_a_tool_call():
    out = repair("I think the fix is in utils.py, around the retry loop.", TOOLS)
    assert not out.ok
    assert not out.rejected


def test_residual_prose_survives_recovery():
    text = 'Let me read it.\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```'
    out = repair(text, TOOLS)
    assert out.ok
    assert "Let me read it." in out.residual_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<tool_call>x</tool_call>", True),
        ('{"name": "read_file"}', True),
        ("“name”: “read_file”", True),
        ('run_bash(command="ls")', True),
        ("The retry loop is in utils.py.", False),
        ("", False),
    ],
)
def test_tool_intent_detection_gates_the_retry(text, expected):
    assert looks_like_tool_intent(text) is expected


def test_balance_closes_truncated_structures():
    assert lenient_loads(_balance('{"a": [1, 2')) == {"a": [1, 2]}
    assert lenient_loads(_balance('{"a": "unterminated')) == {"a": "unterminated"}


def test_find_json_objects_handles_nesting_and_strings():
    text = 'prefix {"a": {"b": 1}} middle {"c": "}"} suffix'
    found = find_json_objects(text)
    assert found == ['{"a": {"b": 1}}', '{"c": "}"}']


def test_tool_call_schema_closes_the_name_enum():
    """Constrained decoding must make tool invention impossible, not merely rare."""
    schema = tool_call_schema(TOOLS)
    names = {branch["properties"]["name"]["const"] for branch in schema["anyOf"]}
    assert names == {"read_file", "run_bash"}
    assert all(b["additionalProperties"] is False for b in schema["anyOf"])


# --- replay (sec 8.5.5) -----------------------------------------------------


def test_replay_preserves_exact_sampled_bytes():
    replay = ReplayMap(max_size=4)
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})
    sampled = (
        '{"name":"read_file",  "arguments":{"path":"a.py"}}'  # note the double space
    )
    replay.put("c1", sampled)
    assert render_call(call, replay) == sampled


def test_replay_falls_back_to_canonical_rendering():
    """A miss must render canonically, not fail: a miss is slow, a wrong prompt is wrong."""
    replay = ReplayMap()
    call = ToolCall(id="missing", name="read_file", arguments={"path": "a.py"})
    assert render_call(call, replay) == 'read_file\n{"path":"a.py"}'


def test_replay_map_is_bounded_lru():
    replay = ReplayMap(max_size=2)
    replay.put("a", "1")
    replay.put("b", "2")
    replay.get("a")  # refresh
    replay.put("c", "3")
    assert "a" in replay and "c" in replay
    assert "b" not in replay


def test_replay_coverage_predicts_cache_reach():
    replay = ReplayMap()
    replay.put("c1", "x")
    messages = [
        Message(role=Role.ASSISTANT, tool_calls=(ToolCall(id="c1", name="read_file"),)),
        Message(role=Role.ASSISTANT, tool_calls=(ToolCall(id="c2", name="run_bash"),)),
    ]
    assert coverage(messages, replay) == (1, 2)
