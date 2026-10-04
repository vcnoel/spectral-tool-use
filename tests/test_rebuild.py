"""Unit tests of the clean pipeline's pure functions (no model, no data download).

    python -m pytest tests/test_rebuild.py -q
"""
import numpy as np
import pytest

from rebuild import bfcl_port, labels, prompts, spans, storage


def test_bfcl_port_self_test():
    assert bfcl_port.self_test()["pass"]


@pytest.mark.parametrize("text, name, values", [
    ('{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius", "days": 3}}',
     "get_weather", ["Paris", "celsius", "3"]),
    ('<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>',
     "get_weather", ["Paris"]),
    ('<function name="get_weather"><param name="city">Paris</param></function>', "get_weather", ["Paris"]),
])
def test_call_roles_dialects(text, name, values):
    calls = spans.call_roles(text)
    assert len(calls) == 1 and calls[0]["name"] == name
    assert [text[v[1]:v[2]] for v in calls[0]["values"]] == values
    assert all(text[v[3]:v[4]] == v[0] for v in calls[0]["values"])
    assert calls[0]["close_span"] is not None and calls[0]["args_span"] is not None


def test_call_roles_list_and_prose():
    two = spans.call_roles('[{"name": "f", "arguments": {"a": 1}}, {"name": "g", "arguments": {"b": "x"}}]')
    assert [c["name"] for c in two] == ["f", "g"]
    assert spans.call_roles("The weather in Paris is mild.") == []


def test_failure_type_taxonomy():
    assert labels.failure_type("missing_args", '{"name":"f","arguments":{"type":"object","properties":{}}}', False) == "schema_echo"
    assert labels.failure_type("missing_args", '{"name":"f","arguments":{}}', False) == "argument_set"
    assert labels.failure_type("wrong_arg_values", "x", True) == "truncated"
    assert labels.failure_type("missing_calls", "x", False) == "dropped_parallel_call"
    assert labels.failure_type("valid_nocall", "", False) == "valid"
    assert labels.clean_prediction("call<|im_end|>\n<|im_start|>user\nmore") == "call"


def test_prompt_spans_and_tool_segments():
    tools = [{"name": "alpha_tool", "description": "Does alpha", "parameters": {"type": "object", "properties": {"x": {}}}},
             {"name": "beta_tool", "description": "Does beta", "parameters": {"type": "object", "properties": {"y": {}}}}]
    text = 'system: tools: {"name": "alpha_tool", "description": "Does alpha", "x"} {"name": "beta_tool", "description": "Does beta", "y"}\nuser: find alpha please\nassistant:'
    cs = prompts.prompt_char_spans(text, tools, "find alpha please")
    assert cs["schema"] is not None and cs["request"] is not None
    assert cs["request"][0] >= cs["schema"][1]
    segs = prompts.tool_segment_spans(text, tools, cs["schema"])
    assert segs[0][0] < segs[1][0] == segs[0][1] and segs[1][1] == cs["schema"][1]


def test_compact_dtype_policy():
    a, d = storage.compact(np.array([1.0, 2.0], np.float32))
    assert d == "float16" and a.dtype == np.float16
    b, d2 = storage.compact(np.array([1e5, 1.0], np.float32))
    assert d2 == "float32"
    c, d3 = storage.compact(np.array([1, 2], np.int32))
    assert d3 == "int32"
