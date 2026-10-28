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
    ('<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>',  # noqa: E501  (fixture literal)
     "get_weather", ["Paris"]),
    ('<function name="get_weather"><param name="city">Paris</param></function>', "get_weather", ["Paris"]),  # noqa: E501  (fixture literal)
])
def test_call_roles_dialects(text, name, values):
    calls = spans.call_roles(text)
    assert len(calls) == 1 and calls[0]["name"] == name
    assert [text[v[1]:v[2]] for v in calls[0]["values"]] == values
    assert all(text[v[3]:v[4]] == v[0] for v in calls[0]["values"])
    assert calls[0]["close_span"] is not None and calls[0]["args_span"] is not None


def test_call_roles_list_and_prose():
    two = spans.call_roles('[{"name": "f", "arguments": {"a": 1}}, {"name": "g", "arguments": {"b": "x"}}]')  # noqa: E501  (fixture literal)
    assert [c["name"] for c in two] == ["f", "g"]
    assert spans.call_roles("The weather in Paris is mild.") == []


def test_failure_type_taxonomy():
    assert labels.failure_type("missing_args", '{"name":"f","arguments":{"type":"object","properties":{}}}', False) == "schema_echo"  # noqa: E501  (fixture literal)
    assert labels.failure_type("missing_args", '{"name":"f","arguments":{}}', False) == "argument_set"  # noqa: E501  (fixture literal)
    assert labels.failure_type("wrong_arg_values", "x", True) == "truncated"
    assert labels.failure_type("missing_calls", "x", False) == "dropped_parallel_call"
    assert labels.failure_type("valid_nocall", "", False) == "valid"
    assert labels.clean_prediction("call<|im_end|>\n<|im_start|>user\nmore") == "call"


def test_prompt_spans_and_tool_segments():
    tools = [{"name": "alpha_tool", "description": "Does alpha", "parameters": {"type": "object", "properties": {"x": {}}}},  # noqa: E501  (fixture literal)
             {"name": "beta_tool", "description": "Does beta", "parameters": {"type": "object", "properties": {"y": {}}}}]  # noqa: E501  (fixture literal)
    text = 'system: tools: {"name": "alpha_tool", "description": "Does alpha", "x"} {"name": "beta_tool", "description": "Does beta", "y"}\nuser: find alpha please\nassistant:'  # noqa: E501  (fixture literal)
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


def test_multi_call_dialects_decode():
    """Every dialect the evaluated models write for two parallel calls must decode to two calls
    after clean_prediction. Added after the 5 Oct 2026 labeller bug (Amendment 3): Llama's
    '<|python_tag|>{..}; {..}' was rejected because the tag was not stripped."""
    from rebuild.labels import clean_prediction
    from spectral_guardrails.probes.labeling import extract_calls
    cases = {
        "llama_semicolon": '<|python_tag|>{"name": "f", "parameters": {"a": 1}}; '
                           '{"name": "f", "parameters": {"a": 2}}<|eom_id|>',
        "qwen_tool_call": '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>\n'
                          '<tool_call>\n{"name": "f", "arguments": {"a": 2}}\n</tool_call><|im_end|>',  # noqa: E501  (fixture literal)
        "json_list": '[{"name": "f", "arguments": {"a": 1}}, {"name": "f", "arguments": {"a": 2}}]',
        "newline_objects": '{"name": "f", "arguments": {"a": 1}}\n{"name": "f", "arguments": {"a": 2}}',  # noqa: E501  (fixture literal)
    }
    for dialect, text in cases.items():
        calls, looked = extract_calls(clean_prediction(text))
        assert looked and calls is not None and len(calls) == 2, (dialect, calls)
    single, _ = extract_calls(clean_prediction('<|python_tag|>{"name": "f", "parameters": {"a": 1}}<|eom_id|>'))  # noqa: E501  (fixture literal)
    assert single is not None and len(single) == 1


def test_default_probe_depth_is_a_stored_layer():
    """The fractional default depths of the published probes must resolve to a stored layer.
    Added after the 28 Oct 2026 pilot gate crash: round(2 L / 3) = 11 on a 16-layer model is
    not stored under --hidden-layers half."""
    from rebuild.loader import CleanRun
    run = object.__new__(CleanRun)
    for L, stored, want_pre, want_last in (
            (16, [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16], 10, 12),
            (28, [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], 18, 22)):
        run.meta, run.hidden_layers = {"n_layers": L}, stored
        assert run.nearest_stored(2 / 3) == want_pre
        assert run.nearest_stored(0.75) == want_last
