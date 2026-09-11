"""Failure-mode labeling (spectral_guardrails.probes.labeling).

These pin the label semantics the result files depend on: which text counts
as a call, and which failure mode each kind of mismatch receives.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectral_guardrails.probes.labeling import (  # noqa: E402
    classify_failure, classify_failure_anyof, extract_calls,
)

GT = '<functioncall> {"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}'


@pytest.mark.parametrize("text", [
    '{"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}',
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}</tool_call>',
    "Sure! <functioncall> {\"name\": \"get_weather\", "
    "\"arguments\": '{\"city\": \"Paris\", \"unit\": \"c\"}'}",
])
def test_extract_calls_handles_call_dialects(text):
    calls, looked = extract_calls(text)
    assert looked
    assert calls and calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"]["city"] == "Paris"


def test_extract_calls_prose_is_not_a_call():
    calls, looked = extract_calls("I'm sorry, I cannot check the weather.")
    assert calls is None and not looked


def test_extract_calls_unparseable_call_shape():
    calls, looked = extract_calls('<tool_call>{"name": "get_weather", "arguments": {</tool_call>')
    assert calls is None and looked


def test_classify_valid():
    pred = '{"name": "get_weather", "arguments": {"city": "paris", "unit": "C"}}'
    assert classify_failure(pred, GT) == (0, "valid")           # case-insensitive values


def test_classify_failure_modes():
    assert classify_failure("I cannot do that.", GT) == (1, "no_call")
    assert classify_failure('{"name": "get_weather", "arguments": {', GT)[1] == "unparseable_call"
    assert classify_failure('{"name": "get_news", "arguments": {"city": "Paris"}}', GT) \
        == (1, "wrong_name")
    assert classify_failure('{"name": "get_weather", "arguments": {"city": "Paris"}}', GT) \
        == (1, "missing_args")
    wrong = '{"name": "get_weather", "arguments": {"city": "London", "unit": "c"}}'
    assert classify_failure(wrong, GT) == (1, "wrong_arg_values")


def test_classify_requires_parseable_ground_truth():
    with pytest.raises(ValueError):
        classify_failure('{"name": "x", "arguments": {}}', "no call here")


def test_anyof_accepts_any_listed_value_and_optional_params():
    gt = [{"get_weather": {"city": ["Paris", "paris"], "unit": ["c", ""]}}]
    assert classify_failure_anyof('{"name": "get_weather", "arguments": {"city": "Paris"}}', gt) \
        == (0, "valid")
    assert classify_failure_anyof('{"name": "get_weather", "arguments": {"city": "Rome"}}', gt) \
        == (1, "wrong_arg_values")


def test_anyof_irrelevance_category():
    gt = [{"get_weather": {"city": ["Paris"]}}]
    assert classify_failure_anyof("There is no suitable tool.", gt, expect_call=False) \
        == (0, "valid_nocall")
    assert classify_failure_anyof('{"name": "get_weather", "arguments": {"city": "Paris"}}',
                                  gt, expect_call=False) == (1, "over_trigger")


def test_anyof_parallel_calls_missing_one():
    gt = [{"a": {"x": ["1"]}}, {"b": {"y": ["2"]}}]
    assert classify_failure_anyof('{"name": "a", "arguments": {"x": "1"}}', gt) \
        == (1, "missing_calls")
