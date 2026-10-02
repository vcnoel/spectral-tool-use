"""
Regression tests for the label defects found in the 29 September audit.

Each test is a correct or wrong call that the previous labeller got wrong.
All of them fell on the JSON call formats, so a mislabel here is a
family-dependent error, not random noise.
"""
from spectral_guardrails.probes.labeling import (
    classify_failure, classify_failure_anyof, extract_calls,
)


def test_list_argument_is_not_collapsed():
    pred = '{"name": "get_route", "parameters": {"coords": [37.7, -122.4], "mode": "car"}}'
    gt = [{"get_route": {"coords": [[37.7, -122.4]], "mode": ["car"]}}]
    assert classify_failure_anyof(pred, gt, True) == (0, "valid")


def test_single_item_list_argument_is_valid():
    pred = '{"name": "f", "parameters": {"tags": ["overview"]}}'
    gt = [{"f": {"tags": [["overview"]]}}]
    assert classify_failure_anyof(pred, gt, True) == (0, "valid")


def test_wrong_list_is_still_wrong():
    pred = '{"name": "get_route", "parameters": {"coords": [37.7, 0.0]}}'
    gt = [{"get_route": {"coords": [[37.7, -122.4]]}}]
    assert classify_failure_anyof(pred, gt, True) == (1, "wrong_arg_values")


def test_list_prefix_is_not_accepted():
    # the old unwrap compared only the first element
    pred = '{"name": "f", "parameters": {"xs": [1, 999]}}'
    gt = [{"f": {"xs": [[1, 2]]}}]
    assert classify_failure_anyof(pred, gt, True) == (1, "wrong_arg_values")


def test_semicolon_separated_parallel_calls_parse():
    text = '{"name": "a", "parameters": {"x": 1}}; {"name": "b", "parameters": {"y": 2}}'
    calls, looked = extract_calls(text)
    assert looked and [c["name"] for c in calls] == ["a", "b"]
    gt = [{"a": {"x": [1]}}, {"b": {"y": [2]}}]
    assert classify_failure_anyof(text, gt, True) == (0, "valid")


def test_list_sent_as_string_matches_list():
    pred = '{"name": "f", "parameters": {"genres": "[\\"War\\"]"}}'
    gt = [{"f": {"genres": [["War"]]}}]
    assert classify_failure_anyof(pred, gt, True) == (0, "valid")


def test_unclosed_tag_does_not_swallow_next_call():
    text = ('<tool_call>{"name": "a", "arguments": {"x": 1}}'
            '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>')
    calls, _ = extract_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]


def test_glaive_list_ground_truth_compares_elementwise():
    gt = '{"name": "f", "arguments": \'{"items": ["a", "b"]}\'}'
    assert classify_failure('{"name": "f", "arguments": {"items": ["a", "b"]}}', gt) == (0, "valid")
    assert classify_failure('{"name": "f", "arguments": {"items": ["a"]}}', gt)[0] == 1


def test_scalars_unchanged():
    gt = [{"f": {"n": [5], "city": ["Paris", "paris"]}}]
    assert classify_failure_anyof('{"name": "f", "parameters": {"n": 5.0, "city": "Paris"}}', gt, True) == (0, "valid")
    assert classify_failure_anyof('{"name": "f", "parameters": {"n": 6, "city": "Paris"}}', gt, True)[0] == 1


def test_minicpm_and_qwen_xml_unchanged():
    gt = [{"f": {"xs": [[1, 2]]}}]
    mini = '<function name="f"><param name="xs">[1, 2]</param></function>'
    qwen = '<tool_call><function=f><parameter=xs>[1, 2]</parameter></function></tool_call>'
    assert classify_failure_anyof(mini, gt, True) == (0, "valid")
    assert classify_failure_anyof(qwen, gt, True) == (0, "valid")


def test_python_literal_list_matches():
    pred = '{"name": "f", "parameters": {"genres": "[\'War\', \'Drama\']"}}'
    gt = [{"f": {"genres": [["War", "Drama"]]}}]
    assert classify_failure_anyof(pred, gt, True) == (0, "valid")


def test_bfcl_string_normalisation():
    gt = [{"f": {"expr": ["x**2 + 3x"], "date": ["April 1, 2024"]}}]
    pred = '{"name": "f", "parameters": {"expr": "x^2 + 3*x", "date": "April 1 2024"}}'
    assert classify_failure_anyof(pred, gt, True) == (0, "valid")
    wrong = '{"name": "f", "parameters": {"expr": "x^3 + 3*x", "date": "April 1 2024"}}'
    assert classify_failure_anyof(wrong, gt, True)[0] == 1


def test_values_read_before_echoed_schema():
    pred = ('{"name": "f", "arguments": {"radius": 10}, '
            '"parameters": {"type": "object", "properties": {"radius": {"type": "integer"}}}}')
    gt = [{"f": {"radius": [10]}}]
    for _ in range(3):
        assert classify_failure_anyof(pred, gt, True) == (0, "valid")
