"""Llama's native multi-call format reaches the parser: <|python_tag|> is stripped first."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pilot_v2 import cut_at_next_turn, strip_control_markers  # noqa: E402
from spectral_guardrails.probes.labeling import classify_failure_anyof  # noqa: E402

CALL = '{"name": "concert_booking.book_ticket", "parameters": {"artist": "Eminem", "city": "%s"}}'
GT = [{"concert_booking.book_ticket": {"artist": ["Eminem"], "city": ["New York City"]}},
      {"concert_booking.book_ticket": {"artist": ["Eminem"], "city": ["Los Angeles"]}}]


def _label(raw):
    return classify_failure_anyof(strip_control_markers(cut_at_next_turn(raw)), GT, True)


def test_python_tag_multi_call_is_parsed():
    raw = "<|python_tag|>" + CALL % "New York City" + "; " + CALL % "Los Angeles" + "<|eom_id|>"
    assert "<|python_tag|>" not in strip_control_markers(raw)
    assert _label(raw) == (0, "valid")


def test_python_tag_wrong_value_is_a_value_error():
    raw = "<|python_tag|>" + CALL % "New York City" + "; " + CALL % "Chicago" + "<|eom_id|>"
    assert _label(raw) == (1, "wrong_arg_values")
