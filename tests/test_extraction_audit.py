"""
Regression tests for the extraction defects found in the 29 September audit,
run against the real tokenizers when they are in the local cache.

- The token-role positions were never found for MiniCPM's call format, so
  its probe read the end-of-sequence state three times.
- Qwen3.5 stopped only on <|endoftext|>, so generations ran past <|im_end|>.
"""
import pytest

from spectral_guardrails.probes.features import find_token_positions_v2
from run_pilot_v2 import stopping_ids


def _tok(name):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        pytest.skip(f"{name} tokenizer not cached")


CALLS = {
    "openbmb/MiniCPM5-2B": '<function name="get_weather"><param name="city">Paris</param>'
                           '<param name="unit">celsius</param></function>',
    "Qwen/Qwen3.5-0.8B": '<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n'
                         '</parameter>\n</function>\n</tool_call>',
    "meta-llama/Llama-3.2-1B-Instruct": '{"name": "get_weather", "parameters": {"city": "Paris"}}',
}


@pytest.mark.parametrize("model", list(CALLS))
def test_positions_found_on_every_dialect(model):
    tok = _tok(model)
    ids = tok(CALLS[model], add_special_tokens=False).input_ids
    pos = find_token_positions_v2(tok, ids, 0)
    assert pos["found"], model
    assert pos["t_func"] != pos["t_end"]
    name_piece = tok.decode(ids[pos["t_func"]:pos["t_func"] + 3])
    assert "get" in name_piece or "weather" in name_piece
    args_text = tok.decode([ids[i] for i in pos["t_args"]])
    assert "Paris" in args_text


def test_prose_is_not_found():
    tok = _tok("Qwen/Qwen3.5-0.8B")
    ids = tok("The weather in Paris is mild today.", add_special_tokens=False).input_ids
    assert not find_token_positions_v2(tok, ids, 0)["found"]


def test_qwen35_stops_at_end_of_turn():
    tok = _tok("Qwen/Qwen3.5-0.8B")

    class _Cfg:
        eos_token_id = tok.convert_tokens_to_ids("<|endoftext|>")

    class _Model:
        config = _Cfg()
        generation_config = None

    ids = stopping_ids(tok, _Model())
    assert tok.convert_tokens_to_ids("<|im_end|>") in ids
