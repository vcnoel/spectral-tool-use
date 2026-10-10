"""
Within-model swap (Oct 2026): CPU check that both arms receive identical inputs.

Renders every BFCL item of the extraction stream through each arm's tokenizer
exactly as run_pilot_v2.extract does (same render_tool_prompt, same flags),
and asserts that the rendered prompt text, the token ids, the prompt hashes,
the stopping ids and the decoding settings are identical across the arms.
No model weights are loaded.

Usage: python scripts_swap/check_arm_identity.py --a Qwen/Qwen3.5-4B-Base --b Qwen/Qwen3.5-4B
       [--benchmark bfcl] [--n 850] [--force-json]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # "" does not hide the GPU on this Windows build
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import run_pilot_v2 as rp  # noqa: E402
from transformers import AutoTokenizer, AutoConfig, GenerationConfig  # noqa: E402


class _Stub:
    """Carries the config/generation_config that stopping_ids reads."""
    def __init__(self, name):
        self.config = AutoConfig.from_pretrained(name)
        try:
            self.generation_config = GenerationConfig.from_pretrained(name)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.config)


def render_all(name, bench, n, force_json):
    rp.FORCE_JSON = force_json
    rp.THINKING_MODE = False
    tok = AutoTokenizer.from_pretrained(name)
    src = (rp.iter_bfcl_examples(n) if bench == "bfcl"
           else rp.iter_bfcl_examples(n, mix=rp.BFCL_LIVE_MIX))
    rows = []
    for ex in src:
        t = rp.render_tool_prompt(tok, ex["tools"], ex["user"])
        if t is None:
            rows.append((None, None, None))
            continue
        ids = tok(t, add_special_tokens=False).input_ids
        rows.append((hashlib.sha256(t.encode()).hexdigest(), len(ids), hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    stub = _Stub(name)
    stop = rp.stopping_ids(tok, stub)
    gc = stub.generation_config.to_dict()
    return tok, rows, stop, gc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--benchmark", default="bfcl")
    ap.add_argument("--n", type=int, default=850)
    ap.add_argument("--force-json", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ta, ra, sa, ga = render_all(a.a, a.benchmark, a.n, a.force_json)
    tb, rb, sb, gb = render_all(a.b, a.benchmark, a.n, a.force_json)
    assert len(ra) == len(rb), (len(ra), len(rb))
    same_text = sum(x[0] == y[0] for x, y in zip(ra, rb))
    same_ids = sum(x[2] == y[2] for x, y in zip(ra, rb))
    over = sum(1 for x in ra if x[1] is not None and x[1] > rp.MAX_PROMPT_TOKENS)
    keys = ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty")
    dec_a = {k: ga.get(k) for k in keys}
    dec_b = {k: gb.get(k) for k in keys}
    rep = {"arm_a": a.a, "arm_b": a.b, "benchmark": a.benchmark, "force_json": a.force_json,
           "n_items": len(ra), "same_prompt_text": same_text, "same_token_ids": same_ids,
           "n_unrenderable_a": sum(x[0] is None for x in ra), "n_unrenderable_b": sum(x[0] is None for x in rb),
           "n_over_max_prompt_tokens": over, "max_prompt_tokens": rp.MAX_PROMPT_TOKENS,
           "stop_ids_a": sa, "stop_ids_b": sb, "stop_ids_equal": sa == sb,
           "generation_config_a": dec_a, "generation_config_b": dec_b,
           "decoding_in_extract": "greedy (do_sample=False), max_new_tokens=%d, eos=stop ids" % rp.MAX_NEW_TOKENS,
           "eos_token_a": ta.eos_token, "eos_token_b": tb.eos_token,
           "pad_used_a": ta.eos_token_id, "pad_used_b": tb.eos_token_id,
           "vocab_equal": ta.get_vocab() == tb.get_vocab(),
           "chat_template_equal": ta.chat_template == tb.chat_template}
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    assert same_text == len(ra) and same_ids == len(ra), "prompts differ between arms"
    assert sa == sb, "stopping ids differ between arms"
    print("IDENTITY OK")


if __name__ == "__main__":
    main()
