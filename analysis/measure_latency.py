"""
What does each judge cost at the moment the call is generated?

A guardrail that judges the call before it executes adds latency to every
call. We time, on one GPU and one model, the pieces a deployment would pay:

  generation            the model producing the call (the baseline cost)
  teacher-forced pass   one forward pass over prompt + call with eager
                        attention, hidden states on; needed by every
                        internal judge because the generation pass under a
                        fused kernel does not expose attention
  per-head features     L x H eigendecompositions on the call span
  LapEigvals features   column sums and a sort per layer
  token-role features   gathering three positions at eight depths
  probe inference       one logistic regression call

Median over repeated prompts, on the BFCL single-turn prompts. Writes
data/theory/latency.json.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from run_pilot_v2 import render_tool_prompt, iter_bfcl_examples  # noqa: E402

from spectral_guardrails.spectral.metrics import (  # noqa: E402
    lapeigvals_diag_profile, per_head_metrics,
)
from spectral_guardrails.spectral.streaming import StreamingAttentionFeatures  # noqa: E402

MODEL = os.environ.get("LAT_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
N_PROMPTS = int(os.environ.get("LAT_N", "20"))
OUT = Path("data/theory/latency.json")


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    L = model.config.num_hidden_layers
    probe_layers = np.linspace(1, L, 8).round().astype(int).tolist()

    times = {k: [] for k in ("generation", "teacher_forced_pass_only",
                             "teacher_forced_pass_with_perhead",
                             "perhead_features_alone", "lapeig_features_alone",
                             "token_role_gather", "gen_tokens", "prompt_tokens")}
    examples = list(iter_bfcl_examples(N_PROMPTS + 6))[:N_PROMPTS + 1]
    for i, ex in enumerate(examples):
        warmup = i == 0          # first call pays kernel and solver start-up
        prompt = render_tool_prompt(tok, ex["tools"], ex["user"])
        ids = tok(prompt, return_tensors="pt").to("cuda")
        P = ids.input_ids.shape[1]

        t0 = sync()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        t1 = sync()
        full = out[:, :]
        T = full.shape[1]
        G = T - P
        span = (P, T)

        # plain teacher-forced pass with hidden states (what a residual probe
        # needs); attention weights are computed by the eager kernel anyway
        with torch.no_grad():
            mo = model(input_ids=full, output_hidden_states=True)
        t2 = sync()

        # the same pass with per-head features reduced inside the hook
        feats = []

        def reducer(layer_idx, w):
            feats.append(per_head_metrics(w[0], span=span))
            return None
        with torch.no_grad(), StreamingAttentionFeatures(model, reducer):
            mo = model(input_ids=full, output_hidden_states=True)
        t3 = sync()

        # feature costs alone, on stored attention of one pass
        with torch.no_grad():
            mo_a = model(input_ids=full, output_attentions=True,
                         output_hidden_states=True)
        attn = [a[0] for a in mo_a.attentions]
        t4 = sync()
        for a in attn:
            per_head_metrics(a, span=span)
        t5 = sync()
        for a in attn:
            # the tensor work of the LapEigvals features only; the stored
            # profile's conversion to Python lists is bookkeeping
            Tn = a.shape[-1]
            denom = torch.arange(1, Tn + 1, device=a.device,
                                 dtype=torch.float32).flip(0)
            lap_diag = a.sum(dim=1, dtype=torch.float32) / denom                 - torch.diagonal(a, dim1=1, dim2=2).to(torch.float32)
            lap_diag.sort(dim=-1, descending=True).values[:, :100]
        t6 = sync()
        hs = mo_a.hidden_states
        _ = torch.cat([torch.cat([hs[l][0, P], hs[l][0, P:T].mean(0), hs[l][0, T - 1]])
                       for l in probe_layers]).float().cpu()
        t7 = sync()

        if warmup:
            continue
        times["generation"].append(t1 - t0)
        times["teacher_forced_pass_only"].append(t2 - t1)
        times["teacher_forced_pass_with_perhead"].append(t3 - t2)
        times["perhead_features_alone"].append(t5 - t4)
        times["lapeig_features_alone"].append(t6 - t5)
        times["token_role_gather"].append(t7 - t6)
        times["gen_tokens"].append(G)
        times["prompt_tokens"].append(P)

    summary = {k: {"median_ms": 1000 * statistics.median(v),
                   "p90_ms": 1000 * float(np.percentile(v, 90))}
               if k not in ("gen_tokens", "prompt_tokens")
               else {"median": statistics.median(v)}
               for k, v in times.items()}
    summary["model"] = MODEL
    summary["n_prompts"] = len(examples) - 1
    summary["layers"] = L
    summary["heads"] = model.config.num_attention_heads
    summary["gpu"] = torch.cuda.get_device_name(0)
    for k, v in summary.items():
        print(f"{k:36s} {v}")
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
