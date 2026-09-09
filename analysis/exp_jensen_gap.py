"""
Measure the head-disagreement gap predicted by the theory section.

Proposition (Jensen gap). For the combinatorial Laplacian, L is linear in W,
and lambda_2(L) = min_{x perp 1, ||x||=1} x^T L x is a pointwise minimum of
linear functionals, hence CONCAVE in L. Therefore

    lambda_2(mean_h L_h)  >=  mean_h lambda_2(L_h),

with equality iff all heads share a Fiedler vector. Head averaging therefore
systematically over-states connectivity relative to the typical head, and the
gap is exactly the head disagreement that averaging discards.

This script measures the gap on real attention, for the combinatorial
Laplacian (where the inequality is a theorem) and for the symmetric
normalized Laplacian (where it is an empirical claim, since the normalized
Rayleigh quotient is not linear in W).

Writes data/theory/jensen_gap.json.
"""
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("data/theory/jensen_gap.json")
MODELS = [
    ("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    ("google/gemma-3-1b-it", "Gemma-3-1B"),
]
PROMPTS = [
    '{"name": "convert_currency", "arguments": {"amount": 500, "from": "USD"}}',
    '{"name": "search_restaurants", "arguments": {"cuisine": "Italian"}}',
    "The capital of France is Paris and the weather there is mild today.",
]


def lam2_combinatorial(W: torch.Tensor) -> float:
    W = W.to(torch.float64).clone()
    W.fill_diagonal_(0.0)
    L = torch.diag(W.sum(-1)) - W
    return float(torch.linalg.eigvalsh(0.5 * (L + L.T))[1].clamp(min=0.0))


def lam2_normalized(W: torch.Tensor) -> float:
    W = W.to(torch.float64).clone()
    W.fill_diagonal_(0.0)
    deg = W.sum(-1)
    inv = torch.where(deg > 1e-12, deg.clamp(min=1e-12).rsqrt(),
                      torch.zeros_like(deg))
    L = torch.eye(W.shape[0], dtype=torch.float64) - inv[:, None] * W * inv[None, :]
    return float(torch.linalg.eigvalsh(0.5 * (L + L.T))[1].clamp(min=0.0))


def fiedler_vector(W: torch.Tensor) -> torch.Tensor:
    W = W.to(torch.float64).clone()
    W.fill_diagonal_(0.0)
    L = torch.diag(W.sum(-1)) - W
    _, vecs = torch.linalg.eigh(0.5 * (L + L.T))
    return vecs[:, 1]


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    records = []
    for model_id, label in MODELS:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, device_map="cpu",
            attn_implementation="eager")
        model.eval()
        for pi, text in enumerate(PROMPTS):
            ids = tok(text, return_tensors="pt").input_ids
            with torch.no_grad():
                attns = model(input_ids=ids, output_attentions=True).attentions
            for li, a in enumerate(attns):
                heads = a[0]                                  # [H, T, T]
                sym = 0.5 * (heads + heads.transpose(-2, -1))
                per_head_comb = [lam2_combinatorial(sym[h]) for h in range(sym.shape[0])]
                per_head_norm = [lam2_normalized(sym[h]) for h in range(sym.shape[0])]
                mean_graph = sym.mean(dim=0)
                # pairwise Fiedler-vector alignment: 1 => heads agree
                fvs = torch.stack([fiedler_vector(sym[h]) for h in range(sym.shape[0])])
                fvs = fvs / fvs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                gram = (fvs @ fvs.T).abs()
                off = gram[~torch.eye(gram.shape[0], dtype=bool)]
                records.append({
                    "model": label,
                    "prompt": pi,
                    "layer": li,
                    "heads": int(sym.shape[0]),
                    "tokens": int(ids.shape[1]),
                    "lam2_comb_avg_graph": lam2_combinatorial(mean_graph),
                    "lam2_comb_mean_of_heads": float(np.mean(per_head_comb)),
                    "lam2_comb_min_head": float(np.min(per_head_comb)),
                    "lam2_norm_avg_graph": lam2_normalized(mean_graph),
                    "lam2_norm_mean_of_heads": float(np.mean(per_head_norm)),
                    "lam2_norm_min_head": float(np.min(per_head_norm)),
                    "per_head_norm_sd": float(np.std(per_head_norm)),
                    "fiedler_alignment_mean": float(off.mean()),
                })
        del model

    comb_gap = np.array([r["lam2_comb_avg_graph"] - r["lam2_comb_mean_of_heads"]
                         for r in records])
    norm_gap = np.array([r["lam2_norm_avg_graph"] - r["lam2_norm_mean_of_heads"]
                         for r in records])
    ratio = np.array([r["lam2_norm_avg_graph"] / max(r["lam2_norm_mean_of_heads"], 1e-12)
                      for r in records])
    align = np.array([r["fiedler_alignment_mean"] for r in records])
    sd = np.array([r["per_head_norm_sd"] for r in records])

    summary = {
        "n_layers_measured": len(records),
        "models": [m[1] for m in MODELS],
        "comb_gap_violations": int((comb_gap < -1e-9).sum()),
        "comb_gap_mean": float(comb_gap.mean()),
        "comb_gap_min": float(comb_gap.min()),
        "norm_gap_violations": int((norm_gap < -1e-9).sum()),
        "norm_gap_mean": float(norm_gap.mean()),
        "norm_gap_median": float(np.median(norm_gap)),
        "norm_ratio_mean": float(ratio.mean()),
        "norm_ratio_median": float(np.median(ratio)),
        "norm_ratio_max": float(ratio.max()),
        "fiedler_alignment_mean": float(align.mean()),
        "fiedler_alignment_max": float(align.max()),
        "per_head_sd_mean": float(sd.mean()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "records": records},
                              indent=2), encoding="utf-8")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
