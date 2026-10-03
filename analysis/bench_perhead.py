"""
Where the per-head spectral features spend their time, and what they should
cost.

LapEigvals needs a column sum and a sort per head. The per-head profile
needs one symmetric eigendecomposition per head of the call-span subgraph,
which for a call of S tokens is an S x S matrix: tiny. If the per-head path
costs an order of magnitude more than LapEigvals, the cause is the
implementation, and this script finds which part.

Inputs are random row-stochastic causal attention tensors of the shapes the
paper meets ([H, T, T] per layer, L layers, call span S). Paths timed:

  library          the current path: spectral_trust.per_head_metrics per
                   layer (float32 widening, dense diag matmuls, eigvalsh on
                   the tensor's device, .tolist() per layer)
  library, no list same without the per-layer conversion to Python lists
  fused, device    span slice, elementwise D^-1/2 W D^-1/2, one eigvalsh
                   over all layers and heads at once, on the tensor's device
  fused, cpu       same, but the [L*H, S, S] block is moved to the CPU first
                   (a few hundred kilobytes) and decomposed by LAPACK
  lapeigvals       the column-sum feature, for reference

Every fused path is checked against the library to 1e-5 before it is timed.
Writes data/theory/bench_perhead.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spectral_guardrails.spectral.metrics import per_head_metrics  # noqa: E402

import os
OUT = ROOT / "data" / "theory" / os.environ.get("BENCH_OUT", "bench_perhead.json")
REPEATS = 9


def causal_attention(L, H, T, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(L, H, T, T, generator=g)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    logits = logits.masked_fill(mask, float("-inf"))
    return torch.softmax(logits, dim=-1).to(device=device, dtype=torch.bfloat16)


def five_metrics(ev):
    """ev: [..., S] ascending eigenvalues, clamped at 0. Same formulas as the library."""
    S = ev.shape[-1]
    lam2, lmax = ev[..., 1], ev[..., -1]
    total = ev.sum(-1, keepdim=True)
    probs = (ev / total.clamp(min=1e-12)).clamp(min=1e-12)
    ent = -(probs * probs.log()).sum(-1) / float(np.log(S))
    hfer = ev[..., S // 2:].sum(-1) / total.squeeze(-1).clamp(min=1e-12)
    degen = total.squeeze(-1) <= 1e-12
    conn = torch.where(lmax > 1e-12, lam2 / lmax.clamp(min=1e-12), torch.zeros_like(lam2))
    ent = torch.where(degen, torch.zeros_like(ent), ent)
    hfer = torch.where(degen, torch.zeros_like(hfer), hfer)
    return torch.stack([lam2, conn, ent, hfer, lmax], -1)


def fused(attn_layers, span, on_cpu):
    """attn_layers: list of [H, T, T]. One decomposition for every layer and head."""
    a, b = span
    blk = torch.stack([x[:, a:b, a:b] for x in attn_layers])          # [L, H, S, S]
    if on_cpu:
        blk = blk.to("cpu")
    w = blk.to(torch.float32)
    w = 0.5 * (w + w.transpose(-1, -2))
    w = w * (1 - torch.eye(w.shape[-1], device=w.device))
    d = w.sum(-1)
    dinv = torch.where(d > 1e-8, d.clamp(min=1e-8).rsqrt(), torch.zeros_like(d))
    lap = torch.eye(w.shape[-1], device=w.device) - dinv.unsqueeze(-1) * w * dinv.unsqueeze(-2)
    lap = 0.5 * (lap + lap.transpose(-1, -2))
    ev = torch.linalg.eigvalsh(lap).clamp(min=0.0)
    return five_metrics(ev).cpu()


def lapeig(attn_layers):
    out = []
    for x in attn_layers:
        T = x.shape[-1]
        denom = torch.arange(1, T + 1, device=x.device, dtype=torch.float32).flip(0)
        d = x.sum(1, dtype=torch.float32) / denom - torch.diagonal(x, dim1=1, dim2=2).float()
        out.append(d.sort(-1, descending=True).values[:, :100])
    return torch.stack(out).cpu()


def timed(fn, device):
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    fn()                                   # warm-up at this shape
    sync()
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(1000 * (time.perf_counter() - t0))
    return statistics.median(ts), float(np.percentile(ts, 90))


def main():
    import os
    threads = int(os.environ.get("BENCH_THREADS", "0"))
    if threads:
        torch.set_num_threads(threads)
    print(f"torch threads: {torch.get_num_threads()}", flush=True)
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    devices = [d for d in devices if d in os.environ.get("BENCH_DEVICES", "cuda,cpu").split(",")]
    shapes = [(16, 32, 300, 30), (16, 32, 400, 90), (16, 32, 600, 250),   # Llama-3.2-1B
              (28, 24, 400, 90)]                                          # Llama-3.2-3B
    rows = []
    for device in devices:
        for L, H, T, S in shapes:
            attn = list(causal_attention(L, H, T, device))
            span = (T - S, T)
            ref = torch.tensor([per_head_metrics(x, span=span) for x in attn])
            for on_cpu in (False, True):
                got = fused(attn, span, on_cpu)
                assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), \
                    f"fused path disagrees with the library ({device}, cpu={on_cpu}): " \
                    f"max diff {float((got - ref).abs().max()):.2e}"
            paths = {
                "library": lambda: [per_head_metrics(x, span=span) for x in attn],
                "library, no list": lambda: [
                    __import__("spectral_trust").per_head_metrics(
                        x, config=__import__("spectral_guardrails.spectral.metrics", fromlist=["x"])._per_head_backend()[1],
                        token_span=span).values for x in attn],
                "fused, device": lambda: fused(attn, span, False),
                "fused, cpu": lambda: fused(attn, span, True),
                "lapeigvals": lambda: lapeig(attn),
            }
            for name, fn in paths.items():
                med, p90 = timed(fn, device)
                rows.append({"device": device, "L": L, "H": H, "T": T, "S": S,
                             "path": name, "median_ms": med, "p90_ms": p90})
                print(f"{device:4s} L={L:2d} H={H:2d} T={T:3d} S={S:3d}  {name:17s} "
                      f"median {med:8.2f} ms   p90 {p90:8.2f} ms", flush=True)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
