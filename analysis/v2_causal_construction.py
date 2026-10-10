"""
v2 (October 2026): a causal two-head construction for Proposition 1 and for the scope
of Proposition 2 (head averaging and algebraic connectivity).

Two causal, row-stochastic heads on T positions:
  previous-token head  A[i, i-1] = 1 (i >= 1), A[0, 0] = 1
  sink head            A[i, 0]   = 1
Configuration one keeps them as two heads. Configuration two replaces both by their
average, which is itself a causal row-stochastic head. Both configurations have the same
head-averaged graph. Each head is symmetrised and stripped of self-loops as in the paper
(W = (A + A^T)/2 - diag). For each T we report lambda_2 of the combinatorial Laplacian
L = D - W and of the symmetric normalised Laplacian I - D^{-1/2} W D^{-1/2}, per head and
for the averaged graph.

Writes results/v2_oct2026/causal_construction.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TS = (5, 10, 50, 200)


def heads(T):
    prev = np.zeros((T, T)); prev[0, 0] = 1.0
    for i in range(1, T):
        prev[i, i - 1] = 1.0
    sink = np.zeros((T, T)); sink[:, 0] = 1.0
    for A in (prev, sink):
        assert np.allclose(A.sum(1), 1.0) and np.allclose(A, np.tril(A))
    return prev, sink


def sym(A):
    W = 0.5 * (A + A.T)
    return W - np.diag(np.diag(W))


def lam2(W, normalised):
    d = W.sum(1)
    if normalised:
        s = 1.0 / np.sqrt(d)
        L = np.eye(len(W)) - s[:, None] * W * s[None, :]
    else:
        L = np.diag(d) - W
    return float(np.sort(np.linalg.eigvalsh(L))[1])


def main():
    out = {}
    for T in TS:
        prev, sink = heads(T)
        avg = 0.5 * (prev + sink)
        Wp, Ws, Wa = sym(prev), sym(sink), sym(avg)
        assert np.allclose(0.5 * (Wp + Ws), Wa)          # the two configurations share the mean graph
        r = {}
        for norm, name in ((False, "combinatorial"), (True, "normalised")):
            lp, ls, la = lam2(Wp, norm), lam2(Ws, norm), lam2(Wa, norm)
            r[name] = {"prev_head": lp, "sink_head": ls, "averaged_head": la,
                       "config_one_mean_over_heads": 0.5 * (lp + ls),
                       "config_one_max_minus_min": ls - lp,
                       "config_two_max_minus_min": 0.0,
                       "averaged_minus_config_one_mean": la - 0.5 * (lp + ls)}
        out[str(T)] = r
        print(T, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r.items()})
    o = ROOT / "results" / "v2_oct2026"
    o.mkdir(parents=True, exist_ok=True)
    (o / "causal_construction.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
