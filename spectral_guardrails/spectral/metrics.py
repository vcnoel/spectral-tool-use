"""
Attention-only spectral metrics (v2 pipeline).

Every metric here is a function of the attention weight matrices ALONE —
no hidden states enter any computation. This restores the paper's
"attention-only" claim, which the v1 pipeline violated (4 of 5 legacy
metrics projected hidden states onto the attention-graph eigenbasis via
spectral_trust.analyze_layer).

Graph construction:
  W  = symmetrized mean-over-heads attention, self-loops removed
  L  = symmetric normalized Laplacian  I - D^{-1/2} W D^{-1/2}

The normalized Laplacian has eigenvalues in [0, 2] regardless of graph
size, which removes the sequence-length dependence that the v1
combinatorial Laplacian metrics carried (a length confound flagged in
the 2026-09 audit).

Per-layer metrics (5, mirroring the v1 metric count):
  fiedler_value        lambda_2 of the normalized Laplacian (algebraic
                       connectivity; low = fragmented routing)
  connectivity_ratio   lambda_2 / lambda_max (scale-free connectivity)
  energy_norm          mean eigenvalue (= trace(L)/T; 1.0 for loop-free
                       graphs by construction, deviates under numerical
                       mass concentration — kept as a sanity feature)
  spectral_entropy_norm  entropy of the eigenvalue distribution
                       normalized by log(T) (in [0, 1])
  hfer                 fraction of eigenvalue mass in the top half of
                       the spectrum
"""
import torch

METRIC_NAMES = [
    "fiedler_value",
    "connectivity_ratio",
    "energy_norm",
    "spectral_entropy_norm",
    "hfer",
]


def build_normalized_laplacian(attn: torch.Tensor,
                               span: tuple[int, int] | None = None) -> torch.Tensor:
    """
    attn: [H, T, T] attention weights for one layer (single sample).
    span: optional (start, end) token range — the induced subgraph is
          extracted BEFORE Laplacian construction (call-local analysis).
    Returns the symmetric normalized Laplacian [S, S] in float32.
    """
    A = attn.to(torch.float32).mean(dim=0)          # [T, T]
    W = 0.5 * (A + A.T)
    if span is not None:
        s, e = span
        W = W[s:e, s:e]
    W = W.clone()
    W.fill_diagonal_(0.0)
    deg = W.sum(dim=-1)
    inv_sqrt = torch.where(deg > 1e-10, deg.clamp(min=1e-10).rsqrt(),
                           torch.zeros_like(deg))
    L = torch.eye(W.shape[0], device=W.device) - inv_sqrt[:, None] * W * inv_sqrt[None, :]
    return L


def spectral_metrics_from_laplacian(L: torch.Tensor) -> dict[str, float]:
    """Eigenvalue-only metrics of a normalized Laplacian. Attention-only."""
    T = L.shape[0]
    if T < 3:
        return {name: 0.0 for name in METRIC_NAMES}
    evals = torch.linalg.eigvalsh(0.5 * (L + L.T))
    evals = evals.clamp(min=0.0)

    lam2 = float(evals[1])
    lam_max = float(evals[-1])
    energy = float(evals.mean())

    total = float(evals.sum())
    if total <= 1e-12:
        entropy_norm, hfer = 0.0, 0.0
    else:
        p = (evals / total).clamp(min=1e-12)
        entropy = float(-(p * p.log()).sum())
        entropy_norm = entropy / max(torch.log(torch.tensor(float(T))).item(), 1e-9)
        hfer = float(evals[T // 2:].sum() / total)

    return {
        "fiedler_value": lam2,
        "connectivity_ratio": lam2 / lam_max if lam_max > 1e-12 else 0.0,
        "energy_norm": energy,
        "spectral_entropy_norm": entropy_norm,
        "hfer": hfer,
    }


def layer_spectral_metrics(attn: torch.Tensor,
                           span: tuple[int, int] | None = None) -> dict[str, float]:
    """Convenience: attention [H, T, T] -> metric dict."""
    return spectral_metrics_from_laplacian(build_normalized_laplacian(attn, span))


# ── rich spectral features (v2.1): eigenvalue vectors, per-head, Gram ─────────

def laplacian_eig_profile(attn: torch.Tensor,
                          span: tuple[int, int] | None = None,
                          k: int = 16) -> list[float]:
    """
    LapEigvals-style profile: k smallest + k largest eigenvalues of the
    normalized Laplacian (head-averaged). Attention-only. Padded with the
    edge value when the graph has fewer than 2k nodes.
    """
    L = build_normalized_laplacian(attn, span)
    T = L.shape[0]
    if T < 3:
        return [0.0] * (2 * k)
    ev = torch.linalg.eigvalsh(0.5 * (L + L.T)).clamp(min=0.0)
    lo = ev[:k].tolist()
    hi = ev[-k:].tolist()
    lo += [lo[-1]] * (k - len(lo))
    hi = [hi[0]] * (k - len(hi)) + hi
    return [float(x) for x in lo + hi]


# Per-head metrics are provided by the spectral-trust library (>=0.3.0), so
# the paper's features come from the published implementation rather than a
# pipeline-local copy. Verified bit-exact against the previous local
# implementation on real attention tensors (2026-09-09).
#
# The library removes self-loops only when asked; the pilot has always built
# the normalized Laplacian on the loop-free graph, so that is pinned here.
from spectral_trust import GSPConfig as _GSPConfig
from spectral_trust import per_head_metrics as _st_per_head_metrics

PER_HEAD_METRICS = ["fiedler_value", "connectivity_ratio",
                    "spectral_entropy_norm", "hfer", "lambda_max"]

_PER_HEAD_CFG = _GSPConfig(normalization="sym", symmetrization="symmetric",
                           remove_self_loops=True)


def per_head_metrics(attn: torch.Tensor,
                     span: tuple[int, int] | None = None) -> list[list[float]]:
    """
    Five eigenvalue-only metrics of each head's OWN normalized Laplacian
    (no head averaging), delegated to spectral_trust.per_head_metrics.

    Head-averaging dilutes the routing anomaly that accompanies a failure,
    because heads are specialized; keeping per-head resolution preserves it.
    All heads are decomposed in one batched eigendecomposition, so five
    metrics cost what one costs. Attention-only.

    Returns [H][5] in PER_HEAD_METRICS order (the library's
    "spectral_radius" is this list's "lambda_max").
    """
    return _st_per_head_metrics(
        attn, config=_PER_HEAD_CFG, token_span=span).values.tolist()


def per_head_fiedler(attn: torch.Tensor,
                     span: tuple[int, int] | None = None) -> list[float]:
    """Back-compat wrapper: lambda_2 per head (first PER_HEAD_METRICS column)."""
    return [row[0] for row in per_head_metrics(attn, span)]


def lapeigvals_diag_profile(attn: torch.Tensor, k_store: int = 100) -> list[list[float]]:
    """
    Faithful re-implementation of the official LapEigvals features
    (Binkowski et al., EMNLP 2025; github.com/graphml-lab-pwr/lapeigvals,
    hallucinations/features/attention_weights.py::laplacian_diagonal_from_attn
    with vertical_edges=False).

    Their insight: causal attention graphs are DAGs, so the Laplacian is
    triangular and its eigenvalues equal its diagonal — no eigendecomposition
    needed. Per layer and PER HEAD:
        diag_j = (sum_i A[h, i, j]) / (T - j)  -  A[h, j, j]
    i.e. position-normalized weighted in-degree minus the self-loop, then the
    values are sorted descending and the top-k kept per (layer, head). The
    official probe is a balanced logistic regression over all layers x heads
    x k, with k swept in {5, 10, 25, 50, 100}; we store the top `k_store`
    so k can be tuned on validation downstream. Attention-only.

    attn: [H, T, T]. Returns [H][k_store] (padded with 0.0 when T < k_store).
    """
    # Reduce in float32 without widening the whole layer first: a full
    # attention tensor at a few thousand tokens is over a gigabyte in
    # float32, and only the column sums and the diagonal are needed.
    H, T, _ = attn.shape
    denom = torch.arange(1, T + 1, device=attn.device,
                         dtype=torch.float32).flip(0)
    col_sum = attn.sum(dim=1, dtype=torch.float32)           # [H, T]
    diag = torch.diagonal(attn, dim1=1, dim2=2).to(torch.float32)
    lap_diag = col_sum / denom - diag                        # [H, T]
    vals = lap_diag.sort(dim=-1, descending=True).values[:, :k_store]
    if vals.shape[1] < k_store:
        pad = torch.zeros(H, k_store - vals.shape[1], device=attn.device)
        vals = torch.cat([vals, pad], dim=1)
    return [[float(x) for x in row] for row in vals]


def gram_spectrum_features(hidden: torch.Tensor, k: int = 8) -> list[float]:
    """
    Spectral features of the RESIDUAL-STREAM vectors themselves (not the
    attention graph): eigenvalues of the Gram matrix of the given token
    span (INSIDE / EigenScore family). Requires hidden-state access.
    hidden: [S, D] residual-stream states for the span.
    Returns [top-k normalized eigenvalues..., effective rank, log-det proxy,
             spectral entropy].
    """
    S = hidden.shape[0]
    if S < 3:
        return [0.0] * (k + 3)
    X = hidden.to(torch.float32)
    X = X - X.mean(dim=0, keepdim=True)
    G = (X @ X.T) / X.shape[1]
    ev = torch.linalg.eigvalsh(G).clamp(min=0.0)
    ev = torch.flip(ev, [0])                       # descending
    total = float(ev.sum())
    if total <= 1e-12:
        return [0.0] * (k + 3)
    p = (ev / total).clamp(min=1e-12)
    entropy = float(-(p * p.log()).sum())
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    logdet = float(torch.log(ev[:min(S, 10)] + 1e-8).mean())
    top = (ev[:k] / total).tolist()
    top += [0.0] * (k - len(top))
    return [float(x) for x in top] + [eff_rank, logdet, entropy]


# ── baseline feature families (attention-only) ────────────────────────────────

def lookback_ratio(attn: torch.Tensor, prompt_len: int,
                   seq_len: int) -> list[list[float]]:
    """
    Lookback Lens features (Chuang et al., 2024): for each head, the share of
    attention mass that generated-token rows place on the prompt versus on
    the generated span. A contextual hallucination is associated with a shift
    of attention away from the provided context.

    attn: [H, T, T] for one layer. Returns [H][2] = [context share,
    generation share], averaged over generated rows. Attention-only.
    """
    H = attn.shape[0]
    rows = attn[:, prompt_len:seq_len, :]                   # generated rows
    if rows.shape[1] == 0:
        return [[0.0, 0.0] for _ in range(H)]
    # reduce straight to float32; the generated rows are a small slice
    ctx = rows[:, :, :prompt_len].sum(-1, dtype=torch.float32)
    gen = rows[:, :, prompt_len:seq_len].sum(-1, dtype=torch.float32)
    total = (ctx + gen).clamp(min=1e-9)
    return [[float((ctx[h] / total[h]).mean()),
             float((gen[h] / total[h]).mean())] for h in range(H)]


def residual_dynamics(hidden_states, prompt_len: int, seq_len: int,
                      k_layers: int = 32) -> list[list[float]]:
    """
    Cross-layer residual-stream dynamics, in the spirit of the ICR probe
    (Zhang et al., 2025): how much each block changes the residual stream
    over the generated span, and how much direction it preserves.

    hidden_states: sequence of [1, T, D] tensors (HF output_hidden_states).
    Returns [L-1][4]: relative update norm (mean and max over the span) and
    cosine similarity with the previous layer (mean and min). Requires
    residual-stream access.
    """
    out = []
    n = min(len(hidden_states), k_layers + 1)
    for i in range(1, n):
        prev = hidden_states[i - 1][0, prompt_len:seq_len].to(torch.float32)
        cur = hidden_states[i][0, prompt_len:seq_len].to(torch.float32)
        if prev.shape[0] == 0:
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        delta = (cur - prev).norm(dim=-1)
        scale = cur.norm(dim=-1).clamp(min=1e-9)
        rel = delta / scale
        cos = torch.nn.functional.cosine_similarity(prev, cur, dim=-1)
        out.append([float(rel.mean()), float(rel.max()),
                    float(cos.mean()), float(cos.min())])
    return out
