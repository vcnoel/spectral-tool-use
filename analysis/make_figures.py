"""
Figures for the paper, drawn from the result files so they cannot disagree
with the numbers. Writes PDFs into paper/iclr/figures/.

  fig_heatmap.pdf    where the per-head signal lives: univariate AUC of each
                     (layer, head) spectral-radius feature on one run
  fig_thinning.pdf   probe AUC against the number of labelled failures, with
                     the confidence baseline as a horizontal line
  fig_jensen.pdf     head-averaged against mean-over-heads connectivity, one
                     point per layer, with the diagonal
  fig_family.pdf     probe-minus-confidence gap per run, grouped by family
  fig_ladder.pdf     the resolution ladder, one bar per rung
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
THEORY = ROOT / "data" / "theory"
FIG = ROOT / "paper" / "iclr" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42})


def heatmap(tag="base_llama1b_glaive", metric_idx=4):
    from sklearn.metrics import roc_auc_score
    from run_pilot_v2 import load_and_relabel, SEMANTIC_MODES
    f = ROOT / "data" / f"pilot_v2_{tag}" / "features.jsonl"
    if not f.exists():
        return
    samples, _ = load_and_relabel(f)
    samples = [s for s in samples if s.get("head_metrics_span") is not None
               and s["failure_mode"] in SEMANTIC_MODES]
    y = np.array([s["label"] for s in samples])
    hm = np.nan_to_num(np.array([s["head_metrics_span"] for s in samples],
                                dtype=np.float32))
    L, H = hm.shape[1], hm.shape[2]
    A = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            x = hm[:, l, h, metric_idx]
            a = roc_auc_score(y, x) if np.std(x) > 0 else 0.5
            A[l, h] = max(a, 1 - a)       # direction-free
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    im = ax.imshow(A, aspect="auto", cmap="viridis", vmin=0.5, vmax=A.max())
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("univariate AUC")
    fig.tight_layout()
    fig.savefig(FIG / "fig_heatmap.pdf")
    plt.close(fig)
    # the spread of the same statistic under head averaging, for the caption
    return {"max": float(A.max()), "median": float(np.median(A)),
            "frac_above_06": float((A > 0.6).mean())}


def thinning():
    f = THEORY / "label_budget.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    curve = d["curve"].get("base_llama1b_bfcl")
    if not curve:
        return
    conf = None
    for r in d.get("cross_run", []):
        if r["run"] == "base_llama1b_bfcl":
            conf = r["confidence"]
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    x = [c["approx_train_positives"] for c in curve]
    yv = [c["auc"] for c in curve]
    ax.plot(x, yv, "o-", color="C0", label="token-role probe")
    if conf is not None:
        ax.axhline(conf, color="C3", ls="--", label="mean log-probability")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("labelled failures available for training")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_thinning.pdf")
    plt.close(fig)


def jensen():
    f = THEORY / "jensen_gap.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    rec = d["records"]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2))
    for ax, key, title in ((axes[0], "comb", "combinatorial Laplacian"),
                           (axes[1], "norm", "normalised Laplacian")):
        x = np.array([r[f"lam2_{key}_mean_of_heads"] for r in rec])
        yv = np.array([r[f"lam2_{key}_avg_graph"] for r in rec])
        models = sorted({r["model"] for r in rec})
        for i, m in enumerate(models):
            sel = np.array([r["model"] == m for r in rec])
            ax.scatter(x[sel], yv[sel], s=6, alpha=0.7, label=m, color=f"C{i}")
        lim = [0, max(x.max(), yv.max()) * 1.05]
        ax.plot(lim, lim, color="k", lw=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(r"mean over heads of $\lambda_2(L_h)$")
        ax.set_ylabel(r"$\lambda_2(\bar L)$ of the averaged graph")
        ax.set_title(title)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_jensen.pdf")
    plt.close(fig)


def family():
    f = THEORY / "family_split.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    rows = d["semantic"]["rows"]
    rows = sorted(rows, key=lambda r: (r["family"] == "recent", r["gap"]))
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    names = [r["run"].replace("base_", "").replace("_", " ") for r in rows]
    gaps = [r["gap"] for r in rows]
    cols = ["C0" if r["family"] == "earlier" else "C1" for r in rows]
    ax.barh(range(len(rows)), gaps, color=cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("probe AUC minus confidence AUC")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="C0", label="Llama / Qwen3 / Gemma"),
                       Patch(color="C1", label="MiniCPM5 / Qwen3.5")],
              frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_family.pdf")
    plt.close(fig)


def ladder():
    f = THEORY / "resolution_ladder.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    sm = d["summary"]
    order = ["graph-averaged (L x 5)", "metric-averaged (L x 5)",
             "1 random heads (L x 1 x 5)", "2 random heads (L x 2 x 5)",
             "4 random heads (L x 4 x 5)", "8 random heads (L x 8 x 5)",
             "per-head (L x H x 5)",
             "noise-padded graph-averaged (L x H x 5)",
             "head-shuffled per-head (L x H x 5)"]
    labels = ["averaged\ngraph", "metric\navg.", "1 head", "2 heads", "4 heads",
              "8 heads", "all heads", "avg. graph\n+ noise",
              "heads\nshuffled"]
    order, labels = zip(*[(o, l) for o, l in zip(order, labels) if o in sm])
    vals = [sm[o]["mean"] for o in order]
    fig, ax = plt.subplots(figsize=(5.0, 2.1))
    cols = ["C7", "C7", "C0", "C0", "C0", "C0", "C0", "C3", "C3"][:len(vals)]
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=0, fontsize=6.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel(f"mean AUC over {len(d['runs'])} runs")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ladder.pdf")
    plt.close(fig)


if __name__ == "__main__":
    stats = {}
    stats["heatmap"] = heatmap()
    thinning()
    jensen()
    family()
    ladder()
    (THEORY / "figure_stats.json").write_text(json.dumps(stats, indent=2),
                                              encoding="utf-8")
    print("figures ->", FIG, json.dumps(stats))
