"""
Figures for the paper, drawn from the result files so they cannot disagree
with the numbers. Writes PDF (for the paper) and PNG (for review) into
paper/iclr/figures/, and figure_stats.json (numbers quoted in captions) into
data/theory/.

  fig1_resolution   what per-head resolution preserves and head averaging
                    destroys, on one layer of one run: the per-head spectral
                    radius of correct and failed calls, head by head, against
                    the single head-averaged value
  fig2_frontier     the centrepiece: every run, the four fixed tier
                    representatives on one AUC axis with the length floor,
                    and the paired-bootstrap margin of each over the floor
  fig3_ladder       the resolution ladder: AUC as head resolution increases,
                    with the two controls that separate resolution from width
  fig4_heatmap      where the per-head signal lives: univariate AUC of each
                    (layer, head) spectral-radius feature, with the layer's
                    head-averaged value beside it
  fig5_jensen       head averaging overstates connectivity: lambda_2 of the
                    averaged graph against the mean over heads, per layer
  fig6_family       the probe-minus-confidence gap is a family property, on
                    both evaluation populations
  fig7_budget       label efficiency: probe AUC against labelled failures
  fig8_multiturn    the cascade: failure rate with clean and corrupted
                    history, and what each detector still sees
  fig9_latency      the cost of each tier per call

Every function returns without writing when its input file is absent, so a
partial result set yields a partial figure set rather than an error.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
THEORY = DATA / "theory"
FIG = ROOT / "paper" / "iclr" / "figures"

# ── palette (validated: adjacent and all-pairs CVD, white surface) ──────────
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# tier colours follow the entity, never the rank
TIER = {"residual": BLUE, "per-head": ORANGE, "LapEigvals": AQUA,
        "logits": INK2, "floor": MUTED}
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "pdf.fonttype": 42, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlelocation": "left",
    "legend.frameon": False,
})

# runs in presentation order, with short labels for axes
RUN_ORDER = [
    ("base_llama1b_glaive", "Llama-1B · Glaive"),
    ("base_llama3b_glaive", "Llama-3B · Glaive"),
    ("base_gemma3_glaive", "Gemma-1B · Glaive"),
    ("qwen35_2b", "Qwen3.5-2B · Glaive"),
    ("base_llama1b_bfcl", "Llama-1B · BFCL"),
    ("base_llama3b_bfcl", "Llama-3B · BFCL"),
    ("base_qwen3_17b_bfcl", "Qwen3-1.7B · BFCL"),
    ("minicpm5_2b_bfcl", "MiniCPM5-2B · BFCL"),
    ("qwen35_08b_bfcl", "Qwen3.5-0.8B · BFCL"),
    ("qwen35_4b_bfcl", "Qwen3.5-4B · BFCL"),
    ("llama1b_live", "Llama-1B · live"),
    ("minicpm5_2b_live", "MiniCPM5-2B · live"),
]
REPS = [("residual", "Hidden token-role [LR]", "token-role probe"),
        ("per-head", "Per-head all metrics (span)", "per-head spectral"),
        ("LapEigvals", "LapEigvals (official code)", "LapEigvals"),
        ("logits", "Mean logprob", "log-probability")]
FLOOR_KEY = "Surface (lengths) [confound]"
PAIRED_VS_FLOOR = {"per-head": "per-head vs surface",
                   "LapEigvals": "LapEigvals vs surface",
                   "residual": "token-role vs surface"}


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _runs(include_all=False):
    """results.json (+ paired.json) for the runs the paper reports, in
    RUN_ORDER. Superseded extractions and auxiliary runs (the confidence-
    summary re-extractions, synthetic tests) are excluded unless
    include_all=True, so a model/benchmark pair is drawn once."""
    order = [t for t, _ in RUN_ORDER]
    labels = dict(RUN_ORDER)
    found = {}
    for p in sorted(DATA.glob("pilot_v2_*/results.json")):
        tag = p.parent.name.replace("pilot_v2_", "")
        if tag not in order and not include_all:
            continue
        found[tag] = (_load_json(p), _load_json(p.parent / "paired.json"))
    tags = [t for t in order if t in found] + [t for t in found if t not in order]
    return [(t, labels.get(t, t.replace("_", " ")), *found[t]) for t in tags]


def _subset(res_json):
    if res_json.get("eval_subset"):
        return res_json["eval_subset"]
    r = res_json["results"]
    return ("call_expected" if r.get("Hidden token-role [LR]", {}).get("call_expected")
            else "semantic")


def _mean(res, key, sub):
    v = [x for x in res.get(key, {}).get(sub, []) if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def _panel_title(ax, text):
    ax.set_title(text, loc="left", pad=3, color=INK)


def _save(fig, name, stats=None):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
def fig1_resolution(tag="base_llama1b_glaive", metric_idx=4, layer=None):
    """Per-head spectral radius of correct and failed calls, head by head, on
    the most discriminative layer, against the head-averaged value."""
    from sklearn.metrics import roc_auc_score
    from run_pilot_v2 import load_and_relabel, SEMANTIC_MODES
    f = DATA / f"pilot_v2_{tag}" / "features.jsonl"
    if not f.exists():
        alt = sorted(DATA.glob("pilot_v2_*/features.jsonl"))
        if not alt:
            return None
        f = alt[0]
        tag = f.parent.name.replace("pilot_v2_", "")
    samples, _ = load_and_relabel(f)
    samples = [s for s in samples if s.get("head_metrics_span") is not None
               and s["failure_mode"] in SEMANTIC_MODES]
    if len(samples) < 20:
        return None
    y = np.array([s["label"] for s in samples])
    hm = np.nan_to_num(np.array([s["head_metrics_span"] for s in samples], dtype=np.float32))
    L, H = hm.shape[1], hm.shape[2]
    avg = np.array([[d.get("fiedler_value", 0.0) for d in s["layer_diagnostics_span"]]
                    for s in samples], dtype=np.float32)          # head-averaged, [N, L]

    def auc_free(x):
        if np.std(x) == 0:
            return 0.5
        a = roc_auc_score(y, x)
        return max(a, 1 - a)

    per_head_auc = np.array([[auc_free(hm[:, li, h, metric_idx]) for h in range(H)]
                             for li in range(L)])
    if layer is None:
        layer = int(per_head_auc.max(1).argmax())
    avg_auc = auc_free(avg[:, layer])

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15),
                             gridspec_kw={"width_ratios": [3.2, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for h in range(H):
        for lab, col, off in ((0, MUTED, -0.18), (1, ORANGE, 0.18)):
            v = hm[y == lab, layer, h, metric_idx]
            ax.scatter(h + off + rng.uniform(-0.07, 0.07, len(v)), v, s=3, alpha=0.35,
                       color=col, linewidths=0, rasterized=True)
    aucs = per_head_auc[layer]
    if H <= 12:
        # the per-head univariate AUC rides under each head index
        ax.set_xticks(range(H))
        ax.set_xticklabels([f"{h}\n{aucs[h]:.2f}" for h in range(H)])
        ax.set_xlabel(f"head (layer {layer}); second row: that head's AUC")
    else:
        # too many heads to label each: index every fourth head, and name the
        # three most discriminative heads with their AUC above the strip
        step = 4 if H <= 32 else 8
        ax.set_xticks(range(0, H, step))
        ax.set_xticklabels([str(h) for h in range(0, H, step)])
        ax.set_xlabel(f"head (layer {layer}); labelled: the three heads with the highest AUC")
        lo0, hi0 = ax.get_ylim()
        for h in np.argsort(aucs)[::-1][:3]:
            ax.annotate(f"AUC {aucs[h]:.2f}", (h, hi0), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.8, color=INK2,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2, shrinkA=0, shrinkB=1))
    ax.set_ylabel(r"spectral radius $\lambda_{\max}$")
    ax.tick_params(axis="x", length=0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.16 * (hi - lo))          # headroom for the labels
    ax.text(0.01, 0.98, "correct", color=MUTED, va="top", ha="left", fontsize=6.8,
            transform=ax.transAxes)
    ax.text(0.01, 0.90, "failed", color=ORANGE, va="top", ha="left", fontsize=6.8,
            transform=ax.transAxes)
    _panel_title(ax, "(a) one layer, every head, every call")

    ax = axes[1]
    ax.hist(avg[y == 0, layer], bins=25, color=MUTED, alpha=0.8, density=True,
            histtype="stepfilled", linewidth=0)
    ax.hist(avg[y == 1, layer], bins=25, color=ORANGE, alpha=0.6, density=True,
            histtype="stepfilled", linewidth=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"head-averaged $\lambda_2$")
    _panel_title(ax, f"(b) averaged: AUC {avg_auc:.2f}")
    fig.tight_layout(w_pad=1.0)
    return _save(fig, "fig1_resolution", {
        "run": tag, "layer": layer, "per_head_auc_max": float(per_head_auc[layer].max()),
        "per_head_auc_median": float(np.median(per_head_auc[layer])),
        "averaged_auc": float(avg_auc), "n": int(len(y))})


# ─────────────────────────────────────────────────────────────────────────────
def fig2_frontier():
    runs = _runs()
    if not runs:
        return None
    n = len(runs)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 0.28 * n + 1.1),
                             gridspec_kw={"width_ratios": [1.55, 1]}, sharey=True)
    ys = np.arange(n)[::-1]
    stats = {"n_runs": n, "n_underpowered": 0, "margin_over_floor": {}}
    ax = axes[0]
    for yi, (tag, label, res_json, paired) in zip(ys, runs):
        res, sub = res_json["results"], _subset(res_json)
        weak = bool(res_json.get("underpowered", False))
        stats["n_underpowered"] += weak
        alpha = 0.45 if weak else 1.0
        floor = _mean(res, FLOOR_KEY, sub)
        if np.isfinite(floor):
            ax.plot([0.5, floor], [yi, yi], color=GRID, lw=3.5, solid_capstyle="butt",
                    zorder=1)
            ax.plot([floor, floor], [yi - 0.28, yi + 0.28], color=MUTED, lw=1.0, zorder=2)
        for tier, key, _ in REPS:
            v = _mean(res, key, sub)
            if np.isfinite(v):
                ax.scatter([v], [yi], s=16, color=TIER[tier], alpha=alpha, zorder=3,
                           edgecolor="white", linewidth=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab + (" †" if r.get("underpowered") else "")
                        for _, lab, r, _ in runs])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC on the evaluated population")
    ax.tick_params(axis="y", length=0)
    ax.axvline(0.5, color=BASE, lw=0.6)
    # identity legend above the plot, spaced by matplotlib so it cannot collide
    # with the data of whichever run sits on top; text in ink, colour on the mark
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MUTED, lw=3, label="length floor")] + [
        Line2D([], [], color=TIER[t], marker="o", ms=4, lw=0, label=nm) for t, _, nm in REPS]
    fig.legend(handles=handles, loc="upper center", ncol=5, handlelength=1.4,
               columnspacing=1.2, bbox_to_anchor=(0.5, 1.0), fontsize=6.4)
    _panel_title(ax, "(a) fixed tier representatives")

    ax = axes[1]
    for yi, (tag, label, res_json, paired) in zip(ys, runs):
        weak = bool(res_json.get("underpowered", False))
        alpha = 0.45 if weak else 1.0
        con = (paired or {}).get("contrasts", {})
        for k, (tier, cname) in enumerate(PAIRED_VS_FLOOR.items()):
            c = con.get(cname)
            off = (k - 1) * 0.22
            if not c:
                continue
            ax.plot([c["ci_lo"], c["ci_hi"]], [yi + off, yi + off], color=TIER[tier],
                    lw=1.1, alpha=alpha, solid_capstyle="butt")
            ax.scatter([c["delta"]], [yi + off], s=9, color=TIER[tier], alpha=alpha,
                       zorder=3, edgecolor="white", linewidth=0.4)
            stats["margin_over_floor"].setdefault(tier, []).append(c["delta"])
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_xlabel(r"$\Delta$AUC vs floor, paired 95% CI")
    _panel_title(ax, "(b) margin over the floor")
    if not any(p for *_, p in runs):
        ax.text(0.5, 0.5, "paired.json missing:\nrun analysis/paired_inference.py",
                transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=6.8)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.96))   # room for the figure legend
    for tier, v in stats["margin_over_floor"].items():
        stats["margin_over_floor"][tier] = {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                            "max": float(np.max(v))}
    return _save(fig, "fig2_frontier", stats)


# ─────────────────────────────────────────────────────────────────────────────
def fig3_ladder():
    d = _load_json(THEORY / "resolution_ladder.json")
    if not d:
        return None
    sm, runs = d["summary"], d.get("runs", {})
    rungs = [("graph-averaged (L x 5)", "averaged\ngraph"),
             ("metric-averaged (L x 5)", "metrics\naveraged"),
             ("1 random heads (L x 1 x 5)", "1 head"),
             ("2 random heads (L x 2 x 5)", "2 heads"),
             ("4 random heads (L x 4 x 5)", "4 heads"),
             ("8 random heads (L x 8 x 5)", "8 heads"),
             ("per-head (L x H x 5)", "all heads")]
    controls = [("noise-padded graph-averaged (L x H x 5)", "averaged\n+ noise"),
                ("head-shuffled per-head (L x H x 5)", "heads\nshuffled")]
    rungs = [(k, lab) for k, lab in rungs if k in sm]
    controls = [(k, lab) for k, lab in controls if k in sm]
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    x = np.arange(len(rungs))
    for tag, r in runs.items():
        ys = [r["auc"].get(k, {}).get("mean", np.nan) for k, _ in rungs]
        ax.plot(x, ys, color=GRID, lw=0.7, zorder=1)
    means = [sm[k]["mean"] for k, _ in rungs]
    ax.plot(x, means, color=INK, lw=1.4, marker="o", ms=3.5, zorder=3)
    for xi, m in zip(x, means):
        ax.annotate(f"{m:.2f}", (xi, m), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6.2, color=INK)
    xc = len(rungs) + 0.6 + np.arange(len(controls))
    for xi, (k, lab) in zip(xc, controls):
        ax.scatter([xi], [sm[k]["mean"]], color=ORANGE, s=18, zorder=3)
        ax.annotate(f"{sm[k]['mean']:.2f}", (xi, sm[k]["mean"]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=6.2, color=ORANGE)
        for tag, r in runs.items():
            v = r["auc"].get(k, {}).get("mean")
            if v is not None:
                ax.scatter([xi], [v], color=ORANGE, s=5, alpha=0.35, zorder=2, linewidths=0)
    ax.set_xticks(list(x) + list(xc))
    ax.set_xticklabels([lab for _, lab in rungs] + [lab for _, lab in controls])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(f"mean AUC over {len(runs)} runs")
    y0 = min(0.5, min(means) - 0.05)
    ax.set_ylim(y0, 1.0)
    if controls:
        ax.axvline(len(rungs) - 0.2, color=GRID, lw=0.6, ls=(0, (2, 2)))
        ax.text(len(rungs) + 0.6 + (len(controls) - 1) / 2, y0 + 0.012,
                "controls: same width,\nno head resolution", ha="center", va="bottom",
                fontsize=6.2, color=ORANGE)
    ax.text(0, 0.98, "grey: individual runs", transform=ax.transAxes, va="top",
            fontsize=6.2, color=MUTED)
    fig.tight_layout()
    return _save(fig, "fig3_ladder",
                 {"n_runs": len(runs), "first": means[0], "last": means[-1]})


# ─────────────────────────────────────────────────────────────────────────────
def fig4_heatmap(tag="base_llama1b_glaive", metric_idx=4):
    from sklearn.metrics import roc_auc_score
    from run_pilot_v2 import load_and_relabel, SEMANTIC_MODES
    f = DATA / f"pilot_v2_{tag}" / "features.jsonl"
    if not f.exists():
        alt = sorted(DATA.glob("pilot_v2_*/features.jsonl"))
        if not alt:
            return None
        f = alt[0]
        tag = f.parent.name.replace("pilot_v2_", "")
    samples, _ = load_and_relabel(f)
    samples = [s for s in samples if s.get("head_metrics_span") is not None
               and s["failure_mode"] in SEMANTIC_MODES]
    if len(samples) < 20:
        return None
    y = np.array([s["label"] for s in samples])
    hm = np.nan_to_num(np.array([s["head_metrics_span"] for s in samples], dtype=np.float32))
    avg = np.array([[d.get("fiedler_value", 0.0) for d in s["layer_diagnostics_span"]]
                    for s in samples], dtype=np.float32)
    L, H = hm.shape[1], hm.shape[2]

    def auc_free(x):
        if np.std(x) == 0:
            return 0.5
        a = roc_auc_score(y, x)
        return max(a, 1 - a)

    A = np.array([[auc_free(hm[:, li, h, metric_idx]) for h in range(H)]
                  for li in range(L)])
    B = np.array([auc_free(avg[:, li]) for li in range(L)])
    fig, axes = plt.subplots(1, 2, figsize=(4.0, 2.5),
                             gridspec_kw={"width_ratios": [H, 1.2], "wspace": 0.08})
    fig.subplots_adjust(left=0.12, right=0.80, bottom=0.2, top=0.86)
    vmax = max(A.max(), B.max(), 0.6)
    im = axes[0].imshow(A, aspect="auto", cmap=SEQ_CMAP, vmin=0.5, vmax=vmax,
                        interpolation="nearest")
    axes[0].set_xlabel("head")
    axes[0].set_ylabel("layer")
    axes[0].set_xticks(range(0, H, max(1, H // 8)))
    _panel_title(axes[0], "(a) per head")
    axes[1].imshow(B[:, None], aspect="auto", cmap=SEQ_CMAP, vmin=0.5, vmax=vmax,
                   interpolation="nearest")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for sp in axes[1].spines.values():
        sp.set_visible(False)
    _panel_title(axes[1], "(b) averaged")
    cax = fig.add_axes([0.84, 0.2, 0.025, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("univariate AUC")
    cb.outline.set_visible(False)
    return _save(fig, "fig4_heatmap", {
        "run": tag, "per_head_max": float(A.max()), "per_head_median": float(np.median(A)),
        "frac_above_06": float((A > 0.6).mean()), "averaged_max": float(B.max())})


# ─────────────────────────────────────────────────────────────────────────────
def fig5_jensen():
    d = _load_json(THEORY / "jensen_gap.json")
    if not d:
        return None
    rec = d["records"]
    models = sorted({r["model"] for r in rec})[:3]      # three slots validate all-pairs
    cols = dict(zip(models, [BLUE, ORANGE, AQUA]))
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3))
    for ax, key, title in ((axes[0], "comb", "(a) combinatorial Laplacian"),
                           (axes[1], "norm", "(b) normalised Laplacian")):
        x = np.array([r[f"lam2_{key}_mean_of_heads"] for r in rec])
        yv = np.array([r[f"lam2_{key}_avg_graph"] for r in rec])
        lim = [0, max(x.max(), yv.max()) * 1.05]
        ax.plot(lim, lim, color=BASE, lw=0.7, zorder=1)
        for m in models:
            sel = np.array([r["model"] == m for r in rec])
            ax.scatter(x[sel], yv[sel], s=7, alpha=0.75, color=cols[m], linewidths=0,
                       zorder=2, rasterized=True)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(r"mean over heads of $\lambda_2(L_h)$")
        ax.set_ylabel(r"$\lambda_2(\bar L)$, averaged graph")
        _panel_title(ax, title)
        ax.text(lim[1] * 0.97, lim[1] * 0.97, "equality", ha="right", va="top",
                fontsize=6.2, color=MUTED, rotation=45, rotation_mode="anchor")
    for i, m in enumerate(models):     # direct legend, top-left of panel (a)
        axes[0].text(0.03, 0.97 - 0.09 * i, m, transform=axes[0].transAxes, va="top",
                     fontsize=6.6, color=cols[m])
    fig.tight_layout(w_pad=1.5)
    return _save(fig, "fig5_jensen", d.get("summary"))


# ─────────────────────────────────────────────────────────────────────────────
def fig6_family():
    d = _load_json(THEORY / "family_split.json")
    if not d:
        return None
    panels = [(k, t) for k, t in (("semantic", "(a) every parseable call"),
                                  ("call_expected", "(b) call-expected items")) if k in d]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels) + 0.3, 2.5),
                             sharey=False, squeeze=False)
    pretty = dict(RUN_ORDER)
    from matplotlib.lines import Line2D
    for ax, (key, title) in zip(axes[0], panels):
        rows = sorted(d[key]["rows"], key=lambda r: (r["family"] == "recent", -r["gap"]))
        names = [pretty.get(r["run"], r["run"].replace("base_", "").replace("_", " "))
                 for r in rows]
        gaps = [r["gap"] for r in rows]
        cols = [BLUE if r["family"] == "earlier" else ORANGE for r in rows]
        ys = np.arange(len(rows))[::-1]
        ax.hlines(ys, 0, gaps, color=cols, lw=1.2, alpha=0.9)
        ax.scatter(gaps, ys, color=cols, s=14, zorder=3, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_yticks(ys)
        ax.set_yticklabels(names)
        for lab, c in zip(ax.get_yticklabels(), cols):   # identity also on the label
            lab.set_color(c)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("probe AUC minus confidence AUC")
        _panel_title(ax, title)
        q = d[key].get("summary", {})
        if q:
            ax.text(0.98, 0.03, f"earlier families {q['gap_earlier_mean']:+.2f}\n"
                                f"recent families {q['gap_recent_mean']:+.2f}\n"
                                f"rank test p = {q['mannwhitney_p']:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
                    color=INK2)
    # family legend below both panels, clear of every bar; the y labels carry
    # the same colour so identity is never colour-alone on the mark
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.5,
                               label="earlier families: Llama / Qwen3 / Gemma"),
                        Line2D([], [], color=ORANGE, lw=1.5,
                               label="recent families: MiniCPM5 / Qwen3.5")],
               loc="lower center", ncol=2, handlelength=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.07, 1, 1))
    return _save(fig, "fig6_family", {k: d[k].get("summary") for k, _ in panels})


# ─────────────────────────────────────────────────────────────────────────────
def fig7_budget(highlight="base_llama1b_bfcl"):
    d = _load_json(THEORY / "label_budget.json")
    if not d or not d.get("curve"):
        return None
    curves = {k: v for k, v in d["curve"].items() if v}
    if highlight not in curves:
        highlight = next(iter(curves))
    conf = {r["run"]: r["confidence"] for r in d.get("cross_run", [])}
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for tag, curve in curves.items():
        x = [c["approx_train_positives"] for c in curve]
        yv = [c["auc"] for c in curve]
        if tag == highlight:
            ax.plot(x, yv, color=BLUE, lw=1.4, marker="o", ms=3, zorder=3)
        else:
            ax.plot(x, yv, color=GRID, lw=0.8, zorder=1)
    if highlight in conf:
        ax.axhline(conf[highlight], color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.text(ax.get_xlim()[1] if False else max(c["approx_train_positives"]
                                                   for c in curves[highlight]),
                conf[highlight], "log-probability ", ha="right", va="bottom",
                fontsize=6.2, color=INK2)
    hx = [c["approx_train_positives"] for c in curves[highlight]]
    ax.set_xscale("log")
    ax.set_xticks(hx)
    ax.set_xticklabels([str(v) for v in hx])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("labelled failures available for training")
    ax.set_ylabel("token-role probe AUC")
    ax.set_ylim(0.5, 1.0)
    ax.text(0.02, 0.97, highlight.replace("base_", "").replace("_", " "),
            transform=ax.transAxes, va="top", fontsize=6.6, color=BLUE)
    ax.text(0.02, 0.89, "other runs", transform=ax.transAxes, va="top", fontsize=6.6,
            color=MUTED)
    fig.tight_layout()
    return _save(fig, "fig7_budget", d.get("cross_run_summary"))


# ─────────────────────────────────────────────────────────────────────────────
def fig8_multiturn():
    d = _load_json(THEORY / "multiturn.json")
    if not d:
        return None
    tags = list(d)
    names = {"mt_llama1b": "Llama-1B", "mt_minicpm": "MiniCPM5-2B"}
    dets = [("token-role probe", BLUE), ("per-head profile", ORANGE),
            ("LapEigvals", AQUA), ("mean log-probability", INK2)]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    top = 0.0
    for i, t in enumerate(tags):
        r = d[t]
        c, k = 100 * r["failure_rate_clean_history"], 100 * r["failure_rate_corrupted_history"]
        top = max(top, c, k)
        ax.plot([i - 0.18, i + 0.18], [c, k], color=INK, lw=1.0)
        ax.scatter([i - 0.18], [c], color=MUTED, s=18, zorder=3)
        ax.scatter([i + 0.18], [k], color=ORANGE, s=18, zorder=3)
        ax.annotate(f"p = {r['fisher_p']:.3f}", (i + 0.18, k), xytext=(5, -1),
                    textcoords="offset points", ha="left", va="center", fontsize=6.0,
                    color=INK2)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([names.get(t, t) for t in tags])
    ax.set_xlim(-0.5, len(tags) - 0.5 + 0.45)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("failure rate, %")
    ax.set_ylim(0, min(100, top * 1.3))             # headroom for the two labels
    ax.text(0.02, 0.97, "clean history", transform=ax.transAxes, va="top", fontsize=6.4,
            color=MUTED)
    ax.text(0.02, 0.89, "corrupted history", transform=ax.transAxes, va="top",
            fontsize=6.4, color=ORANGE)
    _panel_title(ax, "(a) an upstream wrong call cascades")

    ax = axes[1]
    blocks = [("auc_semantic_turn0", "first turn"), ("auc_semantic_history", "with history"),
              ("auc_corrupted_trained_on_clean", "corrupted,\ntrained clean")]
    xs = np.arange(len(blocks))
    for ti, t in enumerate(tags):
        r = d[t]
        for di, (det, col) in enumerate(dets):
            ys = [r[b].get(det, np.nan) for b, _ in blocks]
            ax.plot(xs + (ti - (len(tags) - 1) / 2) * 0.12, ys, color=col, lw=1.0,
                    marker="o" if ti == 0 else "s", ms=3, alpha=0.9 if ti == 0 else 0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in blocks])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    # detector identity as a figure-level legend above the panels, model
    # identity by marker shape; neither can collide with the data
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=col, lw=1.4, label=det) for det, col in dets]
    handles += [Line2D([], [], color=INK2, lw=0, marker="o", ms=3.5,
                       label=names.get(tags[0], tags[0]))]
    if len(tags) > 1:
        handles += [Line2D([], [], color=INK2, lw=0, marker="s", ms=3.5,
                           label=names.get(tags[1], tags[1]))]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=6.2,
               handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    _panel_title(ax, "(b) what each detector still sees")
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.92))
    return _save(fig, "fig8_multiturn", {t: {"fisher_p": d[t]["fisher_p"]} for t in tags})


# ─────────────────────────────────────────────────────────────────────────────
def fig9_latency():
    d = _load_json(THEORY / "latency.json")
    if not d:
        return None
    stages = [("generation", "generate the call", INK2),
              ("teacher_forced_pass_only", "one forward pass", INK2),
              ("token_role_gather", "token-role gather", BLUE),
              ("perhead_features_alone", "per-head spectra", ORANGE),
              ("lapeig_features_alone", "LapEigvals", AQUA)]
    stages = [(k, lab, c) for k, lab, c in stages if k in d]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ys = np.arange(len(stages))[::-1]
    vals = [d[k]["median_ms"] for k, _, _ in stages]
    ax.barh(ys, vals, color=[c for _, _, c in stages], height=0.62)
    for yi, v in zip(ys, vals):
        ax.text(v * 1.08, yi, f"{v:.1f} ms" if v < 10 else f"{v:.0f} ms", va="center",
                fontsize=6.4, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, _ in stages])
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlabel(f"median per call, {d.get('gpu', '').replace('NVIDIA ', '')}")
    ax.set_xlim(right=max(vals) * 4)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return _save(fig, "fig9_latency", {k: d[k]["median_ms"] for k, _, _ in stages})


if __name__ == "__main__":
    stats = {}
    for fn in (fig1_resolution, fig2_frontier, fig3_ladder, fig4_heatmap, fig5_jensen,
               fig6_family, fig7_budget, fig8_multiturn, fig9_latency):
        try:
            out = fn()
        except Exception as e:  # one broken input must not take the others down
            out = {"error": f"{type(e).__name__}: {e}"}
        stats[fn.__name__] = out
        status = ("skipped (no input)" if out is None
                  else out["error"] if "error" in (out or {}) else "ok")
        print(f"{fn.__name__:18s} {status}")
    THEORY.mkdir(parents=True, exist_ok=True)
    (THEORY / "figure_stats.json").write_text(json.dumps(stats, indent=2, default=str),
                                              encoding="utf-8")
    print("figures ->", FIG)
