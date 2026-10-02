"""
Figures for the ICML draft, from the result files. Writes PDFs into
paper/icml/figures/. ICML column width is 3.25 in and text width 6.75 in;
figures are drawn at those widths and inserted unscaled.

  fig1_gap        (a) probe minus confidence per run with tool-resampled
                  intervals; (b) the four tier representatives per run
  fig2_controls   (a) the gap within difficulty strata against the raw gap;
                  (b) the probe's AUC against training failures on Llama,
                  with the other families' gaps as reference lines;
                  (c) gap against the mean log-probability and against the
                  summary chosen on training folds
  fig3_attention  head-averaged, per-head and LapEigvals against the probe
  fig4_cost       per-call cost of each path
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THEORY = ROOT / "data" / "theory"
FIG = ROOT / "paper" / "icml" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COL, TEXT = 3.25, 6.75
INK, GREY, LIGHT = "#2b2b2b", "#8a8a8a", "#cfcfcf"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "xtick.color": INK, "ytick.color": INK})


def rows():
    return json.loads((THEORY / "icml_rows.json").read_text(encoding="utf-8"))


def label(r):
    return f"{r['model']}, {r['bench']}" + (" †" if r["under"] else "")


def fig1():
    R = rows()
    R = sorted(R, key=lambda r: (r["side"] != "internals", -(r["Gap"][0] if r["Gap"] else -9)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(R))[::-1]
    for yi, r in zip(y, R):
        c = BLUE if r["side"] == "internals" else ORANGE
        if r["Gap"] is None:
            continue
        g, lo, hi = r["Gap"]
        alpha = 0.45 if r["under"] else 1.0
        a.plot([lo, hi], [yi, yi], color=c, lw=1.4, alpha=alpha, solid_capstyle="butt")
        a.plot(g, yi, "o", color=c, ms=4.2, alpha=alpha)
    a.axvline(0, color=INK, lw=0.7)
    a.set_yticks(y)
    a.set_yticklabels([label(r) for r in R])
    a.set_xlabel("probe AUC minus confidence AUC, held-out tools")
    a.set_title("(a) the internal advantage", loc="left")
    a.text(0.98, 0.04, "† fewer than 30 failures", transform=a.transAxes, ha="right", fontsize=6.3, color=GREY)
    # (b) tiers
    for yi, r in zip(y, R):
        if r["under"]:
            continue
        att = np.nanmax([r["PerHead"], r["LapEig"]])
        b.plot(r["Floor"], yi, "|", color=GREY, ms=8, mew=1.3)
        b.plot(r["Conf"], yi, "s", color=INK, ms=3.6, mfc="white")
        b.plot(att, yi, "D", color=AQUA, ms=3.4)
        b.plot(r["Probe"], yi, "o", color=BLUE if r["side"] == "internals" else ORANGE, ms=4)
    b.set_yticks(y)
    b.set_yticklabels([])
    b.set_xlabel("AUC")
    b.set_xlim(0.45, 1.0)
    b.set_title("(b) one representative per tier", loc="left")
    top = y[0] + 0.9
    for x, txt, c in ((0.47, "length floor", GREY), (0.63, "confidence", INK), (0.79, "attention", AQUA), (0.95, "probe", BLUE)):
        b.text(x, top, txt, color=c, fontsize=6.3, ha="left" if x < 0.5 else "center", va="bottom")
    b.set_ylim(-0.8, top + 0.9)
    a.set_ylim(-0.8, top + 0.9)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "fig1_gap.pdf")
    plt.close(fig)


def fig2():
    R = rows()
    D = json.loads((THEORY / "difficulty.json").read_text(encoding="utf-8"))
    B = json.loads((THEORY / "budget_matched.json").read_text(encoding="utf-8"))
    C = json.loads((THEORY / "confidence_best.json").read_text(encoding="utf-8"))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXT, 2.4))
    # (a) difficulty
    side = {r["model"]: r["side"] for r in R}
    for m, d in D.items():
        col = BLUE if side.get(m) == "internals" else ORANGE
        a.plot([0, 1], [d["gap"], d["gap_within"]], "-", color=col, lw=1)
        a.plot([0, 1], [d["gap"], d["gap_within"]], "o", color=col, ms=3.5)
        a.text(1.05, d["gap_within"], m, color=col, fontsize=6.2, va="center")
    a.axhline(0, color=INK, lw=0.7)
    a.set_xticks([0, 1])
    a.set_xticklabels(["raw", "within\ndifficulty strata"])
    a.set_xlim(-0.2, 1.9)
    a.set_ylabel("probe minus confidence")
    a.set_title("(a) item difficulty held fixed", loc="left")
    # (b) budget
    for tag, d in B.items():
        pts = sorted(((int(k), v) for k, v in d["curve"].items()))
        x = [p for p, _ in pts]
        b.plot(x, [v["probe"] for _, v in pts], "o-", color=BLUE, ms=3.2, lw=1.1)
        b.plot(x, [v["confidence"] for _, v in pts], "--", color=INK, lw=0.9)
        name = "Llama-1B" if "1b" in tag else "Llama-3B"
        b.text(x[0] * 0.98, pts[0][1]["probe"] + (0.02 if "3b" in tag else -0.025), name, color=BLUE, fontsize=6.2, ha="right", va="center")
        b.text(x[-1] * 1.04, pts[-1][1]["confidence"], name, color=INK, fontsize=6.2, va="center")
    b.set_xscale("log")
    b.set_xlim(38, 330)
    b.set_xticks([51, 91, 157, 236])
    b.set_xticklabels(["51", "91", "157", "236"])
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_xlabel("labelled failures used for training")
    b.set_ylabel("AUC")
    b.set_ylim(0.55, 0.95)
    b.text(0.03, 0.93, "probe (solid), confidence (dashed)", transform=b.transAxes, fontsize=6.2, color=INK)
    b.set_title("(b) labels thinned on Llama", loc="left")
    # (c) summary choice
    for r in R:
        if r["under"] or r["tag"] not in C or C[r["tag"]]["n_summaries"] < 2:
            continue
        col = BLUE if r["side"] == "internals" else ORANGE
        c.plot(r["Gap"][0], C[r["tag"]]["gap_vs_chosen"], "o", color=col, ms=4)
        c.text(r["Gap"][0] + 0.012, C[r["tag"]]["gap_vs_chosen"], r["model"].replace("-3.2", "").replace("5-2B", "5"),
               fontsize=6.0, color=col, va="center")
    lim = [-0.2, 0.42]
    c.plot(lim, lim, color=LIGHT, lw=0.8)
    c.axhline(0, color=INK, lw=0.6)
    c.axvline(0, color=INK, lw=0.6)
    c.set_xlim(lim)
    c.set_ylim(lim)
    c.set_xlabel("gap, mean log-probability")
    c.set_ylabel("gap, summary chosen on training folds")
    c.set_title("(c) confidence summary", loc="left")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "fig2_controls.pdf")
    plt.close(fig)


def fig3():
    R = [r for r in rows() if not r["under"]]
    R = sorted(R, key=lambda r: (r["side"] != "internals", -r["Probe"]))
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    y = np.arange(len(R))[::-1]
    for yi, r in zip(y, R):
        ax.plot([r["HeadAvg"], r["Probe"]], [yi, yi], color=LIGHT, lw=0.9)
        ax.plot(r["Floor"], yi, "|", color=GREY, ms=8, mew=1.2)
        ax.plot(r["HeadAvg"], yi, "x", color=GREY, ms=4.5, mew=1.2)
        ax.plot(r["PerHead"], yi, "D", color=AQUA, ms=3.4)
        ax.plot(r["LapEig"], yi, "^", color=ORANGE, ms=3.6)
        ax.plot(r["Probe"], yi, "o", color=BLUE, ms=3.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label(r) for r in R], fontsize=6.3)
    ax.set_xlabel("AUC")
    ax.set_xlim(0.45, 1.0)
    top = y[0] + 0.9
    for x, txt, c in ((0.5, "floor", GREY), (0.6, "head-averaged", GREY), (0.74, "per-head", AQUA), (0.86, "LapEigvals", ORANGE), (0.96, "probe", BLUE)):
        ax.text(x, top, txt, color=c, fontsize=6.2, ha="center", va="bottom")
    ax.set_ylim(-0.8, top + 1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_attention.pdf")
    plt.close(fig)


def fig4():
    T = json.loads((THEORY / "latency.json").read_text(encoding="utf-8"))
    stages = [("generation", "generate the call", GREY), ("teacher_forced_pass_only", "one forward pass", GREY),
              ("token_role_gather", "token-role gather", BLUE), ("lapeig_features_alone", "LapEigvals", ORANGE),
              ("perhead_features_alone", "per-head spectra", AQUA)]
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    ys = np.arange(len(stages))[::-1]
    vals = [T[k]["median_ms"] for k, _, _ in stages]
    p90 = [T[k]["p90_ms"] for k, _, _ in stages]
    ax.barh(ys, vals, color=[c for _, _, c in stages], height=0.6)
    for yi, v, q in zip(ys, vals, p90):
        ax.plot([v, q], [yi, yi], color=INK, lw=0.8)
        ax.text(max(v, q) * 1.15, yi, f"{v:.1f} ms" if v < 10 else f"{v:.0f} ms", va="center", fontsize=6.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([s for _, s, _ in stages])
    ax.set_xscale("log")
    ax.set_xlim(0.2, 6000)
    ax.set_xlabel("median per call, bar; line to the 90th percentile")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cost.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        try:
            f()
            print(f.__name__, "ok")
        except Exception as e:
            print(f.__name__, "FAILED", type(e).__name__, e)
