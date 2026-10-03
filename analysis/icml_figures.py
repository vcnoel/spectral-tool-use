"""
Figures for the ICML draft, from the result files, in the house figure
grammar: built at the document's measured widths (text 487.8225pt, column
234.8775pt) and inserted unscaled, typeset by LaTeX in the document's font,
colourblind-safe palette, series labelled at their ends, no legend on data.
Every figure is checked afterwards by the layout gate:

    FIGDIR=paper/icml/figures TEXTWIDTH_PT=487.8225 \
        python ~/.claude/skills/research-writing/scripts/check_figures.py

  fig1_gap        (a) the internal advantage per run with its interval
                  (b) one representative per access tier per run
  fig2_controls   (a) advantage raw and within difficulty strata
                  (b) probe and confidence AUC as training failures shrink
                  (c) advantage against two choices of confidence summary
  fig3_attention  head-averaged, per-head, LapEigvals and probe per run
  fig4_cost       per-call cost of each path, median and 90th percentile
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THEORY = ROOT / "data" / "theory"
FIG = ROOT / "paper" / "icml" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TEXT = 487.8225 / 72.27          # inches, measured from the compiled document
COL = 234.8775 / 72.27

INK, MID, FAINT = "#1A1A1A", "#7A7A7A", "0.86"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "text.usetex": True,
    "pgf.preamble": r"\usepackage{times}\usepackage{amsmath}",
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
ANNOT = 6.8


def rows():
    return json.loads((THEORY / "icml_rows.json").read_text(encoding="utf-8"))


def short(model):
    return (model.replace("Llama-3.2-", "Llama ").replace("Gemma-3-", "Gemma ")
            .replace("Qwen3.5-", "Qwen3.5 ").replace("Qwen3-", "Qwen3 ").replace("MiniCPM5-", "MiniCPM5 "))


def run_label(r):
    return f"{short(r['model'])}, {r['bench']}" + (r"$^\dagger$" if r["under"] else "")


def side_colour(r):
    return BLUE if r["side"] == "internals" else ORANGE


def spread(ys, gap, lo=None, hi=None):
    """Move label positions apart until neighbours are at least `gap` apart,
    keeping their order and staying as close as possible to the targets."""
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap:
                mid = 0.5 * (pos[i] + pos[i - 1])
                pos[i - 1], pos[i] = mid - gap / 2, mid + gap / 2
                moved = True
        if lo is not None and pos[0] < lo:
            pos += lo - pos[0]
        if hi is not None and pos[-1] > hi:
            pos -= pos[-1] - hi
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out


def save_at_width(fig, name, target_in):
    """Save, measure the emitted width, and resize the figure until the tight
    output is the target width: labels do not scale with the figure, so a
    fixed guess drifts."""
    import pymupdf
    path = FIG / name
    for _ in range(6):
        fig.savefig(path)
        with pymupdf.open(path) as doc:
            got = doc[0].rect.width / 72.0
        goal = target_in - 0.015       # land just inside the text block
        if goal - 0.01 <= got <= goal:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + (goal - 0.005 - got), h)
    plt.close(fig)


def ordered(R):
    return sorted(R, key=lambda r: (r["side"] != "internals", r["under"], -(r["Gap"][0] if r["Gap"] else 0)))


# ─────────────────────────────────────────────────────────────────────────────
def fig1():
    R = ordered(rows())
    y = np.arange(len(R))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT * 0.935, 2.75), sharey=True,
                               gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.06})
    for yi, r in zip(y, R):
        g, lo, hi = r["Gap"]
        col = MID if r["under"] else side_colour(r)
        a.plot([lo, hi], [yi, yi], color=col, lw=1.2, solid_capstyle="butt")
        a.plot(g, yi, "o", color=col, ms=3.6)
    a.axvline(0, color=INK, lw=0.5)
    a.set_yticks(y)
    a.set_yticklabels([run_label(r) for r in R])
    a.set_xlabel("probe AUC minus confidence AUC")
    a.set_title("(a) internal advantage, held-out tools", loc="left")
    a.set_ylim(-0.7, len(R) - 0.3)

    for yi, r in zip(y, R):
        if r["under"]:
            continue
        att = np.nanmax([r["PerHead"], r["LapEig"]])
        b.plot([min(r["Conf"], r["Probe"]), max(r["Conf"], r["Probe"])], [yi, yi], color=FAINT, lw=2.4,
               solid_capstyle="butt", zorder=1)
        b.plot(r["Floor"], yi, "|", color=MID, ms=7, mew=1.1, zorder=2)
        b.plot(r["Conf"], yi, "s", color=INK, ms=3.4, mfc="white", mew=0.8, zorder=3)
        b.plot(att, yi, "D", color=GREEN, ms=3.2, zorder=3)
        b.plot(r["Probe"], yi, "o", color=side_colour(r), ms=3.6, zorder=4)
    b.set_xlim(0.45, 1.0)
    b.set_xlabel("AUC")
    b.set_title("(b) one representative per access tier", loc="left")
    handles = [plt.Line2D([], [], color=MID, marker="|", ls="", ms=7, mew=1.1, label="length floor"),
               plt.Line2D([], [], color=INK, marker="s", ls="", ms=3.4, mfc="white", mew=0.8, label="log-probability"),
               plt.Line2D([], [], color=GREEN, marker="D", ls="", ms=3.2, label="attention, per head"),
               plt.Line2D([], [], color=INK, marker="o", ls="", ms=3.6, label="residual probe")]
    b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False,
             handletextpad=0.3, columnspacing=0.9)
    save_at_width(fig, "fig1_gap.pdf", TEXT)


# ─────────────────────────────────────────────────────────────────────────────
def fig2():
    R = rows()
    D = json.loads((THEORY / "difficulty.json").read_text(encoding="utf-8"))
    B = json.loads((THEORY / "budget_matched.json").read_text(encoding="utf-8"))
    C = json.loads((THEORY / "confidence_best.json").read_text(encoding="utf-8"))
    side = {r["model"]: r["side"] for r in R}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXT * 0.965, 2.2),
                                  gridspec_kw={"wspace": 1.0, "width_ratios": [1, 1, 1]})

    # (a) difficulty held fixed
    names = list(D)
    ends = [D[m]["gap_within"] for m in names]
    lab = spread(ends, 0.034)
    for m, yl in zip(names, lab):
        d = D[m]
        col = BLUE if side.get(m) == "internals" else ORANGE
        a.plot([0, 1], [d["gap"], d["gap_within"]], "-", color=col, lw=0.9)
        a.plot([0, 1], [d["gap"], d["gap_within"]], "o", color=col, ms=3)
        a.plot([1.04, 1.12], [d["gap_within"], yl], color=col, lw=0.4)
        a.text(1.15, yl, short(m), color=col, fontsize=ANNOT, va="center")
    a.axhline(0, color=MID, lw=0.5)
    a.set_xticks([0, 1])
    a.set_xticklabels(["all items", "within\nstrata"])
    a.set_xlim(-0.15, 1.95)
    a.spines["bottom"].set_bounds(0, 1)
    a.set_ylabel("probe minus confidence")
    a.set_title("(a) difficulty held fixed", loc="left")

    # (b) label budget
    styles = {"1b": INK, "3b": MID}
    lab_pos, lab_txt, lab_col = [], [], []
    xe = 1
    for tag, d in B.items():
        k = "1b" if "1b" in tag else "3b"
        pts = sorted((int(p), v) for p, v in d["curve"].items())
        x = [p for p, _ in pts]
        b.plot(x, [v["probe"] for _, v in pts], "o-", color=styles[k], ms=2.8, lw=0.9)
        b.plot(x, [v["confidence"] for _, v in pts], "--", color=styles[k], lw=0.9)
        name = "1B" if k == "1b" else "3B"
        lab_pos += [pts[-1][1]["probe"], pts[-1][1]["confidence"]]
        lab_txt += [f"{name} probe", f"{name} log-prob"]
        lab_col += [styles[k], styles[k]]
        xe = max(xe, x[-1])
    yl = spread(lab_pos, 0.03)
    for p, t, col in zip(yl, lab_txt, lab_col):
        b.text(xe * 1.12, p, t, color=col, fontsize=ANNOT, va="center")
    b.set_xscale("log")
    b.set_xticks([51, 100, 200])
    b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_xlim(42, xe * 1.08)
    b.set_ylim(0.55, 0.95)
    b.set_xlabel("labelled failures in training")
    b.set_ylabel("AUC")
    b.set_title("(b) failures thinned on Llama", loc="left")

    # (c) choice of confidence summary
    pts = [(r, C[r["tag"]]) for r in R
           if not r["under"] and r["tag"] in C and C[r["tag"]]["n_summaries"] > 1]
    xs = [r["Gap"][0] for r, _ in pts]
    ys = [cc["gap_vs_chosen"] for _, cc in pts]
    lim = [-0.2, 0.42]
    c.plot(lim, lim, color=FAINT, lw=0.8, zorder=0)
    c.axhline(0, color=MID, lw=0.5)
    c.axvline(0, color=MID, lw=0.5)
    for (r, _), xv, yv in zip(pts, xs, ys):
        c.plot(xv, yv, "o", color=side_colour(r), ms=3.4)
    lab = spread(ys, 0.055, lo=lim[0] + 0.02, hi=lim[1] - 0.02)
    for (r, _), xv, yv, yl in zip(pts, xs, ys, lab):
        tx = 0.47
        c.plot([xv + 0.012, tx - 0.01], [yv, yl], color=side_colour(r), lw=0.4)
        c.text(tx, yl, f"{short(r['model'])}, {r['bench']}", color=side_colour(r), fontsize=ANNOT, va="center")
    c.set_xlim(lim)
    c.set_ylim(lim)
    c.set_xticks([-0.2, 0, 0.2, 0.4])
    c.set_yticks([-0.2, 0, 0.2, 0.4])
    c.set_xlabel("against mean log-probability")
    c.set_ylabel("against fold-chosen summary")
    c.set_title("(c) confidence summary", loc="left")
    save_at_width(fig, "fig2_controls.pdf", TEXT)


# ─────────────────────────────────────────────────────────────────────────────
def fig3():
    R = [r for r in ordered(rows()) if not r["under"]]
    y = np.arange(len(R))[::-1]
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    for yi, r in zip(y, R):
        ax.plot([r["HeadAvg"], r["PerHead"]], [yi, yi], color=FAINT, lw=2.2, solid_capstyle="butt", zorder=1)
        ax.plot(r["Floor"], yi, "|", color=MID, ms=7, mew=1.1, zorder=2)
        ax.plot(r["HeadAvg"], yi, "x", color=INK, ms=3.6, mew=0.9, zorder=3)
        ax.plot(r["PerHead"], yi, "D", color=GREEN, ms=3.0, zorder=3)
        ax.plot(r["LapEig"], yi, "^", color=ORANGE, ms=3.4, zorder=3)
        ax.plot(r["Probe"], yi, "o", color=BLUE, ms=3.4, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([run_label(r) for r in R], fontsize=7)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC")
    ax.set_ylim(-0.7, len(R) - 0.3)
    handles = [plt.Line2D([], [], color=MID, marker="|", ls="", ms=7, mew=1.1, label="length floor"),
               plt.Line2D([], [], color=INK, marker="x", ls="", ms=3.6, mew=0.9, label="head-averaged"),
               plt.Line2D([], [], color=GREEN, marker="D", ls="", ms=3.0, label="per-head"),
               plt.Line2D([], [], color=ORANGE, marker="^", ls="", ms=3.4, label="LapEigvals"),
               plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=3.4, label="residual probe")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.32, -0.2), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=0.8, fontsize=7)
    save_at_width(fig, "fig3_attention.pdf", COL)


# ─────────────────────────────────────────────────────────────────────────────
def fig4():
    T = json.loads((THEORY / "latency.json").read_text(encoding="utf-8"))
    stages = [("generation", "generate the call", MID),
              ("teacher_forced_pass_only", "forward pass", MID),
              ("token_role_gather", "token-role gather", BLUE),
              ("lapeig_features_alone", "LapEigvals", ORANGE),
              ("perhead_features_alone", "per-head spectra", GREEN)]
    stages = [s for s in stages if s[0] in T]
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    ys = np.arange(len(stages))[::-1]
    for yi, (k, _, col) in zip(ys, stages):
        med, p90 = T[k]["median_ms"], T[k]["p90_ms"]
        ax.barh(yi, med, color=col, height=0.55)
        ax.plot([med, p90], [yi, yi], color=INK, lw=0.6)
        ax.plot([p90], [yi], "|", color=INK, ms=4, mew=0.6)
        ax.text(p90 * 1.35, yi, f"{med:.1f}" if med < 10 else f"{med:.0f}", va="center", fontsize=ANNOT)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[1] for s in stages], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 1e4)
    ax.set_xlabel("milliseconds per call")
    ax.spines["left"].set_visible(False)
    save_at_width(fig, "fig4_cost.pdf", COL)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        try:
            f()
            print(f.__name__, "ok")
        except Exception as e:  # one broken input must not take the others down
            print(f.__name__, "FAILED", type(e).__name__, e)
