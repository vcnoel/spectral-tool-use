"""
Figures for the v2 ICML draft (paper/icml_v2/figures), in the house grammar of
analysis/icml_figures.py (same widths, fonts, palette and width-fitting save).

  fig1_types     probe minus confidence per run (a) on every failure, (b) on wrong
                 argument values against valid calls, (c) on dropped parallel calls
                 against valid calls, all categories and within parallel categories
  fig2_judges    (a) AUC of the floor, confidence, the output-only judge, the reader
                 model and the probe per run, (b) probe minus each trained comparator
  fig3_controls  (a) advantage within difficulty strata against the raw advantage,
                 (b) AUC against labelled failures in training, (c) advantage under a
                 fold-chosen confidence summary against the mean log-probability
  fig4_attention (a) head-averaged, per-head, LapEigvals and probe AUC per run,
                 (b) cost per call of each path

Gate afterwards:
    FIGDIR=paper/icml_v2/figures TEXTWIDTH_PT=487.8225 python check_figures.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import icml_figures as H  # noqa: E402  (sets the pgf backend and the rcParams)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

FIG = ROOT / "paper" / "icml_v2" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
H.FIG = FIG
AUD = ROOT / "results" / "audit_oct2026"
V2 = ROOT / "results" / "v2_oct2026"
THEORY = ROOT / "data" / "theory"
INK, MID, FAINT, BLUE, ORANGE, GREEN = H.INK, H.MID, H.FAINT, H.BLUE, H.ORANGE, H.GREEN
ANNOT = H.ANNOT
JSONROWS = [("MiniCpmBfclJson", "MiniCPM5 2B, BFCL JSON"), ("QwenThreeFiveBfclJson", "Qwen3.5 0.8B, BFCL JSON")]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def powered():
    return [r for r in H.ordered(H.rows()) if not r["under"]]


def interval(ax, y, g, col, open_=False, ms=3.4):
    ax.plot([g["ci_lo"], g["ci_hi"]], [y, y], color=col, lw=1.1, solid_capstyle="butt")
    ax.plot(g["delta"], y, "o", color=col, ms=ms, mfc="white" if open_ else col, mew=0.9)


# ─────────────────────────────────────────────────────────────────────────────
def fig1():
    R = powered()
    T = load(AUD / "failure_type.json")
    J = load(AUD / "forced_json.json")
    CAT = load(V2 / "redteam_checks.json")["category"]
    labels = [H.run_label(r) for r in R] + [lab for _, lab in JSONROWS]
    keys = [r["key"] for r in R] + [k for k, _ in JSONROWS]
    side = {r["key"]: r["side"] for r in R}
    side.update({k: "confidence" for k, _ in JSONROWS})
    allgap = {r["key"]: {"delta": r["Gap"][0], "ci_lo": r["Gap"][1], "ci_hi": r["Gap"][2]} for r in R}
    allgap["MiniCpmBfclJson"] = J["MiniCPM5-2B"]["json"]["scored_population"]
    allgap["QwenThreeFiveBfclJson"] = J["Qwen3.5-0.8B"]["json"]["scored_population"]
    y = np.arange(len(keys))[::-1].astype(float)
    y[-len(JSONROWS):] -= 0.6            # a gap before the forced-JSON rows
    fig, axs = plt.subplots(1, 3, figsize=(H.TEXT * 0.95, 2.6), sharey=True,
                            gridspec_kw={"wspace": 0.08, "width_ratios": [1, 1, 1]})
    for yi, k in zip(y, keys):
        col = BLUE if side[k] == "internals" else ORANGE
        js = k.endswith("Json")
        interval(axs[0], yi, allgap[k], col, js)
        if "wrong_arg_values" in T.get(k, {}):
            interval(axs[1], yi, T[k]["wrong_arg_values"], col, js)
        if k in CAT:
            c = CAT[k]
            axs[2].plot([min(c["conf_auc"], c["probe_auc"]), max(c["conf_auc"], c["probe_auc"])], [yi, yi],
                        color=FAINT, lw=2.4, solid_capstyle="butt", zorder=1)
            axs[2].plot(c["conf_auc"], yi, "s", color=INK, ms=3.4, mfc="white", mew=0.8, zorder=3)
            axs[2].plot(c["indicator_auc"], yi, "D", color=MID, ms=3.2, zorder=3)
            axs[2].plot(c["probe_auc"], yi, "o", color=col, ms=3.6, mfc="white" if js else col, mew=0.9, zorder=4)
    titles = ["(a) every scored failure", "(b) wrong argument values", "(c) dropped parallel calls, AUC"]
    for ax, t in zip(axs[:2], titles[:2]):
        ax.axvline(0, color=INK, lw=0.5)
        ax.set_xlim(-0.75, 0.75)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_title(t, loc="left")
        ax.set_xlabel("probe AUC minus confidence AUC")
    axs[2].set_title(titles[2], loc="left")
    axs[2].set_xlim(0.55, 1.02)
    axs[2].set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axs[2].set_xlabel("AUC against valid calls")
    yt = y[keys.index("GemmaGlaive")] - 0.1
    for txt, mk, kw, dy in (("probe", "o", dict(color=INK), 0.0), ("confidence", "s", dict(color=INK, mfc="white", mew=0.8), -0.8),
                            ("parallel-category flag", "D", dict(color=MID), -1.6)):
        axs[2].plot(0.6, yt + dy, mk, ms=3.4, **kw)
        axs[2].text(0.625, yt + dy, txt, fontsize=ANNOT, va="center", color=INK)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(labels)
    axs[0].set_ylim(y.min() - 0.7, y.max() + 0.7)
    H.save_at_width(fig, "fig1_types.pdf", H.TEXT)


# ─────────────────────────────────────────────────────────────────────────────
def fig2():
    R = powered()
    F = load(AUD / "floors.json")
    rd = {p.stem: load(p) for p in (AUD / "reader_runs").glob("*.json")}
    y = np.arange(len(R))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(H.TEXT * 0.95, 2.45), sharey=True,
                               gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.06})
    for yi, r in zip(y, R):
        k = r["key"]
        oj = F[k]["judges"]["output_judge"]
        a.plot([min(r["Conf"], r["Probe"]), max(r["Conf"], r["Probe"])], [yi, yi], color=FAINT, lw=2.4,
               solid_capstyle="butt", zorder=1)
        a.plot(r["Floor"], yi, "|", color=MID, ms=7, mew=1.1, zorder=2)
        a.plot(r["Conf"], yi, "s", color=INK, ms=3.4, mfc="white", mew=0.8, zorder=3)
        a.plot(oj["pooled_auc"], yi, "^", color=GREEN, ms=3.6, zorder=3)
        if k in rd:
            a.plot(rd[k]["reader_auc"], yi, "D", color=MID, ms=3.0, zorder=3)
        a.plot(r["Probe"], yi, "o", color=H.side_colour(r), ms=3.6, zorder=4)
        g = oj["probe_minus"]
        b.plot([g["ci_lo"], g["ci_hi"]], [yi + 0.15, yi + 0.15], color=GREEN, lw=1.1, solid_capstyle="butt")
        b.plot(g["delta"], yi + 0.15, "^", color=GREEN, ms=3.6)
        if k in rd:
            g = rd[k]["probe_minus_reader"]
            b.plot([g["ci_lo"], g["ci_hi"]], [yi - 0.18, yi - 0.18], color=MID, lw=1.1, solid_capstyle="butt")
            b.plot(g["delta"], yi - 0.18, "D", color=MID, ms=3.0)
    a.set_yticks(y)
    a.set_yticklabels([H.run_label(r) for r in R])
    a.set_ylim(-0.7, len(R) - 0.3)
    a.set_xlim(0.45, 1.0)
    a.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    a.set_xlabel("AUC")
    a.set_title("(a) AUC of each judge", loc="left")
    b.axvline(0, color=INK, lw=0.5)
    b.set_xlim(-0.2, 0.5)
    b.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
    b.set_xlabel("probe AUC minus comparator AUC")
    b.set_title("(b) probe minus trained output-side judges", loc="left")
    handles = [plt.Line2D([], [], color=MID, marker="|", ls="", ms=7, mew=1.1, label="length floor"),
               plt.Line2D([], [], color=INK, marker="s", ls="", ms=3.4, mfc="white", mew=0.8, label="confidence"),
               plt.Line2D([], [], color=GREEN, marker="^", ls="", ms=3.6, label="output-only judge"),
               plt.Line2D([], [], color=MID, marker="D", ls="", ms=3.0, label="reader model"),
               plt.Line2D([], [], color=INK, marker="o", ls="", ms=3.6, label="residual probe")]
    a.legend(handles=handles, loc="upper center", bbox_to_anchor=(1.0, -0.2), ncol=5, frameon=False,
             handletextpad=0.3, columnspacing=1.0)
    H.save_at_width(fig, "fig2_judges.pdf", H.TEXT)


# ─────────────────────────────────────────────────────────────────────────────
def fig3():
    R = H.rows()
    D = load(THEORY / "difficulty.json")
    B = load(THEORY / "budget_matched.json")
    C = load(THEORY / "confidence_best.json")
    side = {r["model"]: r["side"] for r in R}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(H.TEXT * 0.965, 2.0),
                                  gridspec_kw={"wspace": 1.0, "width_ratios": [1, 1, 1]})
    lim = [-0.15, 0.25]

    def scatter(ax, pts, lim, tx):
        ax.plot(lim, lim, color=FAINT, lw=0.8, zorder=0)
        ax.axhline(0, color=MID, lw=0.5)
        ax.axvline(0, color=MID, lw=0.5)
        ys = [p[1] for p in pts]
        lab = H.spread(ys, 0.16 * (lim[1] - lim[0]) / 1.0 * 0.45, lo=lim[0] + 0.02, hi=lim[1] - 0.02)
        for (xv, yv, txt, col), yl in zip(pts, lab):
            ax.plot(xv, yv, "o", color=col, ms=3.4)
            ax.plot([xv + 0.012 * (lim[1] - lim[0]), tx - 0.012 * (lim[1] - lim[0])], [yv, yl], color=col, lw=0.4)
            ax.text(tx, yl, txt, color=col, fontsize=ANNOT, va="center")
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    pts = [(D[m]["gap"], D[m]["gap_within"], H.short(m), BLUE if side.get(m) == "internals" else ORANGE) for m in D]
    scatter(a, pts, lim, lim[1] + 0.03)
    a.set_xticks([-0.1, 0, 0.1, 0.2])
    a.set_yticks([-0.1, 0, 0.1, 0.2])
    a.set_xlabel("over all BFCL items")
    a.set_ylabel("within difficulty strata")
    a.set_title("(a) advantage, difficulty fixed", loc="left")

    styles = {"1b": INK, "3b": MID}
    lab_pos, lab_txt, lab_col = [], [], []
    xe = 1
    for tag, dd in B.items():
        k = "1b" if "1b" in tag else "3b"
        p_ = sorted((int(p), v) for p, v in dd["curve"].items())
        x = [p for p, _ in p_]
        b.plot(x, [v["probe"] for _, v in p_], "o-", color=styles[k], ms=2.8, lw=0.9)
        b.plot(x, [v["confidence"] for _, v in p_], "--", color=styles[k], lw=0.9)
        name = "1B" if k == "1b" else "3B"
        lab_pos += [p_[-1][1]["probe"], p_[-1][1]["confidence"]]
        lab_txt += [f"{name} probe", f"{name} log-prob"]
        lab_col += [styles[k], styles[k]]
        xe = max(xe, x[-1])
    for p, t, col in zip(H.spread(lab_pos, 0.03), lab_txt, lab_col):
        b.text(xe * 1.12, p, t, color=col, fontsize=ANNOT, va="center")
    b.set_xscale("log")
    b.set_xticks([51, 100, 200])
    b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_xlim(42, xe * 1.08)
    b.set_ylim(0.55, 0.95)
    b.set_xlabel("labelled failures in training")
    b.set_ylabel("AUC")
    b.set_title("(b) AUC by training failures", loc="left")

    lim2 = [-0.2, 0.42]
    pts = [(r["Gap"][0], C[r["tag"]]["gap_vs_chosen"], f"{H.short(r['model'])}, {r['bench']}", H.side_colour(r))
           for r in R if not r["under"] and r["tag"] in C and C[r["tag"]]["n_summaries"] > 1]
    scatter(c, pts, lim2, 0.47)
    c.set_xticks([-0.2, 0, 0.2, 0.4])
    c.set_yticks([-0.2, 0, 0.2, 0.4])
    c.set_xlabel("against mean log-probability")
    c.set_ylabel("against fold-chosen summary")
    c.set_title("(c) advantage, summary chosen", loc="left")
    H.save_at_width(fig, "fig3_controls.pdf", H.TEXT)


# ─────────────────────────────────────────────────────────────────────────────
def fig4():
    R = powered()
    Tl = load(THEORY / "latency.json")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(H.TEXT * 0.95, 2.3),
                                 gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.75})
    y = np.arange(len(R))[::-1]
    for yi, r in zip(y, R):
        ax.plot([r["HeadAvg"], r["PerHead"]], [yi, yi], color=FAINT, lw=2.2, solid_capstyle="butt", zorder=1)
        ax.plot(r["Floor"], yi, "|", color=MID, ms=7, mew=1.1, zorder=2)
        ax.plot(r["HeadAvg"], yi, "x", color=INK, ms=3.6, mew=0.9, zorder=3)
        ax.plot(r["PerHead"], yi, "D", color=GREEN, ms=3.0, zorder=3)
        ax.plot(r["LapEig"], yi, "^", color=ORANGE, ms=3.4, zorder=3)
        ax.plot(r["Probe"], yi, "o", color=BLUE, ms=3.4, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([H.run_label(r) for r in R], fontsize=7)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC")
    ax.set_ylim(-0.7, len(R) - 0.3)
    ax.set_title("(a) attention readouts by run", loc="left")
    handles = [plt.Line2D([], [], color=MID, marker="|", ls="", ms=7, mew=1.1, label="floor"),
               plt.Line2D([], [], color=INK, marker="x", ls="", ms=3.6, mew=0.9, label="head-averaged"),
               plt.Line2D([], [], color=GREEN, marker="D", ls="", ms=3.0, label="per-head"),
               plt.Line2D([], [], color=ORANGE, marker="^", ls="", ms=3.4, label="LapEigvals"),
               plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=3.4, label="probe")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=5, frameon=False,
              handletextpad=0.3, columnspacing=0.7, fontsize=7)

    stages = [("generation", "generate the call", MID), ("teacher_forced_pass_only", "forward pass", MID),
              ("token_role_gather", "token-role gather", BLUE), ("lapeig_features_alone", "LapEigvals", ORANGE),
              ("perhead_features_alone", "per-head spectra", GREEN)]
    stages = [s for s in stages if s[0] in Tl]
    ys = np.arange(len(stages))[::-1]
    for yi, (k, _, col) in zip(ys, stages):
        med, p90 = Tl[k]["median_ms"], Tl[k]["p90_ms"]
        bx.barh(yi, med, color=col, height=0.55)
        bx.plot([med, p90], [yi, yi], color=INK, lw=0.6)
        bx.plot([p90], [yi], "|", color=INK, ms=4, mew=0.6)
        bx.text(p90 * 1.35, yi, f"{med:.1f}" if med < 10 else f"{med:.0f}", va="center", fontsize=ANNOT)
    bx.set_yticks(ys)
    bx.set_yticklabels([s[1] for s in stages], fontsize=7)
    bx.tick_params(axis="y", length=0)
    bx.set_xscale("log")
    bx.set_xlim(0.2, 1e4)
    bx.set_xlabel("milliseconds per call")
    bx.spines["left"].set_visible(False)
    bx.set_title("(b) cost per call, Llama-3.2-1B", loc="left")
    H.save_at_width(fig, "fig4_attention.pdf", H.TEXT)


if __name__ == "__main__":
    import os
    only = sys.argv[1:]
    for f in (fig1, fig2, fig3, fig4):
        if only and f.__name__ not in only:
            continue
        try:
            f()
            print(f.__name__, "ok")
        except Exception as e:
            print(f.__name__, "FAILED", type(e).__name__, e)
