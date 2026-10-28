"""
Audit (October 2026): one markdown summary of the audit result files.

Reads results/audit_oct2026/{recompute,floors,schema_echo}.json and prints
the tables quoted in docs/AUDIT_OCT2026.md; writes them to
results/audit_oct2026/summary.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "audit_oct2026"


def ci(d):
    return "--" if d is None else f"{d['delta']:+.3f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]"


def main():
    rc = json.loads((R / "recompute.json").read_text(encoding="utf-8"))
    fl = json.loads((R / "floors.json").read_text(encoding="utf-8"))
    se = json.loads((R / "schema_echo.json").read_text(encoding="utf-8"))
    L = []
    L.append("### Table A. Headline recomputed, floors and output-side judges (pooled OOF AUC, mean of 5 seeds)\n")
    L.append("| run | side | n+/n- | probe | conf | paper floor | struct floor | conf LR | text(call) | text(call+user) | output judge |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in rc.items():
        f = fl.get(k)
        if f is None:
            continue
        J = f["judges"]
        L.append(f"| {k} | {r['side']} | {r['n_pos']}/{r['n_neg']} | {r['probe_pooled']:.3f} | {r['conf_pooled']:.3f} | "
                 + " | ".join(f"{J[j]['pooled_auc']:.3f}" for j in
                              ("floor_paper", "floor_struct", "conf_lr", "text_call", "text_call_user", "output_judge"))
                 + " |")
    L.append("\n### Table B. Paired, tool-resampled differences (95% percentile interval, draws pooled over 5 seeds)\n")
    L.append("| run | side | probe - conf (paper) | within-fold gap | fusion(probe,conf) - conf | probe - text(call+user) | probe - output judge | fusion(probe,output) - output | gap without schema echo |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, r in rc.items():
        f = fl.get(k)
        if f is None:
            continue
        J = f["judges"]
        L.append(f"| {k} | {r['side']} | {r['gap_pooled']:+.3f} [{r['gap_ci_paired_json'][0]:+.2f}, {r['gap_ci_paired_json'][1]:+.2f}] | "
                 f"{r['gap_fold_mean']:+.3f} | {ci(r['fusion_minus_conf'])} | {ci(J['text_call_user']['probe_minus'])} | "
                 f"{ci(J['output_judge']['probe_minus'])} | {ci(f['fusion_probe_output']['minus_output_judge'])} | "
                 f"{ci(se[k]['gap'])} |")
    L.append("\n### Table C. Within one failure mode (wrong argument values vs valid), stored scores\n")
    L.append("| run | side | n+ | probe | conf | gap |")
    L.append("|---|---|---|---|---|---|")
    for k, f in fl.items():
        w = f["within_wrong_arg_values"]
        L.append(f"| {k} | {f['side']} | {w['n_pos']} | {w['probe']['pooled_auc']:.3f} | {w['conf']['pooled_auc']:.3f} | {ci(w['gap'])} |")
    L.append("\n### Table D. Schema echo and failure-mode mix of the scored population\n")
    L.append("| run | side | echo share of failures | echo share of correct | regex AUC | modes |")
    L.append("|---|---|---|---|---|---|")
    for k, s in se.items():
        modes = fl.get(k, {}).get("mode_counts", {})
        L.append(f"| {k} | {s['side']} | {s['echo_rate_pos']:.2f} | {s['echo_rate_neg']:.2f} | {s['regex_auc']:.3f} | "
                 + ", ".join(f"{m} {n}" for m, n in modes.items()) + " |")
    L.append("\nReverse checks: " + ", ".join(
        f"{k}: floor max|diff| {f['floor_max_abs_diff_vs_stored']:.1e}, folds match {f['folds_match_stored']}"
        for k, f in fl.items()))
    rd = R / "reader_runs"
    if rd.exists():
        L.append("\n### Table E. Reader-model probe (Qwen3.5-0.8B reads request + call text, no schemas)\n")
        L.append("| run | side | reader | own probe | conf | probe - reader | reader - conf |")
        L.append("|---|---|---|---|---|---|---|")
        for k in rc:
            f = rd / f"{k}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            L.append(f"| {k} | {r['side']} | {r['reader_auc']:.3f} | {r['probe_auc']:.3f} | {r['conf_auc']:.3f} | "
                     f"{ci(r['probe_minus_reader'])} | {ci(r['reader_minus_conf'])} |")
    txt = "\n".join(L) + "\n"
    (R / "summary.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
