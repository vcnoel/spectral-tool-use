"""Validation suite of the clean pipeline (docs/PIPELINE_REBUILD.md section 7).

  python -m rebuild.validate_clean render-check                 tokenizer-only route check per model
  python -m rebuild.validate_clean dirty-refusal                the extractor refuses a dirty tree / no pin / pin drift
  python -m rebuild.validate_clean label-parity --tag T         relabel from stored text; BFCL-port agreement
  python -m rebuild.validate_clean drift --a T1 --b T2          two extractions of the same items
  python -m rebuild.validate_clean roundtrip --tag T            storage write/read; loader on a real run
  python -m rebuild.validate_clean baseline-parity              LapEigvals / SinkProbe / Lookback / per-head fixtures
  python -m rebuild.validate_clean audit-sheet --tag T          the 60-item stratified hand-audit CSV (unfilled)
  python -m rebuild.validate_clean post --tag T                 label-parity + roundtrip + audit-sheet (queue CPU lane)
  python -m rebuild.validate_clean all --tag T [--b T2]

Every command writes results/rebuild/validation/<command>[_<tag>].json and prints a verdict.
Runs on CPU; the GPU is hidden.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402

from rebuild import storage, prompts, labels, bfcl_port, registry, benchmarks  # noqa: E402
from rebuild.labels import TYPES  # noqa: E402

OUT = ROOT / "results" / "rebuild" / "validation"
AUDIT_DIR = ROOT / "results" / "rebuild" / "hand_audit"
AUDIT_N = 60
AUDIT_SEED = 20261005


def write(name: str, obj: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    print(f"[validate] wrote {p.relative_to(ROOT)}  ->  {obj.get('verdict')}")
    return obj


# ── render check ─────────────────────────────────────────────────────────────
def render_check(models=None) -> dict:
    from transformers import AutoTokenizer
    items = benchmarks.get("bfcl").load(60)
    probe = [items[0], items[25], items[-1]]        # simple, multiple-ish, parallel-ish categories
    rows = {}
    for mid, fam, params, side, route_exp, slot, note in registry.MODELS:
        if models and mid not in models:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(mid)
        except Exception as e:
            rows[mid] = {"cached": False, "error": type(e).__name__, "expected_route": route_exp}
            continue
        dec = prompts.decide_route(tok, probe)
        sp = {}
        text, _ = prompts.render(tok, probe[0]["tools"], probe[0]["user"], dec["prompt_route"])
        if text:
            cs = prompts.prompt_char_spans(text, probe[0]["tools"], probe[0]["user"])
            offs = [tuple(o) for o in tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
            sp = {k: prompts.char_to_token_span(offs, v) for k, v in cs.items()}
        rows[mid] = {"cached": True, "route": dec["prompt_route"], "expected_route": route_exp,
                     "matches_registration": dec["prompt_route"] == route_exp,
                     "render_check": dec["render_check"], "spans_tok": sp,
                     "spans_found": {k: v is not None for k, v in sp.items()}, "bos": tok.bos_token, "eos": tok.eos_token}
    bad = [m for m, r in rows.items() if r.get("cached") and not r["matches_registration"]]
    missing = [m for m, r in rows.items() if not r.get("cached")]
    return write("render_check", {"rows": rows, "route_mismatch": bad, "not_cached": missing,
                                  "verdict": "PASS" if not bad else f"ROUTE MISMATCH on {bad}"})


# ── dirty-tree refusal ───────────────────────────────────────────────────────
def dirty_refusal() -> dict:
    from rebuild import extract_clean as ec
    res = {}
    # 1. real subprocess on the current tree: must refuse (exit 3) before touching a model
    probe = ROOT / "__dirty_probe__.tmp"
    probe.write_text("x", encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, "-m", "rebuild.extract_clean", "--model", "none", "--benchmark", "bfcl",
                            "--tag", "__refused__", "--pin", "0" * 40], capture_output=True, text=True, cwd=ROOT,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=300)
    finally:
        probe.unlink(missing_ok=True)
    res["dirty_tree"] = {"exit": r.returncode, "refused": r.returncode == ec.EXIT_REFUSED,
                         "stdout_tail": r.stdout.strip().splitlines()[-2:]}
    # 2. in-process with a simulated clean tree: no pin, and code differing from the pin
    real_git = ec.git

    def fake_git(*a):
        if a[:2] == ("status", "--porcelain"):
            return ""
        if a[0] == "diff":
            return "rebuild/extract_clean.py"
        return real_git(*a)
    ec.git = fake_git
    try:
        for label, pin in (("no_pin", None), ("pin_drift", "0" * 40)):
            try:
                ec.require_clean(pin, False)
                res[label] = {"refused": False}
            except SystemExit as e:
                res[label] = {"refused": e.code == ec.EXIT_REFUSED, "exit": e.code}
        st = ec.tree_state(None)
        res["simulated_clean_state"] = {"git_dirty": st["git_dirty"]}
    finally:
        ec.git = real_git
    ok = all(res[k]["refused"] for k in ("dirty_tree", "no_pin", "pin_drift"))
    return write("dirty_refusal", {**res, "verdict": "PASS" if ok else "FAIL"})


# ── label parity ─────────────────────────────────────────────────────────────
def label_parity(tag: str) -> dict:
    recs = storage.read_items(tag)
    meta = storage.read_meta(tag)
    adapter = benchmarks.get(meta["benchmark"])
    mism, type_mism, agree, dis = 0, 0, {"both_valid": 0, "both_invalid": 0, "port_valid_only": 0, "labeller_valid_only": 0}, []
    for r in recs:
        lab = labels.label_item(adapter, {"truth": r["truth"], "expect_call": r["expect_call"], "tools": r["tools"]},
                                r["prediction"], r["truncated"])
        mism += lab["label"] != r["label"]
        type_mism += lab["failure_type"] != r["failure_type"]
        if "bfcl_port_ok" in lab and r["label"] is not None:
            lv, pv = r["label"] == 0, lab["bfcl_port_ok"]
            key = ("both_valid" if lv and pv else "both_invalid" if not lv and not pv
                   else "port_valid_only" if pv else "labeller_valid_only")
            agree[key] += 1
            if key.endswith("only"):
                dis.append({"item_id": r["item_id"], "labeller_mode": r["failure_mode"], "port_reason": lab["bfcl_port_reason"],
                            "prediction": r["prediction"][:300]})
    n_off = sum(agree.values())
    by_item = {r["item_id"]: r for r in recs}
    s_mism, n_s = 0, 0
    for s_ in storage.read_samples(tag):
        r = by_item.get(s_["item_id"])
        if r is None or s_.get("label") is None:
            continue
        lab = labels.label_item(adapter, {"truth": r["truth"], "expect_call": r["expect_call"], "tools": r["tools"]},
                                s_["prediction"], s_.get("gen_tokens", 0) >= 256)
        s_mism += lab["label"] != s_["label"]
        n_s += 1
    out = {"tag": tag, "n": len(recs), "relabel_mismatch": mism, "failure_type_mismatch": type_mism,
           "n_samples": n_s, "sample_relabel_mismatch": s_mism,
           "official_evaluator_installed": bfcl_port.official_available(),
           "comparator": "official bfcl_eval" if bfcl_port.official_available() else
           "REIMPLEMENTATION of the BFCL AST checker rules (rebuild/bfcl_port.py); reference not obtainable offline",
           "port_self_test": bfcl_port.self_test(), "n_with_official_truth": n_off, "agreement": agree,
           "agreement_rate": (agree["both_valid"] + agree["both_invalid"]) / n_off if n_off else None,
           "disagreements": dis[:200]}
    out["verdict"] = ("PASS" if mism == 0 and s_mism == 0 and out["port_self_test"]["pass"] else "FAIL") + \
        (f"; port agreement {out['agreement_rate']:.3f} on {n_off}" if n_off else "; no official truth")
    return write(f"label_parity_{tag}", out)


# ── two-extraction drift ─────────────────────────────────────────────────────
def drift(a: str, b: str) -> dict:
    A = {r["item_id"]: r for r in storage.read_items(a)}
    B = {r["item_id"]: r for r in storage.read_items(b)}
    common = sorted(set(A) & set(B))
    same_text = same_label = same_type = 0
    dv, cos, dh = [], [], []
    ma, mb = storage.read_meta(a), storage.read_meta(b)
    mid = ma["probe_depths_registered"][len(ma["probe_depths_registered"]) // 2]
    for i in common:
        ra, rb = A[i], B[i]
        same_text += ra["prediction"] == rb["prediction"]
        same_label += ra["label"] == rb["label"]
        same_type += ra["failure_type"] == rb["failure_type"]
        va, vb = ra["confidence"].get("value_mean_logprob"), rb["confidence"].get("value_mean_logprob")
        if va is not None and vb is not None:
            dv.append(abs(va - vb))
        ta, tb = storage.read_tensor(a, i), storage.read_tensor(b, i)
        ha, hb = ta["hid"][mid - 1, 0].astype(np.float32), tb["hid"][mid - 1, 0].astype(np.float32)
        cos.append(float(ha @ hb / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-9)))
        if ta["hspec"].shape == tb["hspec"].shape:
            dh.append(float(np.nanmax(np.abs(ta["hspec"].astype(np.float32) - tb["hspec"].astype(np.float32)))))
    n = len(common)
    out = {"a": a, "b": b, "seed_a": ma["seed"], "seed_b": mb["seed"], "device_a": ma["device"], "device_b": mb["device"],
           "n_common": n, "same_generation": same_text / n, "same_label": same_label / n, "same_type": same_type / n,
           "value_logprob_abs_diff_mean": float(np.mean(dv)) if dv else None,
           "name_state_cosine_min": float(min(cos)) if cos else None,
           "per_head_spectra_max_abs_diff": float(max(dh)) if dh else None}
    out["verdict"] = f"{same_label / n:.3f} labels agree, {same_text / n:.3f} generations identical on {n} items"
    return write(f"drift_{a}__{b}", out)


# ── storage round trip ───────────────────────────────────────────────────────
def roundtrip(tag: str | None) -> dict:
    out = {}
    tmp_tag = "__roundtrip__"
    d = storage.run_dir(tmp_tag)
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    rng = np.random.default_rng(0)
    arrays = {"hid": rng.standard_normal((4, 3, 8)).astype(np.float32), "big": np.array([1e5, 1.0], np.float32),
              "ids": np.arange(5, dtype=np.int32), "roles": np.array(["name", "value", "close"])}
    arrays["hid"], d1 = storage.compact(arrays["hid"])
    arrays["big"], d2 = storage.compact(arrays["big"])
    storage.write_item(tmp_tag, "it/0", arrays, {"item_id": "it/0", "x": 1})
    back = storage.read_tensor(tmp_tag, "it/0")
    recs = storage.read_items(tmp_tag)
    out["synthetic"] = {"hid_dtype": d1, "big_dtype": d2,
                        "hid_max_abs_err": float(np.abs(back["hid"].astype(np.float32) - arrays["hid"].astype(np.float32)).max()),
                        "big_exact": bool((back["big"] == arrays["big"]).all()), "ids_exact": bool((back["ids"] == arrays["ids"]).all()),
                        "roles_exact": bool((back["roles"] == arrays["roles"]).all()), "record_back": recs == [{"item_id": "it/0", "x": 1}]}
    ok = (d1 == "float16" and d2 == "float32" and out["synthetic"]["big_exact"] and out["synthetic"]["ids_exact"]
          and out["synthetic"]["roles_exact"] and out["synthetic"]["record_back"] and out["synthetic"]["hid_max_abs_err"] < 1e-3)
    shutil.rmtree(d, ignore_errors=True)
    if tag:
        from rebuild.loader import CleanRun
        t0 = time.time()
        run = CleanRun(tag, allow_unclean=True)
        X = run.probe_matrix()
        H = run.per_head()
        Aa = run.anchored("value")
        out["real"] = {**run.summary(), "probe_matrix_shape": list(X.shape), "probe_finite": bool(np.isfinite(X).all()),
                       "per_head_shape": list(H.shape), "per_head_finite_share": float(np.isfinite(H).mean()),
                       "anchored_value_rows_shape": list(Aa.shape), "anchored_value_finite_share": float(np.isfinite(Aa).mean()),
                       "anchored_value_request_mass_mean": float(np.nanmean(Aa[..., 3])),
                       "anchored_value_schema_gold_mass_mean": float(np.nanmean(Aa[..., 1])),
                       "per_head_full_finite_share": float(np.isfinite(run.per_head_full()).mean()),
                       "attention_margin_finite_share": float(np.isfinite(run.attention_margin()).mean()),
                       "attn_depths": run.depths_attn.tolist(), "load_s": round(time.time() - t0, 2),
                       "refused_as_clean": None}
        try:
            CleanRun(tag)
            out["real"]["refused_as_clean"] = False
        except RuntimeError as e:
            out["real"]["refused_as_clean"] = True
            out["real"]["refusal"] = str(e)[:200]
        ok = ok and out["real"]["probe_finite"] and X.shape[0] == run.n
    return write(f"roundtrip_{tag or 'synthetic'}", {**out, "verdict": "PASS" if ok else "FAIL"})


# ── baseline parity on fixtures ──────────────────────────────────────────────
def baseline_parity() -> dict:
    import torch
    from spectral_guardrails.spectral.metrics import lapeigvals_diag_profile, sink_scores, lookback_ratio, per_head_metrics
    refs = {k: os.environ.get(k) for k in ("REF_LAPEIGVALS", "REF_LOOKBACK_LENS", "REF_SINKPROBE")}
    A = torch.tensor([[[1.0, 0.0, 0.0], [0.4, 0.6, 0.0], [0.2, 0.3, 0.5]]])     # one head, causal, rows sum to 1
    # LapEigvals: l_jj = (sum_i A[i,j])/(T-j) - A[j,j]  (official formula, vertical_edges=False), sorted desc
    lap = np.array(lapeigvals_diag_profile(A)[0][:3])
    lap_expected = np.sort(np.array([(1.6 / 3) - 1.0, (0.9 / 2) - 0.6, 0.5 - 0.5]))[::-1]
    # SinkProbe: s_j = mean_{i>=j} A[i,j], sorted desc; identity l_jj = s_j - a_jj
    s, pos = sink_scores(A)
    s3 = np.array(s[0][:3])
    s_expected = np.sort(np.array([1.6 / 3, 0.9 / 2, 0.5]))[::-1]
    identity = np.allclose(np.sort(np.array([1.6 / 3 - 1.0, 0.45 - 0.6, 0.0])), np.sort(lap), atol=1e-6)
    # Lookback Lens: context share of generated rows (prompt_len = 1): rows 1, 2 -> 0.4, 0.2 -> mean 0.3
    lb = lookback_ratio(A, 1, 3)[0]
    # per-head spectra against the legacy inline eig_feats of extract_perhead.py on a loop-free symmetric graph
    rng = np.random.default_rng(1)
    W = rng.random((2, 12, 12)).astype(np.float32)
    W = torch.tensor(W / W.sum(-1, keepdims=True))
    ph = np.asarray(per_head_metrics(W))
    agree = []
    for h in range(2):
        S = 0.5 * (W[h] + W[h].T).numpy().astype(np.float64)
        np.fill_diagonal(S, 0.0)
        n = S.shape[0]
        dh = 1.0 / np.sqrt(S.sum(1) + 1e-12)
        L = np.eye(n) - dh[:, None] * S * dh[None, :]
        lam = np.clip(np.sort(np.linalg.eigvalsh(L)), 0, None)
        p = lam / lam.sum()
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        agree.append({"fiedler_abs_diff": abs(float(lam[1]) - float(ph[h, 0])),
                      "entropy_abs_diff": abs(ent / np.log(n) - float(ph[h, 2])),
                      "lambda_max_abs_diff": abs(float(lam[-1]) - float(ph[h, 4]))})
    out = {
        "reference_code_available": {k: bool(v and Path(v).exists()) for k, v in refs.items()},
        "status": "REIMPLEMENTATIONS checked against hand-computed fixtures of the published formulas; the authors' "
                  "reference code (graphml-lab-pwr/lapeigvals, voidism/Lookback-Lens, SinkProbe arXiv:2604.10697) "
                  "was not obtainable offline. Set REF_LAPEIGVALS / REF_LOOKBACK_LENS / REF_SINKPROBE to local clones "
                  "to extend this check to their code.",
        "lapeigvals": {"got": lap.tolist(), "expected": lap_expected.tolist(), "pass": bool(np.allclose(lap, lap_expected, atol=1e-6))},
        "sinkprobe": {"got": s3.tolist(), "expected": s_expected.tolist(), "top_pos": pos[0],
                      "pass": bool(np.allclose(s3, s_expected, atol=1e-6)) and pos[0] == 0, "identity_lap_eq_sink_minus_self": identity},
        "lookback": {"got": lb, "expected": [0.3, 0.7], "pass": bool(np.allclose(lb, [0.3, 0.7], atol=1e-6))},
        "per_head_vs_legacy_eig_feats": {"heads": agree, "pass": all(v < 1e-4 for a in agree for v in a.values())},
    }
    ok = all(out[k]["pass"] for k in ("lapeigvals", "sinkprobe", "lookback", "per_head_vs_legacy_eig_feats")) and identity
    return write("baseline_parity", {**out, "verdict": "PASS (fixtures; reimplementation)" if ok else "FAIL"})


# ── hand-audit sheet ─────────────────────────────────────────────────────────
def audit_sheet(tag: str, n: int = AUDIT_N) -> dict:
    recs = storage.read_items(tag)
    rng = np.random.default_rng(AUDIT_SEED)
    by_type = {}
    for r in recs:
        by_type.setdefault(r["failure_type"], []).append(r)
    # at least min(5, available) per type, the rest proportional to type frequency
    alloc = {t: min(5, len(v)) for t, v in by_type.items()}
    rest = n - sum(alloc.values())
    total = sum(len(v) for v in by_type.values())
    for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        extra = min(len(v) - alloc[t], int(round(rest * len(v) / total)))
        alloc[t] += max(0, extra)
    while sum(alloc.values()) > n:
        t = max(alloc, key=lambda k: alloc[k])
        alloc[t] -= 1
    while sum(alloc.values()) < n and any(alloc[t] < len(by_type[t]) for t in alloc):
        t = max((t for t in alloc if alloc[t] < len(by_type[t])), key=lambda k: len(by_type[k]) - alloc[k])
        alloc[t] += 1
    chosen = []
    for t, k in alloc.items():
        idx = rng.choice(len(by_type[t]), size=k, replace=False)
        chosen += [by_type[t][i] for i in sorted(idx)]
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    p = AUDIT_DIR / f"{tag}.csv"
    cols = ["item_id", "category", "tool", "user_request", "generated_call", "ground_truth", "labeller_label",
            "labeller_failure_type", "bfcl_port_ok", "AUTHOR_label(0 valid/1 wrong)", "AUTHOR_failure_type", "AUTHOR_notes"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in chosen:
            gt = r["truth"].get("gt_anyof") if r["truth"]["kind"] == "anyof" else r["truth"].get("gt_text")
            w.writerow([r["item_id"], r["category"], r["tool"], r["user"], r["prediction"], json.dumps(gt) if gt is not None else "",
                        r["label"], r["failure_type"], r.get("bfcl_port_ok", ""), "", "", ""])
    out = {"tag": tag, "csv": str(p.relative_to(ROOT)), "n": len(chosen), "allocation": alloc, "seed": AUDIT_SEED,
           "types_available": {t: len(v) for t, v in by_type.items()},
           "verdict": f"sheet written, {len(chosen)} items, unfilled (the author labels; nothing was labelled by hand here)"}
    return write(f"audit_sheet_{tag}", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render-check", "dirty-refusal", "label-parity", "drift", "roundtrip",
                                    "baseline-parity", "audit-sheet", "post", "all"])
    ap.add_argument("--tag")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args()
    c = a.cmd
    if c == "render-check":
        render_check(a.models)
    elif c == "dirty-refusal":
        dirty_refusal()
    elif c == "label-parity":
        label_parity(a.tag)
    elif c == "drift":
        drift(a.a, a.b)
    elif c == "roundtrip":
        roundtrip(a.tag)
    elif c == "baseline-parity":
        baseline_parity()
    elif c == "audit-sheet":
        audit_sheet(a.tag)
    elif c == "post":
        label_parity(a.tag); roundtrip(a.tag); audit_sheet(a.tag)
    elif c == "all":
        render_check(a.models); dirty_refusal(); baseline_parity()
        if a.tag:
            label_parity(a.tag); roundtrip(a.tag); audit_sheet(a.tag)
        if a.tag and a.b:
            drift(a.tag, a.b)


if __name__ == "__main__":
    main()
