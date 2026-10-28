"""
Budget plan (docs/REGISTRATION_BUDGET.md): shared run table and statistics.

Every statistic reuses the audit's code where one exists:
  G and G_wav      analysis/audit_forced_json.gap (paired tool bootstrap, 1000 draws/seed)
  labels/modes     analysis/audit_meta_extract.process (asserted aligned with scores.npz)
New here: the value-span confidence score (sign and summary fixed on training folds),
a generic paired gap for any two score vectors, a joint tool bootstrap across two runs
that answer the same items, and the exact two-sided Mann-Whitney test.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # "" does not hide the GPU on this machine
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from audit_floors import PROBE, CONF  # noqa: E402
from audit_forced_json import gap as audit_gap  # noqa: E402
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "budget_oct2026"
LOGS = DATA / "budget_logs"
MIN_WAV = 20
MIN_CLASS = 30
WAV = "wrong_arg_values"

# key, tag, checkpoint, family, bench, side, kind, v2 key, v2 tag
B1_RUNS = [
    ("LlamaOneBBfcl", "b1_llama1b_bfcl", "Llama-3.2-1B", "Llama-3.2", "BFCL", "probe", "native", "LlamaOneBBfcl", "v3_llama1b_bfcl"),
    ("LlamaOneBGlaive", "b1_llama1b_glaive", "Llama-3.2-1B", "Llama-3.2", "Glaive", "probe", "native", "LlamaOneBGlaive", "base_llama1b_glaive"),
    ("LlamaOneBLive", "b1_llama1b_live", "Llama-3.2-1B", "Llama-3.2", "BFCL-live", "probe", "native", "LlamaOneBLive", "llama1b_live"),
    ("LlamaThreeBBfcl", "b1_llama3b_bfcl", "Llama-3.2-3B", "Llama-3.2", "BFCL", "probe", "native", "LlamaThreeBBfcl", "base_llama3b_bfcl"),
    ("LlamaThreeBGlaive", "b1_llama3b_glaive", "Llama-3.2-3B", "Llama-3.2", "Glaive", "probe", "native", "LlamaThreeBGlaive", "base_llama3b_glaive"),
    ("GemmaBfcl", "b1_gemma3_1b_bfcl", "Gemma-3-1B", "Gemma-3", "BFCL", "probe", "fallback", "GemmaBfcl", "v3_gemma3_1b_bfcl"),
    ("GemmaGlaive", "b1_gemma3_1b_glaive", "Gemma-3-1B", "Gemma-3", "Glaive", "probe", "fallback", "GemmaGlaive", "base_gemma3_glaive"),
    ("QwenThreeBfcl", "b1_qwen3_17b_bfcl", "Qwen3-1.7B", "Qwen3", "BFCL", "confidence", "native", "QwenThreeBfcl", "base_qwen3_17b_bfcl"),
    ("QwenThreeGlaive", "b1_qwen3_17b_glaive", "Qwen3-1.7B", "Qwen3", "Glaive", "confidence", "native", "QwenThreeGlaive", "v3_qwen3_17b_glaive"),
    ("MiniCpmBfcl", "b1_minicpm5_bfcl", "MiniCPM5-2B", "MiniCPM5", "BFCL", "confidence", "native", "MiniCpmBfcl", "v3_minicpm5_2b_bfcl"),
    ("MiniCpmLive", "b1_minicpm5_live", "MiniCPM5-2B", "MiniCPM5", "BFCL-live", "confidence", "native", "MiniCpmLive", "v3_minicpm5_2b_live"),
    ("MiniCpmGlaive", "b1_minicpm5_glaive", "MiniCPM5-2B", "MiniCPM5", "Glaive", "confidence", "native", "MiniCpmGlaive", "v3_minicpm5_2b_glaive"),
    ("QwenThreeFiveBfcl", "b1_qwen35_08b_bfcl", "Qwen3.5-0.8B", "Qwen3.5", "BFCL", "confidence", "native", "QwenThreeFiveBfcl", "v3_qwen35_08b_bfcl"),
    ("MiniCpmBfclJson", "b1_minicpm5_bfcl_json", "MiniCPM5-2B", "MiniCPM5", "BFCL", "confidence", "json", "MiniCpmBfclJson", "v3_minicpm5_2b_bfcl_json"),
    ("QwenThreeFiveBfclJson", "b1_qwen35_08b_bfcl_json", "Qwen3.5-0.8B", "Qwen3.5", "BFCL", "confidence", "json", "QwenThreeFiveBfclJson", "v3_qwen35_08b_bfcl_json"),
    ("LlamaThreeBBfclFallback", "b1_llama3b_bfcl_fblist", "Llama-3.2-3B", "Llama-3.2", "BFCL", "probe", "control", None, None),
]
B1_OPTIONAL = [("GemmaBfclObject", "b1x_gemma3_1b_bfcl_object", "Gemma-3-1B", "Gemma-3", "BFCL", "probe",
                "attribution", "GemmaBfcl", "v3_gemma3_1b_bfcl")]
# v2 G_wav whose interval excluded zero (REGISTRATION_BUDGET.md, table) -> registered sign
U1_RUNS = {"LlamaOneBGlaive": +1, "LlamaOneBBfcl": +1, "LlamaThreeBGlaive": +1, "LlamaThreeBBfcl": +1,
           "GemmaGlaive": +1, "MiniCpmBfcl": -1, "QwenThreeFiveBfcl": -1, "QwenThreeFiveBfclJson": -1}

# B3: checkpoint, family, predicted side, candidate tags (first existing one is used)
B3_CKPTS = [
    ("Qwen3-4B", "Qwen3", "confidence", ["b3_qwen3_4b_bfcl"]),
    ("Qwen3.5-2B", "Qwen3.5", "confidence", ["b3_qwen35_2b_bfcl"]),
    ("Gemma-2-2B", "Gemma", "probe", ["b3_gemma2_2b_bfcl"]),
    ("Gemma-3-4B-or-270M", "Gemma-3", "probe", ["b3_gemma3_4b_bfcl", "b3_gemma3_270m_bfcl"]),
]


def marker(name: str, kind: str) -> Path:
    return LOGS / "markers" / f"{name}.{kind}"


def run_dir(tag: str) -> Path:
    return DATA / f"pilot_v2_{tag}"


def evaluated(tag: str) -> bool:
    return (run_dir(tag) / "scores.npz").exists() and (DATA / "audit" / f"meta_{tag}.jsonl").exists()


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


class Run:
    """One evaluated run: scores, labels, modes, value-span confidence, metadata."""

    def __init__(self, tag: str, need_value: bool = True):
        self.tag = tag
        d = run_dir(tag)
        self.z = np.load(d / "scores.npz")
        self.R = [json.loads(l) for l in open(DATA / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
        z = self.z
        self.y = z["y"].astype(int)
        sem, ec = z["semantic"].astype(bool), z["expect_call"].astype(bool)
        self.m = sem & ec if not ec.all() else sem
        self.modes = np.array([r["failure_mode"] for r in self.R])
        self.hash = np.array([r["prompt_hash"] for r in self.R])
        self.key = np.array([f"{r.get('user')}||{r['tool']}" for r in self.R])
        self.tools = z["tools"].astype(str)
        self.seeds = [int(s) for s in z["seeds"]]
        assert len(self.R) == len(self.y), (tag, len(self.R), len(self.y))
        mp = d / "run_meta.json"
        self.meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        self.vmean = self.vmin = None
        if need_value:
            self._load_value_conf(d / "features.jsonl")

    def _load_value_conf(self, path: Path):
        vc = {}
        dec = json.JSONDecoder()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                # parse only the two keys needed; full records carry large arrays
                i = line.find('"prompt_hash": ')
                h = dec.raw_decode(line, i + len('"prompt_hash": '))[0] if i >= 0 else None
                j = line.find('"value_conf": ')
                v = dec.raw_decode(line, j + len('"value_conf": '))[0] if j >= 0 else None
                if h is not None and h not in vc:
                    vc[h] = v
        get = lambda k: np.array([(vc.get(h) or {}).get(k, np.nan) if isinstance(vc.get(h), dict) else np.nan
                                  for h in self.hash], dtype=float)
        self.vmean, self.vmin = get("mean"), get("min")
        self.value_cover = float(np.isfinite(self.vmean[self.m]).mean()) if self.m.any() else float("nan")

    # ── populations ───────────────────────────────────────────────────────
    def counts(self):
        m = self.m
        return {"n_items": int(len(self.y)), "n_scored": int(m.sum()), "n_pos": int(self.y[m].sum()),
                "n_neg": int((1 - self.y[m]).sum()), "n_wav": int(((self.modes == WAV) & m).sum()),
                "modes_scored": {k: int(((self.modes == k) & m).sum()) for k in sorted(set(self.modes[m]))}}

    def powered(self):
        c = self.counts()
        return c["n_pos"] >= MIN_CLASS and c["n_neg"] >= MIN_CLASS

    def wav_mask(self):
        return self.m & np.isin(self.modes, ["valid", WAV])

    def wav_identified(self):
        return int(((self.modes == WAV) & self.m).sum()) >= MIN_WAV

    def missing_call_share(self):
        f = self.m & (self.y == 1)
        return float((self.modes[f] == "missing_calls").mean()) if f.any() else float("nan")

    # ── scores ────────────────────────────────────────────────────────────
    def score(self, name, seed):
        return self.z[f"score__{name}__{seed}"]

    def value_scores(self):
        """Out-of-fold value-span confidence per seed (registration: sign and summary
        (mean or min) fixed on the training-fold scored items, training-fold median imputation)."""
        tp = self.z["train_pop__"].astype(bool)
        out, chosen = {}, []
        for s in self.seeds:
            fold = self.z[f"fold__{s}"]
            sc = np.full(len(self.y), np.nan)
            for f in sorted(set(fold[fold >= 0])):
                tr = tp & (fold >= 0) & (fold != f)
                te = fold == f
                best = None
                for nm, v in (("mean", self.vmean), ("min", self.vmin)):
                    med = np.nanmedian(v[tr]) if np.isfinite(v[tr]).any() else 0.0
                    vv = np.where(np.isfinite(v), v, med)
                    a = auc(self.y[tr], vv[tr])
                    if np.isnan(a):
                        continue
                    sgn, a2 = (1.0, a) if a >= 0.5 else (-1.0, 1 - a)
                    if best is None or a2 > best[0]:
                        best = (a2, nm, sgn, vv)
                if best is None:
                    continue
                sc[te] = best[2] * best[3][te]
                chosen.append(best[1])
            out[s] = sc
        self.value_choice = {k: chosen.count(k) for k in ("mean", "min")}
        return out


def paired_gap(run: Run, A: dict, B: dict, mask, n_boot=1000, seed0=7000):
    """AUC(A) - AUC(B) on mask, per seed, tool-resampled paired bootstrap pooled over seeds."""
    y, tools = run.y, run.tools
    draws, deltas, aa, bb = [], [], [], []
    for si, s in enumerate(run.seeds):
        a, b = A[s], B[s]
        ok = mask & np.isfinite(a) & np.isfinite(b) & (run.z[f"fold__{s}"] >= 0)
        if len(np.unique(y[ok])) < 2 or y[ok].sum() < 5:
            return None
        r = paired_bootstrap_delta_auc(y[ok], a[ok], b[ok], n_boot=n_boot, seed=seed0 + si,
                                       return_draws=True, groups=tools[ok])
        deltas.append(r["delta"]); draws.append(r["draws"]); aa.append(r["auc_a"]); bb.append(r["auc_b"])
    d = np.concatenate(draws)
    return {"n_pos": int(y[mask].sum()), "n_neg": int((1 - y[mask]).sum()), "a": float(np.mean(aa)),
            "b": float(np.mean(bb)), "delta": float(np.mean(deltas)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}


def run_summary(run: Run, with_value=True):
    """Every per-run quantity the registration names."""
    c = run.counts()
    r = {"tag": run.tag, **c, "powered": run.powered(), "wav_identified": run.wav_identified(),
         "missing_call_share_of_failures": run.missing_call_share(),
         "git_commit": run.meta.get("git_commit"), "git_dirty": run.meta.get("git_dirty"),
         "code_pin": run.meta.get("code_pin"), "fallback_prompt": run.meta.get("fallback_prompt"),
         "force_json": run.meta.get("force_json"), "model": run.meta.get("model"),
         "model_revision": run.meta.get("model_revision")}
    r["G"] = audit_gap(run.z, run.y, run.m)
    r["G_wav"] = audit_gap(run.z, run.y, run.wav_mask()) if run.wav_identified() else None
    P = {s: run.score(PROBE, s) for s in run.seeds}
    C = {s: run.score(CONF, s) for s in run.seeds}
    if with_value and run.vmean is not None:
        V = run.value_scores()
        r["value_token_coverage_scored"] = run.value_cover
        r["value_summary_chosen_folds"] = run.value_choice
        r["G_val"] = paired_gap(run, P, V, run.m, seed0=7100)
        r["G_wav_val"] = paired_gap(run, P, V, run.wav_mask(), seed0=7200) if run.wav_identified() else None
        r["valuespan_minus_meanlp_wav"] = (paired_gap(run, V, C, run.wav_mask(), seed0=7300)
                                           if run.wav_identified() else None)
    return r


def joint_wav_diff(A: Run, B: Run, n_boot=1000, seed0=9000):
    """D = G_wav(A) - G_wav(B) for two runs over the same benchmark items; each draw resamples
    tools with replacement from the union of tools in either run's wav population."""
    mA, mB = A.wav_mask(), B.wav_mask()
    union = np.array(sorted(set(A.tools[mA]) | set(B.tools[mB])))
    pts, draws = [], []
    for si, s in enumerate(A.seeds):
        rng = np.random.RandomState(seed0 + si)
        sa = {k: A.score(k, s) for k in (PROBE, CONF)}
        sb = {k: B.score(k, s) for k in (PROBE, CONF)}
        okA = mA & np.isfinite(sa[PROBE]) & np.isfinite(sa[CONF]) & (A.z[f"fold__{s}"] >= 0)
        okB = mB & np.isfinite(sb[PROBE]) & np.isfinite(sb[CONF]) & (B.z[f"fold__{s}"] >= 0)
        idxA = {t: np.where(okA & (A.tools == t))[0] for t in union}
        idxB = {t: np.where(okB & (B.tools == t))[0] for t in union}

        def g(run, sc, idx):
            y = run.y[idx]
            if len(np.unique(y)) < 2:
                return np.nan
            return auc(y, sc[PROBE][idx]) - auc(y, sc[CONF][idx])
        allA = np.concatenate([idxA[t] for t in union]); allB = np.concatenate([idxB[t] for t in union])
        pts.append(g(A, sa, allA) - g(B, sb, allB))
        for _ in range(n_boot):
            pick = rng.choice(union, size=len(union), replace=True)
            ia = np.concatenate([idxA[t] for t in pick]); ib = np.concatenate([idxB[t] for t in pick])
            draws.append(g(A, sa, ia) - g(B, sb, ib))
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    return {"delta": float(np.nanmean(pts)), "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)), "n_draws": int(d.size)}


def exact_mw_two_sided(a, b):
    """Exact two-sided Mann-Whitney p over all C(n1+n2, n1) relabellings (statistic U of a)."""
    a, b = list(a), list(b)
    pooled = a + b
    n1 = len(a)

    def U(x, yv):
        return sum((xi > yi) + 0.5 * (xi == yi) for xi in x for yi in yv)
    u_obs = U(a, b)
    mu = n1 * len(b) / 2.0
    dev_obs = abs(u_obs - mu)
    cnt = tot = 0
    for idx in itertools.combinations(range(len(pooled)), n1):
        x = [pooled[i] for i in idx]
        yv = [pooled[i] for i in range(len(pooled)) if i not in idx]
        tot += 1
        cnt += abs(U(x, yv) - mu) >= dev_obs - 1e-12
    return {"U": float(u_obs), "p_two_sided": cnt / tot, "n_assignments": tot,
            "perfect_separation": bool(min(a) > max(b)) if a and b else False}


def fmt(g, k="delta"):
    if not g:
        return "n/a"
    return f"{g[k]:+.3f} [{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}]"


def write(name: str, obj: dict, md_lines: list[str]):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    (OUT / f"{name}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / name}.json and .md")
