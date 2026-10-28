"""
Budget plan B5 (docs/REGISTRATION_BUDGET.md): ablation and steering of the probe's value-error
direction against 2 x 100 matched random directions, teacher-forced on stored generations.

  run     (GPU) one model: direction, nulls, interventions -> data/budget_b5/<tag>.npz + .json
  decide  (CPU) both models: registered rule -> results/budget_oct2026/B5.json and B5.md

Usage:
  python analysis/budget_b5_mechanism.py run --tag b1_llama1b_bfcl --model meta-llama/Llama-3.2-1B-Instruct
  CUDA_VISIBLE_DEVICES=-1 python analysis/budget_b5_mechanism.py decide
Smoke test (CPU, a few items, few directions):
  EXTRACT_DEVICE=cpu python analysis/budget_b5_mechanism.py run --tag ... --model ... --max-items 6 --n-rand 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
os.chdir(ROOT)

import numpy as np  # noqa: E402

N_VALID = 200
N_RAND = 100
ALPHAS = (-2.0, -1.0, 1.0, 2.0)
TOP_PC = 256
OUTD = ROOT / "data" / "budget_b5"
MODELS = {"probe": ("b1_llama1b_bfcl", "meta-llama/Llama-3.2-1B-Instruct"),
          "confidence": ("b1_qwen3_17b_bfcl", "Qwen/Qwen3-1.7B")}


# ── data ─────────────────────────────────────────────────────────────────────
def load_items(tag, max_items=None):
    import budget_common as bc
    run = bc.Run(tag, need_value=False)
    meta = run.meta
    keep = run.m & np.isin(run.modes, ["valid", bc.WAV])
    want = set(run.hash[keep])
    recs = {}
    with open(bc.run_dir(tag) / "features.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            h = r["prompt_hash"]
            if h in want and h not in recs and r.get("value_pos") and r.get("gen_ids") is not None:
                recs[h] = {"hidden": {k: np.asarray(v, dtype=np.float32) for k, v in r["hidden"].items()},
                           "gen_ids": r["gen_ids"], "value_pos": r["value_pos"],
                           "prompt_tokens": r["prompt_tokens"], "value_conf": r.get("value_conf"),
                           "tool": r["tool"], "user": r["user"], "category": r["category"]}
    idx_wav = [i for i in np.where(keep & (run.modes == bc.WAV))[0] if run.hash[i] in recs]
    idx_val = [i for i in np.where(keep & (run.modes == "valid"))[0] if run.hash[i] in recs]
    rng = np.random.RandomState(0)
    if len(idx_val) > N_VALID:
        idx_val = sorted(rng.choice(idx_val, N_VALID, replace=False).tolist())
    idx = sorted(idx_wav + idx_val)
    if max_items:
        idx = sorted(idx_wav[: max_items // 2] + idx_val[: max_items - max_items // 2])
    items = []
    for i in idx:
        r = recs[run.hash[i]]
        r.update({"y": int(run.modes[i] == bc.WAV), "hash": run.hash[i]})
        items.append(r)
    return items, meta


def halves(items, seed=42):
    tools = sorted({it["tool"] for it in items})
    rng = np.random.RandomState(seed)
    rng.shuffle(tools)
    first = set(tools[: len(tools) // 2])
    return np.array([0 if it["tool"] in first else 1 for it in items])


def fit_directions(items, h):
    """Layer L and per-half unit direction w (raw residual space, + = error)."""
    from run_pilot_v2 import fit_lr, auc_safe
    layers = sorted(items[0]["hidden"], key=int)
    d = len(items[0]["hidden"][layers[0]]) // 3
    y = np.array([it["y"] for it in items])
    tools = np.array([it["tool"] for it in items])
    scores, fits = {}, {}
    for li in layers:
        X = np.stack([it["hidden"][li][d:2 * d] for it in items])   # argument-value role
        tot = 0.0
        for hv in (0, 1):
            tr_all = np.where(h == hv)[0]
            ut = sorted(set(tools[tr_all]))
            rng = np.random.RandomState(1000 + hv)
            rng.shuffle(ut)
            val_t = set(ut[: max(1, len(ut) // 5)])
            va = np.array([i for i in tr_all if tools[i] in val_t])
            tr = np.array([i for i in tr_all if tools[i] not in val_t])
            if len(np.unique(y[tr])) < 2 or len(va) == 0 or len(np.unique(y[va])) < 2:
                tot += 0.5
                fits[(li, hv)] = None
                continue
            p = fit_lr(X, y, tr, va)
            v = auc_safe(y[va], p.predict_proba(X[va])[:, 1])
            tot += 0.5 if np.isnan(v) else v
            fits[(li, hv)] = p
        scores[li] = tot
    L = max(scores, key=lambda k: scores[k])
    X = np.stack([it["hidden"][L][d:2 * d] for it in items])
    W = []
    for hv in (0, 1):
        p = fits[(L, hv)]
        if p is None:
            raise SystemExit(f"cannot fit a direction on half {hv} at layer {L}")
        w = p.named_steps["lr"].coef_[0] / p.named_steps["sc"].scale_
        W.append(w / np.linalg.norm(w))
    return int(L), np.stack(W), X, {str(k): float(v) for k, v in scores.items()}


def random_dirs(X, W, n, seed=0):
    rng = np.random.RandomState(seed)
    d = X.shape[1]

    def ok(r):
        return all(abs(float(r @ w)) < 0.1 for w in W)
    iso = []
    while len(iso) < n:
        r = rng.standard_normal(d)
        r /= np.linalg.norm(r)
        if ok(r):
            iso.append(r)
    Xc = X - X.mean(0)
    k = min(TOP_PC, Xc.shape[0] - 1)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    sd = S[:k] / np.sqrt(max(1, Xc.shape[0] - 1))
    cov, tries = [], 0
    while len(cov) < n:
        tries += 1
        r = (rng.standard_normal(k) * sd) @ Vt[:k]
        r /= np.linalg.norm(r)
        if ok(r):
            cov.append(r)
        if tries > 200 * n:
            raise SystemExit("could not draw covariance-matched directions with |cos| < 0.1")
    return np.stack(iso), np.stack(cov), int(k)


# ── model passes ─────────────────────────────────────────────────────────────
class Hook:
    def __init__(self):
        self.mode, self.M = None, None

    def __call__(self, module, args, output):
        if self.mode is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        M = self.M.to(hf.device)                         # (K, d)
        if self.mode == "abl":
            proj = (hf * M[:, None, :]).sum(-1, keepdim=True)
            hf = hf - proj * M[:, None, :]
        else:
            hf = hf + M[:, None, :]
        h2 = hf.to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2


def render_prompts(meta, tok, hashes):
    import hashlib
    import run_pilot_v2 as rp
    rp.FORCE_JSON = bool(meta.get("force_json"))
    rp.FALLBACK_KIND = meta.get("fallback_prompt") or "object"
    rp.THINKING_MODE = False
    bench, n = meta["benchmark"], int(meta["n_requested"])
    src = rp.iter_bfcl_examples(n) if bench == "bfcl" else rp.iter_bfcl_examples(n, mix=rp.BFCL_LIVE_MIX)
    out = {}
    for ex in src:
        t = rp.render_tool_prompt(tok, ex["tools"], ex["user"])
        if t is None:
            continue
        hh = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if hh in hashes:
            out[hh] = t
    return out


def cmd_run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    items, meta = load_items(a.tag, a.max_items)
    h = halves(items)
    L, W, X, layer_scores = fit_directions(items, h)
    n_rand = a.n_rand
    R_iso, R_cov, k = random_dirs(X, W, n_rand)
    yv = np.array([it["y"] for it in items])
    s_half = [float(np.std(X[yv == 0] @ W[hv])) for hv in (0, 1)]
    print(f"[b5] {a.tag}: {len(items)} items ({int(yv.sum())} value errors), layer {L}, "
          f"layer scores {layer_scores}, s {s_half}", flush=True)

    dev = os.environ.get("EXTRACT_DEVICE", "cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev,
                                                 attn_implementation="eager", low_cpu_mem_usage=True).eval()
    n_layers = getattr(model.config, "num_hidden_layers", None) or model.config.get_text_config().num_hidden_layers
    base = model.model if hasattr(model, "model") else model.base_model
    layers = base.layers if hasattr(base, "layers") else base.language_model.layers
    norm = base.norm if hasattr(base, "norm") else base.language_model.norm
    target = norm if L == n_layers else layers[L - 1]      # hidden_states[L]
    hook = Hook()
    handle = target.register_forward_hook(hook)
    prompts = render_prompts(meta, tok, {it["hash"] for it in items})
    missing = [it["hash"] for it in items if it["hash"] not in prompts]
    if missing:
        print(f"[b5] WARNING {len(missing)} prompts not re-rendered; dropped", flush=True)
    half_of = {it["hash"]: int(hv) for it, hv in zip(items, h)}   # keep the fitted split
    items = [it for it in items if it["hash"] in prompts]
    h = np.array([half_of[it["hash"]] for it in items])
    yv = np.array([it["y"] for it in items])

    # direction sets per half: index 0 = w; 1..n = iso; n+1..2n = cov  (ablation, unit)
    dirs = [np.concatenate([W[hv][None], R_iso, R_cov]) for hv in (0, 1)]
    K = a.batch
    n_abl = 1 + 2 * n_rand
    n_steer = len(ALPHAS) * (1 + n_rand)
    lp_clean = np.full(len(items), np.nan)
    lp_abl = np.full((len(items), n_abl), np.nan)
    lp_steer = np.full((len(items), len(ALPHAS), 1 + n_rand), np.nan)
    full_check = []
    use_cache = True

    def suffix_lp(prefix_cache, suffix, tgt_pos, tgt_ids, mode, M):
        """Mean log-prob of tgt_ids at tgt_pos (indices into the suffix logits), per batch row."""
        Kb = M.shape[0] if M is not None else 1
        hook.mode, hook.M = mode, (torch.tensor(M, dtype=torch.float32) if M is not None else None)
        with torch.no_grad():
            if use_cache:
                out = model(input_ids=suffix.repeat(Kb, 1), past_key_values=prefix_cache, use_cache=True)
            else:
                raise RuntimeError("no-cache path is handled by the caller")
        hook.mode = None
        lg = out.logits[:, tgt_pos, :].float().log_softmax(-1)
        tg = torch.tensor(tgt_ids, device=lg.device)
        return lg.gather(-1, tg[None, :, None].expand(lg.shape[0], -1, 1))[..., 0].mean(-1).cpu().numpy()

    for ii, it in enumerate(items):
        ids = tok(prompts[it["hash"]], add_special_tokens=False).input_ids
        assert len(ids) == it["prompt_tokens"], (len(ids), it["prompt_tokens"])
        full = ids + list(it["gen_ids"])
        P = len(ids)
        vpos = [P + v for v in it["value_pos"] if 0 < P + v < len(full)]
        if not vpos:
            continue
        p0, last = vpos[0] - 1, vpos[-1]
        prefix = torch.tensor([full[:p0]], device=model.device)
        suffix = torch.tensor([full[p0:last]], device=model.device)    # positions p0 .. last-1
        tgt_pos = [v - 1 - p0 for v in vpos]
        tgt_ids = [full[v] for v in vpos]
        hv = int(h[ii])
        with torch.no_grad():
            pre = model(input_ids=prefix, use_cache=True)
        cache1 = pre.past_key_values
        # clean, batch 1
        lp_clean[ii] = suffix_lp(cache1, suffix, tgt_pos, tgt_ids, None, None)[0]
        cache1.crop(p0)
        if ii < 3:   # reverse check: cached suffix == one full pass
            with torch.no_grad():
                fo = model(input_ids=torch.tensor([full[:last]], device=model.device))
            lg = fo.logits[0, [v - 1 for v in vpos], :].float().log_softmax(-1)
            ref = float(lg.gather(-1, torch.tensor(tgt_ids, device=lg.device)[:, None])[:, 0].mean())
            full_check.append(abs(ref - lp_clean[ii]))
        cacheK = pre.past_key_values
        cacheK.batch_repeat_interleave(K)
        D = dirs[hv]
        jobs = [("abl", j, D[j]) for j in range(n_abl)]
        sw = s_half[hv]
        for ai, al in enumerate(ALPHAS):
            jobs.append(("add", (ai, 0), al * sw * W[hv]))
            for j in range(n_rand):
                jobs.append(("add", (ai, 1 + j), al * sw * R_cov[j]))
        for mode in ("abl", "add"):
            js = [j for j in jobs if j[0] == mode]
            for b in range(0, len(js), K):
                chunk = js[b: b + K]
                M = np.stack([c[2] for c in chunk])
                if len(chunk) < K:
                    M = np.concatenate([M, np.zeros((K - len(chunk), M.shape[1]))])
                vals = suffix_lp(cacheK, suffix, tgt_pos, tgt_ids, mode, M)
                cacheK.crop(p0)
                for c, v in zip(chunk, vals):
                    if mode == "abl":
                        lp_abl[ii, c[1]] = v
                    else:
                        lp_steer[ii, c[1][0], c[1][1]] = v
        del pre, cache1, cacheK
        if dev == "cuda":
            torch.cuda.empty_cache()
        if ii % 20 == 0:
            print(f"[b5] item {ii + 1}/{len(items)} elapsed {time.time() - t0:.0f}s", flush=True)
    handle.remove()
    stored = np.array([(it.get("value_conf") or {}).get("mean", np.nan) for it in items], dtype=float)
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTD / f"{a.tag}.npz", y=yv, half=h, lp_clean=lp_clean, lp_abl=lp_abl,
                        lp_steer=lp_steer, stored_value_mean=stored, alphas=np.array(ALPHAS),
                        W=W, R_iso=R_iso, R_cov=R_cov)
    info = {"tag": a.tag, "model": a.model, "layer": L, "n_layers": int(n_layers), "layer_scores": layer_scores,
            "n_items": len(items), "n_value_errors": int(yv.sum()), "n_rand": n_rand, "top_pc": k,
            "steer_sd": s_half, "max_cos_w_random": float(max(np.abs(np.concatenate([R_iso, R_cov]) @ W.T).max(), 0)),
            "cached_vs_full_max_abs_diff": float(max(full_check)) if full_check else None,
            "n_dropped_unrendered": len(missing), "wall_s": time.time() - t0,
            "smoke": bool(a.max_items)}
    (OUTD / f"{a.tag}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


# ── decision ─────────────────────────────────────────────────────────────────
def analyse(tag):
    from sklearn.metrics import roc_auc_score
    z = np.load(OUTD / f"{tag}.npz")
    info = json.loads((OUTD / f"{tag}.json").read_text(encoding="utf-8"))
    y = z["y"]
    n = int(info["n_rand"])
    auc = lambda lp: float(roc_auc_score(y, -lp))
    a0 = auc(z["lp_clean"])
    S = np.array([auc(z["lp_abl"][:, j]) - a0 for j in range(z["lp_abl"].shape[1])])
    Sw, Siso, Scov = S[0], S[1:1 + n], S[1 + n:1 + 2 * n]

    def cls(x, null):
        lo, hi = np.percentile(null, 2.5), np.percentile(null, 97.5)
        return {"band": [float(lo), float(hi)], "class": "below" if x < lo else ("above" if x > hi else "inside"),
                "p_below_one_sided": float((np.sum(null <= x) + 1) / (len(null) + 1))}
    ok = np.isfinite(z["stored_value_mean"]) & np.isfinite(z["lp_clean"])
    r = float(np.corrcoef(z["stored_value_mean"][ok], z["lp_clean"][ok])[0, 1]) if ok.sum() > 2 else float("nan")
    # steering slope of the items' mean value log-prob against alpha (alpha = 0 is the clean pass)
    al = np.concatenate([[0.0], z["alphas"]])

    def slope(col):
        m = np.concatenate([[z["lp_clean"].mean()], [z["lp_steer"][:, i, col].mean() for i in range(len(z["alphas"]))]])
        return float(np.polyfit(al, m, 1)[0])
    sw = slope(0)
    sr = np.array([slope(1 + j) for j in range(n)])
    return {"tag": tag, "info": info, "auc_clean": a0, "S_w": float(Sw), "iso": cls(Sw, Siso), "cov": cls(Sw, Scov),
            "sanity_pearson_vs_stored": r, "n_value_errors": int(y.sum()),
            "mean_dlp_w": float(np.mean(z["lp_abl"][:, 0] - z["lp_clean"])),
            "steer_slope_w": sw, "steer": cls(sw, sr)}


def cmd_decide(a):
    import budget_common as bc
    res = {"registration": "docs/REGISTRATION_BUDGET.md B5"}
    for side, (tag, _) in MODELS.items():
        if not (OUTD / f"{tag}.npz").exists():
            res[side] = None
            continue
        res[side] = analyse(tag)
    C, P = res.get("confidence"), res.get("probe")
    reasons = []
    if not C or not P:
        verdict = "NOT RUN"
        reasons.append("a model's run is missing")
    else:
        if (C["iso"]["class"] == "below") != (C["cov"]["class"] == "below"):
            reasons.append("nulls disagree on confidence")
        if P["iso"]["class"] != P["cov"]["class"]:
            reasons.append("nulls disagree on probe")
        for nm, m in (("confidence", C), ("probe", P)):
            if m["n_value_errors"] < 20:
                reasons.append(f"{nm}: fewer than 20 value errors")
            if not (m["sanity_pearson_vs_stored"] >= 0.98):
                reasons.append(f"{nm}: sanity r = {m['sanity_pearson_vs_stored']:.3f} < 0.98")
        if C["auc_clean"] < 0.55:
            reasons.append("confidence-side clean AUC below 0.55")
        c_reads = C["iso"]["class"] == "below" and C["cov"]["class"] == "below"
        c_no = C["iso"]["class"] != "below" and C["cov"]["class"] != "below"
        p_silent = P["iso"]["class"] == "inside" and P["cov"]["class"] == "inside"
        p_reads = P["iso"]["class"] == "below" and P["cov"]["class"] == "below"
        if reasons:
            verdict = "NOT IDENTIFIED"
        elif c_reads and p_silent:
            verdict = "HOLDS"
        elif c_no:
            verdict = "FAILS"
        elif c_reads and p_reads:
            verdict = "BOTH READ"
        else:
            verdict = "NOT IDENTIFIED"
            reasons.append("pattern outside the registered classes")
        steer_consistent = C["steer"]["class"] == "below" and P["steer"]["class"] == "inside"
        res["steering_consistent_secondary"] = bool(steer_consistent)
    res["verdict"] = verdict
    res["reasons"] = reasons
    L = ["# B5: probe-direction ablation and steering vs matched random directions (registered rule)", "",
         f"**Verdict: {verdict}**" + (f" ({'; '.join(reasons)})" if reasons else ""), ""]
    for side in ("confidence", "probe"):
        m = res.get(side)
        if not m:
            L.append(f"- {side}: not run")
            continue
        L.append(f"- {side} ({m['tag']}, layer {m['info']['layer']}, {m['info']['n_items']} items, "
                 f"{m['n_value_errors']} value errors): clean AUC {m['auc_clean']:.3f}, S(w) {m['S_w']:+.4f}; "
                 f"isotropic band [{m['iso']['band'][0]:+.4f}, {m['iso']['band'][1]:+.4f}] -> {m['iso']['class']}; "
                 f"covariance band [{m['cov']['band'][0]:+.4f}, {m['cov']['band'][1]:+.4f}] -> {m['cov']['class']}; "
                 f"steering slope {m['steer_slope_w']:+.4f} ({m['steer']['class']} the band, secondary); "
                 f"sanity r {m['sanity_pearson_vs_stored']:.3f}")
    bc.write("B5", res, L)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--n-rand", type=int, default=N_RAND)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--max-items", type=int, default=None)
    sub.add_parser("decide")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_decide(a)


if __name__ == "__main__":
    main()
