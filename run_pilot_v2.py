"""
v2 pilot — validity-fixed end-to-end pipeline on ONE small model.

Fixes applied relative to the v1 (workshop-paper) pipeline, per the
2026-09 audit:
  1. Greedy decoding (do_sample=False), everything seeded.
  2. Native tool calling: Glaive tool schemas are passed via
     tokenizer.apply_chat_template(tools=...) so the model calls tools in
     its OWN format; labels measure semantics, not Glaive-format
     familiarity.
  3. Format-normalized labeling with failure modes (no_call /
     unparseable_call / wrong_name / missing_args / wrong_arg_values),
     via spectral_guardrails.probes.labeling.classify_failure.
  4. Attention-ONLY spectral metrics on the symmetric normalized
     Laplacian (spectral_guardrails.spectral.metrics) — no hidden states
     touch the spectral features; eigenvalues are scale-normalized to
     kill the sequence-length confound.
  5. Token-role positions restricted to the generated span
     (find_token_positions_v2) — v1 read the system prompt's schema.
  6. Probe layers adaptive to model depth (np.linspace), not hardcoded.
  7. max_new_tokens=256 + truncation flag (v1 truncated at 64 and
     labeled truncations as hallucinations).
  8. No test-set selection anywhere: hyperparameters and directions are
     chosen on train/val only; results averaged over 5 split seeds.
  9. Confound controls: surface-feature baseline (lengths + truncation),
     logprob baseline, and a "semantic subset" evaluation that excludes
     format failures (no_call / unparseable_call).

Usage:
  python run_pilot_v2.py extract --model Qwen/Qwen3.5-2B --n 750
  python run_pilot_v2.py evaluate
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from spectral_guardrails.probes.labeling import (
    classify_failure, extract_calls, extract_glaive_tools,
)
from spectral_guardrails.probes.features import (
    extract_probe_features, find_token_positions_v2,
)
from spectral_guardrails.spectral.metrics import (
    METRIC_NAMES, PER_HEAD_METRICS, layer_spectral_metrics,
    laplacian_eig_profile, per_head_metrics, gram_spectrum_features,
    lapeigvals_diag_profile,
)
from spectral_guardrails.utils.data import load_glaive_data, parse_glaive_chat

OUT_DIR = Path("data/pilot_v2_qwen35_2b")   # overridden by --tag in main()
FEATURES = OUT_DIR / "features.jsonl"
SEED = 42
N_PROBE_LAYERS = 8
MAX_NEW_TOKENS = 256


# ══════════════════════════════════════════════════════════════════════════════
# Extraction
# ══════════════════════════════════════════════════════════════════════════════

def iter_call_examples(limit: int):
    """Yield (system, user_msg, ground_truth) for Glaive examples whose first
    assistant turn is a tool call."""
    pool = load_glaive_data(domain="general", limit=limit * 4, step=13)
    n = 0
    for ex in pool:
        messages = parse_glaive_chat(ex.get("chat", ""))
        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            if "<functioncall>" not in msg["content"]:
                break  # first assistant turn is prose -> not a call example
            if idx == 0 or messages[idx - 1]["role"] != "user":
                break
            gt_calls, _ = extract_calls(msg["content"])
            if gt_calls is None:
                break  # unusable ground truth
            yield ex.get("system", ""), messages[idx - 1]["content"], msg["content"]
            n += 1
            break
        if n >= limit:
            return


def handle_extract(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    if FEATURES.exists() and not args.fresh:
        with open(FEATURES, encoding="utf-8") as f:
            done = {json.loads(line)["prompt_hash"] for line in f if line.strip()}
        print(f"[resume] {len(done)} samples already extracted")
    elif args.fresh and FEATURES.exists():
        FEATURES.unlink()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    probe_layers = sorted(set(np.linspace(1, n_layers, N_PROBE_LAYERS).round().astype(int).tolist()))
    print(f"[extract] {args.model}: {n_layers} layers, probe layers {probe_layers}")

    from tqdm import tqdm
    kept, mode_counts = 0, {}
    pbar = tqdm(iter_call_examples(args.n), total=args.n, desc="extract")
    for system, user, gt in pbar:
        tools = extract_glaive_tools(system)
        if not tools:
            continue
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user}]
        try:
            prompt_text = tok.apply_chat_template(
                msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        except Exception:
            continue
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        if prompt_hash in done:
            kept += 1
            continue

        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        if prompt_len > 2048:
            continue

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tok.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
            )
        gen_ids = out.sequences[0][prompt_len:].tolist()
        # trim trailing pad/eos repeats but keep one eos
        pred = tok.decode(gen_ids, skip_special_tokens=True)
        truncated = tok.eos_token_id not in gen_ids
        if not pred.strip():
            continue

        # mean transition logprob (baseline signal)
        try:
            ts = model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True)
            mean_logprob = float(ts[0].float().mean())
        except Exception:
            mean_logprob = float("nan")

        try:
            label, mode = classify_failure(pred, gt)
        except ValueError:
            continue

        # ── teacher-forced pass: attentions + hidden states ───────────────────
        full_ids = out.sequences[:, : prompt_len + len(gen_ids)]
        with torch.no_grad():
            mo = model(input_ids=full_ids,
                       output_attentions=True, output_hidden_states=True)
        T = full_ids.shape[1]

        span = (prompt_len, T)
        layer_diagnostics, layer_diagnostics_span = [], []
        eig_profile, eig_profile_span, head_fiedler_span = [], [], []
        lapeig_diag = []
        for attn in mo.attentions:               # full-attention layers only
            a = attn[0]                          # [H, T, T]
            layer_diagnostics.append(layer_spectral_metrics(a))
            layer_diagnostics_span.append(layer_spectral_metrics(a, span=span))
            if args.rich:
                eig_profile.append(laplacian_eig_profile(a))
                eig_profile_span.append(laplacian_eig_profile(a, span=span))
                head_fiedler_span.append(per_head_metrics(a, span=span))
                lapeig_diag.append(lapeigvals_diag_profile(a))

        pos = find_token_positions_v2(tok, gen_ids, prompt_len)
        hidden, gram_feats = {}, {}
        for li in probe_layers:
            h = mo.hidden_states[li][0].to(torch.float32).cpu()
            hidden[str(li)] = extract_probe_features(h, pos).tolist()
            if args.rich:
                gram_feats[str(li)] = gram_spectrum_features(
                    mo.hidden_states[li][0][prompt_len:T])

        del mo, out
        torch.cuda.empty_cache()

        rec = {
            "prompt_hash": prompt_hash,
            "user": user,
            "label": int(label),
            "failure_mode": mode,
            "prediction": pred,
            "ground_truth": gt,
            "mean_logprob": mean_logprob,
            "prompt_tokens": int(prompt_len),
            "gen_tokens": len(gen_ids),
            "seq_len": int(T),
            "truncated": bool(truncated),
            "layer_diagnostics": layer_diagnostics,
            "layer_diagnostics_span": layer_diagnostics_span,
            "hidden": hidden,
            "probe_layers": probe_layers,
        }
        if args.rich:
            rec["eig_profile"] = eig_profile
            rec["eig_profile_span"] = eig_profile_span
            # [L][H][5] in PER_HEAD_METRICS order
            rec["head_metrics_span"] = head_fiedler_span
            # [L][H][100] official LapEigvals diagonal profile
            rec["lapeig_diag"] = lapeig_diag
            rec["gram_feats"] = gram_feats
        with open(FEATURES, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        kept += 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        pbar.set_postfix(kept=kept, mode=mode, halluc=f"{label}")
        if kept >= args.n:
            break

    print(f"\n[extract] DONE: {kept} samples -> {FEATURES}")
    print(f"[extract] failure modes this run: {mode_counts}")


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def group_split(samples, seed, train_ratio=0.7, val_ratio=0.15,
                key="prompt_hash"):
    groups = {}
    for i, s in enumerate(samples):
        groups.setdefault(s[key], []).append(i)
    pids = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    n = len(pids)
    a, b = int(train_ratio * n), int((train_ratio + val_ratio) * n)
    tr = [i for p in pids[:a] for i in groups[p]]
    va = [i for p in pids[a:b] for i in groups[p]]
    te = [i for p in pids[b:] for i in groups[p]]
    return np.array(tr), np.array(va), np.array(te)


def grouped_kfold(samples, seed, key="tool", n_folds=5):
    """
    Tool-level grouped K-fold: every sample of a given tool (GT function
    name) lands in exactly one fold, so detectors are always evaluated on
    UNSEEN tools. Kills both the task-identity confound (fn one-hot alone
    scored 0.80 AUC under prompt-hash splits, audit 2026-09) and Glaive's
    massive near-duplicate leakage (574/746 records share an exact GT
    string with another record).
    Yields (train_idx, val_idx, test_idx) per fold; val is a group-level
    20% carve-out of train.
    """
    groups = {}
    for i, s in enumerate(samples):
        groups.setdefault(s[key], []).append(i)
    pids = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    folds = np.array_split(np.arange(len(pids)), n_folds)
    for f in folds:
        test_p = {pids[j] for j in f}
        rest = [p for p in pids if p not in test_p]
        n_val = max(1, len(rest) // 5)
        val_p = set(rest[:n_val])
        tr = np.array([i for p in rest[n_val:] for i in groups[p]])
        va = np.array([i for p in sorted(val_p) for i in groups[p]])
        te = np.array([i for p in sorted(test_p) for i in groups[p]])
        yield tr, va, te


def lmm_matrix(samples, key):
    L = len(samples[0][key])
    M = len(METRIC_NAMES)
    X = np.zeros((len(samples), L * M), dtype=np.float32)
    for i, s in enumerate(samples):
        for li, diag in enumerate(s[key]):
            for mi, m in enumerate(METRIC_NAMES):
                X[i, li * M + mi] = float(diag.get(m, 0.0) or 0.0)
    return X


def sprich_matrix(samples, key):
    """Trajectory statistics over layers (per metric): mean, std, slope,
    delta, range, min-step, max-step — the SpRich family, on 6 layers."""
    N = len(samples)
    L = len(samples[0][key])
    M = len(METRIC_NAMES)
    T = np.zeros((N, L, M))
    for i, s in enumerate(samples):
        for li, diag in enumerate(s[key]):
            for mi, m in enumerate(METRIC_NAMES):
                T[i, li, mi] = float(diag.get(m, 0.0) or 0.0)
    feats = []
    x = np.arange(L, dtype=float)
    for mi in range(M):
        tr = T[:, :, mi]
        d = np.diff(tr, axis=1)
        feats += [tr.mean(1), tr.std(1),
                  np.array([np.polyfit(x, t, 1)[0] for t in tr]),
                  tr[:, -1] - tr[:, 0], tr.max(1) - tr.min(1),
                  d.min(1), d.max(1)]
    return np.nan_to_num(np.column_stack(feats).astype(np.float32))


def hidden_matrix(samples):
    keys = sorted(samples[0]["hidden"].keys(), key=int)
    return np.array([np.concatenate([s["hidden"][k] for k in keys])
                     for s in samples], dtype=np.float32)


def surface_matrix(samples):
    return np.array([[s["prompt_tokens"], s["gen_tokens"],
                      float(s["truncated"])] for s in samples],
                    dtype=np.float32)


def velocity_matrix(samples, key):
    """Layer-to-layer derivative ("spectral velocity") features of every
    metric trajectory: mean, std, min, max, |flux|, argmax position."""
    N = len(samples)
    L = len(samples[0][key])
    M = len(METRIC_NAMES)
    T = np.zeros((N, L, M))
    for i, s in enumerate(samples):
        for li, diag in enumerate(s[key]):
            for mi, m in enumerate(METRIC_NAMES):
                T[i, li, mi] = float(diag.get(m, 0.0) or 0.0)
    feats = []
    for mi in range(M):
        d = np.diff(T[:, :, mi], axis=1)
        feats += [d.mean(1), d.std(1), d.min(1), d.max(1),
                  np.abs(d).sum(1),
                  d.argmax(1).astype(float) / max(d.shape[1] - 1, 1)]
    return np.nan_to_num(np.column_stack(feats).astype(np.float32))


def lapeig_official_scores(lap, y, tr, va, te):
    """
    Official LapEigvals protocol: per-(layer, head) top-k Laplacian-diagonal
    values -> balanced logistic regression, k swept in {5,10,25,50,100} and
    chosen on the validation split.
    lap: (N, L, H, 100) stored diagonal profiles.
    """
    N = lap.shape[0]
    best_scores, best_val = None, -1.0
    for k in [5, 10, 25, 50, 100]:
        Xk = lap[:, :, :, :k].reshape(N, -1)
        p = fit_lr(Xk, y, tr, va)
        v = auc_safe(y[va], p.predict_proba(Xk[va])[:, 1])
        if not np.isnan(v) and v > best_val:
            best_val = v
            best_scores = p.predict_proba(Xk[te])[:, 1]
    return best_scores


def honest_sweep_scores(X, y, tr, va, te):
    """Best single column, direction and column chosen on train+val ONLY."""
    fit = np.concatenate([tr, va])
    best_j, best_sign, best_auc = 0, 1.0, -1.0
    for j in range(X.shape[1]):
        a = auc_safe(y[fit], X[fit, j])
        if np.isnan(a):
            continue
        s, a2 = (1.0, a) if a >= 0.5 else (-1.0, 1 - a)
        if a2 > best_auc:
            best_j, best_sign, best_auc = j, s, a2
    return best_sign * X[te, best_j]


def fit_lr(X, y, tr, va):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    best, best_v = None, -1
    for C in [0.01, 0.1, 1.0, 10.0]:
        p = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(C=C, max_iter=2000,
                                                class_weight="balanced"))])
        p.fit(X[tr], y[tr])
        try:
            v = roc_auc_score(y[va], p.predict_proba(X[va])[:, 1])
        except ValueError:
            v = 0.5
        if v > best_v:
            best, best_v = p, v
    return best


def fit_mlp(X, y, tr, va, seed):
    import torch as th
    from spectral_guardrails.probes.mlp import HallucinationProbe, train_probe
    from sklearn.utils.class_weight import compute_class_weight
    th.manual_seed(seed)
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[tr])
    probe = HallucinationProbe(input_dim=X.shape[1])
    probe, _ = train_probe(probe, X[tr], y[tr], X[va], y[va],
                           epochs=50, patience=5,
                           pos_weight=th.tensor(w[1] / w[0], dtype=th.float32))
    probe.eval()
    dev = next(probe.parameters()).device

    def score(Z):
        with th.no_grad():
            return th.sigmoid(
                probe(th.FloatTensor(Z).to(dev))).cpu().numpy().ravel()
    return score


def auc_safe(y, s):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def handle_evaluate(args):
    from spectral_guardrails.probes.gbt import train_lmm_gbt, predict_lmm
    from spectral_guardrails.probes.labeling import classify_failure, extract_calls

    with open(FEATURES, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # Re-label from stored text with the CURRENT labeler (labels are a
    # function of code version, not of the extraction run) and attach the
    # tool name for tool-level splitting.
    relabel_changes = 0
    for s in samples:
        old = s["label"]
        try:
            s["label"], s["failure_mode"] = classify_failure(
                s["prediction"], s["ground_truth"])
        except ValueError:
            pass
        gt_calls, _ = extract_calls(s["ground_truth"])
        s["tool"] = gt_calls[0]["name"] if gt_calls else "?"
        relabel_changes += (s["label"] != old)

    y = np.array([s["label"] for s in samples])
    modes = np.array([s["failure_mode"] for s in samples])
    tools = np.array([s["tool"] for s in samples])
    semantic = np.isin(modes, ["valid", "wrong_name", "missing_args",
                               "wrong_arg_values"])

    print(f"\nN={len(samples)}  hallucination rate={y.mean():.3f}"
          f"  (relabeling changed {relabel_changes} labels)")
    print("failure modes:", {m: int((modes == m).sum())
                             for m in sorted(set(modes))})
    print(f"unique tools: {len(set(tools))}   semantic subset: "
          f"{int(semantic.sum())} samples, halluc rate {y[semantic].mean():.3f}\n")

    uniq_tools = sorted(set(tools))
    X_fn = np.zeros((len(samples), len(uniq_tools)), dtype=np.float32)
    for i, t in enumerate(tools):
        X_fn[i, uniq_tools.index(t)] = 1.0

    # ── feature families ───────────────────────────────────────────────────────
    X_lmm_full = lmm_matrix(samples, "layer_diagnostics")
    X = {
        "Spectral per-layer (LMM)": X_lmm_full,
        "Spectral per-layer, gen span": lmm_matrix(samples, "layer_diagnostics_span"),
        "Spectral trajectory (SpRich)": sprich_matrix(samples, "layer_diagnostics"),
        "Spectral velocity": velocity_matrix(samples, "layer_diagnostics"),
        "Hidden token-role": hidden_matrix(samples),
        "Surface (lengths) [confound]": surface_matrix(samples),
        "Gen length only [confound]": np.array(
            [[s["gen_tokens"]] for s in samples], dtype=np.float32),
        "Tool one-hot [confound]": X_fn,
    }
    rich = "eig_profile" in samples[0]
    per_head_rows = []
    lap_official = None
    if rich:
        X["Sym-Laplacian eig profile"] = np.array(
            [np.ravel(s["eig_profile"]) for s in samples], dtype=np.float32)
        X["Sym-Laplacian eig profile, gen span"] = np.array(
            [np.ravel(s["eig_profile_span"]) for s in samples], dtype=np.float32)
        if "lapeig_diag" in samples[0]:
            # (N, L, H, 100) official LapEigvals diagonal profiles
            lap_official = np.array([s["lapeig_diag"] for s in samples],
                                    dtype=np.float32)

        if "head_metrics_span" in samples[0]:
            # (N, L, H, 5) in PER_HEAD_METRICS order
            hm = np.array([s["head_metrics_span"] for s in samples],
                          dtype=np.float32)
            N = len(samples)
            for mi, mname in enumerate(PER_HEAD_METRICS):
                row = f"Per-head {mname} (span)"
                X[row] = hm[:, :, :, mi].reshape(N, -1)
                per_head_rows.append(row)
            row = "Per-head all metrics (span)"
            X[row] = hm.reshape(N, -1)
            per_head_rows.append(row)
            per_head_combined = X["Per-head all metrics (span)"]
        else:  # legacy dumps: fiedler only
            row = "Per-head fiedler_value (span)"
            X[row] = np.array([np.ravel(s["head_fiedler_span"])
                               for s in samples], dtype=np.float32)
            per_head_rows.append(row)
            per_head_combined = X[row]

        gkeys = sorted(samples[0]["gram_feats"].keys(), key=int)
        X["Hidden Gram spectra (EigenScore)"] = np.array(
            [np.concatenate([s["gram_feats"][k] for k in gkeys])
             for s in samples], dtype=np.float32)
        X["All attention-spectral combined"] = np.hstack([
            X["Spectral per-layer (LMM)"], X["Spectral per-layer, gen span"],
            X["Spectral trajectory (SpRich)"], X["Spectral velocity"],
            X["Sym-Laplacian eig profile"],
            X["Sym-Laplacian eig profile, gen span"],
            per_head_combined])

    LMM_ROWS = ["Spectral per-layer (LMM)", "Spectral per-layer, gen span"]
    LR_ROWS = ["Spectral trajectory (SpRich)", "Spectral velocity",
               "Surface (lengths) [confound]", "Gen length only [confound]",
               "Tool one-hot [confound]"]
    if rich:
        LR_ROWS += (["Sym-Laplacian eig profile",
                     "Sym-Laplacian eig profile, gen span"]
                    + per_head_rows
                    + ["Hidden Gram spectra (EigenScore)",
                       "All attention-spectral combined"])
    ALL_ROWS = (LMM_ROWS + LR_ROWS +
                ["Hidden token-role [LR]", "Hidden token-role [MLP]",
                 "Best single spectral (honest sweep)", "Mean logprob"])
    if lap_official is not None:
        ALL_ROWS.append("LapEigvals (official code)")

    logprob = np.nan_to_num(
        np.array([s["mean_logprob"] for s in samples]), nan=0.0)

    seeds = [42, 43, 44, 45, 46]
    results = {}   # name -> {"all": [pooled aucs], "semantic": [pooled aucs]}

    def add(name, subset, val):
        results.setdefault(name, {"all": [], "semantic": []})[subset].append(val)

    for seed in seeds:
        # cross-fit: every sample scored exactly once by a model that never
        # saw its tool; ONE pooled AUC per seed over all N samples.
        pooled = {name: np.full(len(samples), np.nan) for name in ALL_ROWS}
        for tr, va, te in grouped_kfold(samples, seed, key="tool"):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            for name in LMM_ROWS:
                m, sc, _ = train_lmm_gbt(X[name], y, tr, va)
                pooled[name][te] = predict_lmm(m, sc, X[name][te])
            for name in LR_ROWS:
                p = fit_lr(X[name], y, tr, va)
                pooled[name][te] = p.predict_proba(X[name][te])[:, 1]
            p = fit_lr(X["Hidden token-role"], y, tr, va)
            pooled["Hidden token-role [LR]"][te] = \
                p.predict_proba(X["Hidden token-role"][te])[:, 1]
            mlp = fit_mlp(X["Hidden token-role"], y, tr, va, seed)
            pooled["Hidden token-role [MLP]"][te] = \
                mlp(X["Hidden token-role"][te])
            pooled["Best single spectral (honest sweep)"][te] = \
                honest_sweep_scores(X_lmm_full, y, tr, va, te)
            if lap_official is not None:
                s_lap = lapeig_official_scores(lap_official, y, tr, va, te)
                if s_lap is not None:
                    pooled["LapEigvals (official code)"][te] = s_lap
            sign = 1.0 if auc_safe(y[tr], -logprob[tr]) >= 0.5 else -1.0
            pooled["Mean logprob"][te] = sign * -logprob[te]

        for name, scores in pooled.items():
            ok = ~np.isnan(scores)
            add(name, "all", auc_safe(y[ok], scores[ok]))
            ok_sem = ok & semantic
            add(name, "semantic", auc_safe(y[ok_sem], scores[ok_sem]))

    # ── report ─────────────────────────────────────────────────────────────────
    n_pos_sem = int(y[semantic].sum())
    lines = ["", "=" * 92,
             f"Pooled cross-fit AUC over N={len(samples)} "
             f"({int(y.sum())} pos; semantic: {n_pos_sem} pos), "
             f"tool-level folds, {len(seeds)} seeds",
             "=" * 92,
             f"{'Detector':<38} {'AUC (all)':>18} {'AUC (semantic only)':>24}",
             "-" * 92]
    for name, r in results.items():
        def fmt(v):
            v = [x for x in v if not np.isnan(x)]
            if not v:
                return "n/a"
            return f"{np.mean(v):.3f} ± {np.std(v):.3f}"
        lines.append(f"{name:<38} {fmt(r['all']):>18} {fmt(r['semantic']):>24}")
    lines.append("=" * 92)
    report = "\n".join(lines)
    print(report)

    out = {
        "n": len(samples),
        "halluc_rate": float(y.mean()),
        "failure_modes": {m: int((modes == m).sum()) for m in sorted(set(modes))},
        "seeds": seeds,
        "results": {k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                    for k, v in results.items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(OUT_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nsaved -> {OUT_DIR / 'results.json'}")


def main():
    global OUT_DIR, FEATURES
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", default="Qwen/Qwen3.5-2B")
    e.add_argument("--n", type=int, default=750)
    e.add_argument("--fresh", action="store_true")
    e.add_argument("--rich", action="store_true", default=True,
                   help="also dump eigenvalue profiles, per-head Fiedler, "
                        "hidden-state Gram spectra")
    e.add_argument("--no-rich", dest="rich", action="store_false")
    e.add_argument("--tag", default=None,
                   help="output subdir tag (default: derived from model id)")
    v = sub.add_parser("evaluate")
    v.add_argument("--tag", default="qwen35_2b")
    args = ap.parse_args()
    if args.cmd == "extract" and args.tag is None:
        args.tag = (args.model.split("/")[-1].lower()
                    .replace("-instruct", "").replace(".", "").replace("-", "_"))
    OUT_DIR = Path(f"data/pilot_v2_{args.tag}")
    FEATURES = OUT_DIR / "features.jsonl"
    if args.cmd == "extract":
        handle_extract(args)
    else:
        handle_evaluate(args)


if __name__ == "__main__":
    main()
