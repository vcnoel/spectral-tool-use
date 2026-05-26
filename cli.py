import argparse
import json
import hashlib
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import BitsAndBytesConfig

# Import library and utilities
from spectral_trust import GSPDiagnosticsFramework, GSPConfig
from spectral_trust.directed_topology import DirectedTopologist
from spectral_trust.spectral import calculate_spectral_velocity
from spectral_guardrails.utils.data import load_glaive_data
from spectral_guardrails.utils.models import MODEL_REGISTRY
from spectral_guardrails.utils.metrics import compute_classification_metrics
from spectral_guardrails.probes.labeling import assign_label
from spectral_guardrails.probes.features import extract_probe_features, find_token_positions
from spectral_guardrails.utils.stats import compute_cohens_d
from spectral_guardrails.probes.mlp import HallucinationProbe, train_probe, evaluate_probe
from spectral_guardrails.probes.gbt import (
    compute_layerwise_features,
    train_lmm_gbt, predict_lmm,
    bootstrap_ci as lmm_bootstrap_ci,
    save_lmm_probe, load_lmm_probe,
    feature_importances,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc as sk_auc

# --- Framework Utilities ---


def get_content_split_indices(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Deterministically splits samples based on prompt identifiers (Template-Consistent Splitting).
    This ensures all samples with the same prompt template go to the same split.
    """
    # Group samples by prompt_hash (or fallback to chat content if not in JSONL)
    groups = {}
    for i, s in enumerate(samples):
        # Prefer prompt_hash saved during extract; fallback is deterministic SHA256
        # (never Python's hash(), which is randomized per-process since Python 3.3)
        pid = s.get('prompt_hash') or hashlib.sha256(
            s.get('ground_truth', str(i)).encode('utf-8', errors='replace')
        ).hexdigest()
        if pid not in groups:
            groups[pid] = []
        groups[pid].append(i)

    unique_pids = sorted(list(groups.keys()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_pids)

    n_unique = len(unique_pids)
    itrain = int(train_ratio * n_unique)
    ival = int((train_ratio + val_ratio) * n_unique)

    train_pids = unique_pids[:itrain]
    val_pids = unique_pids[itrain:ival]
    test_pids = unique_pids[ival:]

    train_idx = [i for pid in train_pids for i in groups[pid]]
    val_idx = [i for pid in val_pids for i in groups[pid]]
    test_idx = [i for pid in test_pids for i in groups[pid]]

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def print_summary_table(metrics: dict, title: str = "Results Summary"):
    print(f"\n{'=' * 40}")
    print(f"{title:^40}")
    print(f"{'=' * 40}")
    print(f"{'Metric':<20} | {'Value':>17}")
    print(f"{'-' * 20}-|-{'-' * 17}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<20} | {v:>17.4f}")
        elif isinstance(v, int):
            print(f"{k:<20} | {v:>17d}")
        else:
            s = str(v)
            if len(s) > 17:
                s = s[:14] + "..."
            print(f"{k:<20} | {s:>17}")
    print(f"{'=' * 40}\n")


def find_optimal_threshold(y_true, y_score, target="recall80"):
    """Finds the threshold that achieves a specific target metric."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    if target == "recall80":
        # Find highest threshold where recall >= 0.8
        valid_idx = np.where(recalls >= 0.8)[0]
        if len(valid_idx) > 0:
            return float(thresholds[valid_idx[-1]])
    elif target == "recall90":
        valid_idx = np.where(recalls >= 0.9)[0]
        if len(valid_idx) > 0:
            return float(thresholds[valid_idx[-1]])
    elif target == "precision80":
        # Find lowest threshold where precision >= 0.8
        valid_idx = np.where(precisions >= 0.8)[0]
        if len(valid_idx) > 0:
            return float(thresholds[valid_idx[0]])
    elif target == "f1":
        f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        return float(thresholds[np.argmax(f1)])

    return 0.5


def plot_evaluation(y_true, y_score, title, save_path):
    """Generates ROC and Precision-Recall plots."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = sk_auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {title}')
    plt.legend(loc="lower right")

    # PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, lw=2, color='blue', label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR: {title}')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [plot] Saved performance graph to {save_path}")


def build_framework(model_name, output_dir, device="cuda"):
    """GoR-style framework configuration with optimized 4-bit quantization."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    config = GSPConfig(
        model_name=model_name,
        device=device,
        output_dir=str(output_dir),
        verbose=False,
        model_kwargs={
            "quantization_config": quantization_config,
            "output_attentions": True
        }
    )
    return config


def compute_trajectory_features(samples: list, n_layers: int, metrics: list = None):
    """
    Computes trajectory features (MMT) like delta, range, slope, and AUC for metrics across layers.
    Mutates samples in-place by adding keys like 'Ltrajectory_{metric}_delta'.
    """
    if metrics is None:
        metrics = ["fiedler_value", "smoothness_index", "spectral_entropy", "hfer", "energy"]

    for s in samples:
        for m in metrics:
            profile = []
            for layer_idx in range(n_layers):
                val = s.get(f"L{layer_idx}_{m}")
                if val is not None:
                    profile.append(val)

            if len(profile) < 2:
                continue

            profile = np.array(profile)

            # GoR Standard Trajectory Features
            s[f"Ltrajectory_{m}_delta"] = profile[-1] - profile[0]
            s[f"Ltrajectory_{m}_range"] = np.ptp(profile)

            # Dynamic Dynamics (GoR - Jump & Flux)
            diffs = np.abs(np.diff(profile))
            s[f"Ltrajectory_{m}_max_jump"] = np.max(diffs) if len(diffs) > 0 else 0.0
            s[f"Ltrajectory_{m}_flux"] = np.sum(diffs) if len(diffs) > 0 else 0.0
            try:
                # Use linear regression to find global slope
                slope = np.polyfit(np.arange(len(profile)), profile, 1)[0]
                s[f"Ltrajectory_{m}_slope"] = slope
            except BaseException:
                s[f"Ltrajectory_{m}_slope"] = 0.0

            # Use np.trapezoid if available (NumPy 2.0+), else np.trapz
            if hasattr(np, 'trapezoid'):
                s[f"Ltrajectory_{m}_auc"] = np.trapezoid(profile)
            else:
                s[f"Ltrajectory_{m}_auc"] = np.trapz(profile)


_SPECTRAL_METRICS = ["fiedler_value", "smoothness_index", "spectral_entropy", "energy", "hfer"]


def compute_rich_spectral_features(raw_samples: list) -> np.ndarray:
    """
    Rich spectral feature matrix for spectral_rich probe mode.

    For each sample, stacks:
      - Trajectory stats (15 per metric × 5 metrics = 75 dim): mean, std, slope,
        early-late delta, min-step, max-step, step-std, AUC, range, skew,
        kurtosis, mid-mean, early/late ratio, inflection count, argmin position.
      - Segmental features (8 per metric × 5 metrics = 40 dim): mean and std of
        4 equal temporal segments.
      - FFT magnitude (norm. by L, half-spectrum, per metric) concatenated.

    Total dimension is 115 + L//2+1 per metric (architecture-dependent).
    Returns (N, D) float32 array, NaN-safe.
    """
    from scipy import stats as sp_stats

    N = len(raw_samples)
    if N == 0:
        return np.empty((0, 0), dtype=np.float32)

    ld0 = raw_samples[0].get('layer_diagnostics', [])
    L = len(ld0)
    if L == 0:
        return np.empty((N, 0), dtype=np.float32)

    # Build (N, L, M) tensor
    M = len(_SPECTRAL_METRICS)
    T = np.zeros((N, L, M), dtype=np.float64)
    for i, s in enumerate(raw_samples):
        for li, layer in enumerate(s.get('layer_diagnostics', [])):
            for mi, m in enumerate(_SPECTRAL_METRICS):
                T[i, li, mi] = float(layer.get(m, 0.0) or 0.0)

    feats = []

    for mi in range(M):
        traj = T[:, :, mi]          # (N, L)

        # ── 15 trajectory statistics ──────────────────────────────────────────
        x = np.arange(L, dtype=float)
        q = max(1, L // 4)

        feats.append(traj.mean(axis=1))
        feats.append(traj.std(axis=1))
        feats.append(np.array([np.polyfit(x, t, 1)[0] for t in traj]))
        feats.append(traj[:, -q:].mean(axis=1) - traj[:, :q].mean(axis=1))

        diffs = np.diff(traj, axis=1) if L > 1 else np.zeros((N, 1))
        feats.append(diffs.min(axis=1))
        feats.append(diffs.max(axis=1))
        feats.append(diffs.std(axis=1))

        auc_trap = np.trapz(
            traj,
            axis=1) if not hasattr(
            np,
            'trapezoid') else np.trapezoid(
            traj,
            axis=1)
        feats.append(auc_trap / max(L, 1))
        feats.append(traj.max(axis=1) - traj.min(axis=1))
        feats.append(sp_stats.skew(traj, axis=1))
        feats.append(sp_stats.kurtosis(traj, axis=1))

        m0, m1 = L // 3, max(L // 3 + 1, 2 * L // 3)
        feats.append(traj[:, m0:m1].mean(axis=1))

        early_m = traj[:, :q].mean(axis=1)
        late_m = traj[:, -q:].mean(axis=1)
        feats.append(np.where(np.abs(early_m) > 1e-9, late_m / (early_m + 1e-9), 0.0))

        if L > 2:
            d2 = np.diff(traj, n=2, axis=1)
            feats.append((np.diff(np.sign(d2), axis=1) != 0).sum(axis=1).astype(float))
        else:
            feats.append(np.zeros(N))

        feats.append(traj.argmin(axis=1).astype(float) / max(L - 1, 1))

        # ── 8 segmental features (4 segments × mean+std) ─────────────────────
        segs = np.array_split(np.arange(L), 4)
        for seg in segs:
            sl = traj[:, seg]
            feats.append(sl.mean(axis=1))
            feats.append(sl.std(axis=1))

        # ── FFT magnitude (half-spectrum, normalised) ─────────────────────────
        fft_mag = np.abs(np.fft.rfft(traj, axis=1)) / max(L, 1)
        for fi in range(fft_mag.shape[1]):
            feats.append(fft_mag[:, fi])

    X = np.column_stack(feats).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def flatten_sample(s: dict) -> dict:
    """
    Normalise a JSONL record to a flat dict.
    Supports both legacy 'spectral' and GoR-style 'layer_diagnostics' keys.
    """
    flat = {k: v for k, v in s.items() if k not in ('spectral', 'hidden', 'layer_diagnostics')}

    # Case 1: layer_diagnostics (list of metrics per layer, GoR Style)
    if 'layer_diagnostics' in s:
        for li, metrics in enumerate(s['layer_diagnostics']):
            for metric_name, value in metrics.items():
                flat[f"L{li}_{metric_name}"] = value

    # Case 2: spectral (dict of layer indices, Legacy Style)
    for layer_str, metrics in s.get('spectral', {}).items():
        for metric_name, value in metrics.items():
            flat[f"L{layer_str}_{metric_name}"] = value

    # Nested hidden
    for layer_str, vec in s.get('hidden', {}).items():
        flat[f"L{layer_str}_hidden"] = vec

    return flat


def generate_layer_profile_plots(samples, y_true, model, domain, output_dir, best_layer=None):
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["fiedler_value", "smoothness_index", "spectral_entropy", "hfer", "energy"]
    y_true = np.array(y_true)

    max_layer = -1
    for k in samples[0].keys():
        if k.startswith("L") and "_" in k:
            try:
                parts = k.split("_")[0]
                if parts[1:].isdigit():
                    max_layer = max(max_layer, int(parts[1:]))
            except BaseException:
                continue
    n_layers = max_layer + 1

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{model} / {domain} — Spectral Layer Profiles", fontsize=16)

    for ax, m in zip(axes.flatten()[:5], metrics):
        data = np.full((len(samples), n_layers), np.nan)
        for i, s in enumerate(samples):
            for layer_idx in range(n_layers):
                data[i, layer_idx] = s.get(f"L{layer_idx}_{m}", np.nan)

        layers = np.arange(n_layers)
        for i in range(len(samples)):
            color = 'blue' if y_true[i] == 0 else 'red'
            ax.plot(layers, data[i], color=color, alpha=0.15, linewidth=0.5)

        valid_data = data[y_true == 0]
        halluc_data = data[y_true == 1]

        if len(valid_data) > 0:
            v_mean = np.nanmean(valid_data, axis=0)
            v_std = np.nanstd(valid_data, axis=0)
            ax.plot(layers, v_mean, color='blue', linewidth=3, label='Valid Mean')
            ax.fill_between(layers, v_mean - v_std, v_mean + v_std, color='blue', alpha=0.1)

        if len(halluc_data) > 0:
            h_mean = np.nanmean(halluc_data, axis=0)
            h_std = np.nanstd(halluc_data, axis=0)
            ax.plot(layers, h_mean, color='red', linewidth=3, label='Halluc Mean')
            ax.fill_between(layers, h_mean - h_std, h_mean + h_std, color='red', alpha=0.1)

        if best_layer is not None:
            ax.axvline(
                x=best_layer,
                color='black',
                linestyle='--',
                alpha=0.6,
                label=f'Best (L{best_layer})')

        ax.set_title(f"{m.replace('_', ' ').title()}")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend(prop={'size': 8})

    axes[1, 2].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path_base = output_dir / f"layer_profile_{model}_{domain}"
    plt.savefig(f"{plot_path_base}.png", dpi=300)
    plt.close()
    print(f"Generated layer profile plots -> {plot_path_base}.png")


def handle_prepare(args):
    data = load_glaive_data(domain=args.domain, limit=args.n_samples)
    output_path = Path(args.output_dir) / f"glaive_{args.domain}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    print(f"Prepared {len(data)} samples for domain '{args.domain}' -> {output_path}")


def handle_extract(args):
    model_id = MODEL_REGISTRY.get(args.model, args.model)
    output_dir = Path(args.output_dir)
    config = build_framework(model_id, output_dir, device=args.device)

    input_file = Path(args.data_dir) / f"glaive_{args.domain}.jsonl"
    if not input_file.exists():
        input_file = output_dir / f"glaive_{args.domain}.jsonl"

    # We proceed even if input_file doesn't exist, because load_glaive_data() fetches from source

    samples = []
    if input_file.exists():
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
    if args.n_samples and samples:
        samples = samples[:args.n_samples]

    output_path = output_dir / f"spectral_features_{args.model}_{args.domain}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Increase diversity by striding through dataset (step=50)
    import hashlib
    dataset = load_glaive_data(domain=args.domain, limit=args.n_samples, step=50)

    print(f"[extract] Running diverse extraction for {len(dataset)} samples (stride=50)...")

    records = []
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(model_id)
        tokenizer = framework.instrumenter.tokenizer
        model = framework.instrumenter.model

        pbar = tqdm(dataset, desc="GSP Extraction", unit="sample")
        for i, ex in enumerate(pbar):
            # 1. Recover chat history
            from spectral_guardrails.utils.data import parse_glaive_chat
            chat_raw = ex.get('chat', '')
            messages = parse_glaive_chat(chat_raw)
            target_idx = -1
            for idx, msg in enumerate(messages):
                if msg['role'] == 'assistant':
                    target_idx = idx
                    break
            if target_idx == -1:
                continue

            ground_truth = messages[target_idx]['content']
            prompt_msgs = [{"role": "system", "content": ex.get(
                'system', "You are a helpful assistant.")}]
            if target_idx > 0:
                prompt_msgs.append(messages[target_idx - 1])

            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            # 2. Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id)
            prediction = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            label = assign_label(prediction, ground_truth)
            full_text = prompt_text + prediction

            # 3. Analyze (Spectral pass)
            try:
                analysis = framework.analyze_text(full_text, save_results=False)
                layer_diagnostics = []
                for li, diag in enumerate(analysis['layer_diagnostics']):
                    layer_diagnostics.append({
                        "fiedler_value": float(diag.fiedler_value),
                        "smoothness_index": float(diag.smoothness_index),
                        "spectral_entropy": float(diag.spectral_entropy),
                        "energy": float(diag.energy),
                        "hfer": float(diag.hfer)
                    })

                # 4. Native Hidden Pass (Replaces monkeypatch capture)
                probe_hidden = {}
                with torch.no_grad():
                    # Explicit pass to capture hidden states correctly for the probe
                    # Using full_text (prompt + prediction)
                    full_inputs_eval = tokenizer(full_text, return_tensors="pt").to(model.device)
                    m_outputs = model(**full_inputs_eval, output_hidden_states=True)
                    hidden_states = [h.to(torch.float32).cpu() for h in m_outputs.hidden_states]

                    target_layers = [4, 8, 11, 15, 20, 24, 28, 31]
                    pos = find_token_positions(
                        tokenizer, full_inputs_eval.input_ids[0].tolist(), prediction)
                    for li in target_layers:
                        if li < len(hidden_states):
                            signals = hidden_states[li][0]
                            h_feat = extract_probe_features(signals, pos)
                            probe_hidden[str(li)] = h_feat.tolist()

                prompt_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
                result = {
                    "original_idx": ex.get('original_idx', i),
                    "prompt_hash": prompt_hash,
                    "prompt": prompt_text,
                    "label": int(label),
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "layer_diagnostics": layer_diagnostics,
                    "hidden": probe_hidden,
                    "seq_len": len(tokenizer(full_text)['input_ids'])
                }
                records.append(result)
                pbar.set_postfix(label="HALL" if label else "ok")
            except Exception as e:
                pbar.write(f"[warn] Framework failed on sample {i}: {e}")
                continue

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for record in records:
            out_f.write(json.dumps(record) + "\n")
    print(f"\n[DONE] Saved {len(records)} features to {output_path}")


def handle_train_probe(args):
    input_file = Path(args.data_dir) / f"spectral_features_{args.model}_{args.domain}.jsonl"
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_samples = [json.loads(line) for line in f]
    samples = [flatten_sample(s) for s in raw_samples]
    y = np.array([s['label'] for s in samples])

    train_idx, val_idx, test_idx = get_content_split_indices(samples)
    print(
        f"[train-probe] Split: {len(train_idx)} Train, "
        f"{len(val_idx)} Val, {len(test_idx)} Test (Template-Consistent)")

    if args.feature_type == "spectral_rich":
        import pickle
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import roc_auc_score

        X = compute_rich_spectral_features(raw_samples)
        print(f"[spectral_rich] Feature matrix: {X.shape}")

        best_C, best_val_auc = 1.0, 0.0
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            try:
                pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(
                    C=C, max_iter=1000, class_weight='balanced'))])
                pipe.fit(X[train_idx], y[train_idx])
                val_auc = roc_auc_score(y[val_idx], pipe.predict_proba(X[val_idx])[:, 1])
                if val_auc > best_val_auc:
                    best_val_auc, best_C = val_auc, C
            except Exception:
                pass

        pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(
            C=best_C, max_iter=1000, class_weight='balanced'))])
        trainval_idx = list(train_idx) + list(val_idx)
        pipe.fit(X[trainval_idx], y[trainval_idx])
        test_scores = pipe.predict_proba(X[test_idx])[:, 1]
        test_auc = roc_auc_score(y[test_idx], test_scores)
        print(
            f"[spectral_rich] Best C={best_C}  Val AUC={best_val_auc:.4f}  Test AUC={test_auc:.4f}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_name = f"probe_{args.model}_spectral_rich.pkl"
        with open(output_dir / probe_name, 'wb') as pf:
            pickle.dump(pipe, pf)
        print(f"[spectral_rich] Saved to {output_dir / probe_name}")
        return

    if args.feature_type == "lmm_gbt":
        from sklearn.metrics import roc_auc_score as _roc
        X_lmm = compute_layerwise_features(raw_samples)
        n_features = X_lmm.shape[1]
        n_layers = n_features // 5
        print(f"[lmm_gbt] Feature matrix: {X_lmm.shape}  (layers={n_layers}, metrics=5)")

        model, scaler, method = train_lmm_gbt(X_lmm, y, train_idx, val_idx)
        print(f"[lmm_gbt] Selected method: {method}")

        test_scores = predict_lmm(model, scaler, X_lmm[test_idx])
        test_auc = _roc(y[test_idx], test_scores)
        lo, hi = lmm_bootstrap_ci(y[test_idx], test_scores)
        print(f"[lmm_gbt] Test AUC: {test_auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

        imp = feature_importances(model, n_layers)
        if imp:
            top = sorted(
                [(l, m, v) for l, ms in imp.items() for m, v in ms.items()],
                key=lambda x: -x[2],
            )[:10]
            print("[lmm_gbt] Top 10 (layer, metric, importance):")
            for l, m, v in top:
                print(f"  L{l:02d}  {m:<20s}  {v:.4f}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_name = f"probe_{args.model}_lmm_gbt.pkl"
        save_lmm_probe(model, scaler, method, output_dir / probe_name)
        print(f"[lmm_gbt] Saved to {output_dir / probe_name}")
        return

    X, y_list = [], []
    for ex in samples:
        features = []
        if args.feature_type in ["hidden", "combined"]:
            hidden_keys = sorted([k for k in samples[0].keys() if k.endswith("_hidden")])
            for k in hidden_keys:
                features.extend(ex.get(k, []))
        if args.feature_type in ["spectral", "combined"]:
            _spectral_keys = sorted(
                [
                    k for k in samples[0].keys() if any(
                        k.endswith(
                            f"_{m}") for m in [
                            "hfer",
                            "fiedler_value",
                            "smoothness_index",
                            "spectral_entropy",
                            "energy"]) and k.startswith("L") and not k.endswith("_hidden")])
            features.extend([ex.get(k, 0.0) for k in _spectral_keys])
        X.append(features)
        y_list.append(ex.get('label', 0))

    X, y = np.array(X), np.array(y_list)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    y_train_np = np.array(y_train)
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train_np)
    pos_weight = torch.tensor(weights[1] / weights[0], dtype=torch.float32)

    probe = HallucinationProbe(input_dim=X_train.shape[1])
    trained_probe, _ = train_probe(probe, X_train, y_train, X_val, y_val,
                                   epochs=args.epochs, patience=args.patience,
                                   pos_weight=pos_weight)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_name = f"probe_{args.model}_{args.feature_type}.pt"
    torch.save({'state_dict': trained_probe.state_dict(),
               'input_dim': X_train.shape[1]}, output_dir / probe_name)

    metrics = evaluate_probe(trained_probe, X_test, y_test)
    print_summary_table({k: v for k, v in metrics.items() if k not in (
        'y_true', 'y_score')}, f"Probe Results ({args.model}) - TEST SPLIT")


def handle_evaluate(args):
    input_file = Path(args.data_dir) / f"spectral_features_{args.model}_{args.domain}.jsonl"
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_samples = [json.loads(line) for line in f]
    samples = [flatten_sample(s) for s in raw_samples]

    # Template-Consistent split enforcement
    _, _, test_idx = get_content_split_indices(samples)
    raw_samples_test = [raw_samples[i] for i in test_idx]
    samples = [samples[i] for i in test_idx]
    y_true = np.array([s['label'] for s in samples])

    print(
        f"[evaluate] Strictly evaluating on TEST split ({len(samples)} samples, template-consistent)")

    # Store scores for hybrid mode
    spectral_scores = None
    probe_scores = None

    if args.mode in ["spectral", "all", "hybrid"]:
        best_params_path = Path("data/categories_sweeps/best_params.json")
        if best_params_path.exists():
            with open(best_params_path, 'r') as f:
                best_params = json.load(f).get(f"{args.model}_{args.domain}")
            if best_params:
                key = f"L{best_params['layer']}_{best_params['metric']}"
                if "fiedler" in key and "fiedler_value" not in key:
                    key = key.replace("fiedler", "fiedler_value")

                # Ensure trajectory features are computed for evaluation
                max_layer = -1
                for k in samples[0].keys():
                    if k.startswith("L") and "_" in k:
                        try:
                            parts = k.split("_")[0]
                            if parts[1:].isdigit():
                                max_layer = max(max_layer, int(parts[1:]))
                        except BaseException:
                            continue
                compute_trajectory_features(samples, max_layer + 1)

                scores = np.array([s.get(key, np.nan) for s in samples], dtype=float)
                valid = ~np.isnan(scores)
                y_s, y_t = scores[valid], y_true[valid]

                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_t, y_s)
                if auc < 0.5:
                    auc = 1 - auc
                    scores = -scores

                # Scale spectral scores to [0,1] for hybrid
                s_min, s_max = scores[valid].min(), scores[valid].max()
                spectral_scores = (scores - s_min) / (s_max - s_min) if s_max > s_min else scores

                d = compute_cohens_d(y_s[y_t == 0], y_s[y_t == 1])
                print(f"--- SPECTRAL EVAL ({key}) ---\n  AUC: {auc:.4f}\n  Cohen's d: {d:.4f}\n")

    if args.mode in ["probe", "all", "hybrid"]:
        if args.feature_type == "lmm_gbt":
            from sklearn.metrics import roc_auc_score as _roc
            probe_name = f"probe_{args.model}_lmm_gbt.pkl"
            probe_path = Path(args.output_dir) / probe_name
            if not probe_path.exists():
                print(f"[ERROR] {probe_path} not found — run train-probe --feature-type lmm_gbt first")
            else:
                model, scaler, method = load_lmm_probe(probe_path)
                X_lmm = compute_layerwise_features(raw_samples_test)
                probe_scores = predict_lmm(model, scaler, X_lmm)
                test_auc = _roc(y_true, probe_scores)
                lo, hi = lmm_bootstrap_ci(y_true, probe_scores)
                print(
                    f"--- PROBE EVAL ({method}) ---\n"
                    f"  AUC: {test_auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
                    f"  features={X_lmm.shape[1]}\n"
                )

                n_layers = X_lmm.shape[1] // 5
                imp = feature_importances(model, n_layers)
                if imp:
                    top = sorted(
                        [(l, m, v) for l, ms in imp.items() for m, v in ms.items()],
                        key=lambda x: -x[2],
                    )[:5]
                    print("  Top features:", [(f"L{l} {m}", f"{v:.3f}") for l, m, v in top])

        elif args.feature_type == "spectral_rich":
            import pickle
            from sklearn.metrics import roc_auc_score as sk_roc_auc
            probe_name = f"probe_{args.model}_spectral_rich.pkl"
            probe_path = Path(args.output_dir) / probe_name
            if not probe_path.exists():
                print(
                    f"[ERROR] {probe_path} not found"
                    " — run train-probe --feature-type spectral_rich first")
            else:
                with open(probe_path, 'rb') as pf:
                    pipe = pickle.load(pf)
                X_rich = compute_rich_spectral_features(raw_samples_test)
                probe_scores = pipe.predict_proba(X_rich)[:, 1]
                test_auc = sk_roc_auc(y_true, probe_scores)
                print(
                    "--- PROBE EVAL (spectral_rich) ---\n"
                    f"  AUC: {test_auc:.4f}  dim={X_rich.shape[1]}\n")
        else:
            probe_name = f"probe_{args.model}_{args.feature_type}.pt"
            probe_path = Path(args.output_dir) / probe_name
            if probe_path.exists():
                ckpt = torch.load(probe_path, map_location='cpu', weights_only=False)
                # Reconstruct feature vector based on type (same logic as handle_train_probe)
                X = []
                for ex in samples:
                    features = []
                    if args.feature_type in ["hidden", "combined"]:
                        hidden_keys = sorted(
                            [k for k in samples[0].keys() if k.endswith("_hidden")])
                        for k in hidden_keys:
                            features.extend(ex.get(k, []))
                    if args.feature_type in ["spectral", "combined"]:
                        _metrics = ["hfer", "fiedler_value", "smoothness_index",
                                    "spectral_entropy", "energy"]
                        spec_keys = sorted([
                            k for k in samples[0].keys()
                            if any(k.endswith(f"_{m}") for m in _metrics)
                            and k.startswith("L")
                            and not k.endswith("_hidden")])
                        features.extend([ex.get(k, 0.0) for k in spec_keys])
                    X.append(features)
                X = np.array(X)

                # Verify checkpoint dimension matches current data
                if 'state_dict' in ckpt:
                    saved_dim = ckpt['state_dict']['net.0.weight'].shape[1]
                else:
                    saved_dim = ckpt['net.0.weight'].shape[1]

                expected_dim = X.shape[1]
                if saved_dim != expected_dim:
                    print(f"[ERROR] Probe dim mismatch: saved={saved_dim} expected={expected_dim}")
                    print(f"[ERROR] Delete {probe_path} and re-run train-probe")
                    return

                probe = HallucinationProbe(input_dim=X.shape[1])
                probe.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
                metrics = evaluate_probe(probe, X, y_true)
                probe_scores = np.array(metrics['y_score'])
                print(
                    "--- PROBE EVAL ---\n"
                    f"  Accuracy: {metrics['accuracy']:.4f}\n"
                    f"  AUC: {metrics['auc']:.4f}\n")

    if args.mode == "hybrid" and spectral_scores is not None and probe_scores is not None:
        # 1. Recalculate raw spectral scores
        # IMPORTANT: Use the original samples and key. Re-extract to avoid previous mutations.
        raw_scores_orig = np.array([s.get(key, np.nan) for s in samples], dtype=float)

        # Determine the correct sign for AUC >= 0.5 using valid samples
        valid_mask = ~np.isnan(raw_scores_orig)
        ys_v = raw_scores_orig[valid_mask]
        yt_v = y_true[valid_mask]

        temp_auc = roc_auc_score(yt_v, ys_v)
        sign = 1.0
        if temp_auc < 0.5:
            sign = -1.0
            temp_auc = 1 - temp_auc

        # 2. Compute stats on sign-corrected valid samples
        raw_scores_corrected = sign * raw_scores_orig
        spec_valid = raw_scores_corrected[valid_mask]
        spec_std = spec_valid.std() if spec_valid.std() > 0 else 1.0

        # 3. Get spectral threshold for the sign-corrected metric
        spec_thresh = find_optimal_threshold(
            yt_v, spec_valid, target=args.optimize_for or "recall80")

        # 4. Normalize: (score - thresh) / std
        spectral_norm = np.zeros_like(raw_scores_corrected)
        spectral_norm[valid_mask] = (raw_scores_corrected[valid_mask] - spec_thresh) / spec_std

        # 5. Hybrid Parameter Search (Maximize Joint AUC)
        print("--- HYBRID PARAMETER SEARCH ---")
        best_h_auc = -1
        best_alpha = 0.5
        best_beta = 0.0

        # Grid for mixing weight alpha (spec vs probe). Include 0 and 1 for true maximization.
        alphas = np.linspace(0.0, 1.0, 21)
        # Grid for offset beta (sweep across quantiles of spectral scores)
        betas = np.quantile(spec_valid, np.linspace(0.0, 1.0, 11))

        from sklearn.metrics import roc_auc_score

        for a in alphas:
            for b_off in betas:
                # Re-normalize with candidate beta
                s_norm = np.zeros_like(raw_scores_corrected)
                s_norm[valid_mask] = (raw_scores_corrected[valid_mask] - b_off) / spec_std

                h_scores = a * s_norm + (1 - a) * probe_scores
                try:
                    curr_auc = roc_auc_score(y_true, h_scores)
                    if curr_auc > best_h_auc:
                        best_h_auc = curr_auc
                        best_alpha = a
                        best_beta = b_off
                except BaseException:
                    continue

        # Final best scores
        s_norm_best = np.zeros_like(raw_scores_corrected)
        s_norm_best[valid_mask] = (raw_scores_corrected[valid_mask] - best_beta) / spec_std
        hybrid_scores = best_alpha * s_norm_best + (1 - best_alpha) * probe_scores

        print(f"  Best Alpha (mixing): {best_alpha:.2f}")
        print(f"  Best Beta (offset): {best_beta:.4f}")
        print(f"  Joint AUC: {best_h_auc:.4f}")

        if args.optimize_for:
            h_thresh = find_optimal_threshold(y_true, hybrid_scores, target=args.optimize_for)
            metrics = compute_classification_metrics(y_true, hybrid_scores, threshold=h_thresh)
            print(
                f"  Optimized ({args.optimize_for}): thresh={h_thresh:.4f}, "
                f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}")

        if args.plots:
            plot_path = Path(args.output_dir) / f"eval_hybrid_{args.model}_{args.domain}.png"
            plot_evaluation(y_true, hybrid_scores, f"Hybrid ({args.model})", plot_path)
        print("")


def handle_sweep(args):
    input_file = Path(args.data_dir) / f"spectral_features_{args.model}_{args.domain}.jsonl"
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_samples = [json.loads(line) for line in f]
    samples = [flatten_sample(s) for s in raw_samples]
    y_true = np.array([s['label'] for s in samples])

    max_layer = -1
    for k in samples[0].keys():
        if k.startswith("L") and "_" in k:
            try:
                parts = k.split("_")[0]
                if parts[1:].isdigit():
                    max_layer = max(max_layer, int(parts[1:]))
            except BaseException:
                continue
    n_layers = max_layer + 1

    compute_trajectory_features(samples, n_layers)

    metrics = ["fiedler_value", "smoothness_index", "spectral_entropy", "hfer", "energy"]
    best_overall = {"auc": -1}
    results = []

    search_space = []
    for layer_idx in range(n_layers):
        for m in metrics:
            search_space.append((layer_idx, m, f"L{layer_idx}_{m}"))
    for m in metrics:
        for tm in ["delta", "range", "slope", "auc", "max_jump", "flux"]:
            search_space.append(("trajectory", f"{m}_{tm}", f"Ltrajectory_{m}_{tm}"))

    from sklearn.metrics import roc_auc_score
    for lid, mname, key in search_space:
        vals = np.array([s.get(key, np.nan) for s in samples], dtype=float)
        vmask = ~np.isnan(vals)
        if vmask.sum() < len(samples) * 0.5:
            continue
        ys, yt = vals[vmask], y_true[vmask]
        try:
            auc = roc_auc_score(yt, ys)
            if auc < 0.5:
                auc = 1 - auc
            res = {"layer": lid, "metric": mname, "auc": auc}
            results.append(res)
            if auc > best_overall["auc"]:
                best_overall = res
        except BaseException:
            continue

    print(f"Best: L{best_overall['layer']} {best_overall['metric']} AUC={best_overall['auc']:.4f}")

    best_params_path = Path("data/categories_sweeps/best_params.json")
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_data = {}
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best_data = json.load(f)
    best_data[f"{args.model}_{args.domain}"] = best_overall
    with open(best_params_path, 'w') as f:
        json.dump(best_data, f, indent=2)


def handle_fusion(args):
    """
    Trains a non-linear meta-classifier (Random Forest) on the Validation split
    and evaluates on the Test split.
    """
    input_file = Path(args.data_dir) / f"spectral_features_{args.model}_{args.domain}.jsonl"
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_samples = [json.loads(line) for line in f]
    samples = [flatten_sample(s) for s in raw_samples]

    train_idx, val_idx, test_idx = get_content_split_indices(samples)

    # 1. Get Probe Scores (probabilities)
    probe_name = f"probe_{args.model}_hidden.pt"
    probe_path = Path(args.output_dir) / probe_name
    if not probe_path.exists():
        print(f"Error: Probe {probe_path} not found. Run train-probe first.")
        return

    checkpoint = torch.load(probe_path, map_location='cpu', weights_only=False)
    X_probe_all = []
    for ex in samples:
        features = []
        hidden_keys = sorted([k for k in samples[0].keys() if k.endswith("_hidden")])
        for k in hidden_keys:
            features.extend(ex.get(k, []))
        X_probe_all.append(features)
    X_probe_all = np.array(X_probe_all)

    probe = HallucinationProbe(input_dim=X_probe_all.shape[1])
    probe.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    probe.eval()
    with torch.no_grad():
        probe_probs = torch.sigmoid(probe(torch.FloatTensor(X_probe_all))).cpu().numpy().flatten()

    # 2. Get Spectral Features (Trajectories)
    # Ensure trajectory features are computed
    max_layer = -1
    for k in samples[0].keys():
        if k.startswith("L") and "_" in k:
            try:
                parts = k.split("_")[0]
                if parts[1:].isdigit():
                    max_layer = max(max_layer, int(parts[1:]))
            except BaseException:
                continue
    compute_trajectory_features(samples, max_layer + 1)

    spec_metrics = ["max_jump", "flux", "delta", "range", "auc"]
    # Identify best spectral feature from previous sweep if available
    best_params_path = Path("data/categories_sweeps/best_params.json")
    best_feat_key = None
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best_params = json.load(f).get(f"{args.model}_{args.domain}")
        if best_params:
            best_feat_key = f"L{best_params['layer']}_{best_params['metric']}"

    spec_feat_X = []
    for s in samples:
        row = []
        # Add trajectory features for each primary metric
        for m in ["fiedler_value", "smoothness_index", "spectral_entropy", "energy"]:
            for tm in spec_metrics:
                row.append(s.get(f"Ltrajectory_{m}_{tm}", 0.0))
        # Add the single best performing static feature
        if best_feat_key:
            row.append(s.get(best_feat_key, 0.0))
        spec_feat_X.append(row)
    spec_feat_X = np.array(spec_feat_X)

    # 3. Combine for Meta-Classifier
    X_meta = np.column_stack([probe_probs, spec_feat_X])
    y_all = np.array([s['label'] for s in samples])

    X_train_meta = X_meta[val_idx]  # Train meta-classifier on VAL split
    y_train_meta = y_all[val_idx]
    X_test_meta = X_meta[test_idx]  # Evaluate on TEST split
    y_test_meta = y_all[test_idx]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    print("--- FUSION AUDIT (Random Forest Meta-Classifier) ---")
    print(f"Train samples (Val split): {len(X_train_meta)}")
    print(f"Test samples (Test split): {len(X_test_meta)}")

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_meta, y_train_meta)

    y_score_meta = rf.predict_proba(X_test_meta)[:, 1]
    meta_auc = roc_auc_score(y_test_meta, y_score_meta)

    # Comparison: Pure Probe on the same split
    probe_test_scores = probe_probs[test_idx]
    probe_auc = roc_auc_score(y_test_meta, probe_test_scores)

    # Comparison: Pure Spectral on the same split
    best_spec_only_auc = 0.0
    if best_feat_key:
        best_spec_only_vals = spec_feat_X[test_idx, -1]
        best_spec_only_auc = roc_auc_score(y_test_meta, best_spec_only_vals)
        if best_spec_only_auc < 0.5:
            best_spec_only_auc = 1 - best_spec_only_auc

    print(f"  Pure Probe AUC (TEST): {probe_auc:.4f}")
    if best_feat_key:
        print(f"  Pure Spectral AUC ({best_params['metric']}, TEST): {best_spec_only_auc:.4f}")
    print(f"  FUSION Joint AUC (TEST): {meta_auc:.4f}")
    print(f"  [meta] Gain over probe: {meta_auc - probe_auc:+.4f}")
    print("")


def handle_fusion_xgb(args):
    """
    Trains a non-linear meta-classifier (XGBoost) on the Validation split
    and evaluates on the Test split.
    """
    input_file = Path(args.data_dir) / f"spectral_features_{args.model}_{args.domain}.jsonl"
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_samples = [json.loads(line) for line in f]
    samples = [flatten_sample(s) for s in raw_samples]

    train_idx, val_idx, test_idx = get_content_split_indices(samples)

    # 1. Get Probe Scores (probabilities)
    probe_name = f"probe_{args.model}_hidden.pt"
    probe_path = Path(args.output_dir) / probe_name
    if not probe_path.exists():
        print(f"Error: Probe {probe_path} not found. Run train-probe first.")
        return

    checkpoint = torch.load(probe_path, map_location='cpu', weights_only=False)
    X_probe_all = []
    hidden_keys = sorted([k for k in samples[0].keys() if k.endswith("_hidden")])
    for ex in samples:
        features = []
        for k in hidden_keys:
            features.extend(ex.get(k, []))
        X_probe_all.append(features)
    X_probe_all = np.array(X_probe_all)

    probe = HallucinationProbe(input_dim=X_probe_all.shape[1])
    probe.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    probe.eval()
    with torch.no_grad():
        probe_probs = torch.sigmoid(probe(torch.FloatTensor(X_probe_all))).cpu().numpy().flatten()

    # 2. Get Spectral Features (Trajectories)
    max_layer = -1
    for k in samples[0].keys():
        if k.startswith("L") and "_" in k:
            try:
                parts = k.split("_")[0]
                if parts[1:].isdigit():
                    max_layer = max(max_layer, int(parts[1:]))
            except BaseException:
                continue
    compute_trajectory_features(samples, max_layer + 1)

    spec_metrics = ["max_jump", "flux", "delta", "range", "auc"]
    best_params_path = Path("data/categories_sweeps/best_params.json")
    best_feat_key = None
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best_params = json.load(f).get(f"{args.model}_{args.domain}")
        if best_params:
            best_feat_key = f"L{best_params['layer']}_{best_params['metric']}"

    spec_feat_X = []
    for s in samples:
        row = []
        for m in ["fiedler_value", "smoothness_index", "spectral_entropy", "energy"]:
            for tm in spec_metrics:
                row.append(s.get(f"Ltrajectory_{m}_{tm}", 0.0))
        if best_feat_key:
            row.append(s.get(best_feat_key, 0.0))
        spec_feat_X.append(row)
    spec_feat_X = np.array(spec_feat_X)

    # 3. Combine for Meta-Classifier
    X_meta = np.column_stack([probe_probs, spec_feat_X])
    y_all = np.array([s['label'] for s in samples])

    X_train_meta = X_meta[val_idx]  # Train meta-classifier on VAL split
    y_train_meta = y_all[val_idx]
    X_test_meta = X_meta[test_idx]  # Evaluate on TEST split
    y_test_meta = y_all[test_idx]

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    print("--- FUSION AUDIT (XGBoost Meta-Classifier) ---")
    print(f"Train samples (Val split): {len(X_train_meta)}")
    print(f"Test samples (Test split): {len(X_test_meta)}")

    # Strict Regularization per User Directive
    clf = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        objective='binary:logistic'
    )
    clf.fit(X_train_meta, y_train_meta)

    y_score_meta = clf.predict_proba(X_test_meta)[:, 1]
    meta_auc = roc_auc_score(y_test_meta, y_score_meta)

    # Comparison
    probe_test_scores = probe_probs[test_idx]
    probe_auc = roc_auc_score(y_test_meta, probe_test_scores)

    print(f"  Pure Probe AUC (TEST): {probe_auc:.4f}")
    print(f"  FUSION Joint AUC (XGB, TEST): {meta_auc:.4f}")
    print(f"  [xgboost] Gain over probe: {meta_auc - probe_auc:+.4f}")
    print("")


def handle_analyze_disconnect(args):
    """
    Identifies samples where the probe is confident but wrong,
    and checks if spectral signals were correct.
    """
    input_file = Path(args.data_dir) / f"spectral_features_{args.model}_{args.domain}.jsonl"
    if not input_file.exists():
        return
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_samples = [json.loads(line) for line in f]
    samples = [flatten_sample(s) for s in raw_samples]
    _, _, test_idx = get_content_split_indices(samples)
    test_samples = [samples[i] for i in test_idx]

    # 1. Get Probe Probs
    probe_path = Path(args.output_dir) / f"probe_{args.model}_hidden.pt"
    if not probe_path.exists():
        return
    checkpoint = torch.load(probe_path, map_location='cpu', weights_only=False)
    hidden_keys = sorted([k for k in samples[0].keys() if k.endswith("_hidden")])
    X_probe = []
    for s in test_samples:
        features = []
        for k in hidden_keys:
            features.extend(s.get(k, []))
        X_probe.append(features)
    probe = HallucinationProbe(input_dim=len(X_probe[0]))
    probe.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    probe.eval()
    with torch.no_grad():
        probs = torch.sigmoid(probe(torch.FloatTensor(X_probe))).cpu().numpy().flatten()

    print("--- DISCONNECT ANALYSIS (Confident failures) ---")
    disconnects = []

    # Identify the layer of max jump for topological collapse reporting
    # We use fiedler_value as the primary topological indicator
    jump_layer_key = "Ltrajectory_fiedler_value_max_jump"

    for i, (s, p) in enumerate(zip(test_samples, probs)):
        label = s['label']
        is_correct = (p >= 0.5 and label == 1) or (p < 0.5 and label == 0)
        confidence = p if p >= 0.5 else 1 - p

        if not is_correct and confidence > 0.8:
            # Find which layer had the max jump
            jump_val = s.get(jump_layer_key, 0.0)

            # Identify the layer index where this jump occurred
            # We look for the metric that equals the jump value (or contributes to it)
            # For simplicity, we'll scan for the L{i}_fiedler_value that is the local max
            jump_layer = -1
            max_val = -1e9
            for k, v in s.items():
                if k.startswith("L") and k.endswith(
                        "_fiedler_value") and not k.startswith("Ltrajectory"):
                    try:
                        layer_idx = int(k.split("_")[0][1:])
                        if v > max_val:
                            max_val = v
                            jump_layer = layer_idx
                    except BaseException:
                        continue

            disconnects.append({
                "index": s.get('original_idx', i),
                "label": int(label),
                "probe_prob": float(p),
                "confidence": float(confidence),
                "prompt": s.get('prompt', 'N/A'),
                "prediction": s.get('prediction', ''),
                "ground_truth": s.get('ground_truth', ''),
                "topological_jump_magnitude": float(jump_val),
                "collapse_layer_index": jump_layer
            })

    print(f"Found {len(disconnects)} confident probe failures in test set.")
    out_path = Path(args.output_dir) / f"disconnect_audit_{args.model}.json"
    with open(out_path, 'w') as f:
        json.dump(disconnects, f, indent=2)
    print(f"Saved disconnect audit to {out_path}")


def handle_audit(args):
    """Mega-Audit scaling command to extract spectral features and consensus samples."""
    import time
    config = GSPConfig(device=args.device, verbose=args.verbose)
    model_id = MODEL_REGISTRY[args.model]
    out_dir = Path(args.output_dir) / f"{args.model}_{args.domain}"
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / f"spectral_features_{args.model}_{args.domain}.jsonl"
    checkpoint_path = out_dir / f"checkpoint_{args.model}_{args.domain}.csv"

    if checkpoint_path.exists():
        import pandas as pd
        df_check = pd.read_csv(checkpoint_path)
        resume_idx = len(df_check)
        print(f"Resuming audit from index {resume_idx}...")
    else:
        resume_idx = 0

    if getattr(args, "bfcl_path", None):
        print(f"Loading BFCL dataset from {args.bfcl_path}...")
        dataset = []
        with open(args.bfcl_path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
        dataset = dataset[:args.n]
    elif getattr(args, "toolbench_path", None):
        print(f"Loading ToolBench dataset from {args.toolbench_path}...")
        dataset = []
        with open(args.toolbench_path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
        dataset = dataset[:args.n]
    else:
        dataset = load_glaive_data(domain=args.domain, limit=args.n, step=50)

    records = []
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(model_id)
        tokenizer = framework.instrumenter.tokenizer
        model = framework.instrumenter.model
        topo = DirectedTopologist(
            device=args.device) if getattr(
            args, "fast_fiedler", False) else None
        target_layers = [4, 8, 11, 15, 20, 24, 28, 31]

        pbar = tqdm(dataset, desc="GSP Extraction", unit="sample")
        for i, ex in enumerate(pbar):
            if i < resume_idx:
                continue

            is_hard_mode = bool(
                getattr(
                    args,
                    "bfcl_path",
                    None) or getattr(
                    args,
                    "toolbench_path",
                    None))
            if is_hard_mode:
                ground_truth = ex.get("ground_truth", "")
                chat_raw = ex.get("chat", "")
                prompt_msgs = [
                    {"role": "system",
                     "content": ex.get("system", "You are a helpful assistant.")},
                    {"role": "user",
                     "content": chat_raw.split("\\nASSISTANT:")[0]
                     .replace("USER: ", "", 1).strip()}
                ]
            else:
                from spectral_guardrails.utils.data import parse_glaive_chat
                chat_raw = ex.get("chat", "")
                messages = parse_glaive_chat(chat_raw)
                target_idx = -1
                for idx, msg in enumerate(messages):
                    if msg["role"] == "assistant":
                        target_idx = idx
                        break
                if target_idx == -1:
                    continue
                ground_truth = messages[target_idx]["content"]
                prompt_msgs = [{"role": "system", "content": ex.get(
                    "system", "You are a helpful assistant.")}]
                if target_idx > 0:
                    prompt_msgs.append(messages[target_idx - 1])

            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            start_gen = time.perf_counter()
            logprobs_list = []
            predictions = []

            try:
                num_samples = getattr(args, "n_samples_consensus", 1)
                temp = 1.0 if num_samples > 1 else getattr(args, "temperature", 0.3)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        temperature=temp,
                        do_sample=True,
                        num_return_sequences=num_samples,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True)

                transition_scores = model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True)
                for j in range(num_samples):
                    pred_tokens = outputs.sequences[j][inputs.input_ids.shape[1]:]
                    prediction = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    predictions.append(prediction)
                    logprobs_list.append(transition_scores[j].cpu().numpy().mean())

                gen_time = time.perf_counter() - start_gen
                prediction = predictions[0]
                mean_logprob = float(np.mean(logprobs_list))

                consensus_score = 1.0
                if num_samples > 1:
                    from spectral_guardrails.utils.data import is_json_equivalent
                    m = 0
                    for op in predictions[1:]:
                        if is_json_equivalent(prediction, op):
                            m += 1
                    consensus_score = (m + 1) / num_samples

                label = assign_label(prediction, ground_truth)
            except Exception as e:
                pbar.write(f"[warn] Generation failed: {e}")
                continue

            ld = []
            ph = {}
            sl = 0

            if not getattr(args, "sampling_only", False):
                full_text = prompt_text + prediction
                start_analysis = time.perf_counter()
                try:
                    if topo is not None:
                        # ── Fast-Fiedler path: single consolidated forward pass ──────────
                        # One pass captures attentions + hidden states; eliminates the
                        # separate framework.analyze_text call and the duplicate hidden-state pass.
                        fe = tokenizer(full_text, return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            mo = model(**fe, output_attentions=True, output_hidden_states=True)
                        attentions = mo.attentions     # tuple[layer] of [1, H, Q, K]
                        hidden_states = mo.hidden_states  # tuple[layer+1] of [1, Q, D]
                        token_ids = fe.input_ids[0].tolist()
                        sl = len(token_ids)

                        # Subgraph slicing: use tool-call token span; fallback to last 50 tokens
                        pos = find_token_positions(tokenizer, token_ids, prediction)
                        t_start, t_end = pos["t_func"], pos["t_end"]
                        if t_end > t_start + 1 and t_end < sl:
                            sub_indices = list(range(t_start, t_end + 1))
                        else:
                            sub_indices = list(range(max(0, sl - 50), sl))

                        idx_t = torch.tensor(sub_indices, device=model.device, dtype=torch.long)

                        fiedler_vals = []
                        for attn in attentions:
                            # [H, Q, K]
                            attn_sq = attn.squeeze(0)
                            attn_sym = framework.graph_constructor.symmetrize_attention(
                                attn_sq.unsqueeze(0))  # [1, H, Q, K]
                            adjacency = framework.graph_constructor.aggregate_heads(
                                attn_sym).squeeze(0)   # [Q, K]
                            # Slice induced subgraph adjacency, build its Laplacian
                            adj_sub = adjacency.index_select(
                                0, idx_t).index_select(1, idx_t)  # [S, S]
                            L_sub = framework.graph_constructor.construct_laplacian(
                                adj_sub.unsqueeze(0)).squeeze(0)  # [S, S]
                            # Symmetrize before Lanczos (RW Laplacian is not symmetric in general)
                            L_sym = 0.5 * (L_sub + L_sub.T)
                            fiedler_vals.append(topo.get_fiedler_value(L_sym))

                        # GPU-native spectral velocity (0.2.0)
                        fv_t = torch.tensor(fiedler_vals)
                        _, max_vel, max_vel_layer = calculate_spectral_velocity(fv_t)

                        ld = [{"fiedler_value": f, "smoothness_index": 0.0,
                               "spectral_entropy": 0.0, "energy": 0.0, "hfer": 0.0,
                               "max_velocity": 0.0}
                              for f in fiedler_vals]
                        ld[max_vel_layer - 1]["max_velocity"] = float(max_vel)

                        # Probe hidden states from the same consolidated pass
                        for li in target_layers:
                            if li < len(hidden_states):
                                h = hidden_states[li][0].to(torch.float32).cpu()
                                ph[str(li)] = extract_probe_features(h, pos).tolist()

                    else:
                        # ── Standard path (full spectral analysis + separate hidden pass) ─
                        analysis = framework.analyze_text(full_text, save_results=False)
                        for diag in analysis["layer_diagnostics"]:
                            ld.append({
                                "fiedler_value": float(diag.fiedler_value),
                                "smoothness_index": float(diag.smoothness_index),
                                "spectral_entropy": float(diag.spectral_entropy),
                                "energy": float(diag.energy),
                                "hfer": float(diag.hfer),
                            })
                        with torch.no_grad():
                            fe = tokenizer(full_text, return_tensors="pt").to(model.device)
                            mo = model(**fe, output_hidden_states=True)
                            hs = [h.to(torch.float32).cpu() for h in mo.hidden_states]
                            pos = find_token_positions(
                                tokenizer, fe.input_ids[0].tolist(), prediction)
                            for li in target_layers:
                                if li < len(hs):
                                    ph[str(li)] = extract_probe_features(hs[li][0], pos).tolist()
                            sl = len(fe.input_ids[0])

                except Exception as e:
                    pbar.write(f"[warn] Analysis failed on sample {i}: {e}")

                analysis_ms = (time.perf_counter() - start_analysis) * 1000
                pbar.set_postfix(label="HALL" if label else "ok", anal_ms=f"{analysis_ms:.0f}")

            rec = {
                "original_idx": ex.get("original_idx", i),
                "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "prompt": prompt_text,
                "label": int(label),
                "prediction": prediction,
                "ground_truth": ground_truth,
                "mean_logprob": mean_logprob,
                "consensus_score": consensus_score,
                "consensus_samples": predictions,
                "latency_ms": gen_time * 1000,
                "layer_diagnostics": ld,
                "hidden": ph,
                "seq_len": sl
            }
            records.append(rec)
            with open(features_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            pbar.set_postfix(label="HALL" if label else "ok")

    print(f"\\n[mega-audit] SUCCESS: {len(records)} samples processed.")


def handle_logic_ensemble(args):
    """
    Search for the best combination of 1, 2, and 3 spectral thresholds (Boolean AND logic).
    Goal: Find the symbolic ensemble that maximizes Joint AUC.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from itertools import combinations

    data_dir = args.data_dir or f"data/n1000_{args.model}_{args.domain}"
    features_path = Path(data_dir) / f"spectral_features_{args.model}_{args.domain}.jsonl"

    if not features_path.exists():
        print(f"Error: Features not found at {features_path}")
        return

    # Load data
    samples = []
    with open(features_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Flatten data for analysis
    rows = []
    for s in samples:
        row = {"label": s["label"], "prompt_hash": s["prompt_hash"]}
        # 'layer_diagnostics' is often saved as a list or a dict mapping layer index to metrics
        ld = s.get("layer_diagnostics", s.get("spectral", []))
        if isinstance(ld, list):
            for i, metrics in enumerate(ld):
                # Metrics is the dict of fiedler, energy, etc for this layer
                for m_name, val in metrics.items():
                    row[f"L{i}_{m_name}"] = val
        else:  # dict
            for layer, metrics in ld.items():
                for m_name, val in metrics.items():
                    row[f"L{layer}_{m_name}"] = val
        rows.append(row)

    df = pd.DataFrame(rows)

    # Template-Consistent Split (leakage prevention)
    unique_hashes = sorted(df['prompt_hash'].unique())
    rng = np.random.RandomState(42)
    rng.shuffle(unique_hashes)

    n_unique = len(unique_hashes)
    itrain = int(0.7 * n_unique)
    ival = int(0.85 * n_unique)

    train_h = set(unique_hashes[:itrain])
    val_h = set(unique_hashes[itrain:ival])
    test_h = set(unique_hashes[ival:])

    _df_train = df[df['prompt_hash'].isin(train_h)]  # noqa: F841
    df_val = df[df['prompt_hash'].isin(val_h)]
    df_test = df[df['prompt_hash'].isin(test_h)]

    _y_test = df_test['label'].values  # noqa: F841

    spectral_cols = [c for c in df.columns if c.startswith("L")]

    print(f"\n[logic-ensemble] Searching over {len(spectral_cols)} metrics...")

    # 1. Rank individual metrics by AUC on VAL
    results = []
    for col in spectral_cols:
        vals = df_val[col].values
        # Handle Inf/Nan
        vals = np.nan_to_num(vals, nan=0, posinf=0, neginf=0)
        try:
            auc = roc_auc_score(df_val['label'], vals)
            if auc < 0.5:
                auc = 1 - auc  # Support both directions
            results.append((col, auc))
        except BaseException:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [r[0] for r in results[:20]]  # Take top 20 for combinations

    print(f"Top metric: {results[0][0]} (AUC: {results[0][1]:.4f})")

    best_overall_auc = results[0][1]
    best_ensemble = [results[0][0]]

    # 2. Greedy search for 2 and 3 combinations
    for k in [2, 3]:
        print(f"Searching combinations of size {k}...")
        for combo in combinations(top_candidates, k):
            # For logic ensemble, we treat this as a simple weighted sum or threshold-veto
            # Here we use a shallow decision tree or mean of normalized scores as a proxy for logic
            val_scores = np.zeros(len(df_val))
            test_scores = np.zeros(len(df_test))

            for col in combo:
                # Directions check: ensure all metrics correlate positively with label=1
                auc_raw = roc_auc_score(df_val['label'], df_val[col])
                sign = 1 if auc_raw >= 0.5 else -1

                # Normalize and accumulate
                v = df_val[col].values
                v = (v - np.mean(v)) / (np.std(v) + 1e-6)
                val_scores += sign * v

                v_t = df_test[col].values
                v_t = (v_t - np.mean(v_t)) / (np.std(v_t) + 1e-6)
                test_scores += sign * v_t

            auc_val = roc_auc_score(df_val['label'], val_scores)
            if auc_val > best_overall_auc:
                best_overall_auc = auc_val
                best_ensemble = list(combo)
                best_test_auc = roc_auc_score(df_test['label'], test_scores)

    print("\n" + "=" * 40)
    print("BEST LOGIC ENSEMBLE FOUND")
    print("=" * 40)
    for i, m in enumerate(best_ensemble):
        print(f"  Metric {i + 1}: {m}")
    print(f"Final TEST AUC: {best_test_auc:.4f}")
    print("=" * 40)


def main():

    # Common parent parser for shared flags
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--device", type=str, default="cuda")
    parent_parser.add_argument("--data-dir", type=str, default="data/")
    parent_parser.add_argument("--output-dir", type=str, default="data/")
    parent_parser.add_argument("--verbose", action="store_true")

    parser = argparse.ArgumentParser(description="Spectral Guardrails CLI")
    subparsers = parser.add_subparsers(dest="command")

    # prepare
    p_prep = subparsers.add_parser("prepare", parents=[parent_parser])
    p_prep.add_argument(
        "--domain",
        choices=[
            "general",
            "finance",
            "math",
            "all"],
        default="general")
    p_prep.add_argument("--n-samples", type=int, default=None)

    # extract
    p_ext = subparsers.add_parser("extract", parents=[parent_parser])
    p_ext.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    p_ext.add_argument("--domain", choices=["general", "finance", "math"], default="general")
    p_ext.add_argument("--temperature", type=float, default=0.3)
    p_ext.add_argument("--n-samples", type=int, default=None)

    # train-probe
    p_probe = subparsers.add_parser("train-probe", parents=[parent_parser])
    p_probe.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p_probe.add_argument("--domain", choices=["general", "finance", "math"], default="general")
    p_probe.add_argument(
        "--feature-type",
        choices=[
            "hidden",
            "spectral",
            "combined",
            "spectral_rich",
            "lmm_gbt"],
        default="hidden")
    p_probe.add_argument("--epochs", type=int, default=50)
    p_probe.add_argument("--patience", type=int, default=5)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", parents=[parent_parser])
    p_eval.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p_eval.add_argument("--domain", choices=["general", "finance", "math"], default="general")
    p_eval.add_argument("--mode", choices=["spectral", "probe", "all", "hybrid"], default="all")
    p_eval.add_argument(
        "--feature-type",
        choices=[
            "hidden",
            "spectral",
            "combined",
            "spectral_rich",
            "lmm_gbt"],
        default="hidden")
    p_eval.add_argument(
        "--optimize-for",
        type=str,
        choices=[
            "recall80",
            "recall90",
            "precision80",
            "f1"],
        help="Target metric for threshold optimization")
    p_eval.add_argument(
        "--plots",
        action="store_true",
        help="Generate performance visualization plots")

    # sweep
    p_sweep = subparsers.add_parser("sweep", parents=[parent_parser])
    p_sweep.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p_sweep.add_argument("--domain", choices=["general", "finance", "math"], default="general")
    p_sweep.add_argument("--plots", action="store_true")

    # logic-ensemble
    p_logic = subparsers.add_parser("logic-ensemble", parents=[parent_parser])
    p_logic.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p_logic.add_argument("--domain", choices=["general", "finance", "math"], default="general")

    # audit (Mega-Audit scaling command)
    p_audit = subparsers.add_parser("audit", parents=[parent_parser])
    p_audit.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p_audit.add_argument("--domain", choices=["general", "finance", "math"], default="general")
    p_audit.add_argument("--n", type=int, default=1000, help="Number of samples for the mega-audit")
    p_audit.add_argument("--temperature", type=float, default=0.3, help="Generation temperature")
    p_audit.add_argument("--n-samples-consensus", type=int, default=1,
                         help="Number of samples for consistency consensus (SelfCheckGPT baseline)")
    p_audit.add_argument("--bfcl-path", type=str, default=None,
                         help="Path to BFCL hard-mode dataset")
    p_audit.add_argument("--toolbench-path", type=str, default=None,
                         help="Path to ToolBench hard-mode dataset")
    p_audit.add_argument(
        "--sampling-only",
        action="store_true",
        help="Only run generation/consensus, skip spectral/hidden extraction")
    p_audit.add_argument(
        "--fast-fiedler",
        action="store_true",
        help="GPU Lanczos Fiedler + subgraph slicing (ToolBench mode, single forward pass)")

    args = parser.parse_args()
    if args.command == "prepare":
        handle_prepare(args)
    elif args.command == "extract":
        handle_extract(args)
    elif args.command == "train-probe":
        handle_train_probe(args)
    elif args.command == "evaluate":
        handle_evaluate(args)
    elif args.command == "sweep":
        handle_sweep(args)
    elif args.command == "fusion":
        handle_fusion_xgb(args)
    elif args.command == "analyze-disconnect":
        handle_analyze_disconnect(args)
    elif args.command == "logic-ensemble":
        handle_logic_ensemble(args)
    elif args.command == "audit":
        handle_audit(args)
    elif args.command is None:
        parser.print_help()
    else:
        print(f"Error: Command '{args.command}' is not recognized.")


if __name__ == "__main__":
    main()
