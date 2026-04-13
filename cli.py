import argparse
import sys
import json
import os
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import BitsAndBytesConfig

# Import library and utilities
from spectral_trust import GSPDiagnosticsFramework, GSPConfig
from spectral_guardrails.utils.data import load_glaive_data
from spectral_guardrails.utils.models import MODEL_REGISTRY, load_model_and_tokenizer
from spectral_guardrails.utils.metrics import compute_classification_metrics
from spectral_guardrails.probes.labeling import assign_label
from spectral_guardrails.probes.features import extract_probe_features, find_token_positions
from spectral_guardrails.utils.stats import compute_cohens_d, bootstrap_auc_ci
from spectral_guardrails.probes.mlp import HallucinationProbe, train_probe, evaluate_probe
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc as sk_auc

# --- Framework Utilities ---

def print_summary_table(metrics: dict, title: str = "Results Summary"):
    print(f"\n{'='*40}")
    print(f"{title:^40}")
    print(f"{'='*40}")
    print(f"{'Metric':<20} | {'Value':>17}")
    print(f"{'-'*20}-|-{'-'*17}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<20} | {v:>17.4f}")
        elif isinstance(v, int):
            print(f"{k:<20} | {v:>17d}")
        else:
            s = str(v)
            if len(s) > 17: s = s[:14] + "..."
            print(f"{k:<20} | {s:>17}")
    print(f"{'='*40}\n")

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
            for l in range(n_layers):
                val = s.get(f"L{l}_{m}")
                if val is not None:
                    profile.append(val)
            
            if len(profile) < 2:
                continue
                
            profile = np.array(profile)
            
            # GoR Standard Trajectory Features
            s[f"Ltrajectory_{m}_delta"] = profile[-1] - profile[0]
            s[f"Ltrajectory_{m}_range"] = np.ptp(profile)
            try:
                # Use linear regression to find global slope
                slope = np.polyfit(np.arange(len(profile)), profile, 1)[0]
                s[f"Ltrajectory_{m}_slope"] = slope
            except:
                s[f"Ltrajectory_{m}_slope"] = 0.0
            
            # Use np.trapezoid if available (NumPy 2.0+), else np.trapz
            if hasattr(np, 'trapezoid'):
                s[f"Ltrajectory_{m}_auc"] = np.trapezoid(profile)
            else:
                s[f"Ltrajectory_{m}_auc"] = np.trapz(profile)

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
            except: continue
    n_layers = max_layer + 1
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{model} / {domain} — Spectral Layer Profiles", fontsize=16)
    
    for ax, m in zip(axes.flatten()[:5], metrics):
        data = np.full((len(samples), n_layers), np.nan)
        for i, s in enumerate(samples):
            for l in range(n_layers):
                data[i, l] = s.get(f"L{l}_{m}", np.nan)
        
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
            ax.axvline(x=best_layer, color='black', linestyle='--', alpha=0.6, label=f'Best (L{best_layer})')
            
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
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    if args.n_samples: samples = samples[:args.n_samples]

    output_path = output_dir / f"spectral_features_{args.model}_{args.domain}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[extract] Running native framework extraction for {len(samples)} samples...")
    
    records = []
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(model_id)
        tokenizer = framework.instrumenter.tokenizer
        model = framework.instrumenter.model

        pbar = tqdm(samples, desc="GSP Extraction", unit="sample")
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
            if target_idx == -1: continue
            
            ground_truth = messages[target_idx]['content']
            prompt_msgs = [{"role": "system", "content": ex.get('system', "You are a helpful assistant.")}]
            if target_idx > 0: prompt_msgs.append(messages[target_idx-1])
            
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            
            # 2. Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, temperature=args.temperature, do_sample=args.temperature > 0, pad_token_id=tokenizer.eos_token_id)
            prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
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
                    
                    target_layers = [4, 8, 16, 20, 24, 28, 31]
                    pos = find_token_positions(tokenizer, inputs.input_ids[0].tolist(), prediction)
                    for li in target_layers:
                        if li < len(hidden_states):
                            signals = hidden_states[li][0]
                            h_feat = extract_probe_features(signals, pos)
                            probe_hidden[str(li)] = h_feat.tolist()

                result = {
                    "original_idx": ex.get('original_idx', i),
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

    X, y = [], []
    for ex in samples:
        features = []
        if args.feature_type in ["hidden", "combined"]:
            hidden_keys = sorted([k for k in samples[0].keys() if k.endswith("_hidden")])
            for k in hidden_keys: features.extend(ex.get(k, []))
        if args.feature_type in ["spectral", "combined"]:
            spec_keys = sorted([k for k in samples[0].keys() if any(k.endswith(f"_{m}") for m in ["hfer", "fiedler_value", "smoothness_index", "spectral_entropy"]) and k.startswith("L") and not k.endswith("_hidden")])
            features.extend([ex.get(k, 0.0) for k in spec_keys])
        X.append(features); y.append(ex.get('label', 0))

    X, y = np.array(X), np.array(y)
    n = len(X); itrain = int(0.7 * n); ival = int(0.85 * n)
    X_train, X_val, X_test = X[:itrain], X[itrain:ival], X[ival:]
    y_train, y_val, y_test = y[:itrain], y[itrain:ival], y[ival:]

    y_train_np = np.array(y_train)
    weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train_np)
    pos_weight = torch.tensor(weights[1] / weights[0], dtype=torch.float32)
    
    probe = HallucinationProbe(input_dim=X_train.shape[1])
    trained_probe, _ = train_probe(probe, X_train, y_train, X_val, y_val, 
                                 epochs=args.epochs, patience=args.patience,
                                 pos_weight=pos_weight)
    
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': trained_probe.state_dict(), 'input_dim': X_train.shape[1]}, output_dir / f"probe_{args.model}.pt")
    
    metrics = evaluate_probe(trained_probe, X_test, y_test)
    print_summary_table({k: v for k, v in metrics.items() if k not in ('y_true', 'y_score')}, f"Probe Results ({args.model})")

def handle_evaluate(args):
    input_file = Path(args.data_dir) / f"spectral_features_{args.model}_{args.domain}.jsonl"
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_samples = [json.loads(line) for line in f]
    samples = [flatten_sample(s) for s in raw_samples]
    y_true = np.array([s['label'] for s in samples])
    
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
                if "fiedler" in key and "fiedler_value" not in key: key = key.replace("fiedler", "fiedler_value")
                
                # Ensure trajectory features are computed for evaluation
                max_layer = -1
                for k in samples[0].keys():
                    if k.startswith("L") and "_" in k:
                        try:
                            parts = k.split("_")[0]
                            if parts[1:].isdigit():
                                max_layer = max(max_layer, int(parts[1:]))
                        except: continue
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
                
                d = compute_cohens_d(y_s[y_t==0], y_s[y_t==1])
                print(f"--- SPECTRAL EVAL ({key}) ---\n  AUC: {auc:.4f}\n  Cohen's d: {d:.4f}\n")

    if args.mode in ["probe", "all", "hybrid"]:
        probe_path = Path(args.output_dir) / f"probe_{args.model}.pt"
        if probe_path.exists():
            checkpoint = torch.load(probe_path, map_location='cpu', weights_only=False)
            
            # Reconstruct feature vector based on type (same logic as handle_train_probe)
            X = []
            for ex in samples:
                features = []
                if args.feature_type in ["hidden", "combined"]:
                    hidden_keys = sorted([k for k in samples[0].keys() if k.endswith("_hidden")])
                    for k in hidden_keys: features.extend(ex.get(k, []))
                if args.feature_type in ["spectral", "combined"]:
                    spec_keys = sorted([k for k in samples[0].keys() if any(k.endswith(f"_{m}") for m in ["hfer", "fiedler_value", "smoothness_index", "spectral_entropy"]) and k.startswith("L") and not k.endswith("_hidden")])
                    features.extend([ex.get(k, 0.0) for k in spec_keys])
                X.append(features)
            X = np.array(X)
            
            # Verify checkpoint dimension matches current data
            ckpt = torch.load(probe_path, map_location='cpu', weights_only=False)
            saved_dim = ckpt['state_dict']['net.0.weight'].shape[1] if 'state_dict' in ckpt else ckpt['net.0.weight'].shape[1]
            expected_dim = X.shape[1]
            if saved_dim != expected_dim:
                print(f"[ERROR] Probe dim mismatch: saved={saved_dim} expected={expected_dim}")
                print(f"[ERROR] Delete {probe_path} and re-run train-probe")
                return
                
            probe = HallucinationProbe(input_dim=X.shape[1])
            probe.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
            metrics = evaluate_probe(probe, X, y_true)
            probe_scores = np.array(metrics['y_score'])
            print(f"--- PROBE EVAL ---\n  Accuracy: {metrics['accuracy']:.4f}\n  AUC: {metrics['auc']:.4f}\n")

    if args.mode == "hybrid" and spectral_scores is not None and probe_scores is not None:
        hybrid_scores = (spectral_scores + probe_scores) / 2
        from sklearn.metrics import roc_auc_score
        h_auc = roc_auc_score(y_true, hybrid_scores)
        print(f"--- HYBRID EVAL ---\n  Joint AUC: {h_auc:.4f}")
        
        if args.optimize_for:
            thresh = find_optimal_threshold(y_true, hybrid_scores, target=args.optimize_for)
            metrics = compute_classification_metrics(y_true, hybrid_scores, threshold=thresh)
            print(f"  Optimized ({args.optimize_for}): thresh={thresh:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}")
            
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
            except: continue
    n_layers = max_layer + 1
    
    compute_trajectory_features(samples, n_layers)
    
    metrics = ["fiedler_value", "smoothness_index", "spectral_entropy", "hfer", "energy"]
    best_overall = {"auc": -1}
    results = []
    
    search_space = []
    for l in range(n_layers):
        for m in metrics: search_space.append((l, m, f"L{l}_{m}"))
    for m in metrics:
        for tm in ["delta", "range", "slope", "auc"]: search_space.append(("trajectory", f"{m}_{tm}", f"Ltrajectory_{m}_{tm}"))

    from sklearn.metrics import roc_auc_score
    for lid, mname, key in search_space:
        vals = np.array([s.get(key, np.nan) for s in samples], dtype=float)
        vmask = ~np.isnan(vals)
        if vmask.sum() < len(samples) * 0.5: continue
        ys, yt = vals[vmask], y_true[vmask]
        try:
            auc = roc_auc_score(yt, ys)
            if auc < 0.5: auc = 1 - auc
            res = {"layer": lid, "metric": mname, "auc": auc}
            results.append(res)
            if auc > best_overall["auc"]: best_overall = res
        except: continue

    print(f"Best: L{best_overall['layer']} {best_overall['metric']} AUC={best_overall['auc']:.4f}")
    
    best_params_path = Path("data/categories_sweeps/best_params.json")
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_data = {}
    if best_params_path.exists():
        with open(best_params_path, 'r') as f: best_data = json.load(f)
    best_data[f"{args.model}_{args.domain}"] = best_overall
    with open(best_params_path, 'w') as f: json.dump(best_data, f, indent=2)

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
    p_prep.add_argument("--domain", choices=["general", "finance", "math", "all"], default="general")
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
    p_probe.add_argument("--feature-type", choices=["hidden", "spectral", "combined"], default="hidden")
    p_probe.add_argument("--epochs", type=int, default=50)
    p_probe.add_argument("--patience", type=int, default=5)
    
    # evaluate
    p_eval = subparsers.add_parser("evaluate", parents=[parent_parser])
    p_eval.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p_eval.add_argument("--domain", choices=["general", "finance", "math"], default="general")
    p_eval.add_argument("--mode", choices=["spectral", "probe", "all", "hybrid"], default="all")
    p_eval.add_argument("--feature-type", choices=["hidden", "spectral", "combined"], default="hidden")
    p_eval.add_argument("--optimize-for", type=str, choices=["recall80", "recall90", "precision80", "f1"], help="Target metric for threshold optimization")
    p_eval.add_argument("--plots", action="store_true", help="Generate performance visualization plots")
    
    # sweep
    p_sweep = subparsers.add_parser("sweep", parents=[parent_parser])
    p_sweep.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p_sweep.add_argument("--domain", choices=["general", "finance", "math"], default="general")
    p_sweep.add_argument("--plots", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare": handle_prepare(args)
    elif args.command == "extract": handle_extract(args)
    elif args.command == "train-probe": handle_train_probe(args)
    elif args.command == "evaluate": handle_evaluate(args)
    elif args.command == "sweep": handle_sweep(args)
    elif args.command is None: parser.print_help()
    else: print(f"Error: Command '{args.command}' is not recognized.")

if __name__ == "__main__":
    main()
