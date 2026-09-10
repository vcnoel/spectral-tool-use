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

Known caveats:
  - Some chat templates inject the current date (Llama-3.x writes
    "Today Date: <today>"), so prompt text -- and therefore prompt_hash --
    changes from day to day. Splits are grouped by TOOL, not prompt_hash,
    so results are unaffected; but byte-identical re-extraction requires
    pinning the date.
  - Per-head spectral features come from spectral_trust (>=0.3.0).

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
    classify_failure, classify_failure_anyof, extract_calls,
    extract_glaive_tools,
)
from spectral_guardrails.probes.features import (
    extract_probe_features, find_token_positions_v2,
)
from spectral_guardrails.spectral.metrics import (
    METRIC_NAMES, PER_HEAD_METRICS, layer_spectral_metrics,
    laplacian_eig_profile, per_head_metrics, gram_spectrum_features,
    lapeigvals_diag_profile, lookback_ratio, residual_dynamics,
)
from spectral_guardrails.utils.data import load_glaive_data, parse_glaive_chat

OUT_DIR = Path("data/pilot_v2_qwen35_2b")   # overridden by --tag in main()
FEATURES = OUT_DIR / "features.jsonl"
SEED = 42
N_PROBE_LAYERS = 8
MAX_NEW_TOKENS = 256
TOKEN_STATE_CAP = 48   # generated tokens kept for the token-level probe


# ══════════════════════════════════════════════════════════════════════════════
# Extraction
# ══════════════════════════════════════════════════════════════════════════════

CONTROL_MARKERS = (
    "<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<|eot_id|>",
    "<|eom_id|>", "<|end|>", "<|end_of_text|>", "</s>", "<|assistant|>",
)


def strip_control_markers(text: str, tok=None) -> str:
    """Remove chat control tokens while keeping structural call tags."""
    for m in CONTROL_MARKERS:
        text = text.replace(m, "")
    if tok is not None:
        for m in (getattr(tok, "eos_token", None), getattr(tok, "pad_token", None)):
            if m:
                text = text.replace(m, "")
    return text.strip()


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


BFCL_DIR = Path("data/bfcl_v4")
BFCL_MIX = [  # (category file stem, expect_call, max examples)
    ("BFCL_v4_simple_python", True, 300),
    ("BFCL_v4_multiple", True, 200),
    ("BFCL_v4_parallel", True, 100),
    ("BFCL_v4_parallel_multiple", True, 100),
    ("BFCL_v4_irrelevance", False, 150),
]

# The "live" categories are real user queries contributed to the leaderboard
# rather than curated ones, so they form a third distribution alongside
# Glaive and the curated BFCL categories.
BFCL_LIVE_MIX = [
    ("BFCL_v4_live_simple", True, 250),
    ("BFCL_v4_live_multiple", True, 350),
    ("BFCL_v4_live_parallel", True, 15),
    ("BFCL_v4_live_parallel_multiple", True, 23),
    ("BFCL_v4_live_irrelevance", False, 200),
]


def _fix_schema(fn: dict) -> dict:
    """BFCL uses 'dict' where JSON schema says 'object'."""
    s = json.loads(json.dumps(fn).replace('"type": "dict"', '"type": "object"'))
    return s


def iter_bfcl_examples(limit: int, mix=None):
    """Yield unified records from the BFCL v4 single-turn categories.
    Each: dict(tools, user, gt_anyof, expect_call, category, tool)."""
    n = 0
    for stem, expect_call, cap in (mix or BFCL_MIX):
        qfile = BFCL_DIR / f"{stem}.json"
        if not qfile.exists():
            continue
        answers = {}
        afile = BFCL_DIR / "possible_answer" / f"{stem}.json"
        if afile.exists():
            for line in open(afile, encoding="utf-8"):
                if line.strip():
                    a = json.loads(line)
                    answers[a["id"]] = a["ground_truth"]
        count = 0
        for line in open(qfile, encoding="utf-8"):
            if count >= cap:
                break
            if not line.strip():
                continue
            q = json.loads(line)
            turns = q["question"]
            first_turn = turns[0] if isinstance(turns[0], list) else turns
            user_msgs = [m["content"] for m in first_turn if m["role"] == "user"]
            if not user_msgs:
                continue
            gt = answers.get(q["id"])
            if expect_call and gt is None:
                continue
            tools = [_fix_schema(f) for f in q["function"]]
            tool_group = (next(iter(gt[0])) if gt else
                          f"irr::{tools[0]['name']}" if tools else "irr::none")
            yield {
                "tools": tools,
                "user": " ".join(user_msgs),
                "gt_anyof": gt,
                "expect_call": expect_call,
                "category": stem.replace("BFCL_v4_", ""),
                "tool": tool_group,
            }
            count += 1
            n += 1
            if n >= limit:
                return


TOOL_PROMPT_FALLBACK = (
    "You are a helpful assistant with access to the following functions.\n"
    "When a function is needed, reply with ONLY a JSON object of the form\n"
    '{{"name": <function name>, "arguments": {{<arg name>: <value>}}}}\n'
    "and nothing else.\n\nAvailable functions:\n{schemas}"
)


def _apply_template(tok, msgs, tools=None):
    """
    apply_chat_template with chain-of-thought disabled where the template
    supports it.

    Qwen3 templates default to thinking mode: the model emits a long
    <think> trace before any tool call, so with a fixed generation budget
    the call is frequently never reached and the example is labelled
    `no_call` — a measurement of reasoning length, not of tool-call
    quality, and not comparable to the non-thinking models in the sweep
    (audit 2026-09-09). Templates that do not accept the flag are
    unaffected.
    """
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if tools is not None:
        kwargs["tools"] = tools
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)


def render_tool_prompt(tok, tools, user):
    """
    Render a tool-calling prompt and VERIFY that the tool schemas actually
    reached it.

    Some chat templates silently ignore the ``tools=`` argument -- notably
    google/gemma-3-*-it. The model then never sees a tool, answers in prose,
    and every example is labelled ``no_call``: 678 such records were produced
    before this guard existed (audit 2026-09-09). When the native template
    drops the tools we fall back to an explicit system-prompt specification.
    If the tool names still do not appear we return None, so an example whose
    label would be an artefact of prompt rendering is never emitted.
    """
    names = [t.get("name", "") for t in tools if t.get("name")]
    try:
        text = _apply_template(
            tok,
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": user}],
            tools=tools)
    except Exception:
        text = None
    if text is not None and all(n in text for n in names):
        return text

    schemas = "\n".join(json.dumps(t) for t in tools)
    sys_msg = TOOL_PROMPT_FALLBACK.format(schemas=schemas)
    for msgs in ([{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user}],
                 [{"role": "user", "content": sys_msg + "\n\n" + user}]):
        try:
            text = _apply_template(tok, msgs)
        except Exception:
            continue
        if all(n in text for n in names):
            return text
    return None


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

    if getattr(args, "benchmark", "glaive") == "bfcl":
        source = iter_bfcl_examples(args.n)
    elif getattr(args, "benchmark", "glaive") == "bfcl_live":
        source = iter_bfcl_examples(args.n, mix=BFCL_LIVE_MIX)
    else:
        def _glaive_source():
            for system, user, gt in iter_call_examples(args.n):
                yield {"tools": extract_glaive_tools(system), "user": user,
                       "gt_text": gt, "gt_anyof": None, "expect_call": True,
                       "category": "glaive", "tool": None}
        source = _glaive_source()

    kept, mode_counts, skipped_prompt = 0, {}, 0
    pbar = tqdm(source, total=args.n, desc="extract")
    for ex in pbar:
        tools, user = ex["tools"], ex["user"]
        if not tools:
            continue
        prompt_text = render_tool_prompt(tok, tools, user)
        if prompt_text is None:
            skipped_prompt += 1
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
        # Decode WITHOUT dropping special tokens, then remove only the chat
        # control markers. Some families (MiniCPM) tokenise the structural
        # tags of a tool call as special tokens, so skipping specials deletes
        # "<function" and "<param" and leaves an unparseable fragment that is
        # then labelled as no call (audit 2026-09).
        pred = strip_control_markers(
            tok.decode(gen_ids, skip_special_tokens=False), tok)
        # Truncation is length against the budget. Testing for eos_token_id is
        # wrong for families that close a tool call with a different marker,
        # such as Llama's <|eom_id|>, which flagged almost every generation.
        truncated = len(gen_ids) >= MAX_NEW_TOKENS
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
            if ex["gt_anyof"] is not None or not ex["expect_call"]:
                label, mode = classify_failure_anyof(
                    pred, ex["gt_anyof"] or [], ex["expect_call"])
                gt_repr = json.dumps(ex["gt_anyof"])
            else:
                label, mode = classify_failure(pred, ex["gt_text"])
                gt_repr = ex["gt_text"]
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
        lapeig_diag, lookback = [], []
        for attn in mo.attentions:               # full-attention layers only
            a = attn[0]                          # [H, T, T]
            layer_diagnostics.append(layer_spectral_metrics(a))
            layer_diagnostics_span.append(layer_spectral_metrics(a, span=span))
            if args.rich:
                eig_profile.append(laplacian_eig_profile(a))
                eig_profile_span.append(laplacian_eig_profile(a, span=span))
                head_fiedler_span.append(per_head_metrics(a, span=span))
                lapeig_diag.append(lapeigvals_diag_profile(a))
                lookback.append(lookback_ratio(a, prompt_len, T))

        pos = find_token_positions_v2(tok, gen_ids, prompt_len)
        # Per-token residual states over the generated span at one mid-late
        # depth, for the token-level probe baseline (Obeso et al., 2025),
        # capped in length and stored at reduced precision to keep the dump
        # tractable.
        token_states, token_layer = None, None
        if args.rich:
            token_layer = max(1, int(round(0.7 * n_layers)))
            span_states = mo.hidden_states[token_layer][0, prompt_len:T]
            # float32, not float16: Llama-class models carry a massive
            # activation in one or two dimensions whose magnitude exceeds the
            # float16 range, which would be stored as inf.
            span_states = span_states[:TOKEN_STATE_CAP].to(torch.float32)
            token_states = span_states.cpu().numpy().round(3).tolist()
        hidden, gram_feats = {}, {}
        for li in probe_layers:
            h = mo.hidden_states[li][0].to(torch.float32).cpu()
            hidden[str(li)] = extract_probe_features(h, pos).tolist()
            if args.rich:
                gram_feats[str(li)] = gram_spectrum_features(
                    mo.hidden_states[li][0][prompt_len:T])

        res_dyn = residual_dynamics(mo.hidden_states, prompt_len, T)             if args.rich else None

        del mo, out
        torch.cuda.empty_cache()

        rec = {
            "prompt_hash": prompt_hash,
            "user": user,
            "label": int(label),
            "failure_mode": mode,
            "prediction": pred,
            "ground_truth": gt_repr,
            "gt_anyof": ex["gt_anyof"],
            "expect_call": ex["expect_call"],
            "category": ex["category"],
            "tool": ex["tool"],
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
            # [L][H][2] Lookback Lens context/generation attention shares
            rec["lookback"] = lookback
            # [L-1][4] cross-layer residual-stream dynamics (ICR-style)
            rec["res_dynamics"] = res_dyn
            # per-token residual states at one depth, for a token-level probe
            rec["token_states"] = token_states
            rec["token_state_layer"] = token_layer
        with open(FEATURES, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        # Record the hash: the benchmark streams repeat prompts (Glaive
        # heavily so), and without this a repeat within the same run is
        # written again -- 13% of each Glaive dump before this fix.
        done.add(prompt_hash)
        kept += 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        pbar.set_postfix(kept=kept, mode=mode, halluc=f"{label}")
        if kept >= args.n:
            break

    print(f"\n[extract] DONE: {kept} samples -> {FEATURES}")
    print(f"[extract] failure modes this run: {mode_counts}")
    if skipped_prompt:
        print(f"[extract] WARNING: {skipped_prompt} examples skipped -- "
              f"tool schemas could not be rendered into the prompt")


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


def token_probe_matrix(samples):
    """
    Token-level probe features (Obeso et al., 2025): residual states of the
    generated tokens at one mid-late depth. Their probe is trained per token
    and the call-level score is the maximum over tokens; here the call-level
    feature is the concatenation of the mean and the max over tokens, which a
    linear model can read as the same pooled statistic without needing
    token-level labels that this benchmark does not provide.
    """
    dim = None
    for s in samples:
        tp = s.get("token_pooled")
        if tp is not None:
            dim = len(tp)
            break
    if dim is None:
        return None
    out = np.zeros((len(samples), dim), dtype=np.float32)
    for i, s in enumerate(samples):
        tp = s.get("token_pooled")
        if tp is not None:
            out[i] = tp
    return out


def residual_dynamics_matrix(samples):
    """Cross-layer residual-stream dynamics (ICR-style)."""
    if not samples[0].get("res_dynamics"):
        return None
    return np.array([np.ravel(s["res_dynamics"]) for s in samples],
                    dtype=np.float32)


def lookback_matrix(samples):
    """Lookback Lens: per-head context/generation attention shares."""
    if not samples[0].get("lookback"):
        return None
    return np.array([np.ravel(s["lookback"]) for s in samples],
                    dtype=np.float32)


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


def fit_lr(X, y, tr, va, c_grid=(0.01, 0.1, 1.0, 10.0)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    best, best_v = None, -1
    for C in c_grid:
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


SEMANTIC_MODES = ["valid", "wrong_name", "missing_args", "wrong_arg_values",
                  "missing_calls", "over_trigger", "valid_nocall"]


def _compact(rec):
    """Replace list-of-float fields with compact arrays, in place.

    Per-token residual states are reduced to the mean-and-max pooling that
    the token-level probe consumes, so the raw states are not retained.
    """
    ts = rec.get("token_states")
    if ts:
        a = np.asarray(ts, dtype=np.float32)
        if not np.isfinite(a).all():
            a = np.nan_to_num(a, nan=0.0, posinf=65504.0, neginf=-65504.0)
        rec["token_pooled"] = np.concatenate([a.mean(0), a.max(0)])
        rec["token_states"] = None
    for key in ("lapeig_diag", "head_metrics_span", "eig_profile",
                "eig_profile_span", "lookback", "res_dynamics"):
        if rec.get(key) is not None:
            rec[key] = np.asarray(rec[key], dtype=np.float32)
    return rec


def load_and_relabel(features_path):
    """Load a feature dump, keep newest schema, re-label from stored text
    with the CURRENT labeler, attach tool names. Returns (samples, changed)."""
    with open(features_path, encoding="utf-8") as f:
        samples = [_compact(json.loads(line)) for line in f if line.strip()]

    if any("head_metrics_span" in s for s in samples):
        n0 = len(samples)
        samples = [s for s in samples if "head_metrics_span" in s]
        if len(samples) < n0:
            print(f"[load] dropped {n0 - len(samples)} stale-schema records")

    # Deduplicate by prompt: repeated prompts were written more than once by
    # runs predating the extraction fix above. The rows are identical, so
    # keeping the first occurrence is lossless; leaving them in silently
    # up-weights those examples.
    seen, unique = set(), []
    for s in samples:
        if s["prompt_hash"] in seen:
            continue
        seen.add(s["prompt_hash"])
        unique.append(s)
    if len(unique) < len(samples):
        print(f"[load] deduplicated {len(samples) - len(unique)} repeated "
              f"prompts ({len(unique)} unique remain)")
    samples = unique

    relabel_changes = 0
    for s in samples:
        old = s["label"]
        # Repair dumps written before the marker and truncation fixes.
        s["prediction"] = strip_control_markers(s["prediction"])
        if s.get("gen_tokens") is not None:
            s["truncated"] = bool(s["gen_tokens"] >= MAX_NEW_TOKENS)
        try:
            if s.get("gt_anyof") is not None or s.get("expect_call") is False:
                s["label"], s["failure_mode"] = classify_failure_anyof(
                    s["prediction"], s.get("gt_anyof") or [],
                    s.get("expect_call", True))
            else:
                s["label"], s["failure_mode"] = classify_failure(
                    s["prediction"], s["ground_truth"])
            # A call that does not parse because our generation budget cut it
            # off is an artefact of the measurement, not a model failure, so
            # it gets its own mode and is excluded from the semantic subset.
            if s["failure_mode"] == "unparseable_call" and s.get("truncated"):
                s["failure_mode"] = "truncated_call"
        except ValueError:
            pass
        if not s.get("tool"):
            gt_calls, _ = extract_calls(s["ground_truth"])
            s["tool"] = gt_calls[0]["name"] if gt_calls else "?"
        relabel_changes += (s["label"] != old)
    return samples, relabel_changes


def handle_evaluate(args):
    from spectral_guardrails.probes.gbt import train_lmm_gbt, predict_lmm

    samples, relabel_changes = load_and_relabel(FEATURES)

    y = np.array([s["label"] for s in samples])
    modes = np.array([s["failure_mode"] for s in samples])
    tools = np.array([s["tool"] for s in samples])
    semantic = np.isin(modes, SEMANTIC_MODES)

    # BFCL dumps carry an irrelevance category (expect_call=False); keep a
    # mask of the call-expected population for a confound-free subset score.
    if any("expect_call" in s for s in samples):
        expect_call_mask = np.array([bool(s.get("expect_call", True))
                                     for s in samples])
        if expect_call_mask.all():
            expect_call_mask = None
    else:
        expect_call_mask = None

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
    extra_rows = []
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

            # Dynamics: layer-to-layer deltas of each head's lambda_max —
            # cross-layer evolution that flat per-layer feature sets
            # (e.g. LapEigvals) do not represent explicitly.
            lmax = hm[:, :, :, 4]                      # (N, L, H)
            if lmax.shape[1] > 1:
                dyn = np.diff(lmax, axis=1).reshape(N, -1)
                X["Per-head lambda_max dynamics"] = dyn
                X["Per-head lambda_max static+dynamics"] = np.hstack(
                    [lmax.reshape(N, -1), dyn])
                per_head_rows += ["Per-head lambda_max dynamics",
                                  "Per-head lambda_max static+dynamics"]
        else:  # legacy dumps: fiedler only
            row = "Per-head fiedler_value (span)"
            X[row] = np.array([np.ravel(s["head_fiedler_span"])
                               for s in samples], dtype=np.float32)
            per_head_rows.append(row)
            per_head_combined = X[row]

        for name, build in (
                ("Token-level probe (Obeso)", token_probe_matrix),
                ("Residual dynamics (ICR-style)", residual_dynamics_matrix),
                ("Lookback Lens", lookback_matrix)):
            M = build(samples)
            if M is not None:
                X[name] = M
                extra_rows.append(name)

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
                    + per_head_rows + extra_rows
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
        results.setdefault(
            name, {"all": [], "semantic": [], "call_expected": []}
        )[subset].append(val)

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
            # BFCL mixes call-expected categories with the irrelevance
            # category. On irrelevance items the label is "called a tool at
            # all", which any call-presence feature predicts trivially, so a
            # mixed-pool AUC is inflated. Score the call-expected subset
            # separately (this is the semantically hard population).
            if expect_call_mask is not None:
                ok_ec = ok_sem & expect_call_mask
                add(name, "call_expected", auc_safe(y[ok_ec], scores[ok_ec]))

    # ── report ─────────────────────────────────────────────────────────────────
    n_pos_sem = int(y[semantic].sum())
    has_ec = expect_call_mask is not None
    if has_ec:
        ec_sem = semantic & expect_call_mask
        hdr_extra = f"{'AUC (call-expected)':>22}"
        info_extra = (f"; call-expected: {int(ec_sem.sum())} samples / "
                      f"{int(y[ec_sem].sum())} pos")
    else:
        hdr_extra, info_extra = "", ""
    width = 92 + (22 if has_ec else 0)
    lines = ["", "=" * width,
             f"Pooled cross-fit AUC over N={len(samples)} "
             f"({int(y.sum())} pos; semantic: {n_pos_sem} pos{info_extra}), "
             f"tool-level folds, {len(seeds)} seeds",
             "=" * width,
             f"{'Detector':<38} {'AUC (all)':>18} {'AUC (semantic only)':>24}"
             + hdr_extra,
             "-" * width]
    for name, r in results.items():
        def fmt(v):
            v = [x for x in v if not np.isnan(x)]
            if not v:
                return "n/a"
            return f"{np.mean(v):.3f} ± {np.std(v):.3f}"
        row = f"{name:<38} {fmt(r['all']):>18} {fmt(r['semantic']):>24}"
        if has_ec:
            row += f"{fmt(r['call_expected']):>22}"
        lines.append(row)
    lines.append("=" * width)
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


def handle_transfer(args):
    """
    Cross-dataset transfer: train detectors on one feature dump, evaluate on
    another (same model, different benchmark). The question a practitioner
    asks: do I need to relabel for my domain, or does a detector trained on
    a generic tool-calling set transfer zero-shot?
    """
    tr_samples, _ = load_and_relabel(Path(f"data/pilot_v2_{args.train_tag}/features.jsonl"))
    te_samples, _ = load_and_relabel(Path(f"data/pilot_v2_{args.test_tag}/features.jsonl"))

    def build(samples):
        N = len(samples)
        hm = np.array([s["head_metrics_span"] for s in samples], dtype=np.float32)
        return {
            "y": np.array([s["label"] for s in samples]),
            "semantic": np.isin(np.array([s["failure_mode"] for s in samples]),
                                SEMANTIC_MODES),
            "Hidden token-role [LR]": hidden_matrix(samples),
            "Per-head all metrics (span)": hm.reshape(N, -1),
            "Per-head lambda_max (span)": hm[:, :, :, 4].reshape(N, -1),
            "lap": np.array([s["lapeig_diag"] for s in samples], dtype=np.float32),
            "Surface (lengths) [confound]": surface_matrix(samples),
            "logprob": np.nan_to_num(
                np.array([s["mean_logprob"] for s in samples]), nan=0.0),
        }

    TR, TE = build(tr_samples), build(te_samples)
    y_tr, y_te = TR["y"], TE["y"]
    print(f"\ntrain={args.train_tag} (N={len(y_tr)}, pos={int(y_tr.sum())})  "
          f"test={args.test_tag} (N={len(y_te)}, pos={int(y_te.sum())})")

    rows = {}
    for seed in [42, 43, 44]:
        itr, iva, _ = group_split(tr_samples, seed, train_ratio=0.85,
                                  val_ratio=0.15, key="tool")
        for name in ["Hidden token-role [LR]", "Per-head all metrics (span)",
                     "Per-head lambda_max (span)",
                     "Surface (lengths) [confound]"]:
            p = fit_lr(TR[name], y_tr, itr, iva)
            sc = p.predict_proba(TE[name])[:, 1]
            rows.setdefault(name, {"all": [], "semantic": []})
            rows[name]["all"].append(auc_safe(y_te, sc))
            rows[name]["semantic"].append(
                auc_safe(y_te[TE["semantic"]], sc[TE["semantic"]]))
        best_scores, best_val = None, -1.0
        for k in [5, 10, 25, 50, 100]:
            Xk_tr = TR["lap"][:, :, :, :k].reshape(len(y_tr), -1)
            Xk_te = TE["lap"][:, :, :, :k].reshape(len(y_te), -1)
            p = fit_lr(Xk_tr, y_tr, itr, iva)
            v = auc_safe(y_tr[iva], p.predict_proba(Xk_tr[iva])[:, 1])
            if not np.isnan(v) and v > best_val:
                best_val = v
                best_scores = p.predict_proba(Xk_te)[:, 1]
        rows.setdefault("LapEigvals (official code)", {"all": [], "semantic": []})
        rows["LapEigvals (official code)"]["all"].append(auc_safe(y_te, best_scores))
        rows["LapEigvals (official code)"]["semantic"].append(
            auc_safe(y_te[TE["semantic"]], best_scores[TE["semantic"]]))

    sign = 1.0 if auc_safe(y_tr, -TR["logprob"]) >= 0.5 else -1.0
    rows["Mean logprob"] = {
        "all": [auc_safe(y_te, sign * -TE["logprob"])],
        "semantic": [auc_safe(y_te[TE["semantic"]],
                              sign * -TE["logprob"][TE["semantic"]])]}

    print("=" * 88)
    print(f"{'Detector':<38} {'AUC (all)':>18} {'AUC (semantic)':>20}")
    print("-" * 88)
    for name, r in rows.items():
        def fmt(v):
            v = [x for x in v if not np.isnan(x)]
            return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "n/a"
        print(f"{name:<38} {fmt(r['all']):>18} {fmt(r['semantic']):>20}")
    print("=" * 88)

    out = Path(f"data/pilot_v2_{args.test_tag}/transfer_from_{args.train_tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                   for k, v in rows.items()}, f, indent=2)
    print(f"saved -> {out}")


def main():
    global OUT_DIR, FEATURES
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", default="Qwen/Qwen3.5-2B")
    e.add_argument("--n", type=int, default=750)
    e.add_argument("--fresh", action="store_true")
    e.add_argument("--benchmark", choices=["glaive", "bfcl", "bfcl_live"],
                   default="glaive")
    e.add_argument("--rich", action="store_true", default=True,
                   help="also dump eigenvalue profiles, per-head Fiedler, "
                        "hidden-state Gram spectra")
    e.add_argument("--no-rich", dest="rich", action="store_false")
    e.add_argument("--tag", default=None,
                   help="output subdir tag (default: derived from model id)")
    v = sub.add_parser("evaluate")
    v.add_argument("--tag", default="qwen35_2b")
    t = sub.add_parser("transfer")
    t.add_argument("--train-tag", required=True)
    t.add_argument("--test-tag", required=True)
    args = ap.parse_args()
    if args.cmd == "transfer":
        handle_transfer(args)
        return
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
