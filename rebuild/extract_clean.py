"""The clean extractor (docs/PIPELINE_REBUILD.md). One extractor, one commit, one storage layout.

  python -m rebuild.extract_clean --model <hf id> --benchmark <adapter> --tag <tag> --pin <commit>

Refuses to run unless the working tree is clean and the code equals the pinned commit
(docs/ and results/ excepted). `--allow-dirty` exists for the CPU smoke test only: the run
is then marked `smoke: true, git_dirty: <actual>` and the loader refuses it by default.

Fixed, identical across arms, recorded in run_meta.json: greedy decoding, 256 new tokens,
2048 prompt-token cap, bfloat16, eager attention, deterministic kernels, reasoning mode off,
pinned template date, seed 42. The prompt route is decided per run (rebuild/prompts.py).

Per item (rebuild/storage.py): the greedy call text and labels; per-token log-probabilities
and entropies with the token roles marked; hidden states at the role positions at the stored
layer set; per-head spectra (call span and whole sequence), head-averaged spectra, the
anchored readout as row role x key span x layer x head, tool-segment masses, LapEigvals,
SinkProbe, Lookback; P(True). With --resample K (default 8, registered): K sampled
generations per call-expected item, labelled and classified (rebuild/resample.py), and the
same features for the kept success and failure samples of within-reach items.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
try:
    import pyarrow  # noqa: F401  (import-order workaround, see run_pilot_v2.py)
except ImportError:
    pass
from spectral_guardrails.utils import determinism  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch)
import torch  # noqa: E402

from rebuild import benchmarks, prompts, spans, labels, attention, storage, ptrue, resample  # noqa: E402
from run_pilot_v2 import stopping_ids  # noqa: E402  (reused: every id that ends an assistant turn)

SEED = 42
MAX_NEW_TOKENS = 256
MAX_PROMPT_TOKENS = 2048
FULL_GRAPH_MAX_TOKENS = 1500     # head-averaged spectra on the whole sequence up to this length
FULL_HEAD_MAX_TOKENS = 384       # per-head spectra on the whole sequence up to this length (cost: H x La eigs)
N_PROBE_DEPTHS = 8
EXIT_REFUSED = 3
DECODING = {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "num_beams": 1, "temperature": None, "top_p": None, "repetition_penalty": 1.0,
            "attn_implementation": "eager", "thinking_mode": False, "template_date": prompts.TEMPLATE_DATE}


def git(*a) -> str:
    return subprocess.check_output(["git", *a], text=True, cwd=ROOT, stderr=subprocess.DEVNULL).strip()


def _tracked(path: str) -> bool:
    try:
        git("rev-parse", "--verify", "-q", f"HEAD:{path}")
        return True
    except subprocess.CalledProcessError:
        return False


def tree_state(pin: str | None) -> dict:
    status = git("status", "--porcelain")
    st = {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(status),
          "git_status_head": status.splitlines()[:20], "code_pin": pin,
          "code_hash": {p: git("rev-parse", f"HEAD:{p}") for p in ("rebuild", "spectral_guardrails")
                        if (ROOT / p).exists() and _tracked(p)}}
    if pin:
        try:
            diff = git("diff", "--name-only", pin, "HEAD", "--", ".", ":(exclude)docs", ":(exclude)results")
            st["code_differs_from_pin"] = diff.splitlines()
            st["pin_is_commit"] = True
        except subprocess.CalledProcessError:        # the pin is not a commit of this repository
            st["code_differs_from_pin"] = [f"<pin {pin} is not a commit>"]
            st["pin_is_commit"] = False
    return st


def require_clean(pin: str | None, allow_dirty: bool) -> dict:
    st = tree_state(pin)
    if allow_dirty:
        print(f"[extract] WARNING smoke mode: tree dirty={st['git_dirty']}; run is marked smoke")
        return st
    if st["git_dirty"]:
        print("[extract] REFUSED: working tree not clean:\n  " + "\n  ".join(st["git_status_head"]))
        raise SystemExit(EXIT_REFUSED)
    if not pin:
        print("[extract] REFUSED: no --pin / REBUILD_PIN (the commit every run must come from)")
        raise SystemExit(EXIT_REFUSED)
    if st["code_differs_from_pin"]:
        print(f"[extract] REFUSED: code differs from pin {pin}: {st['code_differs_from_pin']}")
        raise SystemExit(EXIT_REFUSED)
    return st


def model_revision(model_id: str) -> str | None:
    try:
        from transformers.utils.hub import cached_file
        p = Path(cached_file(model_id, "config.json"))
        return p.parent.name if p.parent.parent.name == "snapshots" else None
    except Exception:
        return None


def versions() -> dict:
    import transformers
    out = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "transformers": transformers.__version__, "numpy": np.__version__}
    for mod in ("spectral_trust", "tokenizers", "datasets", "sklearn", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def head_identity(cfg, attn_depths: list[int]) -> dict:
    """Head identities incl. the KV group of every query head (GQA), per attention layer."""
    H = int(getattr(cfg, "num_attention_heads", 0) or 0)
    Hkv = int(getattr(cfg, "num_key_value_heads", H) or H)
    group = (H // Hkv) if (H and Hkv) else 1
    return {"num_heads": H, "num_kv_heads": Hkv, "heads_per_kv_group": group,
            "kv_group_of_head": [h // group for h in range(H)], "attn_depths": attn_depths,
            "note": "head index h of every stored (La, H, ...) array is the query head; its KV group is kv_group_of_head[h]"}


def hidden_layer_set(n_layers: int, probe_depths: list[int], mode: str) -> list[int]:
    if mode == "all":
        return list(range(1, n_layers + 1))
    return sorted(set(range(2, n_layers + 1, 2)) | {n_layers} | set(probe_depths))


def prompt_offsets(tok, text: str) -> list[tuple[int, int]]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(o) for o in enc["offset_mapping"]]


def step_entropies(scores, row: int = 0) -> np.ndarray:
    return np.array([float(-(torch.softmax(s[row].float(), -1) * torch.log_softmax(s[row].float(), -1)).sum())
                     for s in scores], dtype=np.float32)


def confidence_summaries(logp: np.ndarray, ent: np.ndarray | None, token_role: np.ndarray, value_id: np.ndarray,
                         gen_ids: list[int], special_ids: set, p_true: float | None) -> dict:
    G = len(gen_ids)
    fin = np.isfinite(logp)
    vmask = (value_id >= 0) & fin
    smt = np.isin(token_role, [1, 2]) & fin                           # name and value tokens
    call_tok = [i for i in range(G) if token_role[i] > 0]
    first_c, last_c = (min(call_tok), max(call_tok)) if call_tok else (0, G - 1)
    cmask = np.zeros(G, bool)
    cmask[first_c:last_c + 1] = True
    cmask &= fin & np.array([g not in special_ids for g in gen_ids])
    f = lambda m, fn: (float(fn(logp[m])) if m.any() else None)
    out = {"gnll": f(fin, lambda x: -x.sum()), "gnll_smt": f(smt, lambda x: -x.sum()),
           "nll_max": f(fin, lambda x: -x.min()), "p_true": p_true,
           "mean_logprob": f(fin, np.mean), "min_logprob": f(fin, np.min), "call_mean_logprob": f(cmask, np.mean),
           "value_mean_logprob": f(vmask, np.mean), "value_min_logprob": f(vmask, np.min),
           "n_value_tokens": int(vmask.sum()), "gen_tokens": G}
    if ent is not None and len(ent):
        out.update(mean_entropy=float(ent.mean()), max_entropy=float(ent.max()), entropy_last=float(ent[-1]),
                   value_mean_entropy=float(ent[vmask[:len(ent)]].mean()) if vmask[:len(ent)].any() else None)
    return out


class Extractor:
    """Holds the model and runs the teacher-forced feature pass for any generation of an item."""

    def __init__(self, model, tok, adapter, route: str, device: str, hidden_layers: list[int]):
        self.model, self.tok, self.adapter, self.route, self.device = model, tok, adapter, route, device
        self.hidden_layers = hidden_layers
        self.special_ids = set(tok.all_special_ids)
        self.stop_ids = stopping_ids(tok, model)

    def featurize(self, ex: dict, prompt_text: str, prompt_ids: torch.Tensor, gen_ids: list[int], logp: np.ndarray,
                  ent: np.ndarray | None, with_ptrue: bool) -> tuple[dict, dict]:
        """Everything stored for one generation of one item: (arrays, record fields)."""
        tok, model = self.tok, self.model
        G = len(gen_ids)
        prompt_len = int(prompt_ids.shape[1])
        truncated = G >= MAX_NEW_TOKENS
        raw_text = tok.decode(gen_ids, skip_special_tokens=False)
        pred = labels.clean_prediction(raw_text, tok)
        lab = labels.label_item(self.adapter, ex, pred, truncated)
        roles = spans.role_positions(tok, gen_ids)
        token_role = np.array(roles["token_role"], dtype=np.int8)
        value_id = np.array(roles["value_id"], dtype=np.int16)
        value_toks = [p[4] for p in roles["positions"] if p[0] == "value"] if roles["found"] else []

        cs = prompts.prompt_char_spans(prompt_text, ex["tools"], ex["user"])
        offs = prompt_offsets(tok, prompt_text)
        tspans = {k: prompts.char_to_token_span(offs, v) for k, v in cs.items()}
        tool_spans_tok = [prompts.char_to_token_span(offs, sp)
                          for sp in prompts.tool_segment_spans(prompt_text, ex["tools"], cs["schema"])]
        tool_names = [t.get("name") for t in ex["tools"]]
        gold_tool_index = tool_names.index(ex["tool"]) if ex["tool"] in tool_names else -1
        tspans["system"] = (0, tspans["schema"][0]) if tspans.get("schema") and tspans["schema"][0] > 0 else None

        full_ids = torch.cat([prompt_ids, torch.tensor([gen_ids], device=prompt_ids.device)], 1)
        T = int(full_ids.shape[1])
        full_graph = T <= FULL_GRAPH_MAX_TOKENS
        full_head = T <= FULL_HEAD_MAX_TOKENS
        rows = {"name": [prompt_len + t for c in roles["calls"] for t in c["name_toks"]],
                "value": [prompt_len + t for toks in value_toks for t in toks],
                "close": [prompt_len + t for c in roles["calls"] for t in c["close_toks"]],
                "last": [T - 1], "gen": list(range(prompt_len, T))}
        if not roles["found"]:
            rows.update(name=[T - 1], value=[T - 1], close=[T - 1])
        t1 = time.time()

        def reducer(depth, w):
            return attention.reduce_layer(w, prompt_len, T, rows, value_toks, tspans, tool_spans_tok,
                                          gold_tool_index, full_graph, full_head)
        with torch.no_grad(), attention.StreamedReducer(model, reducer) as stream:
            mo = model(input_ids=full_ids, output_hidden_states=True, use_cache=False)
        depths = stream.depths()
        feats = [stream.results[dp] for dp in depths]
        hs = torch.stack([mo.hidden_states[li][0] for li in self.hidden_layers], 0)      # (Lh, T, d)
        P = roles["positions"]
        cols = []
        for role, ci, vi, tk, toks in P:
            idx = [prompt_len + t for t in toks if prompt_len + t < T] or [T - 1]
            cols.append(hs[:, idx, :].float().mean(1) if len(idx) > 1 else hs[:, idx[0], :].float())
        hid = torch.stack(cols, 1).cpu().numpy()                        # (Lh, P, d)
        del mo, hs
        if self.device != "cpu":
            torch.cuda.empty_cache()
        t_pass = time.time() - t1
        t2 = time.time()
        p_true = (ptrue.p_true(model, tok, ex, pred, self.route, max_prompt_tokens=MAX_PROMPT_TOKENS)
                  if with_ptrue else None)
        t_ptrue = time.time() - t2

        hid_c, hid_dtype = storage.compact(hid)
        stack = lambda k: np.stack([f[k] for f in feats]) if feats and k in feats[0] else np.full((0,), np.nan)
        arrays = {
            "gen_ids": np.array(gen_ids, dtype=np.int32), "logp": logp.astype(np.float32),
            "token_role": token_role, "value_id": value_id, "hid": hid_c,
            "pos_role": np.array([p[0] for p in P]), "pos_call": np.array([p[1] for p in P], dtype=np.int16),
            "pos_value": np.array([p[2] for p in P], dtype=np.int16), "pos_tok": np.array([p[3] for p in P], dtype=np.int32),
            "attn_depths": np.array(depths, dtype=np.int32),
            "sink_top_pos": np.stack([f["sink_top_pos"] for f in feats]).astype(np.int32),
        }
        if ent is not None:
            arrays["entropy"] = ent.astype(np.float16)
        for k in ("hspec", "lspec_span", "anch", "anch_stat", "anch_each", "lapeig", "sink", "lookback", "tool_mass"):
            arrays[k] = storage.compact(stack(k))[0]
        if full_graph:
            arrays["lspec_full"] = storage.compact(stack("lspec_full"))[0]
        if full_head:
            arrays["hspec_full"] = storage.compact(stack("hspec_full"))[0]
        conf = confidence_summaries(logp, ent, token_role, value_id, gen_ids, self.special_ids, p_true)
        rec = {
            "prompt_tokens": prompt_len, "gen_tokens": G, "seq_len": T, "truncated": bool(truncated),
            "prediction": pred, "prediction_raw": raw_text, **lab, "confidence": conf,
            "roles_found": roles["found"], "n_calls_produced": len(roles["calls"]), "n_calls_expected": ex["n_gt_calls"],
            "n_values_total": roles["n_values_total"], "roles_capped": roles["capped"],
            "call_roles_tok": [{"name": c["name"], "name_toks": c["name_toks"], "args_toks": c["args_toks"],
                                "keys": c["value_keys"], "key_toks": c["key_toks"], "value_toks": c["value_toks"],
                                "close_toks": c["close_toks"]} for c in roles["calls"]],
            "dialect": roles["calls"][0]["dialect"] if roles["calls"] else None,
            "prompt_spans_tok": tspans, "prompt_spans_ok": {k: v is not None for k, v in tspans.items()},
            "tool_spans_tok": tool_spans_tok, "gold_tool_index": gold_tool_index,
            "full_graph": full_graph, "full_head_spectra": full_head, "n_attn_layers": len(depths),
            "dtypes": {"hid": hid_dtype}, "t_pass_s": round(t_pass, 3), "t_ptrue_s": round(t_ptrue, 3),
        }
        return arrays, rec

    @torch.no_grad()
    def greedy(self, inputs):
        out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id, eos_token_id=self.stop_ids,
                                  return_dict_in_generate=True, output_scores=True)
        P = inputs.input_ids.shape[1]
        gen_ids = out.sequences[0][P:].tolist()
        if not gen_ids:
            return [], None, None
        ts = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0].float().cpu()
        ent = step_entropies(out.scores)
        del out
        return gen_ids, ts.numpy().astype(np.float32), ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", required=True, choices=benchmarks.names())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=850)
    ap.add_argument("--route", choices=["auto", "native", "fallback_list"], default="auto")
    ap.add_argument("--pin", default=os.environ.get("REBUILD_PIN"))
    ap.add_argument("--allow-dirty", action="store_true", help="smoke test only")
    ap.add_argument("--device", default=os.environ.get("EXTRACT_DEVICE", "cuda"))
    ap.add_argument("--dtype", default=os.environ.get("EXTRACT_DTYPE", "bfloat16"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resample", type=int, default=resample.K, help="K sampled generations per call-expected item (0 = off)")
    ap.add_argument("--temperature", type=float, default=resample.TEMPERATURE)
    ap.add_argument("--hidden-layers", choices=["half", "all"], default="half",
                    help="half: every 2nd block + the last + the 8 registered depths (registered default); all: every block")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--hf-online", action="store_true", help="allow a download (gated weights, a dataset)")
    a = ap.parse_args()
    if not a.hf_online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    t_start = time.time()
    tree = require_clean(a.pin, a.allow_dirty)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    det = determinism.enable_deterministic_torch(warn_only=True)

    adapter = benchmarks.get(a.benchmark)
    items = adapter.load(a.n)
    print(f"[extract] {a.benchmark}: {len(items)} items")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    route = prompts.decide_route(tok, items, None if a.route == "auto" else a.route)
    print(f"[extract] route {route['prompt_route']} ({route['render_check']})")

    rendered, dropped = [], {"unrenderable": 0, "over_prompt_cap": 0}
    for ex in items:
        text, detail = prompts.render(tok, ex["tools"], ex["user"], route["prompt_route"])
        if text is None:
            dropped["unrenderable"] += 1
            continue
        n_tok = len(tok(text, add_special_tokens=False).input_ids)
        if n_tok > MAX_PROMPT_TOKENS:
            dropped["over_prompt_cap"] += 1
            continue
        rendered.append((ex, text, detail, n_tok))
    item_ids = [ex["item_id"] for ex, *_ in rendered]
    prompt_hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for _, t, *_ in rendered]

    d = storage.run_dir(a.tag)
    if a.fresh and d.exists():
        import shutil
        shutil.rmtree(d)
    done = set()
    if (d / "items.jsonl").exists():
        storage.repair_items(a.tag)
        done = {r["item_id"] for r in storage.read_items(a.tag)}
        print(f"[resume] {len(done)} items already extracted")
        old = storage.read_meta(a.tag)
        if old.get("item_digest") != benchmarks.items_digest(item_ids):
            print("[extract] REFUSED: item set differs from the partial run's run_meta.json")
            raise SystemExit(EXIT_REFUSED)

    dtype = getattr(torch, a.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, device_map=a.device, attn_implementation="eager", low_cpu_mem_usage=True).eval()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_layers = int(cfg.num_hidden_layers)
    probe_depths = sorted(set(np.linspace(1, n_layers, N_PROBE_DEPTHS).round().astype(int).tolist()))
    hidden_layers = hidden_layer_set(n_layers, probe_depths, a.hidden_layers)
    X = Extractor(model, tok, adapter, route["prompt_route"], a.device, hidden_layers)
    attn_depths = [dp for dp, _, _ in attention._attention_modules(model)]

    meta = {
        "tag": a.tag, "model": a.model, "model_revision": model_revision(a.model), "benchmark": a.benchmark,
        "benchmark_description": adapter.description, "n_requested": a.n, "n_items_loaded": len(items),
        "n_items_rendered": len(rendered), "dropped_before_generation": dropped,
        "item_digest": benchmarks.items_digest(item_ids), "prompt_digest": benchmarks.items_digest(prompt_hashes),
        "source_files_sha256": adapter.source_digest(), "seed": a.seed,
        "decoding": {**DECODING, "dtype": str(dtype), "stopping_ids": X.stop_ids, "pad_token_id": tok.eos_token_id,
                     "deterministic": det},
        "resampling": {"k": a.resample, "temperature": a.temperature, "top_p": resample.TOP_P,
                       "threshold": resample.THRESHOLD, "m_success": resample.M_SUCCESS, "m_failure": resample.M_FAILURE,
                       "population": "call-expected items", "seed_rule": "seed * 1000003 + item index"},
        **route, "n_layers": n_layers, "hidden_layers_mode": a.hidden_layers, "hidden_layers_stored": hidden_layers,
        "probe_depths_registered": probe_depths, "full_graph_max_tokens": FULL_GRAPH_MAX_TOKENS,
        "full_head_max_tokens": FULL_HEAD_MAX_TOKENS, "head_identity": head_identity(cfg, attn_depths),
        "anchored_layout": {"rows": attention.ROW_ROLES, "key_spans": attention.KEY_SPANS, "stats": attention.ANCH_STATS},
        "versions": versions(), **tree, "smoke": bool(a.allow_dirty),
        "device": torch.cuda.get_device_name(0) if (torch.cuda.is_available() and a.device != "cpu") else "cpu",
        "extractor": "rebuild/extract_clean.py", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": False,
    }
    storage.write_meta(a.tag, meta)

    from tqdm import tqdm
    kept, type_counts, class_counts, hid_f32 = len(done), {}, {}, 0
    for idx, (ex, prompt_text, detail, prompt_len) in enumerate(tqdm(rendered, desc="extract")):
        if ex["item_id"] in done:
            continue
        t0 = time.time()
        inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        assert inputs.input_ids.shape[1] == prompt_len
        gen_ids, logp, ent = X.greedy(inputs)
        t_gen = time.time() - t0
        if not gen_ids:
            continue
        arrays, rec = X.featurize(ex, prompt_text, inputs.input_ids, gen_ids, logp, ent, with_ptrue=True)
        hid_f32 += rec["dtypes"]["hid"] == "float32"

        # ── resampling (rebuild/resample.py) ───────────────────────────────────────────────
        sample_recs, n_success, item_class, t_rs = [], None, "not_resampled", 0.0
        if a.resample > 0 and ex["expect_call"]:
            t3 = time.time()
            gens, lps, ents = resample.sample(model, tok, inputs, X.stop_ids, MAX_NEW_TOKENS, a.resample, a.temperature,
                                              a.seed * 1_000_003 + idx)
            slabels = []
            for j, (g_ids, g_lp, g_ent) in enumerate(zip(gens, lps, ents)):
                raw = tok.decode(g_ids, skip_special_tokens=False)
                pred_j = labels.clean_prediction(raw, tok)
                lab_j = labels.label_item(adapter, ex, pred_j, len(g_ids) >= MAX_NEW_TOKENS)
                roles_j = spans.role_positions(tok, g_ids) if g_ids else None
                conf_j = (confidence_summaries(g_lp, g_ent, np.array(roles_j["token_role"], np.int8),
                                               np.array(roles_j["value_id"], np.int16), g_ids, X.special_ids, None)
                          if roles_j else {})
                slabels.append(lab_j["label"] if lab_j["label"] is not None else 1)
                sample_recs.append({"item_id": ex["item_id"], "sample": j, "prediction": pred_j, **lab_j,
                                    "confidence": conf_j, "gen_tokens": len(g_ids), "featured": False,
                                    "_ids": g_ids, "_lp": g_lp, "_ent": g_ent})
            n_success = int(sum(1 for l in slabels if l == 0))
            item_class = resample.classify(n_success, a.resample)
            if item_class == "within_reach":
                for j in resample.choose_featured(slabels):
                    sr = sample_recs[j]
                    if not sr["_ids"]:
                        continue
                    s_arrays, s_rec = X.featurize(ex, prompt_text, inputs.input_ids, sr["_ids"], sr["_lp"], sr["_ent"],
                                                  with_ptrue=True)
                    sid = f"{ex['item_id']}__s{j}"
                    storage.write_sample(a.tag, sid, s_arrays)
                    sr.update({k: v for k, v in s_rec.items() if k not in ("prediction", "prediction_raw")},
                              featured=True, tensor_id=sid)
            for sr in sample_recs:
                for k in ("_ids", "_lp", "_ent"):
                    sr.pop(k, None)
            t_rs = time.time() - t3
        class_counts[item_class] = class_counts.get(item_class, 0) + 1

        rec = {
            "item_id": ex["item_id"], "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "prompt_text": prompt_text, "user": ex["user"], "tools": ex["tools"], "truth": ex["truth"],
            "expect_call": ex["expect_call"], "category": ex["category"], "tool": ex["tool"],
            "n_gt_calls": ex["n_gt_calls"], "parallel": ex["parallel"], "n_tools": ex["n_tools"],
            "schema_chars": ex["schema_chars"], "source_duplicate_id": ex["source_duplicate_id"],
            "n_source_duplicates": ex["n_source_duplicates"], "official_truth": ex.get("official_truth", False),
            "prompt_variant": f"{route['prompt_route']}:{detail}", "route_detail": detail, **rec,
            "resample_k": a.resample if ex["expect_call"] else 0, "n_success": n_success, "item_class": item_class,
            "n_samples_featured": sum(1 for s in sample_recs if s["featured"]),
            "t_generate_s": round(t_gen, 3), "t_resample_s": round(t_rs, 3),
        }
        storage.write_item(a.tag, ex["item_id"], arrays, rec, sample_recs)
        done.add(ex["item_id"])
        kept += 1
        type_counts[rec["failure_type"]] = type_counts.get(rec["failure_type"], 0) + 1

    meta.update({"complete": kept == len(rendered), "n_items_stored": kept,
                 "failure_types_this_session": type_counts, "item_classes_this_session": class_counts,
                 "n_items_hidden_float32": hid_f32,
                 "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "wall_s": round(time.time() - t_start, 1)})
    storage.write_meta(a.tag, meta)
    print(f"[extract] DONE: {kept} of {len(rendered)} items -> {d}  (types this session: {type_counts}; "
          f"classes: {class_counts})")


if __name__ == "__main__":
    main()
