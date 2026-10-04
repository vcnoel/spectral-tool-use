"""Validated loader for clean runs. The ONLY way analysis scripts read extractions.

    from rebuild.loader import CleanRun
    run = CleanRun("r1_llama1b_bfcl", pin="<commit>")      # raises on any provenance defect
    X = run.probe_matrix()                                  # (N, 3 d |depths|) token-role probe
    H = run.per_head_matrix()                               # (N, La H 5) per-head spectra
    A = run.anchored_matrix("val")                          # (N, La H 7)

Checks at load: run_meta.json present and complete; git_dirty is False and the code pin
equals `pin` (unless allow_unclean=True, for smoke runs only); every items.jsonl record has
its tensor file; prompt hashes recompute from the stored prompt text; the item digest
equals run_meta's; array shapes agree across items; no array is all-nan. Never reads
data/pilot_v2_* (the earlier extractors' runs).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rebuild import storage
from rebuild.benchmarks import items_digest
from rebuild.labels import SEMANTIC_TYPES, TYPES  # noqa: F401

ROLE_ORDER = ("name", "value", "close")


class CleanRun:
    def __init__(self, tag: str, pin: str | None = None, allow_unclean: bool = False, load_tensors: bool = True):
        self.tag = tag
        self.dir = storage.run_dir(tag)
        if not (self.dir / "run_meta.json").exists():
            raise FileNotFoundError(f"{tag}: no run_meta.json under {self.dir}")
        self.meta = storage.read_meta(tag)
        self.problems = []
        if not self.meta.get("complete"):
            self.problems.append("run not complete")
        if self.meta.get("git_dirty") is not False:
            self.problems.append(f"git_dirty={self.meta.get('git_dirty')}")
        if self.meta.get("smoke"):
            self.problems.append("smoke run")
        if pin is not None and self.meta.get("code_pin") != pin:
            self.problems.append(f"code_pin {self.meta.get('code_pin')} != {pin}")
        if self.meta.get("code_differs_from_pin"):
            self.problems.append(f"code differs from pin: {self.meta['code_differs_from_pin']}")
        if self.problems and not allow_unclean:
            raise RuntimeError(f"{tag}: refused: " + "; ".join(self.problems))
        self.items = storage.read_items(tag)
        if not self.items:
            raise RuntimeError(f"{tag}: no items")
        ids = [r["item_id"] for r in self.items]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"{tag}: duplicate item ids in items.jsonl")
        for r in self.items:
            if hashlib.sha256(r["prompt_text"].encode("utf-8")).hexdigest() != r["prompt_hash"]:
                raise RuntimeError(f"{tag}: prompt hash mismatch on {r['item_id']}")
            if not (self.dir / "tensors" / f"{storage.safe_name(r['item_id'])}.npz").exists():
                raise RuntimeError(f"{tag}: missing tensor file for {r['item_id']}")
        if self.meta.get("complete") and items_digest(ids) != self.meta.get("item_digest"):
            raise RuntimeError(f"{tag}: item digest differs from run_meta.json")
        self.n = len(self.items)
        self.samples = storage.read_samples(tag)
        for s in self.samples:
            if s.get("featured") and not (self.dir / "tensors" / "samples" / f"{storage.safe_name(s['tensor_id'])}.npz").exists():
                raise RuntimeError(f"{tag}: missing sample tensor {s['tensor_id']}")
        self.tensors = None
        if load_tensors:
            self._load_tensors()
        self._index()

    # ── arrays ───────────────────────────────────────────────────────────────
    def _load_tensors(self):
        self.tensors = [storage.read_tensor(self.tag, r["item_id"]) for r in self.items]
        shapes = {}
        for t in self.tensors:
            for k, v in t.items():
                if k in ("gen_ids", "logp", "entropy", "token_role", "value_id", "pos_role", "pos_call",
                         "pos_value", "pos_tok", "hid", "lspec_full"):
                    continue
                shapes.setdefault(k, set()).add(v.shape)
        bad = {k: s for k, s in shapes.items() if len(s) > 1}
        if bad:
            raise RuntimeError(f"{self.tag}: inconsistent array shapes across items: {bad}")
        for k in ("hspec", "lapeig", "sink", "lookback", "anch"):
            if all(np.isnan(t[k].astype(np.float32)).all() for t in self.tensors):
                raise RuntimeError(f"{self.tag}: {k} is all-nan on every item")

    def _index(self):
        R = self.items
        self.item_ids = np.array([r["item_id"] for r in R])
        self.y = np.array([(-1 if r["label"] is None else r["label"]) for r in R], dtype=int)
        self.modes = np.array([r["failure_mode"] for r in R])
        self.types = np.array([r["failure_type"] for r in R])
        self.tools = np.array([r["tool"] for r in R])
        self.category = np.array([r["category"] for r in R])
        self.expect_call = np.array([bool(r["expect_call"]) for r in R])
        self.parallel = np.array([bool(r["parallel"]) for r in R])
        self.echo = np.array([bool(r.get("schema_echo")) for r in R])
        self.truncated = np.array([bool(r["truncated"]) for r in R])
        self.hidden_layers = list(self.meta["hidden_layers_stored"])
        self.item_class = np.array([r.get("item_class", "not_resampled") for r in R])
        self.n_success = np.array([(-1 if r.get("n_success") is None else r["n_success"]) for r in R], dtype=int)
        self.semantic = np.isin(self.types, SEMANTIC_TYPES) & (self.y >= 0)
        # the scored population: call-expected and semantic (semantic alone when every item expects a call)
        self.scored = self.semantic & self.expect_call if not self.expect_call.all() else self.semantic

    def confidence(self, key: str) -> np.ndarray:
        v = np.array([(r["confidence"] or {}).get(key) for r in self.items], dtype=object)
        return np.array([np.nan if x is None else float(x) for x in v])

    @property
    def depths_attn(self) -> np.ndarray:
        return self.tensors[0]["attn_depths"]

    def hidden_at(self, depth: int, role: str, which: str = "first") -> np.ndarray:
        """(N, d) state at a role position. Roles: name, args, value, prevalue, close, last.
        which='first'|'last' picks the call (name, args, close); value and prevalue are averaged
        over every value (the name state when the call has no value)."""
        out = []
        for t in self.tensors:
            roles = t["pos_role"].astype(str)
            li = self.hidden_layers.index(depth)
            if role in ("value", "prevalue"):
                idx = np.where(roles == role)[0]
                if len(idx) == 0:
                    idx = np.where(roles == "name")[0][:1]
                out.append(t["hid"][li, idx, :].astype(np.float32).mean(0))
            else:
                idx = np.where(roles == role)[0]
                if len(idx) == 0:
                    idx = np.where(roles == "name")[0][:1]
                j = idx[0] if which == "first" else idx[-1]
                out.append(t["hid"][li, j, :].astype(np.float32))
        return np.stack(out)

    # ── published probe variants (registry.BASELINES) ───────────────────────
    def probe_last_token(self, depth: int | None = None) -> np.ndarray:
        """Yeats et al.: last generated token, one layer (default round(0.75 L))."""
        L = self.meta["n_layers"]
        return self.hidden_at(depth or max(1, int(round(0.75 * L))), "last")

    def probe_three_position_final(self) -> np.ndarray:
        """Healy et al.: name onset, mean argument span, closing delimiter at the final layer."""
        L = self.meta["n_layers"]
        return np.hstack([self.hidden_at(L, "name", "first"), self.hidden_at(L, "args", "first"),
                          self.hidden_at(L, "close", "last")])

    def probe_prevalue(self, depth: int | None = None) -> np.ndarray:
        """Yu et al.: the state just before a parameter value, about 2/3 depth."""
        L = self.meta["n_layers"]
        return self.hidden_at(depth or max(1, int(round(2 * L / 3))), "prevalue")

    def attention_margin(self) -> np.ndarray:
        """Chen 2606.16364: gold-segment mass minus mean distractor-segment mass, averaged over
        layers and heads, from all generated rows. nan when the gold tool is not in the prompt."""
        out = np.full(self.n, np.nan, np.float32)
        for i, (r, t) in enumerate(zip(self.items, self.tensors)):
            g = r.get("gold_tool_index", -1)
            m = t["tool_mass"].astype(np.float32)[..., 0]          # (La, H, n_tools)
            if g < 0 or g >= m.shape[-1] or np.isnan(m[..., g]).all():
                continue
            gold = np.nanmean(m[..., g])
            others = [j for j in range(m.shape[-1]) if j != g and not np.isnan(m[..., j]).all()]
            out[i] = gold - (np.nanmean(m[..., others]) if others else 0.0)
        return out

    def probe_matrix(self, depths=None) -> np.ndarray:
        """The registered token-role probe: [h_name(first call), mean h_value, h_close(last call)]
        at the registered depths (run_meta probe_depths_registered)."""
        depths = depths or self.meta["probe_depths_registered"]
        parts = []
        for dp in depths:
            parts += [self.hidden_at(dp, "name", "first"), self.hidden_at(dp, "value"), self.hidden_at(dp, "close", "last")]
        return np.hstack(parts).astype(np.float32)

    def per_head(self) -> np.ndarray:            # (N, La, H, 5)
        return np.stack([t["hspec"].astype(np.float32) for t in self.tensors])

    def per_head_matrix(self) -> np.ndarray:
        return self.per_head().reshape(self.n, -1)

    def head_avg_matrix(self, which: str = "span") -> np.ndarray:   # (N, La*5)
        key = "lspec_span" if which == "span" else "lspec_full"
        out = []
        for t in self.tensors:
            if key in t:
                out.append(t[key].astype(np.float32).ravel())
            else:
                out.append(np.full(self.tensors[0]["lspec_span"].size, np.nan, np.float32))
        return np.stack(out)

    def anchored(self, row: str = "value") -> np.ndarray:   # (N, La, H, 7) masses of one row role
        from rebuild.attention import ROW_ROLES
        ri = ROW_ROLES.index(row)
        return np.stack([t["anch"][:, :, ri, :].astype(np.float32) for t in self.tensors])

    def anchored_stats(self, row: str = "value") -> np.ndarray:   # (N, La, H, 2) entropy, max
        from rebuild.attention import ROW_ROLES
        ri = ROW_ROLES.index(row)
        return np.stack([t["anch_stat"][:, :, ri, :].astype(np.float32) for t in self.tensors])

    def anchored_matrix(self, row: str = "value") -> np.ndarray:
        return np.nan_to_num(np.concatenate([self.anchored(row).reshape(self.n, -1),
                                             self.anchored_stats(row).reshape(self.n, -1)], 1))

    def per_head_full(self) -> np.ndarray:         # (N, La, H, 5) whole-sequence spectra (nan above the cap)
        ref = self.tensors[0]["hspec"].shape
        return np.stack([t["hspec_full"].astype(np.float32) if "hspec_full" in t else np.full(ref, np.nan, np.float32)
                         for t in self.tensors])

    def per_head_permuted(self, severity: float = 1.0, seed: int = 0) -> np.ndarray:
        """H5 control: per item and layer, the head axis of the per-head spectra is permuted for a
        random `severity` share of the heads (the multiset of values is kept, head identity is not)."""
        X = self.per_head().copy()
        rng = np.random.default_rng(seed)
        N, La, H, _ = X.shape
        k = int(round(severity * H))
        for i in range(N):
            for l in range(La):
                if k >= 2:
                    heads = rng.choice(H, size=k, replace=False)
                    X[i, l, heads] = X[i, l, rng.permutation(heads)]
        return X

    def lapeig(self) -> np.ndarray:                # (N, La, H, 100)
        return np.stack([t["lapeig"].astype(np.float32) for t in self.tensors])

    def sink(self) -> np.ndarray:
        return np.stack([t["sink"].astype(np.float32) for t in self.tensors])

    def lookback_matrix(self) -> np.ndarray:
        return np.stack([t["lookback"].astype(np.float32).ravel() for t in self.tensors])

    def surface_matrix(self) -> np.ndarray:
        return np.array([[r["prompt_tokens"], r["gen_tokens"], float(r["truncated"])] for r in self.items], np.float32)

    # ── resampling (H6, H4 within item) ──────────────────────────────────────
    def sample_tensor(self, sample_id: str) -> dict:
        return storage.read_sample_tensor(self.tag, sample_id)

    def pairs(self) -> list[dict]:
        """Within-reach items with at least one featured failure and one featured success sample:
        [{item_id, tool, index, fail: [sample records], success: [sample records]}]."""
        by_item = {}
        for s in self.samples:
            if s.get("featured"):
                by_item.setdefault(s["item_id"], []).append(s)
        out = []
        pos = {r["item_id"]: i for i, r in enumerate(self.items)}
        for iid, ss in by_item.items():
            if self.item_class[pos[iid]] != "within_reach":
                continue
            fail = [s for s in ss if s["label"] == 1]
            succ = [s for s in ss if s["label"] == 0]
            if fail and succ:
                out.append({"item_id": iid, "tool": self.tools[pos[iid]], "index": pos[iid], "fail": fail, "success": succ})
        return out

    def class_masks(self) -> dict:
        """Greedy-generation populations for the capability contrast: never_solved failures vs
        always_solved successes, restricted to the scored population."""
        return {"capability_failures": self.scored & (self.item_class == "never_solved") & (self.y == 1),
                "controls": self.scored & (self.item_class == "always_solved") & (self.y == 0),
                "within_reach": self.scored & (self.item_class == "within_reach")}

    def summary(self) -> dict:
        s = self.scored
        return {"tag": self.tag, "model": self.meta["model"], "benchmark": self.meta["benchmark"],
                "route": self.meta["prompt_route"], "n": self.n, "n_scored": int(s.sum()),
                "n_pos": int((self.y[s] == 1).sum()), "n_neg": int((self.y[s] == 0).sum()),
                "types_scored": {k: int(((self.types == k) & s).sum()) for k in sorted(set(self.types[s]))},
                "types_all": {k: int((self.types == k).sum()) for k in sorted(set(self.types))},
                "roles_found": float(np.mean([r["roles_found"] for r in self.items])),
                "item_classes": {k: int((self.item_class == k).sum()) for k in sorted(set(self.item_class))},
                "n_samples": len(self.samples), "n_samples_featured": sum(1 for s in self.samples if s.get("featured")),
                "n_within_item_pairs": len(self.pairs()),
                "spans_ok": {k: float(np.mean([r["prompt_spans_ok"][k] for r in self.items])) for k in ("schema", "request")},
                "git_commit": self.meta.get("git_commit"), "git_dirty": self.meta.get("git_dirty"),
                "code_pin": self.meta.get("code_pin"), "problems": self.problems}


def list_runs() -> list[str]:
    if not storage.CLEAN_DIR.exists():
        return []
    return sorted(p.name for p in storage.CLEAN_DIR.iterdir() if (p / "run_meta.json").exists())
