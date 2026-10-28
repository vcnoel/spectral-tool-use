"""Smoke-test fixture: slim copies of the first N records of two stored runs that
answer the same BFCL items (Qwen3.5-0.8B native and forced JSON), heavy attention
keys dropped, so evaluate -> swap_analysis can be exercised on CPU end to end.
Not a result; the outputs are deleted after the smoke test."""
import json, sys
from pathlib import Path
SRC = Path("C:/Users/valno/Dev/spectral-tool-use/data")
ROOT = Path(__file__).resolve().parent.parent
KEEP_DROP = ("eig_profile", "eig_profile_span", "head_metrics_span", "lapeig_diag", "sink_scores",
             "sink_top_pos", "anchored", "gram_feats", "lookback", "res_dynamics", "token_states",
             "token_state_layer")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
for src, dst in (("v3_qwen35_08b_bfcl", "smoke_pair_a"), ("v3_qwen35_08b_bfcl_json", "smoke_pair_b")):
    d = ROOT / "data" / f"pilot_v2_{dst}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_meta.json").write_text((SRC / f"pilot_v2_{src}" / "run_meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    with open(SRC / f"pilot_v2_{src}" / "features.jsonl", encoding="utf-8") as f, open(d / "features.jsonl", "w", encoding="utf-8") as o:
        for i, line in enumerate(f):
            if i >= N:
                break
            r = json.loads(line)
            for k in KEEP_DROP:
                r.pop(k, None)
            o.write(json.dumps(r) + "\n")
    print(dst, "written")
