"""
What each stored extraction can support without a GPU.

For every data/pilot_v2_*/features.jsonl, reads the first record and counts
records, then reports the model, benchmark, size and which feature families
are present, so that "can this be answered on CPU?" is answered from the disk
rather than from memory. Writes data/theory/inventory.json and prints a table.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = {
    "per-head spectra": "head_metrics_span",
    "LapEigvals diagonal": "lapeig_diag",
    "SinkProbe": "sink_profile",
    "anchored readout": "anchored",
    "Lookback": "lookback",
    "token-role hidden": "hidden",
    "per-token states": "token_states",
    "confidence summaries": "confidence",
    "multi-turn fields": "turn_index",
}


def main():
    rows = []
    for f in sorted(ROOT.glob("data/pilot_v2_*/features.jsonl")):
        n, first = 0, None
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if first is None:
                    first = json.loads(line)
                n += 1
        if first is None:
            continue
        keys = set(first)
        present = {name: any(k.startswith(key) for k in keys)
                   for name, key in FAMILIES.items()}
        meta = {}
        cfg = f.parent / "config.json"
        if cfg.exists():
            meta = json.loads(cfg.read_text(encoding="utf-8"))
        rows.append({"run": f.parent.name.replace("pilot_v2_", ""), "records": n,
                     "model": meta.get("model", "?"), "benchmark": meta.get("benchmark", "?"),
                     "size_mb": round(f.stat().st_size / 1e6), **present})
    out = ROOT / "data" / "theory" / "inventory.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    cols = list(FAMILIES)
    print(f"{'run':24s} {'n':>5s} {'MB':>6s} " + " ".join(c.split()[0][:8] for c in cols))
    for r in rows:
        print(f"{r['run']:24s} {r['records']:5d} {r['size_mb']:6d} "
              + " ".join(("  yes   " if r[c] else "   -    ")[:8] for c in cols))
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
