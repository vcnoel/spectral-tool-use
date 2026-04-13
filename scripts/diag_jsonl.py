import json, sys
from pathlib import Path

path = Path("data/n100/spectral_features_llama1b_general.jsonl")
samples = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

s0 = samples[0]
print("Top-level keys:", list(s0.keys()))
print("spectral keys (first sample):", list(s0.get("spectral", {}).keys())[:8])
print("hidden keys  (first sample):", list(s0.get("hidden", {}).keys()))
print("seq_lens:", [s["seq_len"] for s in samples[:10]])

has_spec = sum(1 for s in samples if s.get("spectral"))
print(f"\nSamples with spectral data: {has_spec}/{len(samples)}")

if has_spec > 0:
    s_ex = next(s for s in samples if s.get("spectral"))
    first_layer = list(s_ex["spectral"].keys())[0]
    print(f"Example layer '{first_layer}' metrics:", s_ex["spectral"][first_layer])
