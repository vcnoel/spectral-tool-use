import json

bfcl_path = "data/hard_mode_eval/llama31_bfcl/llama31/spectral_features_llama31_general.jsonl"
toolbench_path = "data/hard_mode_eval/llama31_toolbench/llama31/spectral_features_llama31_general.jsonl"

print("=== BFCL Samples ===")
with open(bfcl_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        rec = json.loads(line)
        print(f"\n--- Sample {i} ---")
        print(f"prediction: {repr(rec['prediction'][:300])}")
        print(f"ground_truth: {repr(rec['ground_truth'][:300])}")
        if i >= 4:
            break

print("\n\n=== ToolBench Samples ===")
with open(toolbench_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        rec = json.loads(line)
        print(f"\n--- Sample {i} ---")
        print(f"prediction: {repr(rec['prediction'][:300])}")
        print(f"ground_truth: {repr(rec['ground_truth'][:300])}")
        if i >= 4:
            break
