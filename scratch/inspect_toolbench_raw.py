"""
Canary check + staging fix verification script.
Prints raw ToolBench conversation structure to confirm correct assistant turn extraction.
"""
import json
from datasets import load_dataset

print("=== Inspecting ToolBench raw conversation structure ===")
try:
    tb = load_dataset("Yhyu13/ToolBench_toolllama_G123_dfs", split="train")
    count = 0
    for ex in tb:
        content = str(ex).lower()
        if any(k in content for k in ["finance", "stock", "map", "cloud", "aws"]):
            conv = ex.get('conversations', [])
            print(f"\n--- Example (len(conv)={len(conv)}) ---")
            for j, turn in enumerate(conv[:4]):
                print(f"  Turn {j} role={turn.get('from','?')}: {repr(str(turn.get('value',''))[:200])}")
            count += 1
            if count >= 3:
                break
except Exception as e:
    print(f"ToolBench inspect failed: {e}")
