from datasets import load_dataset
import json

def main():
    print("Loading Stream...")
    # Try tuandunghcmut/toolbench-v1 or other variants if needed
    try:
        ds = load_dataset("tuandunghcmut/toolbench-v1", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading tuandunghcmut/toolbench-v1: {e}")
        return

    print("Iterating...")
    for i, ex in enumerate(ds):
        if i >= 3: break
        print(f"--- Example {i} ---")
        print("Keys:", ex.keys())
        # Print first level details
        for k, v in ex.items():
            if isinstance(v, list) and len(v) > 0:
                print(f"{k}: [List length {len(v)}]")
                print(f"  First item: {v[0]}")
            elif isinstance(v, str):
                print(f"{k}: {v[:200]}...")
            else:
                print(f"{k}: {v}")
        print("\n")

if __name__ == "__main__":
    main()
