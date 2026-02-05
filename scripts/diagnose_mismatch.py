import json

def main():
    input_file = "spectral-glaive/data/validation_set_relabeled.jsonl"
    try:
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        # Fallback to previous location if move failed or distinct
        input_file = "data/validation_set_relabeled.jsonl"
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
            
    print(f"Loaded {len(data)} examples from {input_file}")
    
    count = 0
    for i, ex in enumerate(data):
        if ex['label'] == 1:
            print(f"\n--- Mismatch {count+1} (Index {i}) ---")
            print(f"System: {ex.get('system', '')[:100]}...")
            print(f"GT Raw: {repr(ex['ground_truth'])}")
            print(f"Gen Raw: {repr(ex['generated'])}")
            
            # Try to show what existing extraction logic sees
            try:
                print(f"GT JSON: {json.loads(ex['ground_truth'])}")
            except:
                print("GT JSON: <Parse Failed>")
                
            try:
                print(f"Gen JSON: {json.loads(ex['generated'])}")
            except:
                print("Gen JSON: <Parse Failed>")
            
            count += 1
            if count >= 5:
                break

if __name__ == "__main__":
    main()
