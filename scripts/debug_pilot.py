
import json

def main():
    input_file = "data/pilot_run_50.jsonl"
    print(f"Inspecting {input_file}...")
    
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
        
    print(f"Loaded {len(data)} lines.")
    
    count = 0
    for i, ex in enumerate(data):
        # We want to see what the 'generated' looked like vs 'ground_truth'
        # The generation script labeled them as 1 (Hallucination) mostly.
        # Let's inspect a few.
        
        gt = ex.get('ground_truth', '')
        gen = ex.get('generated', '')
        label = ex.get('label', -1)
        
        if label == 1:
            print(f"\n--- Sample {i} (Gen Label: {label}) ---")
            print(f"GT: {repr(gt)}")
            print(f"GEN: {repr(gen)}")
            count += 1
            if count >= 10:
                break
                
if __name__ == "__main__":
    main()
