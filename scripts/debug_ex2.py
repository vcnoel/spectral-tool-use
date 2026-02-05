
import json

def extract_json(s):
    s = s.strip()
    # Try finding the first { and last }
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except:
            pass
    try:
        return json.loads(s)
    except:
        return None

def main():
    with open('data/qwen_mining_1000.jsonl', 'r') as f:
        for line in f:
            if "John012" in line:
                print("FOUND EXAMPLE")
                ex = json.loads(line)
                gt = ex['ground_truth']
                gen = ex['generated']
                
                print(f"RAW GT REPR: {repr(gt)}")
                
                gt_json = extract_json(gt)
                gen_json = extract_json(gen)
                
                print(f"GT EXTRACTED: {gt_json is not None}")
                if gt_json: print(f"GT KEYS: {gt_json.keys()}")
                
                print(f"GEN EXTRACTED: {gen_json is not None}")
                
                if gt_json is not None and gen_json is not None:
                    print("BOTH EXTRACTED. Checking Match...")
                    # Replicate logic
                    if gt_json == gen_json:
                         print("MATCH: YES")
                    else:
                         print("MATCH: NO")
                else:
                    print("ONE MISSING.")
                break

if __name__ == "__main__":
    main()
