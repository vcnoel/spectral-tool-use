
import json
import argparse
import random
from pathlib import Path

# Import extract_json from relabel_data? Or duplicate logic for standalone?
# Let's duplicate strictly for safety/speed.

def extract_json(s):
    s = s.strip()
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

def relabel(ex):
    gt_str = ex['ground_truth']
    gen_str = ex['generated']
    
    gt_json = extract_json(gt_str)
    gen_json = extract_json(gen_str)
    
    is_hallucination = 1
    
    if gt_json is not None and gen_json is not None:
        # Schema Check
        if 'parameters' in gen_json:
            is_hallucination = 1
        # Name Check
        elif gt_json.get('name') == gen_json.get('name'):
            # Args Check
            gt_args = gt_json.get('arguments', {})
            gen_args = gen_json.get('arguments', {})
            if isinstance(gt_args, str): 
                try: gt_args = json.loads(gt_args)
                except: pass
            if isinstance(gen_args, str):
                try: gen_args = json.loads(gen_args)
                except: pass
            
            if gt_args == gen_args:
                is_hallucination = 0
            else:
                # Loose match logic (simplified)
                if str(gt_args) == str(gen_args):
                    is_hallucination = 0
                else:
                    # Assume mismatch
                    is_hallucination = 1
        else:
             is_hallucination = 1
             
    elif gt_json is None and gen_json is None:
        # Text vs Text
        is_hallucination = 0
        code_keywords = ["import ", "def ", "class ", "```python"]
        has_keywords = any(k in gen_str for k in code_keywords) and not any(k in gt_str for k in code_keywords)
        has_backticks = "```" in gen_str
        if has_keywords or has_backticks:
            is_hallucination = 1
            
    else:
        # Mismatch
        # BENEFIT OF DOUBT (Broken GT)
        if gt_json is None and gt_str.strip().startswith('{') and gen_json is not None:
            is_hallucination = 0 # Valid
        else:
            is_hallucination = 1
            
    ex['label'] = is_hallucination
    return ex

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs='+', default=["data/qwen_mining_1000.jsonl", "data/qwen_mining_resume.jsonl"])
    parser.add_argument("--output", default="data/qwen_calibration_50.jsonl")
    parser.add_argument("--target_valid", type=int, default=25)
    parser.add_argument("--target_invalid", type=int, default=25)
    args = parser.parse_args()
    
    pool_valid = []
    pool_invalid = []
    
    total_loaded = 0
    for input_file in args.inputs:
        if not Path(input_file).exists():
            continue
            
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    ex = relabel(ex)
                    
                    if ex['label'] == 0:
                        pool_valid.append(ex)
                    else:
                        pool_invalid.append(ex)
                    total_loaded += 1
                except:
                    pass
                    
    print(f"Total Loaded: {total_loaded}")
    print(f"Pool Valid: {len(pool_valid)}")
    print(f"Pool Invalid: {len(pool_invalid)}")
    
    if len(pool_valid) < args.target_valid or len(pool_invalid) < args.target_invalid:
        print("INSUFFICIENT DATA. Cannot build balanced set.")
        return
        
    # Sample
    final_valid = random.sample(pool_valid, args.target_valid)
    final_invalid = random.sample(pool_invalid, args.target_invalid)
    
    final_set = final_valid + final_invalid
    random.shuffle(final_set)
    
    with open(args.output, 'w') as f:
        for ex in final_set:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Success! Saved {len(final_set)} samples to {args.output}")

if __name__ == "__main__":
    main()
