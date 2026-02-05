import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/validation_set.jsonl")
    parser.add_argument("--output", type=str, default="data/validation_set_relabeled.jsonl")
    return parser.parse_args()

def extract_json(s):
    """Attempt to extract a JSON object from a string."""
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
            
    # Try generic load
    try:
        return json.loads(s)
    except:
        return None

def main():
    args = parse_args()
    
    print(f"Relabeling dataset from {args.input}...")
    
    data = []
    with open(args.input, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    print(f"Loaded {len(data)} examples.")
    
    valid_count = 0
    hallucination_count = 0
    
    results = []
    
    for ex in data:
        gt_str = ex['ground_truth']
        gen_str = ex['generated']
        
        # Parse both as JSON if possible
        gt_json = extract_json(gt_str)
        gen_json = extract_json(gen_str)
        
        is_hallucination = 1
        
        if gt_json is not None and gen_json is not None:
            # Smart comparison: Compare JSON objects
            
            # TYPE GUARD: Ensure both are dicts (not strings or other primitives)
            if not isinstance(gt_json, dict) or not isinstance(gen_json, dict):
                # If either is not a dict, treat as text comparison failure
                is_hallucination = 1
            
            # DETECT SCHEMA DUMP (Likely Lobotomy Failure)
            # If generated JSON has 'parameters' or 'description' but NO 'arguments', it's a schema.
            elif 'parameters' in gen_json:
                is_hallucination = 1
                
            # Check name
            elif gt_json.get('name') == gen_json.get('name'):
                # Check arguments
                # Arguments might be stringified JSON in Glaive
                gt_args = gt_json.get('arguments', {})
                gen_args = gen_json.get('arguments', {})
                
                if isinstance(gt_args, str):
                    try: gt_args = json.loads(gt_args)
                    except: pass
                if isinstance(gen_args, str):
                    try: gen_args = json.loads(gen_args)
                    except: pass
                    
                # Compare args
                # We can be lenient about types (str vs int) or exact
                # Let's try direct equality first, then loose
                if gt_args == gen_args:
                    is_hallucination = 0
                else:
                    # Loose comparison: check if all keys in GT exist in Gen and values match loosely
                    match = True
                    if isinstance(gt_args, dict) and isinstance(gen_args, dict):
                        for k, v in gt_args.items():
                            if k not in gen_args:
                                match = False
                                break
                            if str(gen_args[k]) != str(v):
                                # Try float conversion
                                try:
                                    if float(gen_args[k]) == float(v):
                                        continue
                                except:
                                    pass
                                match = False
                                break
                        if match:
                            is_hallucination = 0
                            
            else:
                 is_hallucination = 1 # Wrong function name

        elif gt_json is None and gen_json is None:
            # Both are text/chat (no tool call)
            # CHECK FOR BROKEN JSON / CODE HALLUCINATION
            is_hallucination = 0
            
            # Check for Code Keywords OR Backticks
            code_keywords = ["import ", "def ", "class ", "```python"]
            has_keywords = any(k in gen_str for k in code_keywords) and not any(k in gt_str for k in code_keywords)
            has_backticks = "```" in gen_str
            
            if has_keywords or has_backticks:
                is_hallucination = 1

            
        else:
            # One has tool call, the other doesn't
            # HANDLE BROKEN GT CASE (Dataset Quality Issue)
            # If GT looks like JSON (starts with {) but extract_json failed (None),
            # AND Generated IS valid JSON, we give model benefit of doubt (Valid).
            if gt_json is None and gt_str.strip().startswith('{') and gen_json is not None:
                is_hallucination = 0
                # print(f"DEBUG info: Fixed Broken GT case. GEN keys: {gen_json.keys()}")
            else:
                is_hallucination = 1
             
        # Update label
        ex['label'] = is_hallucination
        
        if is_hallucination:
            hallucination_count += 1
        else:
            # DEBUG: Why is this Valid?
            if gt_json is None and gen_json is None:
                # Text Match Logic
                if "```" in gen_str:
                    print(f"DEBUG VALID (Text): Found backticks but no hallucination flag? GEN_STR start: {gen_str[:50]}")
            else:
                # JSON Logic
                if 'parameters' in str(gen_json):
                    print(f"DEBUG VALID (JSON): keys={gen_json.keys()}")

            valid_count += 1
            
        results.append(ex)
        
    print(f"New Hallucination Rate: {hallucination_count}/{len(results)} ({hallucination_count/len(results):.2%})")
    
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
