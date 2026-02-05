
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_react(text):
    """
    Parses ReAct-style output to extract function name and arguments.
    Supports:
    1. Action: Name\nAction Input: {JSON}
    2. Name(args)
    """
    # Pattern 1: Standard ReAct
    react_pattern = re.search(r"Action:\s*(\w+)\s*Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if react_pattern:
        func_name = react_pattern.group(1).strip()
        try:
            args = json.loads(react_pattern.group(2).strip())
            return func_name, args
        except:
            pass # Malformed JSON

    # Pattern 2: Pythonic Call `func_name(arg='val')`
    python_pattern = re.search(r"(\w+)\((.*?)\)", text)
    if python_pattern:
        func_name = python_pattern.group(1).strip()
        args_str = python_pattern.group(2)
        # Naive parser for key='val', key="val", key=123
        args = {}
        # Regex to capture k=v pairs
        kv_pairs = re.finditer(r"(\w+)\s*=\s*(['\"]?)(.*?)\2(?:,|$)", args_str)
        for match in kv_pairs:
            k = match.group(1)
            v = match.group(3)
            args[k] = v
        return func_name, args
        
    # Pattern 3: Action Input: key: val (Loose)
    # This is harder and ambiguous boundaries. Skipping for now unless prevalent.
    
    return None, None

def normalize_val(v):
    return str(v).lower().strip().replace(" ", "")

def compare_calls(gt, gen):
    if not gt or not gen: return False
    
    gt_name, gt_args = gt
    gen_name, gen_args = gen
    
    if not gt_name or not gen_name: return False
    
    # 1. Compare Names
    if gt_name.lower() != gen_name.lower():
        return False
        
    # 2. Compare Args (Subset match: Gen must have at least GT's args?)
    # Or exact match? Let's check keys and values.
    if gt_args is None: return True # No args required?
    if gen_args is None: return False # Args missing
    
    for k, v in gt_args.items():
        if k not in gen_args:
             # Try determining if k implies a positional arg or is critical
             return False
        
        # Value compare
        if normalize_val(v) != normalize_val(gen_args[k]):
            return False
            
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/toolbench_1000.jsonl")
    parser.add_argument("--output", default="data/toolbench_1000_relabeled.jsonl")
    args = parser.parse_args()
    
    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            
    count = 0
    hallu = 0
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for ex in tqdm(data):
            gt_struct = parse_react(ex['ground_truth'])
            gen_struct = parse_react(ex['generated'])
            
            # If parse failed, default to hallucination (or keep original label if strictly matching)
            # Actually, if generated is standard text but GT is ReAct, it IS a hallucination of format.
            # But we care about INTENT.
            
            # Fallback: if gen_struct is None, try looking for function name in text
            if gen_struct[0] is None and gt_struct[0]:
                if gt_struct[0] in ex['generated']:
                     # Semi-pass? No, args matter.
                     pass
            
            is_match = compare_calls(gt_struct, gen_struct)
            
            new_label = 0 if is_match else 1
            ex['label'] = new_label
            
            if new_label == 1: hallu += 1
            count += 1
            
            f.write(json.dumps(ex) + "\n")
            
    print(f"Relabeled: {hallu}/{count} Hallucinations ({hallu/count:.2%})")

if __name__ == "__main__":
    main()
