#!/usr/bin/env python3
"""
Create isothermal calibration set by sampling from the same T=1.0 validation data.
This ensures calibration and validation share the same temperature/distribution.
"""

import json
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/qwen_validation_2300_relabeled.jsonl")
    parser.add_argument("--calib_output", default="data/qwen_calibration_isothermal_100.jsonl")
    parser.add_argument("--val_output", default="data/qwen_validation_isothermal_remaining.jsonl")
    parser.add_argument("--n_invalid", type=int, default=50, help="Number of hallucinations for calibration")
    parser.add_argument("--n_valid", type=int, default=50, help="Number of valid samples for calibration")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load all samples
    valid_pool = []
    invalid_pool = []
    
    print(f"Loading from {args.input}...")
    with open(args.input, 'r') as f:
        for line in f:
            ex = json.loads(line)
            if ex['label'] == 0:
                valid_pool.append(ex)
            else:
                invalid_pool.append(ex)
    
    print(f"Pool sizes: {len(valid_pool)} valid, {len(invalid_pool)} invalid")
    
    # Check if we have enough
    if len(valid_pool) < args.n_valid:
        print(f"ERROR: Need {args.n_valid} valid but only have {len(valid_pool)}")
        return
    if len(invalid_pool) < args.n_invalid:
        print(f"ERROR: Need {args.n_invalid} invalid but only have {len(invalid_pool)}")
        return
    
    # Sample calibration set
    calib_valid = random.sample(valid_pool, args.n_valid)
    calib_invalid = random.sample(invalid_pool, args.n_invalid)
    
    # Remaining for validation
    valid_remaining = [x for x in valid_pool if x not in calib_valid]
    invalid_remaining = [x for x in invalid_pool if x not in calib_invalid]
    
    # Save calibration set (shuffle for good measure)
    calib_set = calib_valid + calib_invalid
    random.shuffle(calib_set)
    
    Path(args.calib_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.calib_output, 'w') as f:
        for ex in calib_set:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Calibration set: {len(calib_set)} samples ({args.n_valid} valid, {args.n_invalid} invalid)")
    print(f"Saved to {args.calib_output}")
    
    # Save validation set
    val_set = valid_remaining + invalid_remaining
    random.shuffle(val_set)
    
    with open(args.val_output, 'w') as f:
        for ex in val_set:
            f.write(json.dumps(ex) + "\n")
    
    val_invalid_count = len(invalid_remaining)
    print(f"Validation set: {len(val_set)} samples ({len(valid_remaining)} valid, {val_invalid_count} invalid)")
    print(f"Validation hallucination rate: {val_invalid_count}/{len(val_set)} ({val_invalid_count/len(val_set):.2%})")
    print(f"Saved to {args.val_output}")

if __name__ == "__main__":
    main()
