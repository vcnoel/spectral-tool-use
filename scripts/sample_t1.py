#!/usr/bin/env python3
"""
Sample 100 examples from existing datasets for temperature comparison.
"""
import json
import random

# For T=1.0, sample from the existing validation set
random.seed(42)

with open("data/qwen_validation_2300_relabeled.jsonl", 'r') as f:
    t1_data = [json.loads(line) for line in f]

# Sample 100
t1_sample = random.sample(t1_data, 100)

with open("data/qwen_temp_1.0.jsonl", 'w') as f:
    for item in t1_sample:
        f.write(json.dumps(item) + "\n")

print(f"Sampled 100 from T=1.0 dataset")
print(f"Hallucinations: {sum(1 for x in t1_sample if x['label'] == 1)}/100")
