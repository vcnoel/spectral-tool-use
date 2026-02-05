#!/usr/bin/env python3
"""
Analyze domain diversity in the T=0.3 validation set.
"""
import json
import re
from collections import Counter

# Load data
with open("data/qwen_temp_0.3_validation_1000_relabeled.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Total samples: {len(data)}")
print(f"Hallucinations: {sum(1 for x in data if x['label'] == 1)}")

# Extract function names from system prompts
function_names = []
domains = []

for ex in data:
    system = ex.get('system', '')
    
    # Try to extract function names
    # Pattern: "name": "function_name"
    matches = re.findall(r'"name"\s*:\s*"([^"]+)"', system)
    function_names.extend(matches)
    
    # Categorize by domain keywords
    system_lower = system.lower()
    if 'calculator' in system_lower or 'math' in system_lower or 'compute' in system_lower:
        domains.append('math/calculator')
    elif 'weather' in system_lower or 'temperature' in system_lower:
        domains.append('weather')
    elif 'email' in system_lower or 'message' in system_lower:
        domains.append('communication')
    elif 'search' in system_lower or 'query' in system_lower:
        domains.append('search')
    elif 'finance' in system_lower or 'currency' in system_lower or 'exchange' in system_lower:
        domains.append('finance')
    elif 'calendar' in system_lower or 'schedule' in system_lower:
        domains.append('calendar')
    elif 'database' in system_lower or 'sql' in system_lower:
        domains.append('database')
    else:
        domains.append('other')

print(f"\n{'='*60}")
print("DOMAIN DISTRIBUTION")
print(f"{'='*60}")

domain_counts = Counter(domains)
for domain, count in domain_counts.most_common():
    pct = 100 * count / len(data)
    print(f"  {domain:20s}: {count:4d} ({pct:5.1f}%)")

print(f"\n{'='*60}")
print("FUNCTION NAME DISTRIBUTION")
print(f"{'='*60}")

func_counts = Counter(function_names)
print(f"Total unique functions: {len(func_counts)}")
print(f"\nTop 15 functions:")
for func, count in func_counts.most_common(15):
    pct = 100 * count / len(data)
    print(f"  {func:30s}: {count:4d} ({pct:5.1f}%)")

# Check diversity
print(f"\n{'='*60}")
print("DIVERSITY ANALYSIS")
print(f"{'='*60}")

if len(domain_counts) == 1:
    print("❌ WARNING: All samples from ONE domain - detector may not generalize!")
elif len(domain_counts) < 3:
    print("⚠️  CAUTION: Limited diversity (< 3 domains) - test on more domains")
else:
    print(f"✓ Good diversity: {len(domain_counts)} different domains")

if len(func_counts) < 10:
    print("❌ WARNING: Very few unique functions - limited coverage")
elif len(func_counts) < 30:
    print("⚠️  CAUTION: Moderate function diversity")
else:
    print(f"✓ Good function diversity: {len(func_counts)} unique functions")

# Sample a few examples
print(f"\n{'='*60}")
print("SAMPLE EXAMPLES")
print(f"{'='*60}")
for i in range(min(5, len(data))):
    ex = data[i]
    system = ex['system'][:200]
    user = ex['user'][:100]
    print(f"\nExample {i+1}:")
    print(f"  System: {system}...")
    print(f"  User: {user}...")
    print(f"  Label: {'HALLUCINATION' if ex['label'] == 1 else 'VALID'}")
