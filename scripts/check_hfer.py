import json

sample = json.loads(open('data/n100_gor/spectral_features_llama1b_general.jsonl').readline())
print(f"{'Layer':<6} {'hfer':>10} {'entropy':>10} {'smoothness':>12} {'fiedler':>10}")
print('-' * 55)
for layer in range(16):
    m = sample['spectral'][str(layer)]
    hfer = m['hfer']
    entropy = m['spectral_entropy']
    smoothness = m['smoothness_index']
    fiedler = m['fiedler_value']
    hfer_flag = ' <-- OUT OF RANGE' if hfer > 1.0 else ''
    print(f"L{layer:02d}   {hfer:10.6f} {entropy:10.6f} {smoothness:12.6f} {fiedler:10.6f}{hfer_flag}")

print()
print('HFER range check:', 'FAIL' if any(
    sample['spectral'][str(l)]['hfer'] > 1.0 for l in range(16)
) else 'PASS')
