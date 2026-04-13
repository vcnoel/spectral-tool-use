import json
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/verify_extraction.py <path_to_jsonl>")
    sys.exit(1)

path = sys.argv[1]
try:
    with open(path) as f:
        line = f.readline()
        if not line:
            print(f"ERROR: File {path} is empty.")
            sys.exit(1)
        sample = json.loads(line)
except Exception as e:
    print(f"ERROR: Failed to read {path}: {e}")
    sys.exit(1)

spectral = sample.get('spectral', {})
if not spectral:
    print("ERROR: No spectral data in sample. All framework calls failed.")
    sys.exit(1)

print(f"{'Layer':<6} | {'fiedler_value':>13} | {'smoothness_index':>16} | {'spectral_entropy':>16} | {'hfer':>6} | {'energy':>8}")
print('-' * 80)

errors = []
# Assuming 16 layers for Llama-1b or similar
num_layers = len(spectral) 
for layer_idx in range(num_layers):
    l = str(layer_idx)
    if l not in spectral:
        print(f"L{layer_idx:02d}   | MISSING")
        errors.append(f'Layer {layer_idx} missing')
        continue
    m = spectral[l]
    fv = m.get('fiedler_value', float('nan'))
    si = m.get('smoothness_index', float('nan'))
    se = m.get('spectral_entropy', float('nan'))
    h  = m.get('hfer', float('nan'))
    en = m.get('energy', float('nan'))
    print(f"L{layer_idx:02d}   | {fv:13.6f} | {si:16.6f} | {se:16.6f} | {h:6.4f} | {en:8.4f}")

    if not (0 <= si <= 1 + 1e-4) and si == si:
        errors.append(f'L{layer_idx} smoothness out of range: {si}')
    if fv < -1e-4 and fv == fv:
        errors.append(f'L{layer_idx} fiedler negative: {fv}')
    if not (0 <= h <= 1 + 1e-4) and h == h:
        errors.append(f'L{layer_idx} hfer out of range: {h}')
    if se < 0 and se == se:
        errors.append(f'L{layer_idx} entropy negative: {se}')
    if fv != fv:
        errors.append(f'L{layer_idx} fiedler is NaN — framework call failed')

if errors:
    print('\nFAILED:')
    for e in errors: print(f'  {e}')
else:
    print('\nAll sanity checks passed.')
