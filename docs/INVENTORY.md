# Inventory (29 September 2026)

Built by `analysis/inventory.py` from the first record of every
`data/pilot_v2_*/features.jsonl` on this machine; the machine-readable copy is
`data/theory/inventory.json`.

## Models on disk

| model | released | params | runs |
|---|---|---|---|
| Llama-3.2-1B-Instruct | 2024-09 | 1B | Glaive, BFCL, BFCL-live, multi-turn |
| Llama-3.2-3B-Instruct | 2024-09 | 3B | Glaive, BFCL |
| Gemma-3-1B-it | 2025-03 | 1B | Glaive |
| Qwen3-1.7B | 2025-04 | 1.7B | BFCL |
| MiniCPM5-2B | 2026-09 | 2B | BFCL, BFCL-live, multi-turn |
| Qwen3.5-0.8B | 2026-02 | 0.8B | BFCL |
| Qwen3.5-2B | 2026-02 | 2B | Glaive (21 positives, underpowered) |
| Qwen3.5-4B | 2026-02 | 4B | BFCL |

Largest model: 4B. Newest: MiniCPM5-2B. Eight checkpoints, five families.

## What each run stores

Every `base_*`, `conf_*`, `mt_*` and 2026-model run stores: per-head spectral
metrics on the call span `[L][H][5]`, the LapEigvals diagonal `[L][H][100]`,
Lookback ratios, token-role hidden states at eight depths, per-token residual
states at one layer (capped at 48 tokens), mean log-probability, and the
generation text from which labels are recomputed. `conf_*` and `mt_*` also store
nine confidence summaries.

Not stored anywhere on this machine: raw attention rows, SinkProbe scores and
the anchored readout (both computed in the extraction hook only from the 13
September commits onward; the pilot dump `anch_llama1b_glaive` is on the other
machine), and any run above 4B.

## What that means for CPU work

Answerable from disk: any contrast between stored feature families, label
thinning, confidence summaries (on `conf_*`, `mt_*`), family-split statistics,
fusion, resolution controls on per-head features, surface floors, per-token
probes at one layer.

Needs a forward pass: any new anchoring position in the attention rows (for
instance the rows of the argument-value tokens), SinkProbe or the anchored
readout on runs other than the pilot, any model above 4B, the distractor-tool
dose ladder, and the forced-format control.
