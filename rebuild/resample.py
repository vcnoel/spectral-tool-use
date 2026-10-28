"""Model-calibrated construction by resampling (docs/REGISTRATION_REBUILD.md section 4, H6).

For every call-expected item, besides the greedy generation, K samples are drawn at
temperature T (registered: K = 8, T = 0.7, top_p = 1.0, same prompt, stop ids and token budget;
sampling seed = run seed * 1_000_003 + item index, so the samples are reproducible). Each
sample is labelled with the same labeller and tolerance as the greedy call (a sample is a
SUCCESS iff the labeller returns `valid`). With s successes out of K the item is classified,
PER MODEL:

  always_solved   s == K                 controls
  never_solved    s == 0                 capability failures
  within_reach    K > s > K * THRESHOLD  solved in a majority of samples, failed in some
  marginal        0 < s <= K * THRESHOLD mixed below the majority (kept, not within reach)

THRESHOLD = 0.5 (with K = 8: within reach iff s in {5, 6, 7}). Features (the teacher-forced
pass of rebuild/extract_clean.featurize) are extracted for at most M_SUCCESS success samples
and M_FAILURE failure samples of every within-reach item (the first in sample order), so that
the H6 and H4 contrasts are within item and within model: identical prompt, tools, template and
route. Capability failures and controls are represented by their greedy generation, which is
featured for every item anyway. The K sample texts, labels, types and confidence summaries are
stored for every item (samples.jsonl); the per-model item lists and this script are the
artifact to release.

Held-out tools: the classification uses labels only (never features) and defines a population,
like the scored population does; every statistic on it is still computed out of fold under the
tool-grouped folds, and the within-item pairs of a test fold belong to tools unseen in training.
"""
from __future__ import annotations

import numpy as np
import torch

K = 8
TEMPERATURE = 0.7
TOP_P = 1.0
THRESHOLD = 0.5
M_SUCCESS = 2
M_FAILURE = 2
CLASSES = ("always_solved", "never_solved", "within_reach", "marginal", "not_resampled")


def classify(n_success: int, k: int = K, threshold: float = THRESHOLD) -> str:
    if n_success == k:
        return "always_solved"
    if n_success == 0:
        return "never_solved"
    return "within_reach" if n_success > k * threshold else "marginal"


@torch.no_grad()
def sample(model, tok, inputs, stop_ids, max_new_tokens: int, k: int, temperature: float, seed: int):
    """K sampled continuations of one prompt. Returns (gen_id lists, logp arrays, entropy arrays)."""
    g = torch.Generator(device=model.device).manual_seed(seed) if model.device.type == "cuda" else None
    torch.manual_seed(seed)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                         top_p=TOP_P, top_k=0, num_return_sequences=k, pad_token_id=tok.eos_token_id,
                         eos_token_id=stop_ids, return_dict_in_generate=True, output_scores=True)
    P = inputs.input_ids.shape[1]
    seqs = out.sequences[:, P:]
    ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float().cpu().numpy()
    # predictive entropy per step and sample: [k, steps]
    ent = np.stack([(-(torch.softmax(s.float(), -1) * torch.log_softmax(s.float(), -1)).sum(-1)).cpu().numpy()
                    for s in out.scores], 1).astype(np.float32)
    gens, lps, ents = [], [], []
    for j in range(k):
        ids = seqs[j].tolist()
        # trim at the first stopping id (included) and the padding after it
        cut = len(ids)
        for i, t in enumerate(ids):
            if t in stop_ids:
                cut = i + 1
                break
        gens.append(ids[:cut])
        lps.append(ts[j, :cut].astype(np.float32))
        ents.append(ent[j, :cut])
    del out
    return gens, lps, ents


def choose_featured(labels: list[int], m_success: int = M_SUCCESS, m_failure: int = M_FAILURE) -> list[int]:
    """Indices of the samples to feature for a within-reach item: first m successes, first m failures."""
    s = [i for i, l in enumerate(labels) if l == 0][:m_success]
    f = [i for i, l in enumerate(labels) if l == 1][:m_failure]
    return sorted(s + f)
