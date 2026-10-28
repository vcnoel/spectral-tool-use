"""P(True) self-evaluation (Ye et al. 2604.22985, the function-calling UQ study; the prompt
wording below is this pipeline's registered rendering of their definition, since no code was
released): the model is shown the request, the tools and its own call, asked whether the
call is correct, and the probability of the first token of "True" against "False" is read
from one forward pass. Recorded per item as `confidence.p_true`; nan when the generation is
empty or the verification prompt exceeds the prompt cap.
"""
from __future__ import annotations

import math

import torch

from rebuild import prompts

QUESTION = ("Below is a user request and a proposed function call made in response to it.\n\n"
            "User request:\n{user}\n\nProposed function call:\n{call}\n\n"
            "Is the proposed function call correct and complete for this request? "
            "Answer with exactly one word: True or False.")


def _first_token_ids(tok, word: str) -> set:
    ids = set()
    for v in (word, " " + word, word.lower(), " " + word.lower()):
        enc = tok(v, add_special_tokens=False).input_ids
        if enc:
            ids.add(enc[0])
    return ids


def render(tok, ex: dict, call_text: str, route: str):
    user = QUESTION.format(user=ex["user"], call=call_text if call_text.strip() else "(no call was made)")
    if route == "native":
        text, why = prompts.render_native(tok, ex["tools"], user)
        if text is not None:
            return text
    text, _ = prompts.render_fallback(tok, ex["tools"], user)
    return text


@torch.no_grad()
def p_true(model, tok, ex: dict, call_text: str, route: str, max_prompt_tokens: int = 2048) -> float:
    text = render(tok, ex, call_text, route)
    if text is None:
        return float("nan")
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] > max_prompt_tokens:
        return float("nan")
    out = model(input_ids=ids.to(model.device), use_cache=False)
    logits = out.logits[0, -1].float()
    t_ids, f_ids = _first_token_ids(tok, "True"), _first_token_ids(tok, "False")
    lt = torch.logsumexp(logits[sorted(t_ids)], 0)
    lf = torch.logsumexp(logits[sorted(f_ids)], 0)
    return float(torch.sigmoid(lt - lf)) if math.isfinite(float(lt - lf)) else float("nan")
