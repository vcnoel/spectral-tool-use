"""Registered model set (docs/REGISTRATION_REBUILD.md section 1). Written before any clean run.

side: the side predicted by the paper's family rule (Llama-3.2 and Gemma on the probe
side, "internals needed"; Qwen3, Qwen3.5 and MiniCPM5 on the confidence side). The
Qwen3.5-4B Base arm follows docs/REGISTRATION_SWAP.md (no post-training -> probe side).
route: the route the render check predicts (tokenizer-only, 5 Oct 2026); the run records
the route it used.
"""

MODELS = [
    # hf id, family, params (B), predicted side, expected route, slot, notes
    ("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2", 1.2, "probe", "native", "laptop", "paper"),
    ("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2", 3.2, "probe", "native", "laptop", "paper"),
    ("google/gemma-3-1b-it", "Gemma-3", 1.0, "probe", "fallback_list", "laptop", "paper; template drops tools"),
    ("Qwen/Qwen3-1.7B", "Qwen3", 1.7, "confidence", "native", "laptop", "paper"),
    ("openbmb/MiniCPM5-2B", "MiniCPM5", 2.0, "confidence", "native", "laptop", "paper"),
    ("Qwen/Qwen3.5-0.8B", "Qwen3.5", 0.8, "confidence", "native", "laptop", "paper"),
    ("Qwen/Qwen3.5-4B-Base", "Qwen3.5", 4.0, "probe", "native", "laptop", "swap pair, base arm (REGISTRATION_SWAP)"),
    ("Qwen/Qwen3.5-4B", "Qwen3.5", 4.0, "confidence", "native", "laptop", "swap pair, post-trained arm"),
    ("Qwen/Qwen3-4B", "Qwen3", 4.0, "confidence", "native", "laptop", "B3"),
    ("Qwen/Qwen3.5-2B", "Qwen3.5", 2.0, "confidence", "native", "laptop", "B3"),
    ("google/gemma-2-2b-it", "Gemma-2", 2.6, "probe", "fallback_list", "laptop", "B3; template takes no system turn"),
    ("google/gemma-3-4b-it", "Gemma-3", 4.3, "probe", "fallback_list", "laptop", "B3; gated, not cached (8.6 GB)"),
    ("Qwen/Qwen3.5-27B", "Qwen3.5", 27.0, "confidence", "native", "a100", "reserved"),
    ("google/gemma-3-27b-it", "Gemma-3", 27.0, "probe", "fallback_list", "a100", "reserved; gated"),
    # docs/SOTA_REVIEW_2026.md section 4: present in almost every 2025-26 detection paper. Family rule:
    # Llama -> probe side. Local cache deleted; the pod downloads it (gated).
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1", 8.0, "probe", "native", "a100", "cross-paper anchor; gated"),
]
PAPER_SIX = [m[0] for m in MODELS[:6]]
SWAP_PAIR = ["Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-4B"]
B3 = ["Qwen/Qwen3-4B", "Qwen/Qwen3.5-2B", "google/gemma-2-2b-it", "google/gemma-3-4b-it"]
ROUTE_CONTROL_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # supports both routes
EVALUATED = {m[0] for m in MODELS}

# Reader / output-only judge candidates: NOT one of the evaluated checkpoints, and not of an
# evaluated family where avoidable. The author chooses (docs/PIPELINE_REBUILD.md section 8).
READER_CANDIDATES = [
    ("HuggingFaceTB/SmolLM3-3B", "SmolLM3", "cached; outside every evaluated family; tool template renders natively"),
    ("allenai/OLMo-2-0425-1B-Instruct", "OLMo-2", "cached; outside every evaluated family; no tool template (reads text only)"),
    ("microsoft/Phi-4-mini-instruct", "Phi-4", "not cached (download needed)"),
]


def side_of(model_id: str) -> str:
    for m in MODELS:
        if m[0] == model_id:
            return m[3]
    raise KeyError(model_id)


def family_of(model_id: str) -> str:
    for m in MODELS:
        if m[0] == model_id:
            return m[1]
    raise KeyError(model_id)


# Baselines (docs/SOTA_REVIEW_2026.md section 3.3). Every one is computable from the stored
# arrays (rebuild/loader.py) except where noted. Reference code as listed in the review; none
# was obtainable offline on 5 Oct 2026, so each is a reimplementation until a clone is present.
BASELINES = [
    # name, what, stored input, reference
    ("probe_last_token", "linear probe on the last generated token's state, mid-to-late layer (Yeats et al. 2608.27750)",
     "hid[role=last] at every layer", "github.com/Trustworthy-ML-Lab/when2tool (pipeline pattern)"),
    ("probe_three_position_final", "name onset + mean argument span + closing delimiter at the final layer (Healy et al. 2601.05214)",
     "hid[role in name,args,close] at layer L", "none public"),
    ("probe_prevalue_2_3", "state just before each parameter value at about 2/3 depth (Yu et al. 2608.03071)",
     "hid[role=prevalue] at round(2L/3)", "none public"),
    ("probe_token_role_registered", "this paper's token-role probe: name, value mean, close at 8 depths",
     "hid[role in name,value,close]", "this repository"),
    ("gnll", "greedy sequence NLL (Ye et al. 2604.22985)", "logp", "none public"),
    ("gnll_smt", "NLL over semantically meaningful tokens: function name and argument values (Ye et al.)", "logp, token_role", "none public"),
    ("nll_max_avg", "MAX and AVG token NLL (Ye et al.)", "logp", "none public"),
    ("p_true", "P(True) self-evaluation, one extra forward pass (Ye et al.)", "confidence.p_true (rebuild/ptrue.py)", "none public"),
    ("attention_margin", "gold-tool segment mass minus mean distractor-segment mass, averaged over layers and heads (Chen 2606.16364)",
     "tool_mass, gold_tool_index", "none public"),
    ("lapeigvals", "top-k Laplacian diagonal profile over layers and heads (Binkowski et al. 2502.17598)", "lapeig",
     "github.com/graphml-lab-pwr/lapeigvals"),
    ("sinkprobe", "top-k sink scores over layers and heads (Binkowski et al. 2604.10697)", "sink", "github.com/graphml-lab-pwr/sink-probe"),
    ("lookback_lens", "per-head context/generation attention shares (Chuang et al. 2407.07071)", "lookback", "github.com/voidism/Lookback-Lens"),
    ("llm_check_logdet", "attention-kernel log-determinant (Sriramanan et al., NeurIPS 2024); NOT stored: needs the kernel, a GPU pass",
     "not stored", "github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection"),
]
