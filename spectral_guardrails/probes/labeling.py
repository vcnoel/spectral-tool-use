import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ..utils.data import normalize_tool_call

def mask_tool_call(prompt: str, tool_call: str) -> str:
    """
    Remove tool call from prompt, preserve query + context.
    Simply strips the tool call string if present at the end.
    """
    if prompt.endswith(tool_call):
        return prompt[:-len(tool_call)].strip()
    return prompt.strip()

def generate_prediction(model, tokenizer, masked_prompt: str) -> tuple[str, dict]:
    """
    Generates a response and returns the hidden states.
    """
    inputs = tokenizer(masked_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64, # Optimized per user request
            pad_token_id=tokenizer.eos_token_id
        )
    
    predicted_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return predicted_text, None

def extract_function_name(text: str) -> str | None:
    """
    Priority 1: Parse <functioncall> JSON tag
        e.g. '<functioncall> {"name": "get_news_headlines", ...}'
        → "get_news_headlines"
    
    Priority 2: Parse raw JSON object with "name" key
        e.g. '{"name": "calculate_loan_payment", "arguments": {...}}'
        → "calculate_loan_payment"
    
    Priority 3: Return None (model did not invoke any tool)
    
    Never do free-text parsing. If neither JSON pattern is found,
    return None — do not try to extract words from prose.
    """
    import json, re
    
    # Priority 1: <functioncall> tag
    fc_match = re.search(r'<functioncall>\s*(\{.*?\})', text, re.DOTALL)
    if fc_match:
        try:
            obj = json.loads(fc_match.group(1))
            if "name" in obj:
                return obj["name"].lower().replace("_", "")
        except json.JSONDecodeError:
            pass
    
    # Priority 2: Raw JSON object with "name" key
    # Modified regex to be slightly more robust to whitespace
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"', text)
    if json_match:
        return json_match.group(1).lower().replace("_", "")
    
    # Priority 3
    return None

def assign_label(predicted_text: str, ground_truth_text: str) -> int:
    """
    Extract function names from both texts.
    
    Case A — Ground truth is a tool call (has a function name):
        Correct behavior = model also produces matching function name
        label 0 if match, else 1
        
    Case B — Ground truth is a refusal/chat (no function name):
        Correct behavior = model also produces no tool call (None)
        label 0 if None, else 1 (hallucinated tool call)
    """
    truth_name = extract_function_name(ground_truth_text)
    pred_name  = extract_function_name(predicted_text)
    
    # Case B: Ground truth requires NO tool call
    if truth_name is None:
        label = 0 if pred_name is None else 1
    # Case A: Ground truth requires a SPECIFIC tool call
    else:
        if pred_name is None:
            label = 1 # tool bypass = hallucination
        else:
            label = 0 if pred_name == truth_name else 1
            
    # Required DEBUG prints for smoke tests (Disabled for performance)
    # print(f"[DEBUG] raw predicted: '{predicted_text.strip()}'")
    # print(f"[DEBUG] raw ground truth: '{ground_truth_text.strip()}'")
    # print(f"[DEBUG] normalized predicted: '{pred_name}'")
    # print(f"[DEBUG] normalized truth: '{truth_name}'")
    # print(f"[DEBUG] label assigned: {label}")
    
    return label

def build_dataset(model, tokenizer, samples: list[dict], cache_dir: str) -> list[dict]:
    """
    Full pipeline: mask → predict → label → cache hidden states.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    
    for i, ex in enumerate(tqdm(samples)):
        prompt = ex.get('prompt', '')
        gt = ex.get('ground_truth', '')
        
        masked = mask_tool_call(prompt, gt)
        pred, hidden = generate_prediction(model, tokenizer, masked)
        label = assign_label(pred, gt)
        
        # We only need the hidden states from the forward pass of the masked prompt 
        # to get the "pre-generation" features if that's what Healy et al. use.
        # Actually, Healy investigates hidden states at specific positions in the GENERATED sequence.
        
        dataset.append({
            "original_idx": i,
            "predicted": pred,
            "ground_truth": gt,
            "label": label,
            "masked_prompt": masked
        })
        
        # Save to disk
        with open(cache_path / f"sample_{i}.json", 'w') as f:
            json.dump(dataset[-1], f)
            
    return dataset
