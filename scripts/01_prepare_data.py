import json
import random
import re
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# Constants
# Relaxed keywords to match function definitions in system prompt
DOMAIN_KEYWORDS = [
    "calculator", "math", "finance", "currency", "exchange", 
    "loan", "interest", "tax", "convert", "unit", "calculate", "computing"
]

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def normalize_tool_call(tool_call_str):
    """
    Simple normalization to handle whitespace/quote differences.
    Extracts function name and args.
    """
    normalized = "".join(tool_call_str.split())
    return normalized

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2500, help="Number of examples to generate")
    parser.add_argument("--offset", type=int, default=0, help="Start index for selection (skip training data)")
    parser.add_argument("--temp", type=float, default=1.5, help="Temperature for generation (Stress Test: 1.5)")
    parser.add_argument("--output", type=str, default="data/validation_set.jsonl")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling probability")
    parser.add_argument("--ablate_layer", type=int, default=None, help="Layer index to ablate (zero out attention)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--tool_only", action="store_true", help="Filter for tool call examples only")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model ID")
    return parser.parse_args()

def get_ablation_hook():
    def hook(module, input, output):
        # Output of self_attn is typically (attn_output, ...other_stuff)
        # We want to zero out attn_output [batch, seq, hidden]
        if isinstance(output, tuple):
            # Qwen/Llama attention returns (attn_output, past_key_value, attention_weights)
            # We clone the tuple to modify the first element
            attn_output = output[0]
            # Zero it out
            attn_output = torch.zeros_like(attn_output)
            return (attn_output,) + output[1:]
        else:
            return torch.zeros_like(output)
    return hook

def main():
    args = parse_args()
    
    # ... (Load Dataset and Model logic unchanged) ...
    # REDACTED for brevity in tool call, will rely on context matching 
    # Actually I need to match the file context carefully.
    
    # I will split this into two edits to be safe. 
    # Edit 1: Arguments.
    # Edit 2: Hook Logic and Generate Call.


def is_target_domain(example):
    # Disable filtering to get more data
    return True
    # Search for keywords in the system prompt (which contains function descriptions)
    # sys_prompt = example.get('system', '').lower()
    # return any(k in sys_prompt for k in DOMAIN_KEYWORDS)

def parse_chat(chat_str):
    """
    Parses Glaive v2 raw text chat into a list of messages.
    Format: USER: ... \n\n ASSISTANT: ...
    """
    messages = []
    # Split by known roles, keeping delimiters
    # Using a lookahead for split might be cleaner, or simple iterative find
    parts = re.split(r'(?:\n|^)(USER|ASSISTANT|FUNCTION RESPONSE):', chat_str)
    # parts[0] is empty or pre-text
    # parts[1] is role, parts[2] is content, parts[3] is role...
    
    for i in range(1, len(parts), 2):
        role_marker = parts[i].strip()
        content = parts[i+1].strip()
        
        # Clean <|endoftext|>
        content = content.replace("<|endoftext|>", "").strip()
        
        role = "user"
        if role_marker == "ASSISTANT":
            role = "assistant"
        elif role_marker == "FUNCTION RESPONSE":
            role = "tool" # or user? usually tool output comes as tool role
        
        messages.append({"role": role, "content": content})
        
    return messages

def main():
    args = parse_args()
    
    print(f"Loading Glaive v2 dataset...")
    try:
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Filtering for keywords: {DOMAIN_KEYWORDS}...")
    filtered_ds = [ex for ex in ds if is_target_domain(ex)]
    print(f"Found {len(filtered_ds)} matching examples.")
    
    if len(filtered_ds) < 100:
        print("Warning: Very few examples found. Checking 'system' fields of a few examples...")
        for i in range(min(5, len(ds))):
            print(f"Sample {i} system: {ds[i].get('system', 'N/A')}")
            
    if len(filtered_ds) == 0:
        print("Warning: No examples found. Check filters.")
        return

    # selected_ds logic removed as it pre-slices
    print(f"Iterating full dataset (Total: {len(ds)}) looking for {args.limit} valid tool calls...")
    
    print(f"Loading Model: {args.model} (4-bit: {args.load_in_4bit})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    print(f"Generating shadow dataset (Temp={args.temp})...")
    
    # Register Ablation Hook
    hook_handle = None
    if args.ablate_layer is not None:
        print(f"!!! ABLATION ACTIVE: Zeroing output of Layer {args.ablate_layer} !!!")
        try:
            # Try standard HF naming: model.model.layers[i].self_attn
            layer_module = model.model.layers[args.ablate_layer].self_attn
            hook_handle = layer_module.register_forward_hook(get_ablation_hook())
        except AttributeError:
            print("Error: Could not find layer module for ablation. Check model structure.")
            return

    results = []
    hallu_count = 0
    
    # Ensure directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Iterate until we have enough samples
    count = 0
    pbar = tqdm(total=args.limit)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        # Start from offset, iterate indefinitely (until limit)
        for i, ex in enumerate(ds):
            if i < args.offset:
                continue
                
            if count >= args.limit:
                break
                
            # ... (parsing logic) ...
            chat_raw = ex['chat']
            chat = []
            if isinstance(chat_raw, str):
                chat = parse_chat(chat_raw)
            elif isinstance(chat_raw, list):
                chat = chat_raw
            else:
                continue
                
            if not chat:
                continue
    
            if not chat:
                continue
    
            ground_truth_response = None
            input_messages = None
            
            # Find target message
            # If tool_only, we search backwards for the first tool call
            target_idx = -1
            
            for i in range(len(chat) - 1, -1, -1):
                msg = chat[i]
                if msg['role'] == 'assistant':
                    content = msg.get('content', '').strip()
                    if args.tool_only:
                        # Relaxed check: Look for JSON structure
                        if '{' in content and '}' in content:
                            target_idx = i
                            break
                    else:
                        # If not tool_only, take the last assistant message (default behavior)
                        target_idx = i
                        break
            
            if target_idx == -1:
                continue
                
            ground_truth_response = chat[target_idx]['content']
            input_messages = chat[:target_idx]
            
            # Prepare Zero-Shot Input
            last_user_msg = None
            for msg in reversed(input_messages):
                if msg['role'] == 'user':
                    last_user_msg = msg
                    break
                    
            if not last_user_msg:
                continue 
                
            zero_shot_messages = []
            if ex.get('system'):
                zero_shot_messages.append({"role": "system", "content": ex['system']})
            zero_shot_messages.append(last_user_msg)
            
            # Format input
            input_text = tokenizer.apply_chat_template(zero_shot_messages, tokenize=False, add_generation_prompt=True)
            
            # Generate
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=128, 
                    temperature=args.temp, 
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Labeling
            gt_norm = normalize_tool_call(ground_truth_response)
            gen_norm = normalize_tool_call(generated_text)
            
            is_hallucination = 0 if gt_norm == gen_norm else 1
            
            if is_hallucination:
                hallu_count += 1
                
            result_obj = {
                "system": ex.get('system', ''),
                "input_messages": input_messages, # Keep original context for reference? Or zero shot?
                # User might want to see what was actually fed? 
                # But 'input_messages' key is used by downstream script to reconstruct prompt?
                # Downstream script users 'input_messages'. If I change this to zero_shot_messages, 
                # downstream script will analyze zero-shot context.
                # Since we generated with zero shot, we should probably analyze zero shot?
                # The user said "Forces the model to rely entirely on its internal reasoning geometry...".
                # Spectral Analysis should likely run on the SAME context used for generation.
                "input_messages": zero_shot_messages, 
                "ground_truth": ground_truth_response,
                "generated": generated_text,
                "label": is_hallucination,
                "domain_match": True 
            }
            results.append(result_obj)
            
            # Write immediately
            f.write(json.dumps(result_obj) + "\n")
            f.flush()
            
            count += 1
            pbar.update(1)
            
    # Summary
    if len(results) > 0:
        print(f"Hallucination Rate: {hallu_count}/{len(results)} ({hallu_count/len(results):.2%})")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
