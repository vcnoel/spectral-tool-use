
import json
import argparse
import re
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="Number of examples")
    parser.add_argument("--offset", type=int, default=0, help="Start offset")
    parser.add_argument("--temp", type=float, default=0.3, help="Temperature")
    parser.add_argument("--output", type=str, default="data/toolbench_1000.jsonl")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser.parse_args()

def normalize(s):
    # Remove whitespace, converting to single lines
    return "".join(s.split())

def main():
    args = parse_args()

    print(f"Loading ToolBench dataset (stream)...")
    ds = load_dataset("tuandunghcmut/toolbench-v1", split="train", streaming=True)
    
    print(f"Loading Model: {args.model}...")
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

    results = []
    generated_count = 0
    hallu_count = 0 
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(total=args.limit)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, ex in enumerate(ds):
            if i < args.offset: continue
            if generated_count >= args.limit: break
            
            # Parse Columnar Conversations
            # conversations: {'from': [...], 'value': [...]}
            convs = ex.get('conversations', {})
            roles = convs.get('from', [])
            values = convs.get('value', [])
            
            if not roles or not values: continue
            
            # Extract System, User, GT
            sys_prompt = ""
            user_input = ""
            gt_response = ""
            
            # Heuristic: 
            # 0: system
            # 1: user
            # 2: assistant (GT)
            if len(roles) >= 3 and roles[0] == 'system' and roles[1] == 'user' and roles[2] == 'assistant':
                sys_prompt = values[0]
                user_input = values[1]
                gt_response = values[2]
            else:
                # Search
                for r, v in zip(roles, values):
                    if r == 'system': sys_prompt = v
                    elif r == 'user' and not user_input: user_input = v # First user
                    elif r == 'assistant' and not gt_response and user_input: gt_response = v # First assistant after user
            
            if not user_input or not gt_response: 
                continue

            # Construct Messages for Llama
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_input})
            
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=args.temp,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Determine Label
            # ToolBench GT is ReAct. Llama might output ReAct.
            # Comparison: text similarity?
            # Or: if generated contains "Action:" and GT contains "Action:", do they match?
            
            label = 0 # Faithful
            
            # Simple normalization first
            norm_gt = normalize(gt_response)
            norm_gen = normalize(generated_text)
            
            if norm_gt != norm_gen:
                label = 1
                
            # Refinement: If ReAct, check Action Name
            # (TODO: Add smarter parsing if needed)
            
            if label == 1: hallu_count += 1
            
            res = {
                "system": sys_prompt,
                "input_messages": messages, # For analysis script
                "user": user_input,
                "ground_truth": gt_response,
                "generated": generated_text,
                "label": label
            }
            results.append(res)
            f.write(json.dumps(res) + "\n")
            f.flush()
            
            generated_count += 1
            pbar.update(1)
            
    print(f"Hallucination Rate: {hallu_count}/{generated_count} ({hallu_count/generated_count:.2%})")

if __name__ == "__main__":
    main()
