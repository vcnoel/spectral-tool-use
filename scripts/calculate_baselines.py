
import json
import torch
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading data from {args.data}...")
    data = []
    with open(args.data, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    print(f"Loaded {len(data)} examples.")
    
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
    
    results = []
    
    limit = args.limit if args.limit else len(data)
    
    for i, ex in enumerate(tqdm(data[:limit])):
        # Support both 'generated' and 'ground_truth' for comparison? 
        # Usually we want PPL of the GENERATED response to see if model was "unsure" when it hallucinated.
        
        input_messages = ex.get('input_messages', [])
        generated = ex.get('generated', '')
        label = ex.get('label', 0) # 1 = Hallucination
        
        # Sanitize messages for Mistral/Llama chat templates if needed
        # (Same logic as sweep script to be safe)
        sanitized_messages = []
        for msg in input_messages:
            role = msg['role']
            content = msg['content']
            if role in ['tool', 'function', 'function response']:
                role = 'user'
            if sanitized_messages and sanitized_messages[-1]['role'] == role:
                sanitized_messages[-1]['content'] += "\n\n" + content
            else:
                sanitized_messages.append({"role": role, "content": content})
                
        try:
            prompt_text = tokenizer.apply_chat_template(sanitized_messages, tokenize=False, add_generation_prompt=True)
        except:
            # Fallback
            prompt_text = ""
            for m in sanitized_messages:
                prompt_text += f"{m['role'].upper()}: {m['content']}\n"
            prompt_text += "ASSISTANT:"
            
        full_text = prompt_text + generated
        
        # Tokenize
        enc_prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        enc_full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        
        input_ids = enc_full.input_ids.to(model.device)
        len_prompt = enc_prompt.input_ids.shape[1]
        len_full = input_ids.shape[1]
        
        if len_full <= len_prompt:
            # Empty generation or error
            continue
            
        # Labels: -100 for prompt
        labels = input_ids.clone()
        labels[:, :len_prompt] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            # This is average NLL over the target tokens
            
            # Calculate token-wise log probs manually to get min/std?
            # loss is scalar mean.
            # To get individual token probs:
            logits = outputs.logits # [1, seq, vocab]
            # Shift logits and labels
            # logits[..., :-1, :] predicts labels[..., 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # CrossEntropy loss per token (reduction='none')
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Token losses is flattened. We need to mask out -100 ones.
            # But shift_labels has -100 for prompt.
            active_mask = shift_labels.view(-1) != -100
            valid_losses = token_losses[active_mask]
            
            ppl = torch.exp(loss).item()
            mean_logprob = -valid_losses.mean().item()
            min_logprob = -valid_losses.max().item() # Max loss = Min logprob
            
        results.append({
            'is_hallucination': label,
            'ppl': ppl,
            'logprob_mean': mean_logprob,
            'logprob_min': min_logprob,
            'len_gen': len_full - len_prompt
        })
        
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
