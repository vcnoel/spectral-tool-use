import json
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# Assuming spectral_trust is installed and has the necessary API
# User's pseudocode: metrics = spectral_trust.compute_all(...)
import spectral_trust
from spectral_trust import GSPDiagnosticsFramework, GSPConfig, GraphConstructor, SpectralAnalyzer

# Mapping user pseudocode to library calls (assuming library structure from previous turns)
# The user mentioned: metric.hfer, metric.fiedler_value, etc.
# I might need to adapt if the library API is slightly different, but I'll follow the user's snippet.

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/validation_set.jsonl")
    parser.add_argument("--output", type=str, default="data/sweep_results.csv")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading validation set from {args.input}...")
    data = []
    with open(args.input, 'r') as f:
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
        device_map="auto",
        output_attentions=True # Explicitly request attentions in config
    )
    
    results = []
    
    # Initialize Spectral Tools
    config = GSPConfig(model_name=args.model) # Minimal config
    graph_constructor = GraphConstructor(config)
    spectral_analyzer = SpectralAnalyzer(config)
    
    print("Starting Spectral Sweep...")
    
    # Initialize CSV with header
    # We need to know columns ahead of time or just write the first row with header
    # But layer metrics depend on model depth, which we know from loop?
    # Actually, we know 0..31 layers and 4 metrics. We can construct header or just let pandas do it for first row.
    
    # Resume Logic
    start_idx = 0
    file_initialized = False
    
    import os
    if os.path.exists(args.output):
        try:
            existing_df = pd.read_csv(args.output)
            start_idx = len(existing_df)
            print(f"Resuming from index {start_idx} (found {start_idx} rows in {args.output})")
            file_initialized = True
        except:
            print(f"Could not read {args.output}, starting from scratch.")
    
    for idx, example in enumerate(tqdm(data)):
        if args.limit is not None and (idx - start_idx) >= args.limit:
            break
            
        if idx < start_idx:
            continue
            
        # Reconstruct input
        input_messages = example['input_messages'] 
        generated_response = example['generated'] 
        
        # Apply template
        
        # Sanitize messages for chat template (Strict alternation User/Assistant)
        sanitized_messages = []
        for msg in input_messages:
            role = msg['role']
            content = msg['content']
            
            # Map tool/function to user (observation)
            if role in ['tool', 'function', 'function response']:
                role = 'user'
            
            # Merge with previous if same role
            if sanitized_messages and sanitized_messages[-1]['role'] == role:
                sanitized_messages[-1]['content'] += "\n\n" + content
            else:
                sanitized_messages.append({"role": role, "content": content})
                
        # Ensure first message is not System (if template doesn't support it) or Handle System
        # Mistral v0.1 usually handles System but let's be safe. 
        # Actually usually User is first.
        
        # Apply template
        try:
            prompt_text = tokenizer.apply_chat_template(sanitized_messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            # Fallback if template fails: Simple concat
            prompt_text = ""
            for m in sanitized_messages:
                prompt_text += f"{m['role'].upper()}: {m['content']}\n"
            prompt_text += "ASSISTANT:"
        
        # We analyze the generated response context
        full_text = prompt_text + generated_response
        
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        
        # 1. Forward Pass with Hook
        with torch.no_grad():
            outputs = model(
                **inputs, 
                output_attentions=True, 
                output_hidden_states=True
            )
            
        # 2. Iterate every layer (The Grid)
        layer_metrics = {}
        
        # We utilize the full context graph for stability as per standard spectral analysis
        # But we pass the hidden states for the full sequence
        
        for layer_idx in range(len(outputs.attentions)):
            # Shape: [Batch, Heads, Seq, Seq] -> [Heads, Seq, Seq]
            attention = outputs.attentions[layer_idx][0]
            # Shape: [Batch, Seq, Hidden] -> [Seq, Hidden]
            # hidden_states usually includes embeddings at idx 0, causing size mismatch with attention?
            # Transformers outputs.hidden_states is a tuple of (embeddings + layer outputs).
            # outputs.attentions is a tuple of (layer 0 attn, ..., layer N attn).
            # Attention layer i corresponds to input hidden state i (normalized) or output?
            # Usually hidden_states[i+1] is the output of layer i.
            # Spectral Trust framework uses: signals = hidden_states[layer_idx + 1]
            
            # Correction: hidden_states tuple has N+1 elements.
            signals = outputs.hidden_states[layer_idx + 1][0].float()
            
            # Construct Laplacian
            # 1. Symmetrize
            # Cast attention to float32 for spectral ops
            attention_f32 = attention.float()
            attention_sym = graph_constructor.symmetrize_attention(attention_f32.unsqueeze(0)).squeeze(0)
            # 2. Aggregate Heads (Adjacency)
            adjacency = graph_constructor.aggregate_heads(attention_sym.unsqueeze(0)).squeeze(0)
            # 3. Laplacian
            laplacian = graph_constructor.construct_laplacian(adjacency.unsqueeze(0)).squeeze(0)
            
            # Compute Metrics
            try:
                diagnostics = spectral_analyzer.analyze_layer(signals, laplacian, layer_idx)
                
                layer_metrics[f"L{layer_idx}_HFER"] = diagnostics.hfer
                layer_metrics[f"L{layer_idx}_Fiedler"] = diagnostics.fiedler_value
                layer_metrics[f"L{layer_idx}_Smoothness"] = diagnostics.smoothness_index
                layer_metrics[f"L{layer_idx}_Entropy"] = diagnostics.spectral_entropy
            except Exception as e:
                # Fallback or log
                print(f"Error layer {layer_idx}: {e}")
                layer_metrics[f"L{layer_idx}_HFER"] = 0.0
                layer_metrics[f"L{layer_idx}_Fiedler"] = 0.0
                layer_metrics[f"L{layer_idx}_Smoothness"] = 0.0
                layer_metrics[f"L{layer_idx}_Entropy"] = 0.0
            
        row = {
            "is_hallucination": example['label'],
            **layer_metrics
        }
        
        # Save Row Incrementally
        df_row = pd.DataFrame([row])
        if not file_initialized:
            df_row.to_csv(args.output, index=False, mode='w')
            file_initialized = True
        else:
            df_row.to_csv(args.output, index=False, mode='a', header=False)
            
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
