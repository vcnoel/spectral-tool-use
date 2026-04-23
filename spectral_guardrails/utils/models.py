import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_REGISTRY = {
    # Latest Generation (Most Recent)
    "phi4":     "microsoft/phi-4",
    "qwen35":   "Qwen/Qwen3.5-2B",
    "qwen35_2b": "Qwen/Qwen3.5-2B",
    "qwen25":   "Qwen/Qwen2.5-7B-Instruct",
    "llama31":  "meta-llama/Llama-3.1-8B-Instruct",
    "gemma2":   "google/gemma-2-9b-it",
    "gemma2_2b": "google/gemma-2-2b-it",
    
    # Original models (paper)
    "llama":   "meta-llama/Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "qwen":    "Qwen/Qwen2.5-0.5B-Instruct",
    
    # Small models (scaling study)
    "llama3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "llama1b":  "meta-llama/Llama-3.2-1B-Instruct",
    "qwen2b":   "Qwen/Qwen2.5-1.5B-Instruct",
    "mistral2b": "mistralai/Ministral-3-3B-Instruct-2512",
    "gemma_moe": "google/gemma-4-E2B-it",  
    "qwen4b":   "Qwen/Qwen1.5-4B-Chat"
}

def load_model_and_tokenizer(model_key_or_path: str, 
                             load_in_4bit: bool = False,
                             device: str = "auto"):
    """
    Loads model and tokenizer from registry key or direct path.
    """
    model_id = MODEL_REGISTRY.get(model_key_or_path, model_key_or_path)
    
    print(f"Loading Model: {model_id} (4-bit: {load_in_4bit}, device: {device})...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True
    )
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
        attn_implementation="eager",
        trust_remote_code=True
    )
    
    return model, tokenizer
