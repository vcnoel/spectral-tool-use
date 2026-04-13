import json
import re
from datasets import load_dataset
from pathlib import Path

DOMAIN_KEYWORDS = [
    "calculator", "math", "finance", "currency", "exchange", 
    "loan", "interest", "tax", "convert", "unit", "calculate", "computing"
]

def parse_glaive_chat(chat_str):
    """
    Parses Glaive v2 raw text chat into a list of messages.
    """
    messages = []
    parts = re.split(r'(?:\n|^)(USER|ASSISTANT|FUNCTION RESPONSE):', chat_str)
    
    for i in range(1, len(parts), 2):
        role_marker = parts[i].strip()
        content = parts[i+1].strip()
        content = content.replace("<|endoftext|>", "").strip()
        
        role = "user"
        if role_marker == "ASSISTANT":
            role = "assistant"
        elif role_marker == "FUNCTION RESPONSE":
            role = "tool"
        
        messages.append({"role": role, "content": content})
    return messages

def normalize_tool_call(tool_call_str):
    """
    Simple normalization to handle whitespace/quote differences.
    """
    return "".join(tool_call_str.split())

def is_in_domain(example, domain="all"):
    if domain == "all":
        return True
    
    # Simple keyword match in system prompt or chat
    content = (example.get('system', '') + example.get('chat', '')).lower()
    
    if domain == "finance":
        keywords = ["finance", "currency", "exchange", "loan", "interest", "tax"]
    elif domain == "math":
        keywords = ["calculator", "math", "calculate", "computing", "convert", "unit"]
    else:
        keywords = DOMAIN_KEYWORDS
        
    return any(k in content for k in keywords)

def load_glaive_data(domain="all", limit=1000, offset=0):
    """
    Loads and filters Glaive dataset.
    """
    print(f"Loading Glaive v2 dataset (domain={domain}, limit={limit})...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    
    filtered = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        if is_in_domain(ex, domain):
            filtered.append(ex)
        if len(filtered) >= limit:
            break
            
    return filtered
