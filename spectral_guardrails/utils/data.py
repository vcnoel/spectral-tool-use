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

def load_glaive_data(domain="all", limit=1000, offset=0, step=1):
    """
    Loads and filters Glaive dataset with support for strided sampling.
    """
    print(f"Loading Glaive v2 dataset (domain={domain}, limit={limit}, step={step})...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    
    filtered = []
    # Use step-based iteration to ensure we pull from a wider range of the dataset
    for i in range(offset, len(ds), step):
        ex = ds[i]
        if is_in_domain(ex, domain):
            filtered.append(ex)
        if len(filtered) >= limit:
            break
            
    return filtered

def is_json_equivalent(pred1, pred2):
    """
    Checks if two tool calls are semantically equivalent by parsing JSON.
    Assumes tool calls are in a format like: func_name(arg1="val1", arg2=2)
    or just standard JSON if the model outputs it.
    """
    import json
    import re

    def parse_to_dict(s):
        s = s.strip()
        # Handle standard JSON objects
        if s.startswith("{") and s.endswith("}"):
            try: return json.loads(s)
            except: pass
        
        # Handle common tool call format: name(args)
        match = re.search(r"^([a-zA-Z0-9_]+)\((.*)\)$", s)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            # Try to extract key-value pairs
            # This is a heuristic parser for unformatted tool calls
            kv_pairs = re.findall(r"([a-zA-Z0-9_]+)\s*=\s*([^,]+)", args_str)
            if not kv_pairs:
                return {"_name": name, "_raw_args": args_str}
            
            d = {"_name": name}
            for k, v in kv_pairs:
                v = v.strip().strip('"').strip("'")
                # Try numeric conversion
                try:
                    if "." in v: d[k] = float(v)
                    else: d[k] = int(v)
                except:
                    d[k] = v
            return d
            
        return {"_raw": s}

    d1 = parse_to_dict(pred1)
    d2 = parse_to_dict(pred2)
    
    # If both couldn't be parsed meaningfully, fallback to normalized string match
    if len(d1) <= 1 and len(d2) <= 1:
        return "".join(pred1.split()) == "".join(pred2.split())

    return d1 == d2
