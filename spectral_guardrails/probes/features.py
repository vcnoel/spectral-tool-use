import torch
import numpy as np

def find_token_positions(tokenizer, output_ids: list[int], tool_call_str: str) -> dict:
    """
    Locate t_func, T_args, t_end in the JSON token sequence for Glaive.
    Format: <functioncall> {"name": "func_name", "arguments": "..."}
    """
    full_text = tokenizer.decode(output_ids)
    
    # Map token indices to character spans in the decoded text
    token_spans = []
    current_idx = 0
    for tid in output_ids:
        t_text = tokenizer.decode([tid])
        start = full_text.find(t_text, current_idx)
        if start == -1: 
            start = current_idx
        end = start + len(t_text)
        token_spans.append((start, end))
        current_idx = end

    import re
    
    try:
        # Search for "name": "..."
        name_match = re.search(r'"name":\s*"([^"]+)"', tool_call_str)
        if not name_match:
            last = len(output_ids) - 1
            return {'t_func': last, 't_args': [last], 't_end': last}
        
        func_name = name_match.group(1)
        # Find where the function name appears in the full_text
        func_char_start = full_text.find(f'"{func_name}"')
        if func_char_start == -1: func_char_start = full_text.find(func_name)
        
        # Args: heuristic, take everything inside "arguments": "..."
        args_match = re.search(r'"arguments":\s*"?(.*?)"?\s*\}', tool_call_str, re.DOTALL)
        args_char_start = -1
        args_char_end = -1
        if args_match:
            args_str = args_match.group(1)
            args_char_start = full_text.find(args_str, func_char_start)
            args_char_end = args_char_start + len(args_str)

        # End: the tool call closing brace
        end_char = full_text.find('}', max(func_char_start, args_char_end))

        # Map character indices to token indices
        t_func = 0
        for i, (s, e) in enumerate(token_spans):
            if s <= func_char_start < e:
                t_func = i
                break
        
        t_end = len(output_ids) - 1
        for i, (s, e) in enumerate(token_spans):
            if s <= end_char < e:
                t_end = i
                break
                
        t_args = []
        if args_char_start != -1:
            for i, (s, e) in enumerate(token_spans):
                if e > args_char_start and s < args_char_end:
                    t_args.append(i)
        
        if not t_args: t_args = [t_func]

        return {
            't_func': t_func,
            't_args': t_args,
            't_end': t_end
        }
    except:
        last = len(output_ids) - 1
        return {'t_func': last, 't_args': [last], 't_end': last}

def extract_probe_features(hidden_states: torch.Tensor,
                           token_positions: dict,
                           layer: int = -1) -> np.ndarray:
    """
    hidden_states: (Seq, Hidden)
    Returns concat of h_t_func, mean(h_T_args), h_t_end -> (3 * Hidden,)
    """
    h = hidden_states.detach().float().cpu().numpy()
    
    h_func = h[token_positions['t_func']]
    h_args = np.mean(h[token_positions['t_args']], axis=0)
    h_end = h[token_positions['t_end']]
    
    return np.concatenate([h_func, h_args, h_end], axis=0)
