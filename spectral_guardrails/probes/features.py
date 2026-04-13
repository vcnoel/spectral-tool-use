import torch
import numpy as np

def find_token_positions(tokenizer, output_ids: list[int], tool_call_str: str) -> dict:
    """
    Locate t_func, T_args, t_end in the token sequence.
    """
    full_text = tokenizer.decode(output_ids)
    
    # We need to find where tool_call_str starts in the text and map to tokens
    # This is often brittle, so we'll do our best.
    
    # Simple heuristic: find the tokens that decode to parts of the string
    # For now, we'll assume the tool call is at the end of output_ids
    
    # t_func: usually the first subtoken after '(' (or start of tool call)
    # T_args: everything between '(' and ')'
    # t_end: the ')' token
    
    # Actually, let's find the tool call start index in the decoded text
    tc_start_idx = full_text.find(tool_call_str)
    if tc_start_idx == -1:
        return {'t_func': len(output_ids)-1, 't_args': [len(output_ids)-1], 't_end': len(output_ids)-1}
    
    # Map text index to token index (Optimized)
    token_texts = tokenizer.convert_ids_to_tokens(output_ids)
    token_positions = []
    current_char = 0
    for token_text in token_texts:
        # Note: BPE/SentencePiece tokens may have special chars like 'Ġ' or '_'
        clean_text = tokenizer.convert_tokens_to_string([token_text])
        token_positions.append((current_char, current_char + len(clean_text)))
        current_char += len(clean_text)
        
    def get_token_at(char_idx):
        for i, (s, e) in enumerate(token_positions):
            if s <= char_idx < e:
                return i
        return len(output_ids) - 1

    t_func_char = tc_start_idx
    paren_start = tool_call_str.find('(')
    paren_end = tool_call_str.rfind(')')
    
    t_func = get_token_at(tc_start_idx)
    t_end = get_token_at(tc_start_idx + paren_end) if paren_end != -1 else len(output_ids) - 1
    
    t_args = []
    if paren_start != -1 and paren_end != -1:
        arg_start_token = get_token_at(tc_start_idx + paren_start + 1)
        arg_end_token = get_token_at(tc_start_idx + paren_end)
        t_args = list(range(arg_start_token, arg_end_token))
        if not t_args:
            t_args = [arg_start_token]
    else:
        t_args = [t_func]
        
    return {
        't_func': t_func,
        't_args': t_args,
        't_end': t_end
    }

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
