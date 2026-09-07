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
        if func_char_start == -1:
            func_char_start = full_text.find(func_name)

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

        if not t_args:
            t_args = [t_func]

        return {
            't_func': t_func,
            't_args': t_args,
            't_end': t_end
        }
    except BaseException:
        last = len(output_ids) - 1
        return {'t_func': last, 't_args': [last], 't_end': last}


def find_token_positions_v2(tokenizer, generated_ids: list[int],
                            gen_offset: int) -> dict:
    """
    v2 token-role positions — searches ONLY the generated tokens.

    The v1 function searched the function name from character 0 of
    prompt+prediction, so it matched the tool schema inside the SYSTEM
    PROMPT rather than the generated call (audit 2026-09): the token-role
    probe was reading hidden states at prompt positions. Here we decode
    only the generated ids, locate the call structurally, and offset the
    resulting indices by `gen_offset` (= number of prompt tokens) so they
    index into the full teacher-forced sequence.

    Roles:
      t_func  first token of the function-name value
      t_args  tokens spanning the arguments object
      t_end   last structural token of the call (closing brace/tag)
    Fallback when no call is found: last generated token for every role.
    """
    import re

    gen_text = tokenizer.decode(generated_ids)

    # char span per generated token
    spans, cur = [], 0
    for tid in generated_ids:
        piece = tokenizer.decode([tid])
        start = gen_text.find(piece, cur)
        if start == -1:
            start = cur
        end = start + len(piece)
        spans.append((start, end))
        cur = end

    def tok_at(char_idx: int, default: int) -> int:
        for i, (s, e) in enumerate(spans):
            if s <= char_idx < e:
                return i
        return default

    last = len(generated_ids) - 1
    fallback = {"t_func": gen_offset + last,
                "t_args": [gen_offset + last],
                "t_end": gen_offset + last}

    # JSON-style call:  {"name": "fn", "arguments": {...}}
    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', gen_text)
    # XML-style call:   <function=fn> <parameter=k> v </parameter> ...
    xml_m = re.search(r"<function\s*=\s*([\w.\-]+)\s*>", gen_text)

    if name_m is not None:
        t_func = tok_at(name_m.start(1), last)
        args_m = re.search(r'"(?:arguments|parameters)"\s*:\s*(\{.*\}|"[^"]*")',
                           gen_text, re.DOTALL)
        if args_m:
            a_start, a_end = args_m.start(1), args_m.end(1)
        else:
            a_start, a_end = name_m.start(1), name_m.end(1)
        close_idx = gen_text.rfind("}")
        t_end = tok_at(close_idx, last) if close_idx != -1 else last
    elif xml_m is not None:
        t_func = tok_at(xml_m.start(1), last)
        params = list(re.finditer(
            r"<parameter\s*=\s*[\w.\-]+\s*>\s*(.*?)\s*</parameter>",
            gen_text, re.DOTALL))
        if params:
            a_start = params[0].start(1)
            a_end = params[-1].end(1)
        else:
            a_start, a_end = xml_m.start(1), xml_m.end(1)
        close_m = re.search(r"</function>|</tool_call>", gen_text)
        t_end = tok_at(close_m.start(), last) if close_m else last
    else:
        return fallback

    t_args = [i for i, (s, e) in enumerate(spans) if e > a_start and s < a_end]
    if not t_args:
        t_args = [t_func]

    return {"t_func": gen_offset + t_func,
            "t_args": [gen_offset + i for i in t_args],
            "t_end": gen_offset + t_end}


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
