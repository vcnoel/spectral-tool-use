"""Token-role spans of a generated tool call (docs/PIPELINE_REBUILD.md section 4).

For every call in the generation: the function NAME, each ARGUMENT VALUE (keys, quotes,
braces and separators excluded) and the CLOSING delimiter. Three dialects:
  JSON        {"name": "f", "arguments": {"k": v, ...}}  (also "parameters"; lists of calls)
  Qwen XML    <function=f><parameter=k>v</parameter></function>
  MiniCPM XML <function name="f"><param name="k">v</param></function>

Character spans are mapped to generated-token indices through cumulative decoding
(token i ends at len(decode(ids[:i+1]))), which is exact for byte-level tokenizers where
decoding a single token in isolation is not.
"""
from __future__ import annotations

import re

_JSON_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')
_JSON_ARGS = re.compile(r'"(?:arguments|parameters)"\s*:\s*\{')
_JSON_VALUE = re.compile(
    r'"((?:[^"\\]|\\.)*)"\s*:\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\[[^\[\]{}]*\]|-?[\w.+\-]+)')
_XML_FUNC = re.compile(r"<function\s*=\s*([\w.\-]+)\s*>")
_XML_PARAM = re.compile(r"<parameter\s*=\s*([\w.\-]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL)
_XML_FUNC_ATTR = re.compile(r"<function\s+name\s*=\s*[\"']([\w.\-]+)[\"']\s*>")
_XML_PARAM_ATTR = re.compile(r"<param\s+name\s*=\s*[\"']([\w.\-]+)[\"']\s*>\s*(.*?)\s*</param>", re.DOTALL)
_CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)


def matching_close(text: str, i: int) -> int:
    """Index of the bracket closing text[i] ('{' or '['), skipping strings; -1 if none."""
    open_c = text[i]
    close_c = "}" if open_c == "{" else "]"
    depth, in_str, esc = 0, False, False
    for j in range(i, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return j
    return -1


def _json_values(text: str, a: int, b: int):
    """(key, start, end, key_start, key_end) of every value inside the object text[a:b]."""
    out = []
    region = text[a:b + 1]
    for m in _JSON_VALUE.finditer(region):
        key = m.group(1)
        s, e = a + m.start(2), a + m.end(2)
        if text[s] in "\"'" and e - s >= 2:
            s, e = s + 1, e - 1
        if region[m.start(2)] == "{":
            continue
        if e > s:
            out.append((key, s, e, a + m.start(1), a + m.end(1)))
    return out


def _xml_calls(text, func_re, param_re):
    calls = []
    funcs = list(func_re.finditer(text))
    for k, fm in enumerate(funcs):
        body_end = funcs[k + 1].start() if k + 1 < len(funcs) else len(text)
        close = text.find("</function>", fm.end(), body_end)
        close_span = (close, close + len("</function>")) if close >= 0 else None
        body = text[fm.end(): close if close >= 0 else body_end]
        values = []
        for pm in param_re.finditer(body):
            s, e = fm.end() + pm.start(2), fm.end() + pm.end(2)
            cd = _CDATA.search(text, s, e)
            if cd:
                s, e = cd.start(1), cd.end(1)
            if e > s:
                values.append((pm.group(1), s, e, fm.end() + pm.start(1), fm.end() + pm.end(1)))
        pms = list(param_re.finditer(body))
        args_span = ((fm.end() + pms[0].start(), fm.end() + pms[-1].end()) if pms else None)
        calls.append({"name": fm.group(1), "name_span": (fm.start(1), fm.end(1)),
                      "values": values, "close_span": close_span, "args_span": args_span, "dialect": "xml"})
    return calls


def call_roles(gen_text: str) -> list[dict]:
    """Every call in the generation with its name span, value spans and closing span."""
    if "<function=" in gen_text:
        return _xml_calls(gen_text, _XML_FUNC, _XML_PARAM)
    if re.search(r"<function\s+name\s*=", gen_text):
        return _xml_calls(gen_text, _XML_FUNC_ATTR, _XML_PARAM_ATTR)
    calls = []
    for nm in _JSON_NAME.finditer(gen_text):
        obj_start = gen_text.rfind("{", 0, nm.start())
        if obj_start < 0:
            continue
        obj_end = matching_close(gen_text, obj_start)
        if obj_end < 0:
            obj_end = len(gen_text) - 1
        if calls and calls[-1]["_obj"] == (obj_start, obj_end):
            continue
        values, args_span = [], None
        am = _JSON_ARGS.search(gen_text, nm.end(), obj_end + 1)
        if am:
            a = am.end() - 1
            b = matching_close(gen_text, a)
            if b < 0 or b > obj_end:
                b = obj_end
            values = _json_values(gen_text, a, b)
            args_span = (a, b + 1)
        calls.append({"name": nm.group(1), "name_span": (nm.start(1), nm.end(1)), "values": values,
                      "close_span": (obj_end, obj_end + 1), "args_span": args_span, "dialect": "json",
                      "_obj": (obj_start, obj_end)})
    for c in calls:
        c.pop("_obj", None)
    return calls


def token_char_ends(tok, gen_ids: list[int]) -> list[int]:
    """Char end offset of every generated token in decode(gen_ids) (specials kept)."""
    ends = []
    for i in range(len(gen_ids)):
        ends.append(len(tok.decode(gen_ids[: i + 1], skip_special_tokens=False)))
    return ends


def tokens_in(ends: list[int], span) -> list[int]:
    if span is None:
        return []
    s, e = span
    out = []
    prev = 0
    for i, end in enumerate(ends):
        if end > s and prev < e and end > prev:
            out.append(i)
        prev = end
    return out


def role_positions(tok, gen_ids: list[int], max_calls: int = 4, max_values: int = 12) -> dict:
    """Token-role positions of the generation.

    Roles (docs/PIPELINE_REBUILD.md 4.3): per call `name` (first token of the function name),
    `args` (mean over the whole arguments object, Healy et al.'s argument span), each `value`
    (mean over the value's tokens), each `prevalue` (the token just before the value's first
    token, Yu et al.'s pre-parameter-value position), `close` (closing delimiter); and one
    `last` (the final generated token, Yeats et al.'s last-token probe).

    Returns dict(found, calls=[{name_tok, value_toks:[[...]], value_keys, close_tok}],
                 positions=[(role, call_idx, value_idx, token_index or -1, token_list)],
                 token_role (per generated token: 0 none, 1 name, 2 value, 3 close, 4 argument name),
                 value_id (per token: value index or -1), n_values_total, capped).
    Fallback when no call is found: the last generated token carries every role."""
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)
    ends = token_char_ends(tok, gen_ids)
    G = len(gen_ids)
    token_role = [0] * G
    value_id = [-1] * G
    calls = call_roles(gen_text)
    last = G - 1
    if not calls or G == 0:
        pos = [("name", 0, -1, last, [last]), ("args", 0, -1, -1, [last]), ("value", 0, 0, -1, [last]),
               ("prevalue", 0, 0, last, [last]), ("close", 0, -1, last, [last]), ("last", -1, -1, last, [last])]
        return {"found": False, "calls": [], "positions": pos, "token_role": token_role,
                "value_id": value_id, "n_values_total": 0, "capped": False, "gen_text": gen_text}
    positions, out_calls, n_val, vid = [], [], 0, 0
    capped = len(calls) > max_calls
    for ci, c in enumerate(calls[:max_calls]):
        nt = tokens_in(ends, c["name_span"])
        name_tok = nt[0] if nt else last
        for t in nt:
            token_role[t] = 1
        vt_list, keys, kt_list = [], [], []
        at = tokens_in(ends, c.get("args_span"))
        positions.append(("name", ci, -1, name_tok, [name_tok]))
        positions.append(("args", ci, -1, -1, at or [name_tok]))
        for vi, (key, s, e, ks, ke) in enumerate(c["values"]):
            n_val += 1
            toks = tokens_in(ends, (s, e))
            if not toks:
                continue
            if vid >= max_values:
                capped = True
                continue
            for t in toks:
                token_role[t] = 2
                value_id[t] = vid
            kt = tokens_in(ends, (ks, ke))
            for t in kt:
                if token_role[t] == 0:
                    token_role[t] = 4
            vt_list.append(toks)
            kt_list.append(kt)
            keys.append(key)
            positions.append(("value", ci, vid, -1, toks))
            positions.append(("prevalue", ci, vid, max(0, toks[0] - 1), [max(0, toks[0] - 1)]))
            vid += 1
        ct = tokens_in(ends, c["close_span"])
        close_tok = ct[-1] if ct else last
        for t in ct:
            if token_role[t] == 0:
                token_role[t] = 3
        positions.append(("close", ci, -1, close_tok, [close_tok]))
        out_calls.append({"name": c["name"], "name_tok": name_tok, "name_toks": nt or [name_tok],
                          "args_toks": at, "value_toks": vt_list, "key_toks": kt_list,
                          "value_keys": keys, "close_toks": ct or [close_tok], "close_tok": close_tok,
                          "dialect": c["dialect"]})
    positions.append(("last", -1, -1, last, [last]))
    return {"found": True, "calls": out_calls, "positions": positions, "token_role": token_role,
            "value_id": value_id, "n_values_total": n_val, "capped": capped, "gen_text": gen_text}
