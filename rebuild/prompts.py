"""Prompt routes (docs/PIPELINE_REBUILD.md section 3).

Two routes, decided per RUN (never per item) and recorded in run_meta.json:

  native         the model's own chat template renders the tool schemas (tools=...).
                 Accepted only if every tool name appears in the rendered text.
  fallback_list  the registered system-prompt specification below, which PERMITS a
                 list of calls. Used when the template drops the tools (Gemma-3,
                 SmolLM2), when it rejects a system turn (Gemma-2: the specification
                 is then prepended to the user turn), or when forced with --route
                 fallback_list for the native-vs-fallback control.

The earlier pipeline's one-object fallback (TOOL_PROMPT_FALLBACK in run_pilot_v2.py)
is NOT offered: it asked for a single JSON object and induced dropped parallel calls.

The chat template's reasoning mode is disabled wherever the template accepts the flag,
and the date a template stamps into its system header is pinned, so that prompt text
and prompt hash are stable across days.
"""
from __future__ import annotations

import hashlib
import json

TEMPLATE_DATE = "01 Jan 2026"

FALLBACK_LIST_SPEC = (
    "You are a helpful assistant with access to the following functions.\n"
    "When a function is needed, reply with ONLY a JSON list of one or more objects of the form\n"
    '[{"name": <function name>, "arguments": {<arg name>: <value>}}, ...]\n'
    "with one object per call, and nothing else.\n\nAvailable functions:\n"
)
FALLBACK_SPEC_SHA256 = hashlib.sha256(FALLBACK_LIST_SPEC.encode("utf-8")).hexdigest()

ROUTES = ("native", "fallback_list")


def apply_template(tok, msgs, tools=None):
    kwargs = dict(tokenize=False, add_generation_prompt=True, date_string=TEMPLATE_DATE)
    if tools is not None:
        kwargs["tools"] = tools
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)


def schemas_text(tools) -> str:
    return "\n".join(json.dumps(t) for t in tools)


def render_native(tok, tools, user):
    """Rendered prompt through the native template, or None when the template drops a tool."""
    names = [t.get("name", "") for t in tools if t.get("name")]
    try:
        text = apply_template(
            tok, [{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": user}], tools=tools)
    except Exception:
        return None, "template_error"
    if not all(n in text for n in names):
        return None, "tools_dropped"
    return text, "ok"


def render_fallback(tok, tools, user):
    """Rendered prompt through the registered list specification.

    Returns (text, placement) with placement 'system' or 'user' (templates without a
    system role), or (None, reason)."""
    names = [t.get("name", "") for t in tools if t.get("name")]
    sys_msg = FALLBACK_LIST_SPEC + schemas_text(tools)
    for placement, msgs in (
            ("system", [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]),
            ("user", [{"role": "user", "content": sys_msg + "\n\n" + user}])):
        try:
            text = apply_template(tok, msgs)
        except Exception:
            continue
        if all(n in text for n in names):
            return text, placement
    return None, "unrenderable"


def render(tok, tools, user, route: str):
    """Render under a FIXED route. Returns (text, detail) or (None, reason)."""
    if route == "native":
        return render_native(tok, tools, user)
    if route == "fallback_list":
        return render_fallback(tok, tools, user)
    raise ValueError(route)


def decide_route(tok, items, requested: str | None = None, sample: int = 50) -> dict:
    """Decide the route for a run from a render check over `items` (the first `sample`).

    requested=None: native if EVERY sampled item renders natively, else fallback_list.
    requested='native' asserts that every sampled item renders natively.
    requested='fallback_list' forces the fallback (the route control).
    Returns a dict recorded in run_meta.json."""
    rows = list(items)[:sample]
    native_ok = 0
    reasons = {}
    placement = {}
    for ex in rows:
        t, why = render_native(tok, ex["tools"], ex["user"])
        if t is not None:
            native_ok += 1
        else:
            reasons[why] = reasons.get(why, 0) + 1
        _, pl = render_fallback(tok, ex["tools"], ex["user"])
        placement[pl] = placement.get(pl, 0) + 1
    all_native = native_ok == len(rows) and rows
    if requested == "native" and not all_native:
        raise SystemExit(f"[route] native requested but {len(rows) - native_ok} of {len(rows)} "
                         f"sampled items do not render natively: {reasons}")
    route = requested or ("native" if all_native else "fallback_list")
    return {"prompt_route": route, "route_requested": requested,
            "render_check": {"n_sampled": len(rows), "native_ok": native_ok, "native_failures": reasons,
                             "fallback_placement": placement},
            "fallback_spec_sha256": FALLBACK_SPEC_SHA256 if route == "fallback_list" else None,
            "template_date": TEMPLATE_DATE, "thinking_mode": False}


# ── spans inside the rendered prompt ───────────────────────────────────────────

def _all_occurrences(hay: str, needle: str, start: int = 0):
    i = hay.find(needle, start)
    while i >= 0:
        yield i
        i = hay.find(needle, i + 1)


def prompt_char_spans(prompt_text: str, tools, user: str) -> dict:
    """Character spans of the tool-schema region and of the user request in the prompt.

    schema: from the earliest occurrence of any tool name to the latest occurrence of any
            tool name, description or parameter name (first occurrence after that tool's
            name). Covers the rendered schemas on every route checked (Llama, Qwen3,
            Qwen3.5, MiniCPM5, SmolLM3 natively; the JSON specification on the fallback).
    request: the LAST occurrence of the user text that does not overlap the schema span
            (the user turn follows the schemas in every template checked).
    Both may be None; the flags say so and the item is kept."""
    starts, ends = [], []
    for t in tools:
        name = t.get("name") or ""
        if not name:
            continue
        i = prompt_text.find(name)
        if i < 0:
            continue
        starts.append(i)
        ends.append(i + len(name))
        desc = (t.get("description") or "").strip()
        if desc:
            j = prompt_text.find(desc[:60], i)
            if j >= 0:
                ends.append(j + min(60, len(desc)))
        props = ((t.get("parameters") or {}).get("properties") or {})
        for p in props:
            j = prompt_text.find(p, i)
            if j >= 0:
                ends.append(j + len(p))
    schema = (min(starts), max(ends)) if starts else None
    request = None
    u = user.strip()
    if u:
        cands = [k for k in _all_occurrences(prompt_text, u)]
        if not cands and len(u) > 80:   # templates that re-wrap long text: anchor on a prefix
            cands = [k for k in _all_occurrences(prompt_text, u[:80])]
            u = u[:80]
        for k in reversed(cands):
            if schema is None or k >= schema[1] or k + len(u) <= schema[0]:
                request = (k, k + len(u))
                break
    return {"schema": schema, "request": request}


def char_to_token_span(offsets, span):
    """Half-open token range covering a half-open char span, from a tokenizer offset mapping."""
    if span is None:
        return None
    s, e = span
    toks = [i for i, (a, b) in enumerate(offsets) if b > s and a < e and b > a]
    if not toks:
        return None
    return (toks[0], toks[-1] + 1)


def tool_segment_spans(prompt_text: str, tools, schema_span) -> list:
    """Character span of each tool's definition segment inside the schema span (Chen 2606.16364,
    'harness attention allocation'): from the tool's first name occurrence to the next tool's
    first name occurrence (or the schema end), in prompt order. None for a tool not found."""
    if schema_span is None:
        return [None] * len(tools)
    starts = []
    for i, t in enumerate(tools):
        name = t.get("name") or ""
        k = prompt_text.find(name, schema_span[0]) if name else -1
        starts.append((k if 0 <= k < schema_span[1] else None, i))
    order = sorted([(k, i) for k, i in starts if k is not None])
    spans = [None] * len(tools)
    for j, (k, i) in enumerate(order):
        end = order[j + 1][0] if j + 1 < len(order) else schema_span[1]
        spans[i] = (k, end)
    return spans
