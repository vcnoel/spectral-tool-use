"""
Multi-turn tool calling from the BFCL v4 multi-turn categories.

The benchmark's multi-turn examples are stateful: a turn's correct calls
depend on what earlier turns did. Executing them properly needs the
benchmark's simulated back ends, which is a different engineering problem
from the one this paper studies. We therefore use a teacher-forced protocol:
the conversation is replayed with the GROUND-TRUTH calls of the earlier turns
already in it, and the model is asked only for the calls of the current turn.
The label is whether those calls match the ground truth for that turn.

This measures what a guardrail would face in an agent loop, namely a call
produced with a history behind it, while keeping the label unambiguous.

It also enables the experiment the single-turn setting cannot support. By
replacing one earlier turn's calls with a wrong call, we can ask two
questions that matter for a deployed agent: does the model's failure rate at
the current turn rise when its history already contains a failure, and does a
detector trained on clean histories still work once the history is corrupted.
"""
import ast
import json
import random
from pathlib import Path

BFCL_DIR = Path("data/bfcl_v4")
FUNC_DOC_DIR = BFCL_DIR / "multi_turn_func_doc"

# involved_classes entry -> function-document file
CLASS_TO_DOC = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
    "WebSearchAPI": "web_search",
    "MemoryKV": "memory_kv",
    "MemoryVector": "memory_vector",
    "MemoryRecSum": "memory_rec_sum",
}

MULTI_TURN_FILES = [
    ("BFCL_v4_multi_turn_base", "base"),
    ("BFCL_v4_multi_turn_miss_param", "miss_param"),
    ("BFCL_v4_multi_turn_miss_func", "miss_func"),
    ("BFCL_v4_multi_turn_long_context", "long_context"),
]

_doc_cache: dict[str, list[dict]] = {}


def load_function_docs(classes) -> list[dict]:
    """Tool schemas for the classes an example involves."""
    out = []
    for cls in classes or []:
        stem = CLASS_TO_DOC.get(cls)
        if stem is None:
            continue
        if stem not in _doc_cache:
            f = FUNC_DOC_DIR / f"{stem}.json"
            if not f.exists():
                _doc_cache[stem] = []
            else:
                rows = [json.loads(line) for line in
                        open(f, encoding="utf-8") if line.strip()]
                for r in rows:
                    r.pop("response", None)
                _doc_cache[stem] = rows
        out.extend(_doc_cache[stem])
    return out


def parse_python_call(text: str):
    """
    Parse the benchmark's call syntax, e.g. ``mv(source='a', destination='b')``,
    into ``{"name": ..., "arguments": {...}}``. Returns None if it will not
    parse, which keeps a malformed ground truth out of the data rather than
    turning it into a spurious label.
    """
    try:
        node = ast.parse(text.strip(), mode="eval").body
        if not isinstance(node, ast.Call):
            return None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            return None
        args = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                args[kw.arg] = ast.unparse(kw.value)
        # positional arguments appear in the corpus, e.g. sort('f.pdf')
        for i, a in enumerate(node.args):
            try:
                args[f"__pos{i}"] = ast.literal_eval(a)
            except Exception:
                args[f"__pos{i}"] = ast.unparse(a)
        return {"name": name, "arguments": args}
    except Exception:
        return None


def render_history(turns, ground_truth, upto, corrupt_turn=None,
                   corrupt_with=None, keep_last=2):
    """
    Build the message history for turns [0, upto).

    Each earlier turn contributes its user message and an assistant message
    carrying that turn's ground-truth calls, so the model sees a correct
    trajectory. When ``corrupt_turn`` is given, that turn's assistant message
    carries ``corrupt_with`` instead, which is how the cascade condition is
    constructed.
    """
    # Only the most recent turns are replayed. The attention tensor grows
    # with the square of the prompt length, and a full history with every
    # tool schema runs past four thousand tokens, which cannot be held for
    # every layer and head. Keeping the last two turns bounds that while
    # preserving what the current turn depends on most directly.
    msgs = []
    start = max(0, upto - keep_last)
    for t in range(start, upto):
        for m in turns[t]:
            if m.get("role") == "user":
                msgs.append({"role": "user", "content": m["content"]})
        calls = ground_truth[t]
        if corrupt_turn == t and corrupt_with is not None:
            calls = corrupt_with
        if calls:
            rendered = ", ".join(calls)
            msgs.append({"role": "assistant",
                         "content": f"<tool_call>{rendered}</tool_call>"})
            msgs.append({"role": "user",
                         "content": "The tool calls completed successfully."})
    return msgs


def iter_multiturn_examples(limit: int, corrupt_fraction: float = 0.0,
                            seed: int = 0, max_turn: int = 4,
                            keep_last: int = 2, max_tools: int = 22):
    """
    Yield one record per (example, turn) pair.

    ``corrupt_fraction`` of the yielded records have one earlier turn's calls
    replaced by a call drawn from a different example, which is the cascade
    condition. Records carry ``corrupted`` so the two conditions can be
    compared on identical examples.
    """
    rng = random.Random(seed)
    pool, n = [], 0

    for stem, category in MULTI_TURN_FILES:
        qf, af = BFCL_DIR / f"{stem}.json", BFCL_DIR / "possible_answer" / f"{stem}.json"
        if not (qf.exists() and af.exists()):
            continue
        answers = {}
        for line in open(af, encoding="utf-8"):
            if line.strip():
                a = json.loads(line)
                answers[a["id"]] = a["ground_truth"]
        for line in open(qf, encoding="utf-8"):
            if not line.strip():
                continue
            q = json.loads(line)
            gt = answers.get(q["id"])
            if not gt:
                continue
            tools = load_function_docs(q.get("involved_classes"))
            # the tool schemas dominate the prompt, so examples whose tool
            # set is very large are skipped rather than truncated, which
            # would change the task
            if not tools or len(tools) > max_tools:
                continue
            pool.append((q, gt, tools, category))

    rng.shuffle(pool)
    # a bank of wrong calls to inject, taken from other examples
    bank = [c for _, gt, _, _ in pool for turn in gt for c in turn][:2000]

    for q, gt, tools, category in pool:
        turns = q["question"]
        for t in range(min(len(turns), len(gt), max_turn)):
            if not gt[t]:
                continue
            user_msgs = [m["content"] for m in turns[t] if m.get("role") == "user"]
            if not user_msgs:
                continue
            gt_calls = [parse_python_call(c) for c in gt[t]]
            if any(c is None for c in gt_calls):
                continue

            corrupted = t > 0 and rng.random() < corrupt_fraction
            corrupt_with = None
            corrupt_turn = None
            if corrupted and bank:
                corrupt_turn = rng.randrange(max(0, t - keep_last), t)
                corrupt_with = [rng.choice(bank)]

            history = render_history(turns, gt, t, corrupt_turn, corrupt_with,
                                     keep_last=keep_last)
            # the injected call must be inside the replayed window to have
            # any chance of affecting the current turn
            if corrupted and corrupt_turn is not None and                     corrupt_turn < max(0, t - keep_last):
                corrupted = False
            yield {
                "tools": tools,
                "user": " ".join(user_msgs),
                "history": history,
                "gt_anyof": [{c["name"]: {k: [v] for k, v in c["arguments"].items()}}
                             for c in gt_calls],
                "expect_call": True,
                "category": f"mt_{category}",
                "tool": gt_calls[0]["name"],
                "turn_index": t,
                "n_turns": len(gt),
                "corrupted": bool(corrupted),
            }
            n += 1
            if n >= limit:
                return
