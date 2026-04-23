"""
Fixed Hard Mode Dataset Staging Script.

BFCL:
  - Downloads question data AND the official answer key JSON from gorilla-llm's
    eval_checker ground_truth files, matching by ID.
  - chat format: USER: {question}\nASSISTANT: (model generates here)
  - ground_truth: the JSON string of the expected function call(s) from the answer key

ToolBench:
  - Extracts the first `assistant` turn (by role) as ground_truth.
  - chat format: USER: {user_task}\nASSISTANT: (model generates here)
  - ground_truth: first assistant turn content (Thought/Action style)
"""
import json
import requests
from datasets import load_dataset
from pathlib import Path


def prepare_bfcl(data_root: Path, limit: int = 500):
    print("Preparing BFCL (Berkeley)...")
    data_root.mkdir(parents=True, exist_ok=True)

    # Question files and their matching answer key files — confirmed working URLs
    BFCL_PAIRS = [
        (
            "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_parallel.json",
            "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/BFCL_v3_parallel.json",
            "parallel"
        ),
        (
            "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_multiple.json",
            "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/BFCL_v3_multiple.json",
            "multiple"
        ),
        (
            "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_parallel_multiple.json",
            "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/BFCL_v3_parallel_multiple.json",
            "parallel_multiple"
        ),
    ]

    samples = []
    for q_url, ans_url, category in BFCL_PAIRS:
        if len(samples) >= limit:
            break

        print(f"  Fetching {category} from {q_url.split('/')[-1]}...")
        q_resp = requests.get(q_url, timeout=60)
        a_resp = requests.get(ans_url, timeout=60)

        if q_resp.status_code != 200:
            print(f"  FAILED questions: HTTP {q_resp.status_code}")
            continue
        if a_resp.status_code != 200:
            print(f"  FAILED answer key: HTTP {a_resp.status_code}")
            continue

        def parse_jsonl_or_json(resp):
            try:
                data = resp.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    # single entry
                    return [data]
            except ValueError:
                pass
            return [json.loads(l) for l in resp.text.strip().splitlines() if l.strip()]

        questions = parse_jsonl_or_json(q_resp)
        answers   = parse_jsonl_or_json(a_resp)

        # Build ID → answer map
        ans_by_id = {a.get("id", ""): a for a in answers if isinstance(a, dict)}
        print(f"  Loaded {len(questions)} questions, {len(answers)} answer entries.")

        for q in questions:
            if len(samples) >= limit:
                break
            qid = q.get("id", "")

            # question field is a list of message lists
            q_field = q.get("question", [])
            if isinstance(q_field, list) and q_field:
                inner = q_field[0]
                if isinstance(inner, list) and inner:
                    question_text = inner[0].get("content", "")
                elif isinstance(inner, dict):
                    question_text = inner.get("content", "")
                else:
                    question_text = str(inner)
            else:
                question_text = str(q_field)

            # function schema — include in system so model knows what tools exist
            func_schema = json.dumps(q.get("function", []))

            ans_entry = ans_by_id.get(qid, {})
            gt = ans_entry.get("ground_truth", "")
            if not gt:
                continue
            # ground_truth is a list of {tool_name: {arg: [val], ...}} dicts
            if not isinstance(gt, str):
                gt = json.dumps(gt)

            samples.append({
                "id": qid,
                "system": (
                    "You are a helpful assistant with access to tools. "
                    "Respond ONLY with a valid JSON list of function calls matching this schema: "
                    + func_schema
                ),
                "chat": f"USER: {question_text}\nASSISTANT: ",
                "ground_truth": gt,
                "category": category,
            })

    out_path = data_root / "bfcl_hard.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(samples)} BFCL samples to {out_path}")
    return len(samples)


def prepare_toolbench(data_root: Path, limit: int = 500):
    print("Preparing ToolBench...")
    data_root.mkdir(parents=True, exist_ok=True)

    high_stakes_kw = [
        "finance", "economy", "stock", "portfolio", "loan", "tax",
        "map", "logistic", "route", "cloud", "ec2", "aws", "gcp",
        "azure", "admin", "server",
    ]

    try:
        tb = load_dataset("Yhyu13/ToolBench_toolllama_G123_dfs", split="train")
    except Exception as e:
        print(f"ToolBench load failed: {e}")
        return 0

    samples = []
    for ex in tb:
        if len(samples) >= limit:
            break
        content = str(ex).lower()
        if not any(k in content for k in high_stakes_kw):
            continue

        conv = ex.get("conversations", [])

        # Find the first user task turn (skip system)
        user_text = None
        for turn in conv:
            if turn.get("from") == "user":
                val = turn.get("value", "").strip()
                # Skip the "This is not the first time..." retry prompts
                if val and "not the first time" not in val:
                    user_text = val
                    break

        # Find the first assistant turn — this is the expected tool response
        gt_text = None
        for turn in conv:
            if turn.get("from") == "assistant":
                gt_text = turn.get("value", "").strip()
                break

        if not user_text or not gt_text:
            continue  # skip malformed entries

        samples.append({
            "system": "You are an expert assistant managing complex tool-use environments.",
            "chat": f"USER: {user_text}\nASSISTANT: ",
            "ground_truth": gt_text,
            "category": "toolbench_high_stakes",
        })

    out_path = data_root / "toolbench_hard.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(samples)} ToolBench samples to {out_path}")
    return len(samples)


if __name__ == "__main__":
    data_root = Path("data/hard_mode_datasets")
    n_bfcl = prepare_bfcl(data_root, limit=500)
    n_tb   = prepare_toolbench(data_root, limit=500)
    print(f"\nDone. BFCL={n_bfcl}, ToolBench={n_tb}")
