
import json

def main():
    try:
        with open('data/qwen_tuning_200.jsonl', 'r') as f:
            count = 0
            for line in f:
                data = json.loads(line)
                gt = data['ground_truth']
                # Heuristic: contains "{" or "tool"
                # if "{" in gt:
                print(f"GT Ex {count}: {gt[:200]}")
                count += 1
                if count >= 5:
                    break
    except FileNotFoundError:
        print("File not found.")

if __name__ == "__main__":
    main()
