
import json

def main():
    with open('data/qwen_mining_1000.jsonl', 'r') as f:
        count = 0
        for line in f:
            ex = json.loads(line)
            if ex['label'] == 1:
                print(f"LABEL: {ex['label']}")
                print(f"GT: {ex['ground_truth'][:100]}")
                print(f"GEN: {ex['generated'][:100]}")
                print("-" * 40)
                count += 1
                if count >= 3:
                    break

if __name__ == "__main__":
    main()
