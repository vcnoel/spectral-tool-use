from datasets import load_dataset
import json

def main():
    print("Loading dataset...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    
    print(f"Features: {ds.features}")
    
    # Print first few examples
    for i in range(3):
        print(f"\n--- Example {i} ---")
        ex = ds[i]
        print(f"Keys: {ex.keys()}")
        print(f"System: {ex.get('system', 'N/A')}")
        chat = ex.get('chat', 'N/A')
        print(f"Chat Type: {type(chat)}")
        print(f"Chat Content: {chat}")

if __name__ == "__main__":
    main()
