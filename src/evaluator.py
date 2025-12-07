import json
import os
import argparse
import re
from src.agent import Agent

# Load dev data
DATA_PATH = "cse476_final_project_dev_data.json"

def load_data(path=DATA_PATH):
    with open(path, 'r') as f:
        return json.load(f)

def normalize_text(s: str) -> str:
    s = (str(s) or "").strip().lower()
    # Remove surrounding punctuation and extra whitespace
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_number(s: str):
    # Try multiple strategies to find the final answer
    if not s:
        return None
        
    s = str(s)
    
    # 1. Boxed latex
    boxed = re.search(r"\\boxed\{(.*?)\}", s)
    if boxed:
        s = boxed.group(1)
        
    # 2. Extract all numbers and take the last one
    # Note: this regex handles integers and floats
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if matches:
        return matches[-1]
        
    return None

def grade_answer(expected: str, got: str, domain: str) -> bool:
    if domain == "math":
        # Try numeric match first
        exp_num = extract_number(expected)
        got_num = extract_number(got)
        if exp_num is not None and got_num is not None:
             # check float equality with tolerance
            try:
                if abs(float(exp_num) - float(got_num)) < 1e-9:
                    return True
            except:
                pass
        
    return normalize_text(got) == normalize_text(expected)

def run_evaluation(limit=None, batch_size=10, strict=False):
    data = load_data()
    agent = Agent()
    
    # If limit is explicitly None, run all. 
    # If limit is an int > 0, run that many.
    if limit:
        data = data[:limit]
        print(f"Running evaluation on first {limit} items...")
    else:
        print(f"Running evaluation on FULL dataset ({len(data)} items)...")
        
    print(f"Settings: batch_size={batch_size}, strict={strict}")
    
    results = []
    total_correct = 0
    total_evaluated = 0
    
    
    # Domain stats
    domain_stats = {} # {domain: {'total': 0, 'correct': 0}}

    # Process in batches
    for b_start in range(0, len(data), batch_size):
        batch = data[b_start : b_start + batch_size]
        batch_num = (b_start // batch_size) + 1
        
        print(f"\n\n{'='*20} BATCH {batch_num} (Items {b_start+1}-{b_start+len(batch)}) {'='*20}")
        
        batch_correct = 0
        
        for i, item in enumerate(batch):
            abs_idx = b_start + i
            total_evaluated += 1
            
            print(f"\n--- Problem {abs_idx+1} ---")
            prompt = item['input']
            expected = item['output']
            domain = item.get('domain', 'unknown')
            
            # Init domain stats
            if domain not in domain_stats:
                domain_stats[domain] = {'total': 0, 'correct': 0}
            domain_stats[domain]['total'] += 1
            
            print(f"Domain: {domain}")
            # print(f"Prompt: {prompt[:100]}...")
            
            try:
                prediction = agent.solve(item)
                print(f"Prediction: {prediction}")
                print(f"Expected: {expected}")
                
                is_correct = grade_answer(expected, prediction, domain)
                
                if is_correct:
                    print(f"Result: CORRECT (Batch: {batch_correct+1}/{i+1})")
                    batch_correct += 1
                    total_correct += 1
                    item['correct'] = True
                    domain_stats[domain]['correct'] += 1
                else:
                    print(f"Result: INCORRECT (Batch: {batch_correct}/{i+1})")
                    item['correct'] = False
                
                item['prediction'] = prediction
                results.append(item)
                
            except Exception as e:
                print(f"Error executing problem {abs_idx}: {e}")
        
        # Batch Complete Checks
        print(f"\n>>> Batch {batch_num} Summary: {batch_correct}/{len(batch)} correct.")
        
        if strict and batch_correct < len(batch):
            print(f"[STRICT MODE] Batch failed. Stopping evaluation.")
            break
            
    print(f"\nTotal Summary: {total_correct}/{total_evaluated} correct ({len(results)} items processed).")
    print("\nDomain Breakdown:")
    for dom, stats in domain_stats.items():
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {dom}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Number of items to test (0 for all)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--strict", action="store_true", help="Stop if a batch fails")
    args = parser.parse_args()
    
    # Passing None if limit is 0 to trigger full run logic
    limit_arg = args.limit if args.limit > 0 else None
    run_evaluation(limit=limit_arg, batch_size=args.batch_size, strict=args.strict)
