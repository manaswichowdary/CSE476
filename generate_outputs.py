
import json
import argparse
import os
import signal
import sys
from tqdm import tqdm
from src.agent import Agent

# Input/Output paths
INPUT_PATH = "cse476_final_project_dev_data.json"
OUTPUT_PATH = "cse476_final_project_answers.json"

results = []
interrupted = False

def save_results(partial=False):
    """Saves current results to the output file."""
    if partial:
        print(f"\nSaving partial results ({len(results)} items)...")
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

def handle_interrupt(sig, frame):
    """Handles Ctrl+C to save before exiting."""
    global interrupted
    print("\n\nProcess interrupted! Saving progress...")
    save_results(partial=True)
    interrupted = True
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, handle_interrupt)

def generate_outputs(limit=None):
    global results
    
    # Load input data
    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)
    
    # Load existing results if resuming
    processed_count = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    results = existing_data
                    processed_count = len(results)
                    print(f"Resuming from existing output file. Found {processed_count} already processed items.")
        except json.JSONDecodeError:
            print("Warning: Output file exists but is corrupted or empty. Starting from scratch.")
    
    if limit:
        data = data[:limit]
        print(f"Limiting to first {limit} items.")
    
    # Determine which items still need to be processed
    # We assume sequential processing 1:1 with input data
    remaining_data = data[processed_count:]
    
    if not remaining_data:
        print("All items already processed!")
        return

    agent = Agent()
    print(f"Generating outputs for {len(remaining_data)} remaining items...")
    
    for i, item in enumerate(tqdm(remaining_data, initial=processed_count, total=len(data))):
        if interrupted:
            break
            
        try:
            # Solve
            prediction = agent.solve(item)
            
            # Create result object
            result_item = item.copy()
            result_item['prediction'] = prediction
            
            # Append and Save immediately (Robustness)
            results.append(result_item)
            
            if i % 1 == 0: # Save every item to ensure no data loss
                save_results()
            
        except Exception as e:
            print(f"Error on item index {processed_count + i}: {e}")
            item['prediction'] = "Error"
            results.append(item)
            save_results()

    # Final Save
    save_results()
    print(f"\nDone. Saved {len(results)} outputs to {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of items for testing")
    args = parser.parse_args()
    
    generate_outputs(limit=args.limit)
