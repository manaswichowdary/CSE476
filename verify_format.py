import json
import sys
import argparse
from pathlib import Path

# Paths
INPUT_PATH = Path("cse476_final_project_submission/cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse476_final_project_submission/cse_476_final_project_answers.json")

def verify_format(limit=None):
    print(f"Verifying {OUTPUT_PATH} against {INPUT_PATH}...")
    
    # Load Input
    try:
        with INPUT_PATH.open("r") as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_PATH} not found.")
        return

    # Load Output
    try:
        with OUTPUT_PATH.open("r") as f:
            answers = json.load(f)
    except FileNotFoundError:
        print(f"Error: Output file {OUTPUT_PATH} not found.")
        return
    except json.JSONDecodeError:
        print("Error: Output file is not valid JSON.")
        return
        
    print(f"Number of questions: {len(questions)}")
    print(f"Number of answers:   {len(answers)}")
    
    expected_count = len(questions)
    if limit:
        print(f"Checking against limit: {limit}")
        expected_count = limit

    # Check 1: Length Mismatch
    if len(answers) != expected_count:
        if limit:
             print(f"⚠️  WARNING: Answers count ({len(answers)}) does not match limit ({expected_count}).")
        else:
             print(f"⚠️  WARNING: Length mismatch! Answers ({len(answers)}) do not cover all questions ({len(questions)}).")
        print("   (This is expected if generation is currently in progress)")
    else:
        print("✅ Lengths match.")

    # Check 2: Formatting per item
    issues = 0
    for i, item in enumerate(answers):
        # Check Dict
        if not isinstance(item, dict):
            print(f"Item {i}: Not a dictionary.")
            issues += 1
            continue
            
        # Check Key
        if 'output' not in item:
            print(f"Item {i}: Missing 'output' key.")
            issues += 1
            continue
            
        # Check Type
        val = item['output']
        if not isinstance(val, str):
            print(f"Item {i}: 'output' is not a string (Type: {type(val)}).")
            issues += 1
            continue
        
        # Check Length
        if len(val) >= 5000:
            print(f"Item {i}: Answer too long ({len(val)} chars). Max 5000.")
            issues += 1
            
    if issues == 0:
        print("\n✅ Internal format check passed (Structure/Types/Limits OK).")
    else:
        print(f"\n❌ Internal format check FAILED: Found {issues} issues.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Expected number of items")
    args = parser.parse_args()
    
    verify_format(limit=args.limit)
