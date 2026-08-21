#!/usr/bin/env python3
"""
Runner script for pipeline-11
"""

import json
import sys
import os

# Import the pipeline module
# Need to be in the same directory as pipeline-11.py to import it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
from importlib import import_module
pipeline = import_module('pipeline-11')

def main():
    print("Loading catalog from lp_catalog.json...")
    with open('lp_catalog.json') as f:
        catalog = json.load(f)

    print(f"Loaded {len(catalog)} catalog records")
    print("Starting classification pipeline...")

    # Run the pipeline
    result = pipeline.classify_catalog(catalog)

    # Save results
    print("\nSaving results to results_t11.json...")
    with open('results_t11.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Ruled in: {len(result['ruled_in'])}")
    print(f"Ruled out: {len(result['ruled_out'])}")
    print(f"Coverage: {result['metadata']['coverage_ratio']:.4f}")
    print(f"API calls: {result['metadata']['api_calls_made']}")
    print(f"\nResults saved to results_t11.json")

if __name__ == '__main__':
    main()
