#!/usr/bin/env python3
"""Runner for pipeline 9."""

import json
import time
import sys
import archive.cataloger.pipeline_runs.pipeline_9 as pipeline_9

def main():
    print("Loading catalog...")
    with open('lp_catalog.json', 'r') as f:
        catalog = json.load(f)

    print(f"Catalog loaded: {len(catalog)} records")
    print(f"Starting pipeline 9...")

    start_time = time.time()

    try:
        result = pipeline_9.classify_catalog(catalog)
        runtime = time.time() - start_time

        print(f"\nPipeline 9 COMPLETED!")
        print(f"Runtime: {runtime:.2f}s")
        print(f"Ruled in: {len(result['ruled_in'])}")
        print(f"Ruled out: {len(result['ruled_out'])}")
        print(f"Coverage: {result['metadata']['coverage_ratio']:.4f}")
        print(f"API calls: {result['metadata']['api_calls_made']}")

        result['metadata']['runtime_seconds'] = runtime
        with open('results_9.json', 'w') as f:
            json.dump(result, f, indent=2)

        return 0

    except Exception as e:
        runtime = time.time() - start_time
        print(f"\nPipeline 9 FAILED after {runtime:.2f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        error_info = {
            'status': 'failed',
            'error': str(e),
            'runtime_seconds': runtime,
        }
        with open('results_9.json', 'w') as f:
            json.dump(error_info, f, indent=2)

        return 1

if __name__ == '__main__':
    sys.exit(main())
