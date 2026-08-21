"""
Main evaluation script for Discogs record classification pipeline.

Evaluates pipeline on 395,508 enriched catalog with:
- 75% coverage requirement (296,631+ records)
- 90% precision on both classes
- 70% recall on both classes (63,170 TPs + 213,686 TNs)
- 1,000 API call budget
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

from archive.cataloger.eval_files.test_api_client import TestAPIClient
from archive.cataloger.eval_files.metrics import calculate_metrics, format_metrics_report


def load_data():
    """Load catalog with ground truth."""
    print("Loading data files...")

    # Catalog with ground truth (395,508 enriched records)
    catalog_file = Path(__file__).parent / 'lp_catalog.json'
    with open(catalog_file) as f:
        catalog = json.load(f)
    print(f"  Catalog: {len(catalog):,} records")

    return catalog


def find_pipeline(pipeline_path):
    """
    Resolve pipeline.py path. All files are in a flat directory structure,
    so we check the given path and same directory as evaluate.py.
    """
    candidates = [
        Path(pipeline_path),                    # As given
        Path(__file__).parent / pipeline_path,  # Relative to eval script
        Path(__file__).parent / 'pipeline.py',  # Same dir as eval script
    ]

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except (PermissionError, OSError):
            continue

    raise FileNotFoundError(
        f"Could not find pipeline.py. Tried: {pipeline_path} and same directory as evaluate.py. "
        "Pass the explicit path with --pipeline /path/to/pipeline.py"
    )


def evaluate_pipeline(pipeline_path='pipeline.py'):
    """
    Run evaluation of the pipeline.

    Args:
        pipeline_path: Path to pipeline.py module (searched broadly if not found)

    Returns:
        dict with evaluation results and exit code
    """
    print("=" * 70)
    print("DISCOGS PIPELINE EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    catalog = load_data()

    # Count ground truth
    positives = sum(1 for r in catalog if r.get('wants', 0) > r.get('haves', 0))
    negatives = len(catalog) - positives

    print(f"\nGround Truth:")
    print(f"  Total records:  {len(catalog):,}")
    print(f"  Positives:      {positives:,} ({positives/len(catalog)*100:.1f}%)")
    print(f"  Negatives:      {negatives:,} ({negatives/len(catalog)*100:.1f}%)")
    print()

    # Import pipeline
    print("Importing pipeline...")
    pipeline_module_path = find_pipeline(pipeline_path)
    print(f"  Found pipeline at: {pipeline_module_path}")
    sys.path.insert(0, str(pipeline_module_path.parent))

    from archive.cataloger.pipeline import classify_catalog

    # Strip wants/haves from catalog (so pipeline can't cheat)
    test_catalog = []
    for record in catalog:
        test_record = {k: v for k, v in record.items() if k not in ['wants', 'haves']}
        test_catalog.append(test_record)

    print(f"\nPrepared test catalog: {len(test_catalog):,} records (wants/haves stripped)")
    print()

    # Create API client backed by full ground truth catalog
    # (Pipeline can query any of the 395,508 records, max 1,000 calls)
    api_client = TestAPIClient(catalog, max_calls=1000)

    # Run pipeline
    print("=" * 70)
    print("RUNNING PIPELINE")
    print("=" * 70)
    print()

    results = classify_catalog(test_catalog, api_client)

    # Calculate metrics
    print("\n" + "=" * 70)
    print("CALCULATING METRICS")
    print("=" * 70)
    print()

    metrics = calculate_metrics(results, catalog)

    print(format_metrics_report(metrics))

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'catalog_size': len(catalog),
        'ground_truth': {
            'positives': positives,
            'negatives': negatives,
            'positive_rate': positives / len(catalog)
        },
        'metrics': metrics
    }

    output_file = Path(__file__).parent / 'evaluation_report.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Check pass/fail criteria
    print("\n" + "=" * 70)
    print("PASS/FAIL CRITERIA")
    print("=" * 70)
    print()

    coverage_pct = metrics['coverage']['coverage_pct']
    prec_in = metrics['precision']['ruled_in']
    prec_out = metrics['precision']['ruled_out']
    recall_in = metrics['recall']['ruled_in']
    recall_out = metrics['recall']['ruled_out']
    api_calls = metrics['coverage']['api_calls_made']

    # Display checks
    checks = [
        ("Coverage", coverage_pct, 0.75, "≥", lambda x: f"{x:.1%}"),
        ("Precision (ruled_in)", prec_in, 0.90, "≥", lambda x: f"{x:.1%}"),
        ("Precision (ruled_out)", prec_out, 0.90, "≥", lambda x: f"{x:.1%}"),
        ("Recall (ruled_in)", recall_in, 0.70, "≥", lambda x: f"{x:.1%}"),
        ("Recall (ruled_out)", recall_out, 0.70, "≥", lambda x: f"{x:.1%}"),
        ("API Budget", api_calls, 1000, "≤", lambda x: f"{x} calls"),
    ]

    all_pass = True
    for name, value, threshold, op, fmt in checks:
        if op == "≥":
            passed = value >= threshold
        else:  # "≤"
            passed = value <= threshold

        status = "✓" if passed else "✗"
        print(f"  {name}: {fmt(value)} {status}")

        if not passed:
            all_pass = False

    # Add pass field to output and resave
    output['pass'] = all_pass
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    if all_pass:
        print("=" * 70)
        print("RESULT: PASS ✓")
        print("=" * 70)
        return output, 0
    else:
        print("=" * 70)
        print("RESULT: FAIL ✗")
        print("=" * 70)
        return output, 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate Discogs record classification pipeline'
    )
    parser.add_argument(
        '--pipeline',
        default='pipeline.py',
        help='Path to pipeline.py (default: pipeline.py, searched broadly if not found)'
    )

    args = parser.parse_args()

    try:
        results, exit_code = evaluate_pipeline(pipeline_path=args.pipeline)
        sys.exit(exit_code)

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
