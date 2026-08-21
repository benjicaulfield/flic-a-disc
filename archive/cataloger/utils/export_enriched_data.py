"""
Export enriched records from Django database to JSON for mock API.

This script connects to the ml/ Django project and exports all enriched records
to a JSON file that the mock API can load.

Usage:
    python export_enriched_data.py [--output data/enriched_training.json]
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add ml/ to Python path to import Django
ML_DIR = Path(__file__).parent.parent / 'ml'
sys.path.insert(0, str(ML_DIR))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from bandit.models import Record


def export_enriched_records(output_path: str, format: str = 'jsonl'):
    """
    Export all enriched records to JSON.

    Args:
        output_path: Path to output file
        format: 'jsonl' (JSON lines) or 'array' (single JSON array)
    """
    print("=" * 70)
    print("EXPORTING ENRICHED DATA FOR MOCK API")
    print("=" * 70)

    # Query all enriched records (api_enriched=True, wanted=False for unbiased samples)
    print("\n📊 Querying enriched records...")
    records = Record.objects.filter(
        api_enriched=True,
        wanted=False  # Only unbiased random samples
    ).values(
        'discogs_id', 'artist', 'title', 'label', 'catno', 'year',
        'genres', 'styles', 'format', 'wants', 'haves'
    )

    records_list = list(records)
    print(f"✓ Found {len(records_list):,} enriched records")

    # Calculate stats
    positives = sum(1 for r in records_list if r['wants'] > r['haves'])
    negatives = len(records_list) - positives
    pos_rate = positives / len(records_list) if records_list else 0

    print(f"\n📈 Dataset stats:")
    print(f"   Positives (wants>haves): {positives:,} ({pos_rate:.1%})")
    print(f"   Negatives (wants<=haves): {negatives:,} ({1-pos_rate:.1%})")

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Export
    print(f"\n💾 Exporting to {output_path}...")

    if format == 'jsonl':
        # JSON lines format (one record per line)
        with open(output_file, 'w') as f:
            for record in records_list:
                # Convert to simpler format for mock API
                simplified = {
                    'discogs_id': record['discogs_id'],
                    'artist': record['artist'],
                    'title': record['title'],
                    'label': record['label'],
                    'year': record['year'],
                    'genres': record['genres'],
                    'styles': record['styles'],
                    'wants': record['wants'],
                    'haves': record['haves'],
                }
                f.write(json.dumps(simplified) + '\n')

    else:  # array format
        # Single JSON array
        simplified_records = [
            {
                'discogs_id': r['discogs_id'],
                'artist': r['artist'],
                'title': r['title'],
                'label': r['label'],
                'year': r['year'],
                'genres': r['genres'],
                'styles': r['styles'],
                'wants': r['wants'],
                'haves': r['haves'],
            }
            for r in records_list
        ]

        with open(output_file, 'w') as f:
            json.dump(simplified_records, f, indent=2)

    print(f"✓ Exported {len(records_list):,} records")

    # Calculate file size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ File size: {size_mb:.2f} MB")

    print("\n" + "=" * 70)
    print("✓ EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Start mock API: uvicorn mock_api:app --reload --port 8001")
    print(f"  2. Test endpoint: curl http://localhost:8001/stats")
    print(f"  3. Query release: curl http://localhost:8001/releases/<release_id>")


def main():
    parser = argparse.ArgumentParser(
        description="Export enriched records from Django database to JSON for mock API"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/enriched_training.json',
        help='Output file path (default: data/enriched_training.json)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['jsonl', 'array'],
        default='array',
        help='Output format: jsonl (JSON lines) or array (single JSON array)'
    )

    args = parser.parse_args()

    try:
        export_enriched_records(args.output, args.format)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
