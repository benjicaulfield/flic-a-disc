"""
Convert catalog from JSON lines to JSON array format.

Reads lp_catalog_filtered_four.json (JSON lines) and converts to JSON array.
"""
import json
from pathlib import Path

def main():
    input_file = Path('discogs/data/lp_catalog_filtered_four.json')
    output_file = Path('discogs/data/lp_catalog.json')

    print("=" * 70)
    print("CONVERTING CATALOG TO JSON ARRAY")
    print("=" * 70)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")

    # Read JSON lines
    print(f"\n📂 Reading JSON lines...")
    records = []
    with open(input_file) as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                records.append(json.loads(line))
            if i % 500000 == 0:
                print(f"   Loaded {i:,} records...")

    print(f"✓ Loaded {len(records):,} records")

    # Write as JSON array
    print(f"\n💾 Writing JSON array...")
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2)

    # Check file size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(records):,} records ({size_mb:.2f} MB)")

    print("\n" + "=" * 70)
    print("✓ CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\nFile: {output_file}")
    print(f"Records: {len(records):,}")
    print(f"Format: JSON array")

if __name__ == '__main__':
    main()
