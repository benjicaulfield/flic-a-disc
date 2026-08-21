"""
Export test set from Django database to JSON.

Creates a balanced 25k test set with 29% positive, 71% negative split
(matching the base rate from enriched_training.json).
"""

import os
import sys
import json
import django
from pathlib import Path

# Add ml directory to path for Django imports
ml_dir = Path(__file__).parent.parent / 'ml'
sys.path.insert(0, str(ml_dir))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.db.models import F
from bandit.models import Record


def export_test_set(
    output_file='data/test_set.json',
    total_size=25000,
    positive_rate=0.29
):
    """
    Export balanced test set from database.

    Args:
        output_file: Path to output JSON file
        total_size: Total number of records in test set
        positive_rate: Proportion of positives (default: 0.29)
    """
    print("=" * 70)
    print("EXPORTING TEST SET FROM DATABASE")
    print("=" * 70)

    # Calculate split
    n_positives = int(total_size * positive_rate)
    n_negatives = total_size - n_positives

    print(f"Target: {total_size:,} records")
    print(f"  Positives: {n_positives:,} ({positive_rate:.1%})")
    print(f"  Negatives: {n_negatives:,} ({1-positive_rate:.1%})")
    print()

    # Load enriched_training IDs to exclude
    training_file = Path(__file__).parent / 'data' / 'enriched_training.json'
    print(f"Loading training IDs from {training_file}...")

    with open(training_file) as f:
        training_data = json.load(f)
    training_ids = {str(r['discogs_id']) for r in training_data}
    print(f"  Excluding {len(training_ids):,} training records\n")

    # Query available records (not in training, not skipped)
    available = Record.objects.filter(
        skipped=False
    ).exclude(
        discogs_id__in=training_ids
    )

    # Sample negatives
    print(f"Sampling {n_negatives:,} negatives...")
    negatives = list(
        available.filter(wants__lte=F('haves'))
        .order_by('?')  # Random order
        .values(
            'discogs_id', 'artist', 'title', 'label', 'catno',
            'wants', 'haves', 'genres', 'styles', 'year', 'format'
        )[:n_negatives]
    )
    print(f"  ✓ Sampled {len(negatives):,} negatives")

    # Sample positives
    print(f"Sampling {n_positives:,} positives...")
    positives = list(
        available.filter(wants__gt=F('haves'))
        .order_by('?')  # Random order
        .values(
            'discogs_id', 'artist', 'title', 'label', 'catno',
            'wants', 'haves', 'genres', 'styles', 'year', 'format'
        )[:n_positives]
    )
    print(f"  ✓ Sampled {len(positives):,} positives\n")

    # Combine and shuffle
    test_set = negatives + positives
    import random
    random.shuffle(test_set)

    # Verify split
    actual_positives = sum(1 for r in test_set if r['wants'] > r['haves'])
    actual_rate = actual_positives / len(test_set)

    print(f"Final test set:")
    print(f"  Total: {len(test_set):,}")
    print(f"  Positives: {actual_positives:,} ({actual_rate:.1%})")
    print(f"  Negatives: {len(test_set) - actual_positives:,} ({1-actual_rate:.1%})")
    print()

    # Save to JSON
    output_path = Path(__file__).parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(test_set, f, indent=2)

    print(f"  ✓ Saved {len(test_set):,} records")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 70)
    print("✓ Test set export complete!")
    print("=" * 70)

    return test_set


if __name__ == '__main__':
    export_test_set()
