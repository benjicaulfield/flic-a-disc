"""
Generate balanced training CSV from enriched records.
- All negatives (wants <= haves)
- Positives sampled to achieve 29.1% positive rate
"""
import os
import sys
import django
import pandas as pd
from pathlib import Path

# Setup Django
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from bandit.models import Record
from django.db.models import F

def main():
    print("=" * 70)
    print("GENERATING BALANCED TRAINING CSV")
    print("=" * 70)

    # Get all negatives (wants <= haves, unbiased random samples only)
    print("\n📊 Querying negatives (wants <= haves)...")
    negatives = list(Record.objects.filter(
        wanted=False,
        wants__lte=F('haves')
    ).values(
        'discogs_id', 'artist', 'title', 'label', 'catno', 'year',
        'genres', 'styles', 'format', 'wants', 'haves'
    ))

    print(f"✓ Found {len(negatives):,} negatives")

    # Get all positives (wants > haves, unbiased random samples only)
    print("\n📊 Querying positives (wants > haves)...")
    positives = list(Record.objects.filter(
        wanted=False,
        wants__gt=F('haves')
    ).values(
        'discogs_id', 'artist', 'title', 'label', 'catno', 'year',
        'genres', 'styles', 'format', 'wants', 'haves'
    ))

    print(f"✓ Found {len(positives):,} positives")

    # Calculate how many positives needed for 29.1% rate
    # positives / (positives + negatives) = 0.291
    # positives = 0.291 * (positives + negatives)
    # positives = (0.291 / 0.709) * negatives
    target_positive_rate = 0.291
    num_negatives = len(negatives)
    num_positives_needed = int((target_positive_rate / (1 - target_positive_rate)) * num_negatives)

    print(f"\n🎯 Target: {target_positive_rate:.1%} positive rate")
    print(f"   Negatives: {num_negatives:,}")
    print(f"   Positives needed: {num_positives_needed:,}")

    if len(positives) < num_positives_needed:
        print(f"\n⚠️  WARNING: Only {len(positives):,} positives available, need {num_positives_needed:,}")
        print(f"   Will use all {len(positives):,} positives")
        num_positives_needed = len(positives)

    # Randomly sample positives
    import random
    random.seed(42)
    sampled_positives = random.sample(positives, num_positives_needed)

    print(f"\n✓ Sampled {len(sampled_positives):,} positives")

    # Combine
    all_records = negatives + sampled_positives
    random.shuffle(all_records)  # Shuffle to mix positives and negatives

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Add label column
    df['label'] = (df['wants'] > df['haves']).astype(int)

    # Calculate actual positive rate
    actual_rate = df['label'].mean()

    print(f"\n📈 Final dataset:")
    print(f"   Total records: {len(df):,}")
    print(f"   Positives: {df['label'].sum():,}")
    print(f"   Negatives: {(~df['label'].astype(bool)).sum():,}")
    print(f"   Positive rate: {actual_rate:.1%}")

    # Save to CSV
    output_path = Path('discogs/data/training_data.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved to {output_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
