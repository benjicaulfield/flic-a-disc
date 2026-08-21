"""
Count positive and negative records in database that aren't in enriched_training.json.

This shows what's available for building a new test set.
"""

import json
from pathlib import Path
from django.core.management.base import BaseCommand
from bandit.models import DiscogsRecord


class Command(BaseCommand):
    help = 'Count available positive and negative records for test set'

    def handle(self, *args, **options):
        # Load enriched_training.json to get IDs to exclude
        data_dir = Path(__file__).parent.parent.parent.parent / 'discogs' / 'data'
        training_file = data_dir / 'enriched_training.json'

        self.stdout.write(f"Loading training IDs from {training_file}...")
        with open(training_file, 'r') as f:
            training_data = json.load(f)

        training_ids = {str(record['discogs_id']) for record in training_data}
        self.stdout.write(f"  Excluding {len(training_ids):,} training records\n")

        # Query database for available records (not in training)
        self.stdout.write("Querying database for available records...")
        available = DiscogsRecord.objects.exclude(
            discogs_id__in=training_ids
        )

        total_available = available.count()
        self.stdout.write(f"  Found {total_available:,} available records\n")

        if total_available == 0:
            self.stdout.write(self.style.WARNING("No available records found!"))
            return

        # Count positives and negatives
        self.stdout.write("Counting positives and negatives...")
        positives = available.filter(wants__gt=0).extra(
            where=['wants > haves']
        ).count()

        negatives = total_available - positives

        # Display results
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("AVAILABLE RECORDS (not in training set)"))
        self.stdout.write("=" * 60)
        self.stdout.write(f"Total:     {total_available:,}")
        self.stdout.write(f"Positives: {positives:,} ({positives/total_available*100:.1f}%) - wants > haves")
        self.stdout.write(f"Negatives: {negatives:,} ({negatives/total_available*100:.1f}%) - wants <= haves")
        self.stdout.write("=" * 60 + "\n")
