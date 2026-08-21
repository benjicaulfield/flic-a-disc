"""
Build test set by collecting negatives from seller inventory.
Randomly selects a seller from ml/sellers.json.
Skips all positives to avoid expensive suggested_price API calls.
"""

import json
import random
from pathlib import Path
from django.core.management.base import BaseCommand
from django.db.models import F
from bandit.utils.get_user_inventory import build_test_set_negatives


class Command(BaseCommand):
    help = 'Build test set negatives from random seller inventory (skips positives)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--negative-limit',
            type=int,
            default=21000,
            help='Stop after collecting this many negatives (default: 21000)'
        )
        parser.add_argument(
            '--seller',
            type=str,
            default=None,
            help='Specific seller username (optional - defaults to random from sellers.json)'
        )

    def handle(self, *args, **options):
        from bandit.models import Record
        from time import monotonic

        negative_limit = options['negative_limit']
        specific_seller = options.get('seller')

        overall_start = monotonic()
        sellers_processed = []
        total_negatives_added = 0

        # Load enriched_training.json to exclude those IDs
        data_dir = Path(__file__).parent.parent.parent.parent / 'discogs' / 'data'
        training_file = data_dir / 'enriched_training.json'

        self.stdout.write(f"Loading training IDs to exclude...")
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        training_ids = {str(record['discogs_id']) for record in training_data}

        # Load sellers list
        sellers_file = Path(__file__).parent.parent.parent.parent / 'sellers.json'

        try:
            with open(sellers_file, 'r') as f:
                sellers_data = json.load(f)
            usernames = [s['username'] for s in sellers_data if 'username' in s]

            if not usernames:
                self.stdout.write(self.style.ERROR("No usernames found in sellers.json"))
                return

            self.stdout.write(f"Loaded {len(usernames)} sellers from sellers.json")

        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.stdout.write(self.style.ERROR(f"Error loading sellers: {e}"))
            return

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("BUILDING TEST SET NEGATIVES")
        self.stdout.write("=" * 70)
        self.stdout.write(f"Target: {negative_limit:,} negatives total")
        self.stdout.write(f"Strategy: Process sellers sequentially until target reached")
        self.stdout.write("=" * 70 + "\n")

        # Loop through sellers until we hit the target
        while True:
            # Check current count in database
            current_count = Record.objects.filter(
                wants__lte=F('haves')
            ).exclude(
                discogs_id__in=training_ids
            ).count()

            if current_count >= negative_limit:
                self.stdout.write(f"\n✓ Target reached! {current_count:,} negatives in database")
                break

            remaining = negative_limit - current_count

            # Select seller
            if specific_seller:
                username = specific_seller
                self.stdout.write(f"\nUsing specified seller: {username}")
            else:
                # Avoid re-processing same sellers if possible
                available = [u for u in usernames if u not in sellers_processed]
                if not available:
                    self.stdout.write("\n✗ All sellers processed but target not reached")
                    self.stdout.write(f"  Current: {current_count:,} / {negative_limit:,}")
                    break

                username = random.choice(available)
                self.stdout.write(f"\nRandomly selected seller: {username}")
                self.stdout.write(f"  ({len(sellers_processed)+1} sellers processed, {len(available)-1} remaining)")

            self.stdout.write(f"  Current progress: {current_count:,} / {negative_limit:,}")
            self.stdout.write(f"  Need {remaining:,} more negatives\n")

            try:
                # Process this seller's inventory
                result = build_test_set_negatives(username, negative_limit=negative_limit)

                sellers_processed.append(username)
                negatives_from_seller = result['negatives']
                total_negatives_added += negatives_from_seller

                self.stdout.write(f"\n  ✓ Completed seller: {username}")
                self.stdout.write(f"    Negatives from this seller: {negatives_from_seller:,}")
                self.stdout.write(f"    Positives skipped: {result['positives_skipped']:,}")
                self.stdout.write(f"    Time: {result['elapsed_minutes']:.1f} minutes")

                # If specific seller provided, only do one iteration
                if specific_seller:
                    break

            except Exception as e:
                self.stdout.write(f"\n  ✗ Error processing seller {username}: {e}")
                sellers_processed.append(username)  # Don't retry failed sellers
                continue

        # Final summary
        overall_elapsed = (monotonic() - overall_start) / 60
        final_count = Record.objects.filter(
            wants__lte=F('haves')
        ).exclude(
            discogs_id__in=training_ids
        ).count()

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("FINAL SUMMARY")
        self.stdout.write("=" * 70)
        self.stdout.write(f"Sellers processed: {len(sellers_processed)}")
        self.stdout.write(f"  {', '.join(sellers_processed)}")
        self.stdout.write(f"Total negatives in database: {final_count:,}")
        self.stdout.write(f"Negatives added this run: {total_negatives_added:,}")
        self.stdout.write(f"Total time: {overall_elapsed:.1f} minutes")
        self.stdout.write("=" * 70)
