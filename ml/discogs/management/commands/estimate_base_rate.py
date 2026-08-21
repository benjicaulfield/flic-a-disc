import json                                                                                                                                                
import random
import time                                                                                                                                                
import sys                                                                            
from pathlib import Path
from datetime import datetime
from math import sqrt

from django.core.management.base import BaseCommand
from django.utils import timezone

from bandit.models import Record
from bandit.discogs_client import authenticate_client

class Command(BaseCommand):
    help = 'Estimate base rate of wants>haves in catalog via random sampling'

    def add_arguments(self, parser):
        parser.add_argument(
            '--sample-size',
            type=int,
            default=5000,
            help='Number of random records to sample (default: 5000)'
        )
        parser.add_argument(
            '--catalog-path',
            type=str,
            default='discogs/data/lp_catalog_filtered_thrice.json',
            help='Path to LP catalog file'
        )
        parser.add_argument(
            '--resume',
            action='store_true',
            help='Resume from previous run (skip already enriched records)'
        )

    def handle(self, *args, **options):
        sample_size = options['sample_size']
        catalog_path = Path(options['catalog_path'])
        resume = options['resume']

        self.stdout.write("="*70)
        self.stdout.write(f"BASE RATE ESTIMATION VIA RANDOM SAMPLING")
        self.stdout.write("="*70)
        self.stdout.write(f"Sample size: {sample_size}")
        self.stdout.write(f"Catalog: {catalog_path}")
        self.stdout.write(f"Estimated time: {sample_size / 120:.1f} minutes (120/min)")
        self.stdout.write("="*70)

        # Load catalog
        self.stdout.write("\n📂 Loading catalog...")
        catalog = []
        with open(catalog_path) as f:
            for line in f:
                if line.strip():
                    catalog.append(json.loads(line))

        self.stdout.write(f"✓ Loaded {len(catalog):,} records")

        # Random sample
        self.stdout.write(f"\n🎲 Sampling {sample_size} random records...")
        sample = random.sample(catalog, min(sample_size, len(catalog)))

        # Filter out already enriched (if resume mode)
        if resume:
            existing_ids = set(Record.objects.values_list('discogs_id', flat=True))
            sample = [r for r in sample if r['release_id'] not in existing_ids]
            self.stdout.write(f"✓ Resume mode: {len(sample)} records remaining after dedup")

        # Initialize Discogs client
        d = authenticate_client()

        # Tracking stats
        enriched_count = 0
        wants_gt_haves_count = 0
        errors = 0

        self.stdout.write("\n" + "="*70)
        self.stdout.write("ENRICHMENT IN PROGRESS")
        self.stdout.write("="*70)
        self.stdout.write("")

        start_time = time.time()

        for i, record in enumerate(sample, 1):
            release_id = record['release_id']

            try:
                # API call
                release = d.release(int(release_id))
                wants = release.community.want if hasattr(release, 'community') else 0
                haves = release.community.have if hasattr(release, 'community') else 0

                # Log each record
                artist = record.get('artist', 'Unknown')
                title = record.get('title', 'Unknown')
                keeper_status = '✓ KEEPER' if wants > haves else '  skip'

                self.stdout.write(
                    f"  [{i:4d}] {keeper_status} | "
                    f"W:{wants:4d} H:{haves:4d} | "
                    f"{artist[:30]:30s} - {title[:40]:40s}"
                )

                # Save to database
                Record.objects.update_or_create(
                    discogs_id=release_id,
                    defaults={
                        'artist': record.get('artist', 'Unknown'),
                        'title': record.get('title', 'Unknown'),
                        'format': ['Vinyl', 'LP'],
                        'label': record.get('label', ''),
                        'catno': record.get('catalog_number'),
                        'wants': wants,
                        'haves': haves,
                        'genres': record.get('genre', []),
                        'styles': record.get('style', []),
                        'year': int(record['year']) if record.get('year') else None,
                        'wanted': False,
                        'evaluated': True,
                    }
                )

                enriched_count += 1
                if wants > haves:
                    wants_gt_haves_count += 1

                # Running statistics (print every 50 records)
                if i % 50 == 0 or i == len(sample):
                    self._print_stats(i, len(sample), enriched_count, wants_gt_haves_count,
                                    errors, start_time)

                # Rate limiting (120/min = 0.5 per second)
                time.sleep(0.5)

            except Exception as e:
                errors += 1
                self.stdout.write(f"  ⚠️   Error on {release_id}: {e}")
                continue

        # Final report
        self.stdout.write("\n" + "="*70)
        self.stdout.write("FINAL RESULTS")
        self.stdout.write("="*70)

        total_time = time.time() - start_time
        positive_rate = wants_gt_haves_count / enriched_count if enriched_count > 0 else 0
        ci_lower, ci_upper = self._confidence_interval(positive_rate, enriched_count)

        self.stdout.write(f"\n✓ Enriched: {enriched_count:,} records")
        self.stdout.write(f"✓ Errors: {errors}")
        self.stdout.write(f"✓ Time elapsed: {total_time/60:.1f} minutes")
        self.stdout.write(f"\n📊 ESTIMATED BASE RATE (wants > haves):")
        self.stdout.write(f"   Point estimate: {positive_rate:.1%}")
        self.stdout.write(f"   95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
        self.stdout.write(f"\n💡 Interpretation:")
        self.stdout.write(f"   In the full 2.9M catalog, approximately {positive_rate:.1%} of records")
        self.stdout.write(f"   are desirable (wants > haves).")
        self.stdout.write(f"   That's roughly {int(2_900_000 * positive_rate):,} records.")
        self.stdout.write("="*70 + "\n")

    def _print_stats(self, current, total, enriched, positives, errors, start_time):
        """Print running statistics"""
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0

        positive_rate = positives / enriched if enriched > 0 else 0
        ci_lower, ci_upper = self._confidence_interval(positive_rate, enriched)

        # Progress bar
        pct = current / total
        bar_width = 40
        filled = int(bar_width * pct)
        bar = '█' * filled + '░' * (bar_width - filled)

        self.stdout.write(f"\r[{bar}] {current}/{total} ({pct:.1%})", ending='')
        self.stdout.flush()

        # Detailed stats every 50
        if current % 50 == 0:
            self.stdout.write("")  # New line
            self.stdout.write(f"  Enriched: {enriched:,} | Errors: {errors}")
            self.stdout.write(f"  Wants>Haves: {positives:,} ({positive_rate:.1%})")
            self.stdout.write(f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
            self.stdout.write(f"  Rate: {rate:.1f} rec/sec | ETA: {eta/60:.1f} min")
            self.stdout.write("")

    def _confidence_interval(self, p, n, z=1.96):
        """Calculate 95% confidence interval for proportion"""
        if n == 0:
            return 0.0, 0.0

        # Wilson score interval (better for small n or extreme p)
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denominator
        margin = z * sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denominator

        return max(0, centre - margin), min(1, centre + margin)