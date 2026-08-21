import time
from datetime import datetime
from django.core.management.base import BaseCommand
from django.db.models import Q, F
from bandit.models import DiscogsRecord
from bandit.discogs_client import authenticate_client


class Command(BaseCommand):
    help = 'Backfill suggested_price for desirable records (wants > haves)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit number of records to process (for testing)'
        )
        parser.add_argument(
            '--resume',
            action='store_true',
            help='Resume from last checkpoint'
        )

    def handle(self, *args, **options):
        limit = options.get('limit')
        resume = options.get('resume')

        self.stdout.write(self.style.SUCCESS('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS('BACKFILL SUGGESTED PRICES'))
        self.stdout.write(self.style.SUCCESS('='*60 + '\n'))

        # Initialize Discogs client
        self.stdout.write("Authenticating with Discogs API...")
        client = authenticate_client()
        self.stdout.write(self.style.SUCCESS("✓ Authenticated\n"))

        # Query records needing backfill
        self.stdout.write("Querying records needing backfill...")
        records = DiscogsRecord.objects.filter(
            wants__gt=F('haves')
        ).filter(
            Q(suggested_price__isnull=True) | Q(suggested_price='')
        ).order_by('discogs_id')  # Consistent ordering for resume

        total = records.count()
        self.stdout.write(self.style.SUCCESS(f"✓ Found {total:,} records\n"))

        if limit:
            records = records[:limit]
            self.stdout.write(f"Limiting to {limit:,} records for testing\n")

        if total == 0:
            self.stdout.write(self.style.SUCCESS("No records need backfilling!"))
            return

        # Stats
        processed = 0
        updated = 0
        skipped = 0
        errors = 0
        start_time = time.time()

        self.stdout.write(f"Starting backfill...")
        self.stdout.write(f"Rate limit: 60 calls/min")
        self.stdout.write(f"Estimated time: ~{total/60:.1f} minutes ({total/60/60:.1f} hours)\n")

        for record in records:
            try:
                # Rate limiting: 60 calls/min = 1 call per second
                time.sleep(1)

                # Fetch release from API
                release = client.release(int(record.discogs_id))

                # Get VG+ price
                try:
                    vg_plus_price = release.price_suggestions.very_good_plus.value

                    # Update record
                    record.suggested_price = str(vg_plus_price)
                    record.save(update_fields=['suggested_price'])

                    updated += 1

                except AttributeError:
                    # No price suggestions available
                    skipped += 1

            except Exception as e:
                self.stderr.write(f"Error on {record.discogs_id}: {e}")
                errors += 1

            processed += 1

            # Progress update every 60 records (~1 minute)
            if processed % 60 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = total - processed
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60

                self.stdout.write(
                    f"[{processed:,}/{total:,}] "
                    f"Updated: {updated:,} | Skipped: {skipped:,} | Errors: {errors:,} | "
                    f"Rate: {rate*60:.1f}/min | ETA: {eta_minutes:.1f} min"
                )

            # Checkpoint every 500 records
            if processed % 500 == 0:
                self.stdout.write(self.style.WARNING(f"✓ Checkpoint: {processed:,} processed"))

        # Final summary
        elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS('BACKFILL COMPLETE'))
        self.stdout.write(self.style.SUCCESS('='*60))
        self.stdout.write(f"Total processed: {processed:,}")
        self.stdout.write(f"Updated: {updated:,}")
        self.stdout.write(f"Skipped (no price): {skipped:,}")
        self.stdout.write(f"Errors: {errors:,}")
        self.stdout.write(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed/60/60:.1f} hours)")
        self.stdout.write(self.style.SUCCESS('='*60 + '\n'))
