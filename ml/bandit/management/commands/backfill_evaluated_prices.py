"""
Backfill suggested_price for evaluated records that are missing it.

Run with:
    uv run python manage.py backfill_evaluated_prices
    uv run python manage.py backfill_evaluated_prices --limit 100   # test run
    uv run python manage.py backfill_evaluated_prices --resume      # pick up where left off
"""
import time
from pathlib import Path
from django.core.management.base import BaseCommand
from django.db.models import Q
from bandit.models import DiscogsRecord
from bandit.discogs_client import authenticate_client


CHECKPOINT_FILE = Path(__file__).parent / '.backfill_evaluated_checkpoint'


class Command(BaseCommand):
    help = 'Backfill suggested_price for evaluated records missing it'

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=None)
        parser.add_argument('--resume', action='store_true')

    def handle(self, *args, **options):
        limit = options.get('limit')
        resume = options.get('resume')

        self.stdout.write('\n' + '='*60)
        self.stdout.write('BACKFILL EVALUATED RECORD PRICES')
        self.stdout.write('='*60 + '\n')

        client = authenticate_client()
        self.stdout.write('✓ Authenticated\n')

        qs = DiscogsRecord.objects.filter(
            evaluated=True,
        ).filter(
            Q(suggested_price__isnull=True) | Q(suggested_price='')
        ).order_by('discogs_id')

        total = qs.count()
        self.stdout.write(f'Found {total:,} evaluated records missing suggested_price')

        start_after = None
        if resume and CHECKPOINT_FILE.exists():
            start_after = CHECKPOINT_FILE.read_text().strip()
            self.stdout.write(f'Resuming after discogs_id={start_after}')
            qs = qs.filter(discogs_id__gt=start_after)

        if limit:
            qs = qs[:limit]
            self.stdout.write(f'Limiting to {limit:,} records\n')

        count = qs.count()
        self.stdout.write(f'Processing {count:,} records')
        self.stdout.write(f'Estimated time: ~{count/60:.0f} min ({count/60/60:.1f} hours)\n')

        processed = updated = skipped = errors = 0
        start_time = time.time()

        for record in qs.iterator(chunk_size=100):
            try:
                time.sleep(1)
                release = client.release(int(record.discogs_id))

                try:
                    vg_plus = release.price_suggestions.very_good_plus
                    if vg_plus is None or vg_plus.data is None:
                        raise AttributeError('no price data')
                    record.suggested_price = str(vg_plus.value)
                    updated += 1
                except AttributeError:
                    record.suggested_price = 'N/A'
                    skipped += 1

                record.save(update_fields=['suggested_price'])
                CHECKPOINT_FILE.write_text(str(record.discogs_id))

            except Exception as e:
                self.stderr.write(f'  Error on {record.discogs_id}: {e}')
                errors += 1

            processed += 1

            if processed % 60 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (count - processed) / rate / 60 if rate > 0 else 0
                self.stdout.write(
                    f'[{processed:,}/{count:,}] '
                    f'updated={updated:,} skipped={skipped:,} errors={errors:,} '
                    f'ETA={eta:.0f}m'
                )

        elapsed = time.time() - start_time
        self.stdout.write('\n' + '='*60)
        self.stdout.write('DONE')
        self.stdout.write(f'Processed: {processed:,}')
        self.stdout.write(f'Updated:   {updated:,}')
        self.stdout.write(f'No price:  {skipped:,} (set to N/A)')
        self.stdout.write(f'Errors:    {errors:,}')
        self.stdout.write(f'Time:      {elapsed/60:.1f} min')
        self.stdout.write('='*60 + '\n')

        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
