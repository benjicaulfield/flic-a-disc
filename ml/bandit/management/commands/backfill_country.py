import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction

from bandit.models import DiscogsRecord

CATALOG = Path(__file__).parent.parent.parent / 'data' / 'lp_catalog.json'
CHUNK   = 1000


class Command(BaseCommand):
    help = 'Backfill country field from lp_catalog.json'

    def handle(self, *args, **options):
        self.stdout.write("Loading catalog...")
        with open(CATALOG) as f:
            catalog = json.load(f)

        # Build release_id -> country map, skip blanks
        country_map = {
            str(r['release_id']): r['country']
            for r in catalog
            if r.get('country')
        }
        self.stdout.write(f"Catalog entries with country: {len(country_map):,}")

        self.stdout.write("Fetching all DB record IDs...")
        qs = DiscogsRecord.objects.values_list('id', 'discogs_id')
        total = qs.count()
        self.stdout.write(f"Total records: {total:,}")

        updated = 0
        batch = []

        for record_id, discogs_id in qs.iterator(chunk_size=CHUNK):
            country = country_map.get(str(discogs_id))
            if not country:
                continue
            batch.append(DiscogsRecord(id=record_id, country=country))
            if len(batch) >= CHUNK:
                with transaction.atomic():
                    DiscogsRecord.objects.bulk_update(batch, ['country'])
                updated += len(batch)
                batch = []
                self.stdout.write(f"  Updated: {updated:,} / {total:,}", ending='\r')
                self.stdout.flush()

        if batch:
            with transaction.atomic():
                DiscogsRecord.objects.bulk_update(batch, ['country'])
            updated += len(batch)

        self.stdout.write(f"\nDone. Updated {updated:,} records.")
