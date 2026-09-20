"""
Loads ml/bandit/discogs_scrapes/wantlist_listings.csv into DiscogsRecord /
DiscogsListing, backfilling wants/haves/suggested_price from the Discogs
API for any release that doesn't already have them cached.

Run:
  cd ml && .venv/bin/python3 manage.py import_wantlist_listings
"""
import csv
from pathlib import Path

from django.core.management.base import BaseCommand
from django.conf import settings

from bandit.wantlist_import import import_rows

DEFAULT_CSV = Path(settings.BASE_DIR) / "bandit" / "discogs_scrapes" / "wantlist_listings.csv"


class Command(BaseCommand):
    help = "Import wantlist_listings.csv into DiscogsRecord/DiscogsListing, backfilling missing data from the Discogs API."

    def add_arguments(self, parser):
        parser.add_argument('--csv', default=str(DEFAULT_CSV), help='Path to wantlist_listings.csv')
        parser.add_argument('--skip-enrich', action='store_true', help='Skip Discogs API backfill of wants/haves/suggested_price')
        parser.add_argument('--limit-enrich', type=int, default=None, help='Cap how many releases get API-enriched (for testing)')

    def handle(self, *args, **options):
        csv_path = Path(options['csv'])
        if not csv_path.exists():
            self.stderr.write(self.style.ERROR(f"No CSV at {csv_path}"))
            return

        with csv_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.stdout.write(f"Loaded {len(rows)} listing rows from {csv_path}")

        summary = import_rows(
            rows,
            skip_enrich=options['skip_enrich'],
            limit_enrich=options['limit_enrich'],
            log=self.stdout.write,
        )
        self.stdout.write(self.style.SUCCESS(
            f"Done: {summary['listings_created']} new listings, {summary['listings_updated']} updated"
        ))
