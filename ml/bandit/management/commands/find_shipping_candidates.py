"""
Scans the wantlist scrape pages for foreign sellers (ships_from != United
States) that aren't yet in ANY bucket of sellers_sorted.json, and appends
them to a new "candidates" bucket at the bottom of that file for manual
review -- same workflow as the existing free_shipping notes column, just
unverified until a human checks the seller's real policy and fills in
notes.

Does not touch the DB. Pure scrape-JSON -> sellers_sorted.json candidate
surfacing; run sync_seller_shipping separately once you've promoted a
candidate into free_shipping or flat_rate.

Run:
  cd ml && .venv/bin/python3 manage.py find_shipping_candidates
"""
import json
import glob
from pathlib import Path

from django.core.management.base import BaseCommand
from django.conf import settings

DEFAULT_JSON = Path(settings.BASE_DIR) / "sellers_sorted.json"
DEFAULT_PAGES_GLOB = str(
    Path(settings.BASE_DIR) / "bandit" / "wantlist_scrape" / "pages" / "*.json"
)


def load_page_objects(path):
    """Handles both a single {"items": [...]} object and the older
    comma-joined-multi-object page dumps."""
    with open(path) as f:
        raw = f.read().strip()
    if not raw:
        return []
    content = raw.rstrip().rstrip(',')
    try:
        return [json.loads(content)]
    except json.JSONDecodeError:
        try:
            return json.loads('[' + content + ']')
        except json.JSONDecodeError:
            return []


class Command(BaseCommand):
    help = "Surface new foreign-seller shipping candidates from the wantlist scrape into sellers_sorted.json."

    def add_arguments(self, parser):
        parser.add_argument('--json', default=str(DEFAULT_JSON), help='Path to sellers_sorted.json')
        parser.add_argument('--pages', default=DEFAULT_PAGES_GLOB, help='Glob of scraped page JSON files')

    def handle(self, *args, **options):
        json_path = Path(options['json'])
        if not json_path.exists():
            self.stderr.write(self.style.ERROR(f"No file at {json_path}"))
            return

        with json_path.open() as f:
            data = json.load(f)
        buckets = data[0]

        known = set()
        for bucket_name in ('free_shipping', 'flat_rate', 'candidates'):
            for entry in buckets.get(bucket_name, []):
                known.add(entry['username'])

        candidates = {}
        files = [p for p in glob.glob(options['pages']) if not p.endswith('.bak')]
        for path in files:
            for page in load_page_objects(path):
                for item in page.get('items') or []:
                    seller = item.get('seller') or {}
                    name = seller.get('name')
                    ships_from = seller.get('shipsFrom')
                    if not name or not ships_from:
                        continue
                    if ships_from == 'United States':
                        continue
                    if name in known or name in candidates:
                        continue

                    shipping = item.get('shipping') or {}
                    price = item.get('price') or {}
                    candidates[name] = {
                        'username': name,
                        'ships_from': ships_from,
                        # Same shape as a promoted flat_rate entry (rate_5lp)
                        # so no renaming is needed during annotation. Left
                        # null -- the scrape only ever observes a single-item
                        # order, which isn't the same quantity as a 5-LP
                        # order cost, so it can't be prefilled honestly.
                        'rate_5lp': None,
                        'currency': price.get('currencyCode'),
                        # Reference only: what a single item from this
                        # seller cost to ship in the one listing we saw.
                        'observed_single_item_price': shipping.get('shippingPrice'),
                        'observed_free_shipping_min': shipping.get('freeShippingMin'),
                        'notes': '',
                    }

        if not candidates:
            self.stdout.write('No new candidates found.')
            return

        buckets.setdefault('candidates', []).extend(candidates.values())
        with json_path.open('w') as f:
            json.dump(data, f, indent=2)

        self.stdout.write(self.style.SUCCESS(
            f"Added {len(candidates)} new candidates to {json_path} (bottom of 'candidates' bucket)."
        ))
