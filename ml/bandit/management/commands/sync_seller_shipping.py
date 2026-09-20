"""
Syncs the hand-curated shipping policies in sellers_sorted.json onto
DiscogsSeller, then bulk-propagates those same fields onto every
DiscogsListing row for that seller (shipping policy is a seller-level
attribute, so it's cheaper to copy it down than to join at query time).

Reads sellers_sorted.json's "free_shipping" array (free-shipping-above-X
sellers) and "flat_rate" array (flat-rate-under-a-cap foreign sellers).
Both optional, missing ones are skipped. Does NOT touch the "candidates"
bucket (see find_shipping_candidates), since those are unverified.

A flat_rate entry looks like:
  {"username": "...", "rate_5lp": 28.0, "currency": "EUR", "notes": "DONE"}
rate_5lp is the annotated cost of shipping 5 LPs together in one order --
a single reference-order-size number instead of a base+per-item model.

Creates DiscogsSeller rows for usernames that don't exist yet, so a
seller's policy is on record even before we've imported any of their
listings.

Run:
  cd ml && .venv/bin/python3 manage.py sync_seller_shipping
"""
import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.conf import settings

from bandit.models import DiscogsSeller, DiscogsListing
from bandit.utils.get_exchange_rates import get_exchange_rates, convert_to_usd

DEFAULT_JSON = Path(settings.BASE_DIR) / "sellers_sorted.json"


def to_usd(amount, currency, rates, fallback_usd):
    try:
        return convert_to_usd(amount, currency, rates)
    except (KeyError, ZeroDivisionError, TypeError):
        return fallback_usd


class Command(BaseCommand):
    help = "Sync sellers_sorted.json shipping policies onto DiscogsSeller and propagate to DiscogsListing."

    def add_arguments(self, parser):
        parser.add_argument('--json', default=str(DEFAULT_JSON), help='Path to sellers_sorted.json')

    def handle(self, *args, **options):
        json_path = Path(options['json'])
        if not json_path.exists():
            self.stderr.write(self.style.ERROR(f"No file at {json_path}"))
            return

        with json_path.open() as f:
            data = json.load(f)
        buckets = data[0] if data else {}

        rates = get_exchange_rates()

        sellers_created = 0
        sellers_updated = 0
        listings_touched = 0

        for entry in buckets.get('free_shipping', []):
            seller, created = self._upsert_seller(entry, rates, kind='free_shipping')
            sellers_created += created
            sellers_updated += not created
            listings_touched += self._propagate(seller, kind='free_shipping')

        for entry in buckets.get('flat_rate', []):
            seller, created = self._upsert_seller(entry, rates, kind='flat_rate')
            sellers_created += created
            sellers_updated += not created
            listings_touched += self._propagate(seller, kind='flat_rate')

        self.stdout.write(self.style.SUCCESS(
            f"Sellers: {sellers_created} created, {sellers_updated} updated. "
            f"Listings touched: {listings_touched}"
        ))

    def _upsert_seller(self, entry, rates, kind):
        username = entry['username']
        currency = entry.get('currency') or ''
        notes = entry.get('notes') or ''

        if kind == 'free_shipping':
            amount = entry.get('amount')
            usd = to_usd(amount, currency, rates, entry.get('usd'))
        else:
            amount = entry.get('rate_5lp')
            usd = to_usd(amount, currency, rates, entry.get('rate_5lp_usd'))

        seller, created = DiscogsSeller.objects.get_or_create(
            name=username, defaults={'currency': currency}
        )

        if kind == 'free_shipping':
            seller.free_shipping_min_amount = amount
            seller.free_shipping_min_currency = currency
            seller.free_shipping_min_usd = usd
        else:
            seller.flat_rate_amount = amount
            seller.flat_rate_currency = currency
            seller.flat_rate_usd = usd

        if notes:
            seller.shipping_notes = notes
        seller.save()

        return seller, created

    def _propagate(self, seller, kind):
        if kind == 'free_shipping':
            fields = {
                'free_shipping_min_amount': seller.free_shipping_min_amount,
                'free_shipping_min_currency': seller.free_shipping_min_currency,
                'free_shipping_min_usd': seller.free_shipping_min_usd,
            }
        else:
            fields = {
                'flat_rate_amount': seller.flat_rate_amount,
                'flat_rate_currency': seller.flat_rate_currency,
                'flat_rate_usd': seller.flat_rate_usd,
            }
        return DiscogsListing.objects.filter(seller=seller).update(**fields)
