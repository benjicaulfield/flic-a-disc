import json
import logging
from pathlib import Path

from discogs_client.exceptions import HTTPError
from django.core.management.base import BaseCommand

from .get_user_inventory import get_inventory
from .rate_limiter import RateLimiter
from ..models import DiscogsListing, Record

logger = logging.getLogger(__name__)

USERNAMES_PATH = Path(__file__).parent / "usernames.json"

class Populator:
    def __init__(self, command):
        self.stdout = command.stdout
        self.style = command.style
        self.rate_limiter = RateLimiter()
        self.usernames = self.load_usernames()

    def run(self):
        for user in list(self.usernames):
            self.process_user(user)
            self.usernames.remove(user)
            self.save_usernames()

    def process_user(self, user):
        self.stdout.write(f"- {user}")

        try:
            inventory = get_inventory(user)
            if not inventory:
                self.stdout.write("No inventory found")
                return

        except HTTPError as e:
            self._error(user, f"HTTPError: {e}")
            return
        except Exception as e:
            self._error(user, f"Unexpected error: {e}")
            return

        if not inventory:
            self.stdout.write("No inventory found")
            return
        
        processed = 0
        errors = 0

        for record_data in inventory:
            try:
                record, _ = self.process_record(record_data)
                listing, _ = self.process_listing(record, record_data)
                processed += 1
            except Exception as e:
                errors += 1
                logger.error(f"Error processing record: {e}")
                self.stdout.write(self.style.ERROR(str(e)))

        self.stdout.write(
            f"Done: {processed} processed, {errors} errors"
        )

    def process_record(self, record_data):
        discogs_id = self._get(record_data, "discogs_id")
        if not discogs_id:
            raise ValueError("Missing discogs_id")
        
        return Record.objects.get_or_create(
            discogs_id=discogs_id,
            defaults={
                "artist": self.force_primitive(self._get(record_data, "artist", "")),
                "title": self.force_primitive(self._get(record_data, "title", "")),
                "label": self.force_primitive(self._get(record_data, "label", "")),
                "catno": self.force_primitive(self._get(record_data, "catno", "")),
                "wants": self._get(record_data, "wants", 0),
                "haves": self._get(record_data, "haves", 0),
                "genres": self.force_primitive(self._get(record_data, "genres", [])),
                "styles": self.force_primitive(self._get(record_data, "styles", [])),
                "year": self._get(record_data, "year"),
                "suggested_price": self.force_primitive(self._get(record_data, "suggested_price")),
            },
        )
    
    def force_primitive(self, value):
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)
    
    def process_listing(self, record, record_data):
        # Discogs inventory objects usually look like record_data.seller.id
        seller = self._get(record_data, "seller")

        if isinstance(seller, dict):
            seller_id = seller.get("id")
        else:
            seller_id = getattr(seller, "id", None)

        if not seller_id:
            seller_id = 42

        record_price = self.force_primitive(self._get(record_data, "record_price"))
        media_condition = self.force_primitive(self._get(record_data, "media_condition"))

        if not record_price:
            raise ValueError("Missing record_price")
        if not media_condition:
            raise ValueError("Missing media_condition")

        return DiscogsListing.objects.get_or_create(
            seller_id=seller_id,
            record=record,
            defaults={
                "record_price": str(record_price),
                "media_condition": media_condition,
            },
        )


    def load_usernames(self):
        if not USERNAMES_PATH.exists():
            return []
        return json.loads(USERNAMES_PATH.read_text())

    def save_usernames(self):
        tmp = USERNAMES_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.usernames, indent=2))
        tmp.replace(USERNAMES_PATH)

    def _get(self, obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _error(self, user, message):
        logger.error(f"{user}: {message}")
        self.stdout.write(self.style.ERROR(f"{user}: {message}"))
