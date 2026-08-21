import os
import re
import csv
import glob
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from bandit.models import EbayListing

class Command(BaseCommand):

    def normalize_title(self, text):
        if not text: return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    def parse_price(self, price_str):
        if not price_str: return 0.0
        try:
            cleaned = re.sub(r"[^0-9.]", "", price_str)
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    def handle(self, *args, **kwargs):
        raw = glob.glob(os.path.join(settings.BASE_DIR, 'bandit/ebay/ebay_training_data/ebay_auctions_2026*.csv'))
        raw += glob.glob(os.path.join(settings.BASE_DIR, 'bandit/ebay/ebay_training_data/ebay_bin_2026*.csv'))
        annotated = glob.glob(os.path.join(settings.BASE_DIR, 'bandit/ebay/ebay_training_data/ebay_auctions_key_*.csv'))
        annotated += glob.glob(os.path.join(settings.BASE_DIR, 'bandit/ebay/ebay_training_data/ebay_bin_key_*.csv'))

        raw_rows = []
        for path in raw:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for line in reader:
                    if len(line) < 3:
                        continue
                    ebay_id, ebay_title, price = line[0], line[1], line[2]
                    raw_rows.append({"ebay_id": ebay_id, "ebay_title": ebay_title, "price": price})
        
        listings = [
            EbayListing(
                ebay_id=row['ebay_id'],
                ebay_title=self.normalize_title(row['ebay_title']),
                price=self.parse_price(row['price']),
            )
            for row in raw_rows
        ]
        print(f"listings: {len(listings)}")
        EbayListing.objects.bulk_create(listings, ignore_conflicts=True)

        keeper_ids = set()
        for path in annotated:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for line in reader:
                    if len(line) < 3:
                        continue
                    keeper_ids.add(line[0])
        
        EbayListing.objects.filter(ebay_id__in=keeper_ids).update(evaluated=True, wanted=True)

        all_ids = {row['ebay_id'] for row in raw_rows}
        non_keeper_ids = all_ids - keeper_ids
        print(f"nonkeeperids: {len(non_keeper_ids)}")
        EbayListing.objects.filter(ebay_id__in=non_keeper_ids).update(evaluated=True, wanted=False)
        

        
            


        

