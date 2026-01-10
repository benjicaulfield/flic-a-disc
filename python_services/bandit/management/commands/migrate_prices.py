from django.core.management.base import BaseCommand
from bandit.models import DiscogsListing

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        listings = DiscogsListing.objects.all()[1:]
        total = listings.count()
        print(total)

        for i, listing in enumerate(listings.iterator(chunk_size=1000)):
            try:
                price, currency = listing.record_price.split(', ')
                listing.price = float(price)
                listing.currency = currency
                listing.save()

                if i % 1000 == 0:
                    print(f"Processed {i}/{total}")
            except ValueError:
                print(f"Failed on row {listing.id}: {listing.record_price}")
            
        print("donezo")
