from django.core.management.base import BaseCommand
import pandas as pd
from bandit.models import EbayBasicListing

class Command(BaseCommand):
    help = 'Import eBay training data from CSV'

    def handle(self, *args, **options):
        csv_path = 'data/ebay_training_data.csv'
        df = pd.read_csv(csv_path)
        self.stdout.write(f"Importing {len(df)} listings...")
        
        created = 0
        for _, row in df.iterrows():
            _, was_created = EbayBasicListing.objects.get_or_create(
                ebay_title=row['Title'],
                defaults={
                    'price': row['Price'],
                    'wanted': row['wanted'],
                }
            )
            if was_created:
                created += 1
            
            if created % 1000 == 0:
                self.stdout.write(f"Created {created} listings...")
        
        self.stdout.write(self.style.SUCCESS(
            f"✅ Imported {created} new listings "
            f"({df['wanted'].sum()} keepers, {(~df['wanted']).sum()} skips)"
        ))