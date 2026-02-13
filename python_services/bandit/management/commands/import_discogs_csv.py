# bandit/management/commands/import_discogs_csv.py
from django.core.management.base import BaseCommand
import csv
from django.utils import timezone
from bandit.models import Record
from bandit.utils.get_user_inventory import authenticate_client
import time

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--inventory', type=str, help='Path to inventory CSV')
        parser.add_argument('--wantlist', type=str, help='Path to wantlist CSV')
    
    def handle(self, *args, **options):
        self.client = authenticate_client()
        
        if options['inventory']:
            self.import_inventory(options['inventory'])
        if options['wantlist']:
            self.import_wantlist(options['wantlist'])
    
    def fetch_release_data(self, release_id):
        """Fetch full data from Discogs API"""
        try:
            release = self.client.release(release_id)
            stats = release.data.get('community', {}) if release.data else {}
            
            # Get suggested price
            try:
                vg_plus = release.price_suggestions.very_good_plus
                suggested_price = str(vg_plus)
            except:
                suggested_price = ''
            
            return {
                'genres': release.genres or [],
                'styles': release.styles or [],
                'year': release.year,
                'wants': stats.get('want', 0),
                'haves': stats.get('have', 0),
                'suggested_price': suggested_price,
                'evaluated': True,
            }
        except Exception as e:
            self.stdout.write(f"Error fetching {release_id}: {e}")
            return {'evaluated': True}
    
    def import_inventory(self, filepath):
        self.stdout.write("Importing inventory...")
        count = 0
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                api_data = self.fetch_release_data(row['release_id'])
                
                Record.objects.get_or_create(
                    discogs_id=row['release_id'],
                    defaults={
                        'artist': row['artist'],
                        'title': row['title'],
                        'label': row['label'],
                        'catno': row['catno'],
                        'format': row['format'],
                        'added': timezone.now(),
                        'wanted': False,
                        'genres': api_data.get('genres', []),
                        'styles': api_data.get('styles', []),
                        'wants': api_data.get('wants', 0),
                        'haves': api_data.get('haves', 0),
                        'suggested_price': api_data.get('suggested_price', ''),
                        'evaluated': api_data.get('evaluated', True),
                    }
                )
                count += 1
                if count % 10 == 0:
                    self.stdout.write(f"Processed {count}...")
                    time.sleep(1)
        
        self.stdout.write(f"Imported {count} inventory records")
    
    def import_wantlist(self, filepath):
        self.stdout.write("Importing wantlist...")
        count = 0
        seen = set()
        
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['Artist'], row['Title'])
                
                if key in seen:
                    continue
                
                seen.add(key)
                
                api_data = self.fetch_release_data(row['release_id'])
                
                Record.objects.get_or_create(
                    discogs_id=row['release_id'],
                    defaults={
                        'artist': row['Artist'],
                        'title': row['Title'],
                        'label': row['Label'],
                        'catno': row['Catalog#'],
                        'format': row['Format'],
                        'year': int(row['Released']) if row['Released'] else None,
                        'added': timezone.now(),
                        'wanted': True,
                        'genres': api_data.get('genres', []),
                        'styles': api_data.get('styles', []),
                        'wants': api_data.get('wants', 0),
                        'haves': api_data.get('haves', 0),
                        'suggested_price': api_data.get('suggested_price', ''),
                        'evaluated': api_data.get('evaluated', True),
                    }
                )
                count += 1
                if count % 10 == 0:
                    self.stdout.write(f"Processed {count}...")
                    time.sleep(1)
        
        self.stdout.write(f"Imported {count} unique wantlist records")