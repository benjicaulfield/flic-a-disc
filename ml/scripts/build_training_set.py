"""
Select 6000 enriched records from each vote bucket, flag them training_set=True,
then fetch suggested_price for those missing it.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

import requests
from bandit.models import Record

BUCKET_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'catalog/data/vote_buckets.jsonl')
SAMPLE_SIZE = 6000
DISCOGS_TOKEN = os.environ.get('DISCOGS_TOKEN', '')

def get_suggested_price(discogs_id):
    url = f'https://api.discogs.com/marketplace/stats/{discogs_id}'
    headers = {'Authorization': f'Discogs token={DISCOGS_TOKEN}', 'User-Agent': 'flic-a-disc/1.0'}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            price = data.get('lowest_price') or data.get('median_price') or {}
            if isinstance(price, dict):
                val = price.get('value')
                return str(val) if val else ''
        return ''
    except Exception as e:
        print(f'  Error fetching {discogs_id}: {e}')
        return ''

# Load buckets
buckets = {}
with open(BUCKET_FILE) as f:
    for line in f:
        obj = json.loads(line)
        buckets[obj['votes']] = obj['records']

# Step 1: mark training_set=True for 6000 enriched records per bucket
print('Marking training set...')
all_selected_ids = []
for votes in sorted(buckets):
    ids = buckets[votes]
    selected = list(
        Record.objects.filter(discogs_id__in=ids, wants__gt=0)
        .order_by('?')
        .values_list('id', flat=True)[:SAMPLE_SIZE]
    )
    Record.objects.filter(id__in=selected).update(training_set=True)
    all_selected_ids.extend(selected)
    print(f'  votes={votes}: marked {len(selected)} records')

print(f'Total marked: {len(all_selected_ids)}')

# Step 2: fetch suggested_price for those missing it
missing = list(
    Record.objects.filter(training_set=True, suggested_price='')
    .values_list('id', 'discogs_id')
)
print(f'\nFetching suggested price for {len(missing)} records...')

for i, (record_id, discogs_id) in enumerate(missing):
    price = get_suggested_price(discogs_id)
    if price:
        Record.objects.filter(id=record_id).update(suggested_price=price)
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{len(missing)} done')
    time.sleep(1.1)  # Discogs rate limit: 60/min

print('Done.')
