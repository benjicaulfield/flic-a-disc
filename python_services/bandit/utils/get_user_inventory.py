import os
import json
from decouple import config
import discogs_client
from discogs_client.exceptions import HTTPError
from datetime import datetime
from ..models import Record 
from .rate_limiter import RateLimiter, rate_limit_client

consumer_key = config('DISCOGS_CONSUMER_KEY')
consumer_secret = config('DISCOGS_CONSUMER_SECRET')
TOKEN_FILE = 'discogs_token.json'

INVENTORY_FILE = 'user_inventories.json'
INVENTORIES_FOLDER = 'inventories'

LIMITER = RateLimiter(60)

def load_inventory_json():
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_inventory_json(inventory):
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f, indent=4)

def update_user_inventory(username, record_ids):
    data = load_inventory_json()
    today = datetime.now().strftime('%Y-%m-%d')
    if username not in data:
        data[username] = {
            "last_inventory": today,
            "record_ids": record_ids
        }
    else:
        existing_ids = data[username]['record_ids']
        all_ids = record_ids + [rid for rid in existing_ids if rid not in record_ids]
        data[username] = {
            "last_inventory": today,
            "record_ids": all_ids[:50]
        }
    
    save_inventory_json(data)

def load_tokens():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return json.load(f)
    return None

def save_tokens(token, secret):
    with open(TOKEN_FILE, 'w') as f:
        json.dump({'token': token, 'secret': secret}, f)

def authenticate_client():
    d = discogs_client.Client('wantlist/1.0')
    d.set_consumer_key(consumer_key, consumer_secret)
    rate_limit_client(d, LIMITER)
    tokens = load_tokens()

    if tokens:
        d.set_token(tokens['token'], tokens['secret'])
    else:
        token, secret, url = d.get_authorize_url()
        print(f"Please visit this URL to authorize: {url}")
        verifier = input("Enter the verifier code: ")
        access_token, access_secret = d.get_access_token(verifier)
        d.set_token(access_token, access_secret)
        save_tokens(access_token, access_secret)

    return d

def get_inventory(username):
    d = authenticate_client()
    records = []
    user = d.user(username)
    inventory = user.inventory    
    inventory.per_page = 250

    for i in range(1, 100):
        try:
            print(f"Fetching page {i}...")
            page = inventory.page(i)
            print(f"Page {i} loaded successfully")
            page_records = filter_page(page)
            records += page_records

        except HTTPError as e:
            if e.status_code == 404:
                print(f"Page {i} for user {username}: All done.")
                break  
            else:
                print(f"HTTPError occurred on page {i} for user {username}: {e}. Skipping page.")
                continue  
        
        except Exception as e:
            print(f"Error fetching page {i} for user {username}: {e}. Skipping page.")
            continue  

    return records

def filter_page(page):
    keepers = []
    
    release_ids = []
    for listing in page:
        try:
            release_ids.append(listing.release.id)
        except:
            continue
    
    existing_records = {
        r.discogs_id: r 
        for r in Record.objects.filter(discogs_id__in=release_ids)
    }
    
    for listing in page:
        try:
            release_id = listing.release.id
            
            if release_id in existing_records:
                record = existing_records[release_id]
                parsed = {
                    'discogs_id': record.discogs_id,
                    'media_condition': listing.condition,
                    'record_price': f"{listing.price.value}, {listing.price.currency}",
                    'seller': listing.seller.username,
                    'artist': record.artist,
                    'title': record.title,
                    'label': record.label,
                    'catno': record.catno,
                    'wants': record.wants,
                    'haves': record.haves,
                    'genres': record.genres,
                    'styles': record.styles,
                    'year': record.year,
                    'suggested_price': get_suggested_price(listing.release, listing.condition)
                }
                keepers.append(parsed)
            else:
                data = listing.data or {}
                release = data.get('release') or {}
                fmt = release.get('format') or []
                format_str = ' '.join(fmt) if isinstance(fmt, list) else str(fmt)

                if 'LP' in format_str and wanted(listing):
                    parsed = parse_listing(listing)
                    if isinstance(parsed, dict):
                        keepers.append(parsed)
        except Exception as e:
            print(f"Error filtering listing: {e}")
            continue
    
    return keepers

def parse_listing(l):
    try:
        _id = l.release.id
        _media_condition = l.condition
        _price = (l.price.value, l.price.currency)
        _seller = l.seller.username

        rd = l.release.data or {}     # ← coalesce once
        _artist = rd.get('artist', '')
        _title  = rd.get('title', '')
        _label  = rd.get('label', '')
        _catno  = rd.get('catalog_number', '')

        stats = (rd.get('stats') or {}).get('community') or {}
        _wants = stats.get('in_wantlist', 0)
        _haves = stats.get('in_collection', 0)

        _genres = l.release.genres or []
        _styles = l.release.styles or []
        _year   = l.release.year

        return {
            'discogs_id': _id,
            'media_condition': _media_condition,
            'record_price': f"{_price[0]}, {_price[1]}",
            'seller': _seller,
            'artist': _artist,
            'title': _title,
            'label': _label,
            'catno': _catno,
            'wants': _wants,
            'haves': _haves,
            'genres': _genres,
            'styles': _styles,
            'year': _year,
            'suggested_price': get_suggested_price(l.release, _media_condition)
        }
    except Exception as e:
        print(f"Error parsing listing: {e}")
        return None


def wanted(listing):
    try:
        data = listing.data or {}
        release = data.get('release') or {}
        stats = release.get('stats') or {}
        community = stats.get('community') or {}
        in_wantlist = community.get('in_wantlist', 0)
        in_collection = community.get('in_collection', 0)
        return in_wantlist > in_collection
    except Exception:
        return False

def get_suggested_price(release, condition):
    try:
        record = Record.objects.filter(discogs_id=release.id).first()

        if record and record.suggested_price:
            # Parse "<Price 149.5 'USD'>" format
            price_str = str(record.suggested_price)
            if '<Price' in price_str:
                # Extract number from "<Price 149.5 'USD'>"
                import re
                match = re.search(r'<Price ([0-9.]+)', price_str)
                vg_plus = float(match.group(1)) if match else None
            elif ', ' in price_str:
                vg_plus = float(price_str.split(', ')[0])
            else:
                vg_plus = float(price_str)
        else:
            vg_plus = release.price_suggestions.very_good_plus.value
        
        MULTIPLIERS = {
            'Near Mint (NM or M-)': 1.31,
            'Very Good Plus (VG+)': 1.0,
            'Very Good (VG)': 0.69,
            'Good Plus (G+)': 0.38,
        }
        
        return vg_plus * MULTIPLIERS.get(condition, 1.0)
    except Exception as e:
        print(f"ERROR in get_suggested_price for {release.id}: {e}")
        return None