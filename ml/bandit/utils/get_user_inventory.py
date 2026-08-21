import os
import json
from decouple import config
import discogs_client
from discogs_client.exceptions import HTTPError
from datetime import datetime
from ..models import DiscogsRecord 
from .rate_limiter import RateLimiter, rate_limit_client

consumer_key = config('DISCOGS_CONSUMER_KEY')
consumer_secret = config('DISCOGS_CONSUMER_SECRET')
TOKEN_FILE = 'discogs_token.json'

INVENTORY_FILE = 'user_inventories.json'
INVENTORIES_FOLDER = 'inventories'

LIMITER = RateLimiter(60)

VINYL_FORMATS = {'LP', '7"', '12"', '10"', 'Vinyl'}

def normalize_format(fmt):
    """Return a flat list of individual format tokens.

    The marketplace listing's embedded ``release.format`` is a comma-separated
    string (e.g. ``"LP, Album, RE"``), while full-release data is already a list.
    Both must be reduced to individual tokens (``["LP", "Album", "RE"]``) before
    intersecting with VINYL_FORMATS, otherwise the whole comma-joined string is
    compared as a single token and never matches.
    """
    items = fmt if isinstance(fmt, list) else [fmt]
    tokens = []
    for item in items:
        tokens.extend(t.strip() for t in str(item).split(',') if t.strip())
    return tokens

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

def get_inventory(username, filter_by_demand=True, save_as_we_go=True):
    """
    Fetch inventory for a seller.

    Args:
        username: Discogs username
        filter_by_demand: If True, only include records where wants > haves (default)
                         If False, include all vinyl LPs regardless of demand
        save_as_we_go: If True, save records to database after each page (default)
    """
    from ..knapsack import save_listings

    d = authenticate_client()
    records = []
    seen_listing_ids = set()  # dedupe identical listings seen across pages (pagination drift)

    try:
        print(f"Fetching user: {username}")
        user = d.user(username)

        # Try to access user data first to verify user exists
        try:
            username_check = user.username
            print(f"User found: {username_check}")
        except Exception as e:
            print(f"User validation failed: {e}")
            raise Exception(f"Seller '{username}' not found or inaccessible. Check the username spelling.")

        print(f"Fetching inventory for {username}... (filter_by_demand={filter_by_demand})")
        inventory = user.inventory
        inventory.per_page = 250
        print(f"Inventory object created successfully")
    except Exception as e:
        print(f"\nError fetching inventory for user '{username}': {e}")
        raise

    total_pages = inventory.pages if hasattr(inventory, 'pages') else 100

    # Cap at 100 pages max
    total_pages = min(total_pages, 80)
    print(f"Total pages: {total_pages}")

    # Fetch pages in REVERSE order (newest first)
    for i in range(total_pages, 0, -1):
        try:
            print(f"Fetching page {i}...")
            page = inventory.page(i)
            print(f"Page {i} loaded successfully")
            page_records = filter_page(page, filter_by_demand=filter_by_demand)

            # Drop listings already seen on an earlier page (pagination can shift
            # as items sell during the multi-minute fetch, repeating a listing).
            # Dedupe on marketplace listing_id so genuine multiple copies of the
            # same release (distinct listings) are preserved.
            deduped = []
            for r in page_records:
                lid = r.get('listing_id')
                if lid is not None and lid in seen_listing_ids:
                    continue
                if lid is not None:
                    seen_listing_ids.add(lid)
                deduped.append(r)
            page_records = deduped

            # Save high-demand records from this page immediately
            if save_as_we_go and page_records:
                wanted_records = [
                    r for r in page_records
                    if r.get('wants', 0) > r.get('haves', 0) * 0.75
                ]
                if wanted_records:
                    save_listings(wanted_records)
                    print(f"  Saved {len(wanted_records)} records from page {i}")

            records += page_records

        except HTTPError as e:
            if e.status_code == 404:
                # Page doesn't exist, keep going backwards
                continue  
            else:
                print(f"HTTPError occurred on page {i} for user {username}: {e}. Skipping page.")
                continue  
        
        except Exception as e:
            print(f"Error fetching page {i} for user {username}: {e}. Skipping page.")
            continue  

    return records

def calculate_suggested_price_from_db(record, condition):
    """Calculate suggested price from DB record without API calls"""
    try:
        # Get VG+ price from record
        if not record.suggested_price:
            print(f"  No suggested_price in DB for {record.discogs_id}")
            return None

        price_str = str(record.suggested_price)
        print(f"  Found suggested_price in DB for {record.discogs_id}: {price_str}")

        # Parse different formats
        if '<Price' in price_str:
            # Extract from "<Price 149.5 'USD'>"
            import re
            match = re.search(r'<Price ([0-9.]+)', price_str)
            vg_plus = float(match.group(1)) if match else None
        elif ', ' in price_str:
            vg_plus = float(price_str.split(', ')[0])
        else:
            vg_plus = float(price_str)

        if not vg_plus:
            return None

        # Apply condition multiplier
        MULTIPLIERS = {
            'Near Mint (NM or M-)': 1.31,
            'Very Good Plus (VG+)': 1.0,
            'Very Good (VG)': 0.69,
            'Good Plus (G+)': 0.38,
        }

        return vg_plus * MULTIPLIERS.get(condition, 1.0)
    except Exception as e:
        print(f"Error calculating suggested price for {record.discogs_id}: {e}")
        return None

def filter_page(page, filter_by_demand=True):
    keepers = []

    # Collect release IDs as STRINGS to match database CharField
    release_ids = []
    for listing in page:
        try:
            release_ids.append(str(listing.release.id))
        except:
            continue

    # Batch query for all existing records
    existing_records = {
        r.discogs_id: r
        for r in DiscogsRecord.objects.filter(discogs_id__in=release_ids)
    }

    for listing in page:
        try:
            release_id = str(listing.release.id)

            if release_id in existing_records:
                # Record exists in DB - use that data (faster, more complete)
                record = existing_records[release_id]

                # Check format from listing data, not DB (DB format field is often empty)
                fmt = (listing.data or {}).get('release', {}).get('format') or []
                fmt_list = normalize_format(fmt)
                if fmt_list and not VINYL_FORMATS.intersection(fmt_list):
                    continue

                # Demand filter: wants > haves * 0.75
                # Close to strict wants > haves but catches near-misses
                if filter_by_demand and record.wants <= record.haves * 0.75:
                    continue

                # Calculate suggested price from DB data
                suggested_price = calculate_suggested_price_from_db(record, listing.condition)

                # If no suggested_price in DB, fetch from API and save it
                # This is slow but ESSENTIAL data - will be cached for next run
                if suggested_price is None:
                    try:
                        suggested_price = get_suggested_price(listing.release, listing.condition)
                        print(f"  Fetched and saved suggested_price for {release_id}")
                    except Exception as e:
                        print(f"  Error fetching suggested_price from API for {release_id}: {e}")
                        suggested_price = listing.price.value

                parsed = {
                    'listing_id': listing.id,
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
                    'genres': list(record.genres) if record.genres else [],
                    'styles': list(record.styles) if record.styles else [],
                    'year': record.year,
                    'suggested_price': suggested_price,
                    'format': fmt_list,
                }
                keepers.append(parsed)
            else:
                # Record not in DB - need to evaluate
                data = listing.data or {}
                release = data.get('release') or {}
                fmt = release.get('format') or []
                fmt_list = normalize_format(fmt)
                if fmt_list and not VINYL_FORMATS.intersection(fmt_list):
                    continue

                # Get wants/haves to decide if we should skip
                stats = (release.get('stats') or {}).get('community') or {}
                wants = stats.get('in_wantlist', 0)
                haves = stats.get('in_collection', 0)

                if filter_by_demand and wants <= haves * 0.75:
                    continue

                # Passes threshold - parse full record with API calls
                parsed = parse_listing(listing)
                if isinstance(parsed, dict):
                    keepers.append(parsed)
                    print(f"  ✓ Parsed and added {release_id}")
                else:
                    print(f"  ✗ Parse returned None for {release_id}")
        except Exception as e:
            print(f"Error filtering listing: {e}")
            continue

    return keepers

def parse_listing(l):
    try:
        _id = str(l.release.id)  # Convert to string for consistency with DB
        _media_condition = l.condition
        _price = (l.price.value, l.price.currency)
        _seller = l.seller.username

        rd = l.release.data or {}     # ← coalesce once
        _artist = rd.get('artist', '')
        _title  = rd.get('title', '')
        _label  = rd.get('label', '')
        _catno  = rd.get('catalog_number', '')
        _fmt    = rd.get('format') or []
        _fmt    = _fmt if isinstance(_fmt, list) else [str(_fmt)]

        stats = (rd.get('stats') or {}).get('community') or {}
        _wants = stats.get('in_wantlist', 0)
        _haves = stats.get('in_collection', 0)

        _genres = l.release.genres or []
        _styles = l.release.styles or []
        _year   = l.release.year

        return {
            'listing_id': l.id,
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
            'format': _fmt,
            'suggested_price': get_suggested_price(l.release, _media_condition)
        }
    except Exception as e:
        print(f"Error parsing listing: {e}")
        return None


def wanted(listing):
    """Check if record meets demand threshold"""
    try:
        data = listing.data or {}
        release = data.get('release') or {}
        stats = release.get('stats') or {}
        community = stats.get('community') or {}
        in_wantlist = community.get('in_wantlist', 0)
        in_collection = community.get('in_collection', 0)
        # Threshold: wants > haves * 0.75
        return in_wantlist > in_collection * 0.75
    except Exception:
        return False

def get_suggested_price(release, condition):
    try:
        record = DiscogsRecord.objects.filter(discogs_id=release.id).first()

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
            # Fetch from API and save to DB for next time
            vg_plus_obj = release.price_suggestions.very_good_plus
            if vg_plus_obj is None or vg_plus_obj.data is None:
                if record:
                    record.suggested_price = 'N/A'
                    record.save(update_fields=['suggested_price'])
                return None
            vg_plus = vg_plus_obj.value
            if record:
                record.suggested_price = str(vg_plus)
                record.save(update_fields=['suggested_price'])
                print(f"  Saved suggested_price for {record.discogs_id}: ${vg_plus}")

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


def build_test_set_negatives(username, negative_limit=21000):
    """
    Fetch inventory to collect negative examples (wants <= haves) for test set.

    Only saves NEGATIVES with full metadata. Skips all positives completely
    to avoid expensive suggested_price API calls.

    Args:
        username: Discogs username
        negative_limit: Stop after collecting this many negatives (default: 21000)

    Returns:
        dict with counts: {'negatives': int, 'skipped_positives': int}
    """
    from time import monotonic
    from django.utils import timezone

    d = authenticate_client()

    # Counters
    positive_skipped = 0
    negative_count = 0
    start_time = monotonic()

    print("=" * 70)
    print(f"BUILDING TEST SET NEGATIVES FROM SELLER: {username}")
    print("=" * 70)
    print(f"Target: {negative_limit:,} negatives")
    print(f"Saving: Full metadata for negatives ONLY (skipping positives)")
    print(f"API calls: None (all data from inventory listing)")
    print()

    try:
        print(f"Fetching user: {username}")
        user = d.user(username)

        try:
            username_check = user.username
            print(f"User found: {username_check}")
        except Exception as e:
            print(f"User validation failed: {e}")
            raise Exception(f"Seller '{username}' not found or inaccessible.")

        print(f"Fetching inventory...")
        inventory = user.inventory
        inventory.per_page = 250
        print(f"Inventory object created successfully\n")

    except Exception as e:
        print(f"\nError fetching inventory for user '{username}': {e}")
        raise

    # Get total pages
    first_page = inventory.page(1)
    total_pages = inventory.pages if hasattr(inventory, 'pages') else 100
    total_pages = min(total_pages, 100)  # Cap at 100
    print(f"Total pages: {total_pages} (capped at 100)")
    print()

    # Process pages in reverse order (newest first)
    for page_num in range(total_pages, 0, -1):
        if negative_count >= negative_limit:
            print(f"\n✓ Reached negative limit ({negative_limit:,}). Stopping.")
            break

        try:
            page_start = monotonic()
            print(f"[PAGE {page_num}/{total_pages}] Fetching...")
            page = inventory.page(page_num)
            page_fetch_time = monotonic() - page_start

            page_positives_skipped = 0
            page_negatives = 0

            for listing in page:
                try:
                    data = listing.data or {}
                    release = data.get('release') or {}
                    fmt = release.get('format') or []
                    fmt_list = normalize_format(fmt)
                    if fmt_list and not VINYL_FORMATS.intersection(fmt_list):
                        continue

                    # Get wants/haves (FREE from listing.data)
                    stats = (release.get('stats') or {}).get('community') or {}
                    wants = stats.get('in_wantlist', 0)
                    haves = stats.get('in_collection', 0)

                    release_id = str(listing.release.id)
                    is_positive = wants > haves

                    # Skip all positives (don't save, don't make API calls)
                    if is_positive:
                        page_positives_skipped += 1
                        positive_skipped += 1
                        continue

                    # Skip if we've hit negative limit
                    if negative_count >= negative_limit:
                        continue

                    # Parse full metadata for negatives only
                    rd = listing.release.data or {}
                    artist = rd.get('artist', '')
                    title = rd.get('title', '')
                    label = rd.get('label', '')
                    catno = rd.get('catalog_number', '')
                    genres = listing.release.genres or []
                    styles = listing.release.styles or []
                    year = listing.release.year

                    # Save negative record to database (no API calls needed)
                    DiscogsRecord.objects.get_or_create(
                        discogs_id=release_id,
                        defaults={
                            'artist': artist,
                            'title': title,
                            'label': label,
                            'catno': catno,
                            'wants': wants,
                            'haves': haves,
                            'genres': genres,
                            'styles': styles,
                            'year': year,
                            'format': fmt if isinstance(fmt, list) else [fmt],
                            'suggested_price': '',  # No suggested_price for negatives
                        }
                    )

                    negative_count += 1
                    page_negatives += 1

                except Exception as e:
                    print(f"  ✗ Error processing listing: {e}")
                    continue

            # Page summary
            page_total_time = monotonic() - page_start
            print(f"[PAGE {page_num}/{total_pages}] Complete in {page_total_time:.1f}s (fetch: {page_fetch_time:.1f}s)")
            print(f"  Negatives: +{page_negatives} saved")
            print(f"  Positives: {page_positives_skipped} skipped (not saved)")

            # Running totals
            elapsed = monotonic() - start_time
            rate = negative_count / (elapsed / 60) if elapsed > 0 else 0

            print(f"\n  RUNNING TOTALS:")
            print(f"    Negatives saved:  {negative_count:,} / {negative_limit:,} ({negative_count/negative_limit*100:.1f}%)")
            print(f"    Positives skipped: {positive_skipped:,}")
            print(f"    Rate:             {rate:.1f} negatives/min")
            print(f"    Elapsed:          {elapsed/60:.1f} minutes")

            # ETA calculation
            if negative_count < negative_limit and rate > 0:
                remaining_negatives = negative_limit - negative_count
                eta_minutes = remaining_negatives / rate
                print(f"    ETA:              {eta_minutes:.1f} minutes (to reach {negative_limit:,} negatives)")

            print()

        except HTTPError as e:
            if e.status_code == 404:
                continue
            else:
                print(f"  HTTPError on page {page_num}: {e}")
                continue
        except Exception as e:
            print(f"  Error on page {page_num}: {e}")
            continue

    # Final summary
    elapsed = monotonic() - start_time

    print("=" * 70)
    print("TEST SET NEGATIVES COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Negatives saved:   {negative_count:,}")
    print(f"Positives skipped: {positive_skipped:,}")
    print(f"Time:              {elapsed/60:.1f} minutes")
    print(f"Rate:              {negative_count/(elapsed/60):.1f} negatives/min")
    print("=" * 70)

    return {
        'negatives': negative_count,
        'positives_skipped': positive_skipped,
        'elapsed_minutes': elapsed / 60
    }