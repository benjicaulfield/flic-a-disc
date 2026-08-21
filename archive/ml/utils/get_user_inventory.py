import json
import os
import datetime

INVENTORY_FILE = ""

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
                    # Check if LP format
                    data = listing.data or {}
                    release = data.get('release') or {}
                    fmt = release.get('format') or []
                    format_str = ' '.join(fmt) if isinstance(fmt, list) else str(fmt)

                    if 'LP' not in format_str:
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
                    Record.objects.get_or_create(
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
                            'added': timezone.now(),
                            'skipped': False,
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