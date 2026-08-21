"""Debug script to trace where knapsack filters out records"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from bandit.utils.get_user_inventory import get_inventory
from bandit.knapsack import score_and_filter_seller_listings

seller = "AndMoreAgainRecords"
budget = 200

# Track a specific record you expect to see
EXPECTED_PRICE = "CA$59.99"  # Change this to match exactly what you see

print(f"\n{'='*60}")
print(f"🔍 DEBUGGING KNAPSACK FOR {seller}")
print(f"{'='*60}\n")

# Step 1: Get inventory
print("📦 Step 1: Fetching inventory...")
inventory = get_inventory(seller)
print(f"✅ Fetched {len(inventory)} total listings\n")

if len(inventory) > 0:
    print("First 5 listings:")
    for i, item in enumerate(inventory[:5], 1):
        price = item.get('record_price', 'N/A')
        marker = "👀 EXPECTED!" if EXPECTED_PRICE in str(price) else ""
        print(f"  {i}. {item.get('artist', 'Unknown')[:30]} - {price} - {item.get('media_condition')} - wants:{item.get('wants')} haves:{item.get('haves')} {marker}")
    print()

    # Find the expected record
    expected = None
    for item in inventory:
        if EXPECTED_PRICE in str(item.get('record_price', '')):
            expected = item
            print("🎯 FOUND EXPECTED RECORD:")
            for key in ['discogs_id', 'artist', 'title', 'media_condition', 'record_price', 'wants', 'haves', 'suggested_price']:
                print(f"  {key}: {item.get(key)}")
            print()

# Step 2: Filter by condition
ALLOWED_CONDITIONS = ['Near Mint (NM or M-)', 'Very Good Plus (VG+)']
condition_filtered = [
    item for item in inventory
    if item.get('media_condition') in ALLOWED_CONDITIONS
]
print(f"🎯 Step 2: After condition filter (NM/VG+ only)")
print(f"   Before: {len(inventory)} → After: {len(condition_filtered)}")

# Check if expected record survived
expected_survived = any(EXPECTED_PRICE in str(item.get('record_price', '')) for item in condition_filtered)
if expected_survived:
    print(f"   ✅ Expected record PASSED condition filter")
else:
    print(f"   ❌ Expected record REMOVED by condition filter")
    for item in inventory:
        if EXPECTED_PRICE in str(item.get('record_price', '')):
            print(f"      Condition was: '{item.get('media_condition')}'")
            print(f"      Allowed: {ALLOWED_CONDITIONS}")

if len(inventory) > len(condition_filtered):
    removed = len(inventory) - len(condition_filtered)
    print(f"   ❌ Removed {removed} items due to condition")
    conditions = {}
    for item in inventory:
        cond = item.get('media_condition', 'Unknown')
        conditions[cond] = conditions.get(cond, 0) + 1
    print("   Condition breakdown:")
    for cond, count in sorted(conditions.items(), key=lambda x: -x[1])[:5]:
        kept = "✅" if cond in ALLOWED_CONDITIONS else "❌"
        print(f"     {kept} {cond}: {count}")
print()

# Step 3: Check format (for DB records)
# This happens in filter_page for existing records
print(f"🎵 Step 3: Checking formats")
for item in condition_filtered[:5]:
    print(f"   {item.get('artist')} - {item.get('title')}")
print()

# Step 4: Score and filter
print("🔢 Step 4: Running score_and_filter_seller_listings...")
try:
    scored = score_and_filter_seller_listings(condition_filtered)
    print(f"   Before: {len(condition_filtered)} → After: {len(scored)}")

    if len(scored) == 0:
        print("\n   ❌ ALL RECORDS FILTERED OUT!")
        print("   Checking why...")

        # Check wants/haves
        print("\n   Wants vs Haves analysis:")
        wanted_count = 0
        for item in condition_filtered[:10]:
            wants = item.get('wants', 0)
            haves = item.get('haves', 0)
            is_wanted = wants > haves
            wanted_count += is_wanted
            status = "✅" if is_wanted else "❌"
            print(f"     {status} {item.get('artist', 'Unknown')[:30]:30} wants={wants:4} haves={haves:4}")

        print(f"\n   Total wanted (wants > haves): {wanted_count}/{len(condition_filtered)}")

    else:
        print(f"\n   ✅ Got {len(scored)} scored records")

        # Check if expected record is in scored results
        expected_in_scored = any(EXPECTED_PRICE in str(item.get('record_price', '')) for item in scored)
        if expected_in_scored:
            print(f"   ✅ Expected record IS in scored results")
            for item in scored:
                if EXPECTED_PRICE in str(item.get('record_price', '')):
                    print(f"      Score: {item['score']:.3f}, Price: ${item['price']:.2f}")
        else:
            print(f"   ❌ Expected record NOT in scored results (filtered during scoring)")

        print("\n   Top 5 by score:")
        for item in sorted(scored, key=lambda x: x['score'], reverse=True)[:5]:
            marker = "👀" if EXPECTED_PRICE in str(item.get('record_price', '')) else ""
            print(f"     {item['score']:.3f} - {item['artist'][:30]} - {item['title'][:30]} - ${item['price']:.2f} {marker}")

        # Run knapsack
        from bandit.knapsack import knapsack
        print(f"\n🎒 Step 5: Running knapsack optimization (budget=${budget})...")
        selected = knapsack(scored, budget)
        print(f"   Selected {len(selected)} items")

        # Check if expected is selected
        expected_selected = any(EXPECTED_PRICE in str(item.get('record_price', '')) for item in selected)
        if expected_selected:
            print(f"   ✅ Expected record WAS SELECTED by knapsack!")
        else:
            print(f"   ❌ Expected record was NOT selected by knapsack")
            print(f"      (It's in the pool but wasn't optimal)")

        total_cost = sum(item['price'] for item in selected)
        total_score = sum(item['score'] for item in selected)
        print(f"   Total cost: ${total_cost:.2f} / ${budget}")
        print(f"   Total score: {total_score:.2f}")

        print("\n   Selected items:")
        for item in sorted(selected, key=lambda x: x['score'], reverse=True):
            marker = "👀" if EXPECTED_PRICE in str(item.get('record_price', '')) else ""
            print(f"     {item['score']:.3f} - ${item['price']:6.2f} - {item['artist'][:25]} {marker}")

except Exception as e:
    print(f"\n   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print("🔍 DEBUG COMPLETE")
print(f"{'='*60}\n")
