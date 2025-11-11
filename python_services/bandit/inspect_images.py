import random
from .discogs_client import authenticate_client
from .models import Record, DiscogsListing


# Get 5 random recent listings
recent_listings = DiscogsListing.objects.order_by('-id')[:500]
sample_listings = random.sample(list(recent_listings), min(5, len(recent_listings)))

print("\n" + "="*80)
print("Checking images for 5 random records")
print("="*80)

d = authenticate_client()

for listing in sample_listings:
    record = listing.record
    print(f"\n{'='*80}")
    print(f"Record: {record.artist} - {record.title}")
    print(f"Discogs ID: {record.discogs_id}")
    print(f"Current record_image in DB: {record.record_image}")
    
    if record.discogs_id:
        try:
            release = d.release(record.discogs_id)
            
            print(f"\nType of release.images: {type(release.images)}")
            print(f"release.images value: {release.images}")
            print(f"Is None? {release.images is None}")
            print(f"Is truthy? {bool(release.images)}")
            
            if release.images is not None:
                print(f"Length: {len(release.images)}")
                if len(release.images) > 0:
                    print(f"\nFirst image object: {release.images[0]}")
                    print(f"Type: {type(release.images[0])}")
                    if isinstance(release.images[0], dict):
                        print(f"Keys: {release.images[0].keys()}")
                        print(f"'uri' key exists? {'uri' in release.images[0]}")
                        print(f"'resource_url' key exists? {'resource_url' in release.images[0]}")
                else:
                    print("Empty list!")
                    
        except Exception as e:
            print(f"Error fetching release: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No discogs_id available")

print("\n" + "="*80)
print("Done!")
print("="*80 + "\n")