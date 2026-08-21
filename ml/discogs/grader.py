import json
import os
import sys
import django
import time
import random

# Django setup
sys.path.insert(0, os.path.abspath('..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from bandit.models import Record
from bandit.discogs_client import authenticate_client
from django.utils import timezone

# Initialize Discogs client
d = authenticate_client()

# Load both JSON files
with open('data/query.json', 'r') as f:
    query_ids = json.load(f)

with open('skipped.json', 'r') as f:
    skipped_ids = json.load(f)

# Random sample 500 from each
print(f"Loaded {len(query_ids)} query IDs and {len(skipped_ids)} skipped IDs")
query_sample = random.sample(query_ids, min(500, len(query_ids)))
skipped_sample = random.sample(skipped_ids, min(500, len(skipped_ids)))
print(f"Sampled {len(query_sample)} query IDs and {len(skipped_sample)} skipped IDs")

# Create list with labels: (id, predicted_positive)
all_ids = [(rid, True) for rid in query_sample] + [(rid, False) for rid in skipped_sample]
print(f"\nProcessing {len(query_sample)} query IDs (predicted positive) + {len(skipped_sample)} skipped IDs (predicted negative) = {len(all_ids)} total\n")

# Track results
correct = 0
incorrect = 0
errors = 0
already_in_db = 0

for idx, (release_id, predicted_positive) in enumerate(all_ids, 1):
    try:
        # Check if already in database
        existing = Record.objects.filter(discogs_id=release_id).first()
        if existing:
            already_in_db += 1
            actual_positive = existing.wants > existing.haves
            is_correct = (actual_positive == predicted_positive)

            if is_correct:
                correct += 1
                status = "✓"
            else:
                incorrect += 1
                status = "✗"

            # Log progress
            total_graded = correct + incorrect
            accuracy = (correct / total_graded * 100) if total_graded > 0 else 0
            pred_label = "POS" if predicted_positive else "NEG"
            actual_label = "pos" if actual_positive else "neg"
            print(f"[{idx}/{len(all_ids)}] {status} DB {release_id} | Pred:{pred_label} Act:{actual_label} | Accuracy: {accuracy:.1f}% ({correct}✓/{total_graded})")
            continue

        # Fetch from API
        release = d.release(release_id)
        wants = release.community.want if hasattr(release, 'community') else 0
        haves = release.community.have if hasattr(release, 'community') else 0

        actual_positive = wants > haves
        is_correct = (actual_positive == predicted_positive)

        # Save to database (both positives and negatives)
        # Extract data from release
        artist_name = 'Unknown'
        if hasattr(release, 'artists') and release.artists:
            artist_name = release.artists[0].name if hasattr(release.artists[0], 'name') else 'Unknown'

        label_name = ''
        catno = ''
        if hasattr(release, 'labels') and release.labels:
            label_name = release.labels[0].name if hasattr(release.labels[0], 'name') else ''
            catno = release.labels[0].catno if hasattr(release.labels[0], 'catno') else ''

        genres = getattr(release, 'genres', [])
        styles = getattr(release, 'styles', [])
        year = getattr(release, 'year', None)

        # Update or create record
        record, created = Record.objects.update_or_create(
            discogs_id=release_id,
            defaults={
                'artist': artist_name,
                'title': getattr(release, 'title', 'Unknown'),
                'format': ['Vinyl', 'LP'],
                'label': label_name,
                'catno': catno,
                'wants': wants,
                'haves': haves,
                'genres': genres if isinstance(genres, list) else [],
                'styles': styles if isinstance(styles, list) else [],
                'year': int(year) if year else None,
                'wanted': False,
                'evaluated': True,
            }
        )

        if is_correct:
            correct += 1
            status = "✓"
        else:
            incorrect += 1
            status = "✗"

        # Log every record with running accuracy
        total_graded = correct + incorrect
        accuracy = (correct / total_graded * 100) if total_graded > 0 else 0
        pred_label = "POS" if predicted_positive else "NEG"
        actual_label = "pos" if actual_positive else "neg"
        print(f"[{idx}/{len(all_ids)}] {status} {release_id}: W={wants:,} H={haves:,} | Pred:{pred_label} Act:{actual_label} | Accuracy: {accuracy:.1f}% ({correct}✓/{total_graded})")

        # Rate limiting: 60 calls/min = 1 per second
        time.sleep(1)

    except Exception as e:
        errors += 1
        total_graded = correct + incorrect
        accuracy = (correct / total_graded * 100) if total_graded > 0 else 0
        pred_label = "POS" if predicted_positive else "NEG"
        print(f"[{idx}/{len(all_ids)}] ⚠️  {release_id}: {e} | Pred:{pred_label} | Accuracy: {accuracy:.1f}% ({correct}✓/{total_graded})")
        continue

# Summary
print("\n" + "="*60)
print("GRADING COMPLETE")
print("="*60)
print(f"Total processed: {len(all_ids)}")
print(f"Already in database: {already_in_db}")
print(f"Predictions from query.json (predicted positive): {len(query_ids)}")
print(f"Predictions from skipped.json (predicted negative): {len(skipped_ids)}")
print(f"\nResults:")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Errors: {errors}")

total_graded = correct + incorrect
if total_graded > 0:
    accuracy = correct / total_graded
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total_graded})")

# Calculate precision and recall for positives
query_correct = 0
query_total = 0
print(f"\nDetailed breakdown available in database (api_enriched=True records)")
