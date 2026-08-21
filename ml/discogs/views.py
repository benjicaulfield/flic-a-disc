import json
import random
import torch
import threading
import time
from pathlib import Path


from django.db import models as django_models
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from bandit.models import BanditModel, DiscogsRecord as RecordModel
from bandit.training import BanditTrainer
from bandit.bandit_selection import adaptive_batch_selection
from bandit.discogs_client import authenticate_client
from .models import CatalogBatch

# Global lock to prevent multiple background batch creation threads
_replenishment_lock = threading.Lock()
_replenishment_active = False

def create_single_batch():
    """Create one batch of 20 enriched records"""
    print("🔨 Creating single batch...")

    # Get 1000 records
    records, remaining = get_thousand_records()
    if records is None:
        print("❌ No records available")
        return None

    # Load model and get predictions
    trainer = BanditTrainer()
    trainer.load_latest_model()

    features = trainer.feature_extractor.extract_batch_features(records)
    features_tensor = torch.FloatTensor(features)

    mean_probs, variances = trainer.model.predict_with_uncertainty(features_tensor)
    mean_probs = mean_probs.cpu().numpy()
    variances = variances.cpu().numpy()

    # Select 20 records
    selected_indices = adaptive_batch_selection(
        candidates=list(range(len(records))),
        predictions=mean_probs,
        uncertainties=variances,
        batch_num=0,
        total_batch_size=20,
        total_batches=100,
        random_count=3
    )

    selected_20 = [records[i] for i in selected_indices]

    # Get predictions/uncertainties for the selected records
    selected_predictions = [mean_probs[i] for i in selected_indices]
    selected_mean_predictions = [mean_probs[i] for i in selected_indices]
    selected_uncertainties = [variances[i] for i in selected_indices]

    # Enrich with API calls
    d = authenticate_client()
    enriched_20 = []

    for i, record in enumerate(selected_20, 1):
        # Rename fields
        record['genres'] = record.pop('genre', [])
        record['styles'] = record.pop('style', [])
        record['discogs_id'] = record['release_id']
        record['id'] = i

        try:
            r = d.release(record['release_id'])

            # Check if release data was returned
            if r is None:
                raise ValueError("Release not found or API returned None")

            record['wants'] = r.community.want if hasattr(r, 'community') else 0
            record['haves'] = r.community.have if hasattr(r, 'community') else 0

            suggested_price = ''
            price_suggestions = getattr(r, 'price_suggestions', None)
            if price_suggestions and hasattr(price_suggestions, 'very_good_plus'):
                price = price_suggestions.very_good_plus
                suggested_price = str(price.value) if hasattr(price, 'value') else str(price)
            elif hasattr(r, 'lowest_price'):
                price = r.lowest_price
                suggested_price = str(price.value) if hasattr(price, 'value') else str(price)

            record['suggested_price'] = suggested_price

            # Rate limiting: ~1 request per second to stay under 60/min
            time.sleep(1)

        except Exception as e:
            print(f"  Error enriching {record.get('release_id')}: {e}")
            record['wants'] = 0
            record['haves'] = 0
            record['suggested_price'] = ''

        enriched_20.append(record)

    print(f"✅ Batch created with {len(enriched_20)} records")
    return {
        'records': enriched_20,
        'predictions': [float(p) for p in selected_predictions],
        'mean_predictions': [float(p) for p in selected_mean_predictions],
        'uncertainties': [float(u) for u in selected_uncertainties]
    }


def create_batches_background(count):
    """Background thread to create N batches"""
    global _replenishment_active

    print(f"\n{'='*60}")
    print(f"🔧 BACKGROUND BATCH CREATION: {count} batches")
    print(f"{'='*60}\n")

    try:
        for i in range(count):
            print(f"[{i+1}/{count}] Creating batch...")

            batch_data = create_single_batch()
            if batch_data is None:
                print("❌ Failed to create batch, stopping")
                break

            # Save to database
            CatalogBatch.objects.create(
                records=batch_data['records'],
                predictions=batch_data['predictions'],
                mean_predictions=batch_data['mean_predictions'],
                uncertainties=batch_data['uncertainties']
            )
            print(f"✅ Batch {i+1}/{count} saved to database")

            # Small delay between batches
            if i < count - 1:
                time.sleep(2)

        print(f"\n{'='*60}")
        print(f"✅ BACKGROUND CREATION COMPLETE: {count} batches")
        print(f"{'='*60}\n")
    finally:
        # Clear the flag when done (even if there was an error)
        with _replenishment_lock:
            _replenishment_active = False
            print("🔓 Replenishment lock released")


def get_thousand_records():
    data_dir = Path(__file__).parent / 'data'
    high_path = data_dir / 'bandit_predictions_high.json'
    medium_path = data_dir / 'bandit_predictions_medium.json'

    # Load both files with error handling
    try:
        with open(high_path) as f:
            high = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error loading high predictions: {e}")
        print("⚠️  File corrupted. Re-run predict_with_bandit.py to regenerate.")
        high = []

    try:
        with open(medium_path) as f:
            medium = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error loading medium predictions: {e}")
        print("⚠️  File corrupted. Re-run predict_with_bandit.py to regenerate.")
        medium = []

    # Combine pools
    total_pool = high + medium

    if len(total_pool) < 1000:
        return None, total_pool, None

    # Sample 1000 and convert year to int
    selected = random.sample(total_pool, 1000)

    # Convert year from string to int for feature extraction
    for record in selected:
        if 'year' in record and record['year']:
            try:
                if isinstance(record['year'], str):
                    record['year'] = int(record['year'])
            except (ValueError, TypeError):
                record['year'] = None

    selected_ids = {r['release_id'] for r in selected}

    # Remove selected from both pools
    high = [r for r in high if r['release_id'] not in selected_ids]
    medium = [r for r in medium if r['release_id'] not in selected_ids]

    # Save remaining back to files using atomic writes
    import tempfile
    import shutil

    # Atomic write for high (write to temp file, then move)
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=data_dir, suffix='.tmp') as tmp:
            json.dump(high, tmp)
            tmp_path = tmp.name
        shutil.move(tmp_path, high_path)
    except Exception as e:
        print(f"⚠️  Error saving high predictions: {e}")

    # Atomic write for medium
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=data_dir, suffix='.tmp') as tmp:
            json.dump(medium, tmp)
            tmp_path = tmp.name
        shutil.move(tmp_path, medium_path)
    except Exception as e:
        print(f"⚠️  Error saving medium predictions: {e}")

    remaining = high + medium
    return selected, remaining

def enrich_missing(records):
    """For any record missing wants/haves/suggested_price, check DB first then hit the API."""
    needs_api = [r for r in records if not r.get('wants') and not r.get('haves')]
    if not needs_api:
        return records

    # DB pass — fill from already-enriched records
    ids = [str(r.get('discogs_id') or r.get('release_id', '')) for r in needs_api]
    db_map = {
        str(rec.discogs_id): rec
        for rec in RecordModel.objects.filter(discogs_id__in=ids)
        if rec.wants or rec.haves
    }

    still_needs_api = []
    for r in needs_api:
        rid = str(r.get('discogs_id') or r.get('release_id', ''))
        if rid in db_map:
            db_rec = db_map[rid]
            r['wants'] = db_rec.wants
            r['haves'] = db_rec.haves
            r['suggested_price'] = db_rec.suggested_price or ''
        else:
            still_needs_api.append(r)

    if not still_needs_api:
        return records

    # API pass — only for records not in DB
    d = authenticate_client()
    for r in still_needs_api:
        release_id = r.get('discogs_id') or r.get('release_id')
        try:
            release = d.release(release_id)
            r['wants'] = release.community.want if hasattr(release, 'community') else 0
            r['haves'] = release.community.have if hasattr(release, 'community') else 0
            suggested_price = ''
            price_suggestions = getattr(release, 'price_suggestions', None)
            if price_suggestions and hasattr(price_suggestions, 'very_good_plus'):
                price = price_suggestions.very_good_plus
                if price and price.data:
                    suggested_price = str(price.value)
            r['suggested_price'] = suggested_price
            # Save to DB so we don't call the API again
            RecordModel.objects.filter(discogs_id=str(release_id)).update(
                wants=r['wants'],
                haves=r['haves'],
                suggested_price=suggested_price or RecordModel.objects.filter(discogs_id=str(release_id)).values_list('suggested_price', flat=True).first()
            )
            time.sleep(1)
        except Exception as e:
            print(f"  enrich_missing: error on {release_id}: {e}")
            r.setdefault('wants', 0)
            r.setdefault('haves', 0)
            r.setdefault('suggested_price', '')

    return records


@api_view(['POST'])
def select_batch(request):
    print("\n" + "="*60)
    print("🎬 CATALOG BATCH SELECTION (FROM POOL)")
    print("="*60)

    # Get random unused batch from pool
    batch = CatalogBatch.objects.filter(used=False).order_by('?').first()

    if not batch:
        print("❌ No pre-computed batches available! Creating one on-demand...")
        # Fallback: create batch on-demand (slow)
        batch_data = create_single_batch()
        if batch_data is None:
            return Response({
                'error': 'No batches available and failed to create new one'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        return Response({
            'records': batch_data['records'],
            'predictions': batch_data['predictions'],
            'mean_predictions': batch_data['mean_predictions'],
            'uncertainties': batch_data['uncertainties'],
            'pool_remaining': 0,
            'source': 'on-demand'
        })

    # Mark batch as used
    batch.used = True
    batch.save()
    print(f"✅ Serving batch {batch.id} from pool")

    # Check pool size and trigger replenishment if needed
    global _replenishment_active
    unused_count = CatalogBatch.objects.filter(used=False).count()
    print(f"📊 Pool status: {unused_count} unused batches remaining")

    if unused_count < 5:
        # Check if replenishment is already running
        with _replenishment_lock:
            if _replenishment_active:
                print("⏳ Pool low but replenishment already in progress, skipping")
            else:
                print("⚠️  Pool low! Triggering background replenishment of 5 batches...")
                _replenishment_active = True
                # Start background thread to create more batches
                thread = threading.Thread(target=create_batches_background, args=(5,))
                thread.daemon = True  # Thread dies when main process dies
                thread.start()
                print("✅ Background replenishment started")

    print(f"✅ Serving {len(batch.records)} records from batch {batch.id}")
    print("="*60)
    print("🎬 BATCH SERVED FROM POOL (INSTANT!)")
    print("="*60 + "\n")

    return Response({
        'records': batch.records,
        'predictions': batch.predictions,
        'mean_predictions': batch.mean_predictions,
        'uncertainties': batch.uncertainties,
        'pool_remaining': unused_count - 1,
        'source': 'pool'
    })


OOF_FILE = Path(__file__).parent.parent / 'bandit' / 'data' / 'oof_predictions.jsonl'

@api_view(['GET'])
def oof_batch(request):
    """GET /ml/discogs/oof/?offset=0
    Returns 20 records from oof_predictions.jsonl starting at offset,
    enriched with metadata from DB (API call if missing).
    """
    offset = int(request.query_params.get('offset', 0))
    page_size = 20

    with open(OOF_FILE) as f:
        all_lines = f.readlines()

    total = len(all_lines)
    page_lines = all_lines[offset:offset + page_size]

    if not page_lines:
        return Response({'records': [], 'total': total, 'offset': offset})

    release_ids = [json.loads(l)['release_id'] for l in page_lines]

    db_map = {
        str(r.discogs_id): r
        for r in RecordModel.objects.filter(discogs_id__in=release_ids)
    }

    needs_api = [rid for rid in release_ids if rid not in db_map]

    if needs_api:
        d = authenticate_client()
        for rid in needs_api:
            try:
                r = d.release(int(rid))
                rd = r.data or {}
                wants = r.community.want if hasattr(r, 'community') else 0
                haves = r.community.have if hasattr(r, 'community') else 0
                price_suggestions = getattr(r, 'price_suggestions', None)
                suggested_price = ''
                if price_suggestions and hasattr(price_suggestions, 'very_good_plus'):
                    price = price_suggestions.very_good_plus
                    if price and price.data:
                        suggested_price = str(price.value)
                rec, _ = RecordModel.objects.get_or_create(
                    discogs_id=rid,
                    defaults={
                        'artist': ', '.join(a.get('name', '') for a in rd.get('artists', [])),
                        'title': rd.get('title', ''),
                        'label': rd.get('labels', [{}])[0].get('name', '') if rd.get('labels') else '',
                        'wants': wants,
                        'haves': haves,
                        'genres': rd.get('genres', []),
                        'styles': rd.get('styles', []),
                        'year': rd.get('year'),
                        'suggested_price': suggested_price,
                    }
                )
                db_map[rid] = rec
            except Exception as e:
                print(f"oof_batch: error on {rid}: {e}")

    records = []
    for rid in release_ids:
        rec = db_map.get(rid)
        if not rec:
            continue
        records.append({
            'id': rec.id,
            'discogs_id': rec.discogs_id,
            'artist': rec.artist,
            'title': rec.title,
            'label': rec.label,
            'year': rec.year,
            'wants': rec.wants,
            'haves': rec.haves,
            'genres': list(rec.genres) if rec.genres else [],
            'styles': list(rec.styles) if rec.styles else [],
            'suggested_price': rec.suggested_price or '',
            'evaluated': rec.evaluated,
            'wanted': rec.wanted,
        })

    return Response({'records': records, 'total': total, 'offset': offset})


@api_view(['POST'])
def enrich_records(request):
    """POST /ml/discogs/enrich/
    Body: {"discogs_ids": ["123", "456", ...]}
    Returns wants/haves/suggested_price for each, making API calls only when missing.
    """
    discogs_ids = [str(i) for i in request.data.get('discogs_ids', [])]
    if not discogs_ids:
        return Response({'enriched': []})

    db_map = {
        str(r.discogs_id): r
        for r in RecordModel.objects.filter(discogs_id__in=discogs_ids)
    }

    enriched = []
    needs_api = []

    for did in discogs_ids:
        rec = db_map.get(did)
        if rec and (rec.wants or rec.haves) and rec.suggested_price:
            enriched.append({
                'discogs_id': did,
                'wants': rec.wants,
                'haves': rec.haves,
                'suggested_price': rec.suggested_price,
            })
        else:
            needs_api.append(did)

    if needs_api:
        d = authenticate_client()
        for did in needs_api:
            try:
                r = d.release(int(did))
                wants = r.community.want if hasattr(r, 'community') else 0
                haves = r.community.have if hasattr(r, 'community') else 0
                suggested_price = ''
                price_suggestions = getattr(r, 'price_suggestions', None)
                if price_suggestions and hasattr(price_suggestions, 'very_good_plus'):
                    price = price_suggestions.very_good_plus
                    if price and price.data:
                        suggested_price = str(price.value)
                RecordModel.objects.filter(discogs_id=did).update(
                    wants=wants,
                    haves=haves,
                    suggested_price=suggested_price or django_models.F('suggested_price'),
                )
                enriched.append({
                    'discogs_id': did,
                    'wants': wants,
                    'haves': haves,
                    'suggested_price': suggested_price,
                })
                time.sleep(1)
            except Exception as e:
                print(f"enrich_records: error on {did}: {e}")
                enriched.append({'discogs_id': did, 'wants': 0, 'haves': 0, 'suggested_price': ''})

    return Response({'enriched': enriched})









