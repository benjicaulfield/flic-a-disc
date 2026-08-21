"""
Streaming Data Filtering Pipeline for Vinyl LP Classification

Core approach:
1. Feature hashing LightGBM model trained on 30k enriched records
   - Uses FeatureHasher for artist/label (avoids target encoding leakage)
   - Genre/style multi-hot, year, country, frequency features
2. API-calibrated thresholds (~200 calls for rank-stratified calibration)
3. Strategic API querying (~800 calls on uncertain positives)
4. Training-data propagation via master_id
5. Adaptive coverage management
"""

import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import lightgbm as lgb
from sklearn.feature_extraction import FeatureHasher
import warnings
import logging
from datetime import datetime
import os
import sys

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_t10.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add ml directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../ml'))

# Setup Django before importing models
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from bandit.utils.get_user_inventory import authenticate_client
from bandit.models import Record
from django.utils import timezone

logger.info("Initializing authenticated Discogs client...")
d = authenticate_client()
logger.info("Client authenticated successfully")

def clean_list_field(lst):
    """Clean genre/style lists that may contain double-wrapped JSON strings."""
    cleaned = []
    if not isinstance(lst, list):
        return []
    for item in lst:
        if isinstance(item, str) and item.startswith('['):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, list):
                    cleaned.extend(parsed)
                else:
                    cleaned.append(item)
            except:
                cleaned.append(item)
        else:
            cleaned.append(item)
    return cleaned


def build_model_and_score(training_data, catalog_records):
    """
    Build feature hashing LightGBM model and score all catalog records.

    Returns: (cat_probs, cat_release_ids, cat_master_ids)
    """
    logger.info("Building training dataframe...")
    train_df = pd.DataFrame(training_data)
    train_df['wants'] = train_df['wants'].astype(int)
    train_df['haves'] = train_df['haves'].astype(int)
    train_df['positive'] = (train_df['wants'] > train_df['haves']).astype(int)
    train_df['_source'] = 'train'
    logger.info(f"Training dataframe: {len(train_df)} rows")

    logger.info("Building catalog dataframe...")
    cat_df = pd.DataFrame(catalog_records)
    cat_df['_source'] = 'catalog'
    logger.info(f"Catalog dataframe: {len(cat_df)} rows")

    logger.info("Concatenating dataframes...")
    all_df = pd.concat([train_df, cat_df], ignore_index=True)
    logger.info(f"Combined dataframe: {len(all_df)} rows")
    
    # --- Feature engineering ---
    logger.info("Starting feature engineering...")
    logger.info("  Cleaning genre/style lists...")
    all_df['genre_clean'] = all_df['genre'].apply(clean_list_field)
    all_df['style_clean'] = all_df['style'].apply(clean_list_field)

    logger.info("  Creating year/decade features...")
    all_df['year_int'] = pd.to_numeric(all_df['year'], errors='coerce').fillna(0).astype(int)
    all_df['has_year'] = (all_df['year_int'] > 0).astype(int)
    all_df['decade'] = (all_df['year_int'] // 10 * 10).clip(lower=0)

    logger.info("  Creating country features...")
    all_df['country_clean'] = all_df['country'].fillna('').astype(str)
    all_df['has_country'] = (all_df['country_clean'] != '').astype(int)

    # Genre multi-hot
    logger.info("  Building genre multi-hot encoding...")
    all_genres = sorted(set(g for gl in all_df['genre_clean'] for g in gl))
    logger.info(f"    Found {len(all_genres)} unique genres")
    genre_cols = []
    for g in all_genres:
        col = f'genre_{g}'
        all_df[col] = all_df['genre_clean'].apply(lambda x, g=g: int(g in x))
        genre_cols.append(col)
    logger.info(f"    Created {len(genre_cols)} genre columns")

    # Style multi-hot (top 150 by frequency)
    logger.info("  Building style multi-hot encoding...")
    style_counts = Counter(s for sl in all_df['style_clean'] for s in sl)
    top_styles = [s for s, c in style_counts.most_common(150)]
    logger.info(f"    Using top {len(top_styles)} styles")
    style_cols = []
    for s in top_styles:
        col = f'style_{s}'.replace(' ', '_').replace('/', '_').replace('&', 'and')
        col = col.replace("'", "").replace(',', '').replace('(', '').replace(')', '')
        all_df[col] = all_df['style_clean'].apply(lambda x, s=s: int(s in x))
        style_cols.append(col)
    logger.info(f"    Created {len(style_cols)} style columns")
    
    logger.info("  Creating count features...")
    all_df['n_genres'] = all_df['genre_clean'].apply(len)
    all_df['n_styles'] = all_df['style_clean'].apply(len)
    all_df['has_master'] = (all_df['master_id'].astype(str) != '0').astype(int)

    # Frequency encodings (label-free, no leakage)
    logger.info("  Computing frequency encodings...")
    all_df['country_freq'] = all_df['country_clean'].map(
        all_df['country_clean'].value_counts().to_dict())
    all_df['label_freq'] = all_df['label'].fillna('').map(
        all_df['label'].fillna('').value_counts().to_dict())
    all_df['artist_freq'] = all_df['artist'].fillna('').map(
        all_df['artist'].fillna('').value_counts().to_dict())

    # Feature hashing for artist and label (key innovation: no target encoding leakage)
    logger.info("  Starting feature hashing...")
    n_artist_hash = 512
    n_label_hash = 256

    logger.info(f"    Hashing artists ({n_artist_hash} features)...")
    artist_hasher = FeatureHasher(n_features=n_artist_hash, input_type='string')
    ah = artist_hasher.transform(
        all_df['artist'].fillna('').apply(lambda x: [x])).toarray()
    logger.info(f"    Artist hash matrix shape: {ah.shape}")

    logger.info(f"    Hashing labels ({n_label_hash} features)...")
    label_hasher = FeatureHasher(n_features=n_label_hash, input_type='string')
    lh = label_hasher.transform(
        all_df['label'].fillna('').apply(lambda x: [x])).toarray()
    logger.info(f"    Label hash matrix shape: {lh.shape}")

    logger.info("  Adding hash features to dataframe...")
    a_cols = [f'ah_{i}' for i in range(n_artist_hash)]
    l_cols = [f'lh_{i}' for i in range(n_label_hash)]
    for i, col in enumerate(a_cols):
        all_df[col] = ah[:, i]
    for i, col in enumerate(l_cols):
        all_df[col] = lh[:, i]
    logger.info("  Hash features added to dataframe")

    # Country label encoding
    logger.info("  Creating country encoding...")
    cmap = {c: i for i, c in enumerate(all_df['country_clean'].unique())}
    all_df['country_enc'] = all_df['country_clean'].map(cmap)
    
    logger.info("Feature engineering complete!")

    feature_cols = (
        ['year_int', 'has_year', 'decade', 'has_country', 'country_enc',
         'n_genres', 'n_styles', 'has_master',
         'country_freq', 'label_freq', 'artist_freq'] +
        genre_cols + style_cols + a_cols + l_cols
    )
    logger.info(f"Total features: {len(feature_cols)}")

    logger.info("Splitting train/catalog data...")
    train_mask = all_df['_source'] == 'train'
    cat_mask = all_df['_source'] == 'catalog'

    logger.info("Extracting feature matrices...")
    train_features = all_df[train_mask][feature_cols].values.astype(np.float32)
    train_labels = all_df[train_mask]['positive'].values
    logger.info(f"Training features shape: {train_features.shape}")

    cat_features = all_df[cat_mask][feature_cols].values.astype(np.float32)
    cat_release_ids = all_df[cat_mask]['release_id'].values.astype(str)
    cat_master_ids = all_df[cat_mask]['master_id'].astype(str).values
    logger.info(f"Catalog features shape: {cat_features.shape}")

    # Train LightGBM
    logger.info("Training LightGBM model (800 trees)...")
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        is_unbalance=True,
    )
    model.fit(train_features, train_labels)
    logger.info("Model training complete!")

    logger.info("Scoring catalog records...")
    cat_probs = model.predict_proba(cat_features)[:, 1]
    logger.info(f"Scored {len(cat_probs)} records")

    return cat_probs, cat_release_ids, cat_master_ids


def classify_catalog(catalog):
    """
    Classify records from catalog using up to 1,000 API calls.

    Uses global authenticated Discogs client `d` for API calls.
    All API calls are logged to file and saved to Record table (get_or_create,
    updating wants/haves/api_enriched for existing records).

    Args:
        catalog: List of records from lp_catalog.json (without wants/haves)

    Returns:
        {
            'ruled_in': [release_ids],      # Confident positives (wants > haves)
            'ruled_out': [release_ids],      # Confident negatives (wants <= haves)
            'verified': [release_ids],       # IDs that were queried via API
            'metadata': {
                'api_calls_made': int,
                'coverage_ratio': float,
                'approach': str
            }
        }
    """
    # ============================================================
    # PHASE 0: Load training data and build model
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 0: Loading training data and building model")
    logger.info("=" * 60)

    with open('enriched_training.json') as f:
        training_data = json.load(f)
    logger.info(f"Loaded {len(training_data)} training records")
    
    total_catalog = len(catalog)
    catalog_records = catalog  # Keep reference for database saves

    # Build model and score catalog
    cat_probs, cat_release_ids, cat_master_ids = \
        build_model_and_score(training_data, catalog)
    
    # Build training propagation lookup (master_id)
    master_labels = defaultdict(list)
    for r in training_data:
        mid = str(r.get('master_id', '0'))
        if mid != '0' and mid.strip():
            pos = 1 if int(r['wants']) > int(r['haves']) else 0
            master_labels[mid].append(pos)
    
    # Compute percentile ranks (0 = highest prob, 1 = lowest)
    rank_order = np.argsort(cat_probs)[::-1]
    ranks = np.empty(len(cat_probs), dtype=int)
    ranks[rank_order] = np.arange(len(rank_order))
    pct_ranks = ranks / total_catalog
    
    # ============================================================
    # PHASE 1: API calibration (~200 calls)
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: API calibration (target: ~200 calls)")
    logger.info("=" * 60)

    api_calls = 0
    verified_ids = []
    api_results = {}  # catalog_index -> {'positive', 'prob', 'pct_rank'}
    max_api_calls = 1000
    n_calibration = 200
    
    np.random.seed(42)
    
    # Stratified sampling across rank percentiles
    cal_bands = [
        (0.00, 0.05, 20), (0.05, 0.10, 20), (0.10, 0.15, 20),
        (0.15, 0.20, 20), (0.20, 0.25, 20), (0.25, 0.35, 20),
        (0.35, 0.50, 20), (0.50, 0.70, 20), (0.70, 0.85, 20),
        (0.85, 1.01, 20),
    ]
    
    cal_indices = []
    for lo, hi, n_want in cal_bands:
        in_band = np.where((pct_ranks >= lo) & (pct_ranks < hi))[0]
        if len(in_band) > 0:
            n_s = min(n_want, len(in_band))
            sampled = np.random.choice(in_band, size=n_s, replace=False)
            cal_indices.extend(sampled.tolist())
    
    for idx in cal_indices:
        if api_calls >= n_calibration:
            break
        rid = int(cat_release_ids[idx])
        try:
            logger.info(f"API call {api_calls+1} (calibration): Querying release {rid}")
            result = d.release(rid)
            api_calls += 1
            verified_ids.append(str(rid))
            is_positive = int(result['wants'] > result['haves'])
            logger.info(f"API call {api_calls} (calibration): Release {rid} - wants={result['wants']}, haves={result['haves']}, positive={is_positive}")

            # Save to database: get_or_create Record and update wants/haves
            record, created = Record.objects.get_or_create(
                discogs_id=str(rid),
                defaults={
                    'artist': catalog_records[idx].get('artist', ''),
                    'title': catalog_records[idx].get('title', ''),
                    'label': catalog_records[idx].get('label', ''),
                    'catno': catalog_records[idx].get('catno', ''),
                    'wants': result['wants'],
                    'haves': result['haves'],
                    'added': timezone.now(),
                    'genres': catalog_records[idx].get('genre', []),
                    'styles': catalog_records[idx].get('style', []),
                    'year': catalog_records[idx].get('year'),
                    'api_enriched': True,
                }
            )
            if not created:
                record.wants = result['wants']
                record.haves = result['haves']
                record.api_enriched = True
                record.save()

            api_results[idx] = {
                'positive': is_positive,
                'prob': float(cat_probs[idx]),
                'pct_rank': float(pct_ranks[idx]),
            }
        except Exception as e:
            logger.error(f"API call failed at {api_calls+1} (calibration): {e}")
            break
    
    # ============================================================
    # PHASE 2: Find optimal thresholds from calibration data
    # ============================================================
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Finding optimal thresholds from {len(api_results)} calibration samples")
    logger.info("=" * 60)

    cal_pct = np.array([api_results[i]['pct_rank'] for i in api_results])
    cal_lab = np.array([api_results[i]['positive'] for i in api_results])
    
    # Find positive threshold: largest top-X% with >= 90% precision
    best_pos_pct = 0.0
    for pct in np.arange(0.005, 0.40, 0.005):
        top_mask = cal_pct <= pct
        n_above = top_mask.sum()
        if n_above < 5:
            continue
        precision = cal_lab[top_mask].mean()
        if precision >= 0.90:
            best_pos_pct = pct
    
    # Find negative threshold: largest bottom-X% with >= 90% neg precision
    best_neg_pct = 0.0
    for neg_frac in np.arange(0.005, 0.95, 0.005):
        bot_mask = cal_pct >= (1.0 - neg_frac)
        n_below = bot_mask.sum()
        if n_below < 5:
            continue
        neg_precision = (1 - cal_lab[bot_mask]).mean()
        if neg_precision >= 0.90:
            best_neg_pct = neg_frac
    
    # Convert to probability thresholds
    pos_prob_thresh = float('inf')
    neg_prob_thresh = -float('inf')
    
    if best_pos_pct > 0:
        k_pos = max(1, int(total_catalog * best_pos_pct))
        pos_prob_thresh = cat_probs[rank_order[k_pos - 1]]
    
    if best_neg_pct > 0:
        k_neg_start = int(total_catalog * (1.0 - best_neg_pct))
        neg_prob_thresh = cat_probs[rank_order[min(total_catalog - 1, k_neg_start)]]

    logger.info(f"Threshold calibration complete:")
    logger.info(f"  Positive threshold: top {best_pos_pct*100:.2f}% (prob >= {pos_prob_thresh:.4f})")
    logger.info(f"  Negative threshold: bottom {best_neg_pct*100:.2f}% (prob <= {neg_prob_thresh:.4f})")

    # ============================================================
    # PHASE 3: Strategic API queries on uncertain positives (~800 calls)
    # ============================================================
    logger.info("=" * 60)
    logger.info(f"PHASE 3: Strategic API queries (budget: {max_api_calls - api_calls} calls)")
    logger.info("=" * 60)

    remaining_budget = max_api_calls - api_calls
    queried_set = set(api_results.keys())
    
    if remaining_budget > 0 and best_pos_pct > 0:
        # Query records just below the positive threshold (highest-value uncertain records)
        query_lo = best_pos_pct
        query_hi = min(best_pos_pct + 0.25, 0.50)
        
        in_band = np.where((pct_ranks >= query_lo) & (pct_ranks < query_hi))[0]
        # Sort by model probability descending
        in_band_sorted = in_band[np.argsort(cat_probs[in_band])[::-1]]
        
        for idx in in_band_sorted:
            if api_calls >= max_api_calls:
                break
            if idx in queried_set:
                continue

            rid = int(cat_release_ids[idx])
            try:
                logger.info(f"API call {api_calls+1} (strategic_query): Querying release {rid}")
                result = d.release(rid)
                api_calls += 1
                verified_ids.append(str(rid))
                is_positive = int(result['wants'] > result['haves'])
                logger.info(f"API call {api_calls} (strategic_query): Release {rid} - wants={result['wants']}, haves={result['haves']}, positive={is_positive}")

                # Save to database: get_or_create Record and update wants/haves
                record, created = Record.objects.get_or_create(
                    discogs_id=str(rid),
                    defaults={
                        'artist': catalog_records[idx].get('artist', ''),
                        'title': catalog_records[idx].get('title', ''),
                        'label': catalog_records[idx].get('label', ''),
                        'catno': catalog_records[idx].get('catno', ''),
                        'wants': result['wants'],
                        'haves': result['haves'],
                        'added': timezone.now(),
                        'genres': catalog_records[idx].get('genre', []),
                        'styles': catalog_records[idx].get('style', []),
                        'year': catalog_records[idx].get('year'),
                        'api_enriched': True,
                    }
                )
                if not created:
                    record.wants = result['wants']
                    record.haves = result['haves']
                    record.api_enriched = True
                    record.save()

                api_results[idx] = {
                    'positive': is_positive,
                    'prob': float(cat_probs[idx]),
                    'pct_rank': float(pct_ranks[idx]),
                }
                queried_set.add(idx)
            except Exception as e:
                logger.error(f"API call failed at {api_calls+1} (strategic_query): {e}")
                break
    elif remaining_budget > 0:
        # No positive threshold found - query top-ranked records
        for idx in rank_order:
            if api_calls >= max_api_calls:
                break
            if idx in queried_set:
                continue
            rid = int(cat_release_ids[idx])
            try:
                logger.info(f"API call {api_calls+1} (fallback): Querying release {rid}")
                result = d.release(rid)
                api_calls += 1
                verified_ids.append(str(rid))
                is_positive = int(result['wants'] > result['haves'])
                logger.info(f"API call {api_calls} (fallback): Release {rid} - wants={result['wants']}, haves={result['haves']}, positive={is_positive}")

                # Save to database: get_or_create Record and update wants/haves
                record, created = Record.objects.get_or_create(
                    discogs_id=str(rid),
                    defaults={
                        'artist': catalog_records[idx].get('artist', ''),
                        'title': catalog_records[idx].get('title', ''),
                        'label': catalog_records[idx].get('label', ''),
                        'catno': catalog_records[idx].get('catno', ''),
                        'wants': result['wants'],
                        'haves': result['haves'],
                        'added': timezone.now(),
                        'genres': catalog_records[idx].get('genre', []),
                        'styles': catalog_records[idx].get('style', []),
                        'year': catalog_records[idx].get('year'),
                        'api_enriched': True,
                    }
                )
                if not created:
                    record.wants = result['wants']
                    record.haves = result['haves']
                    record.api_enriched = True
                    record.save()

                api_results[idx] = {
                    'positive': is_positive,
                    'prob': float(cat_probs[idx]),
                    'pct_rank': float(pct_ranks[idx]),
                }
                queried_set.add(idx)
            except Exception as e:
                logger.error(f"API call failed at {api_calls+1} (fallback): {e}")
                break
    
    # ============================================================
    # PHASE 4: Final classification
    # ============================================================
    logger.info("=" * 60)
    logger.info(f"PHASE 4: Final classification (used {api_calls}/{max_api_calls} API calls)")
    logger.info("=" * 60)

    ruled_in = set()
    ruled_out = set()
    
    # 1. API-verified records (100% accurate)
    for idx, result in api_results.items():
        rid = str(cat_release_ids[idx])
        if result['positive']:
            ruled_in.add(rid)
        else:
            ruled_out.add(rid)
    
    # 2. Training propagation: master_id with strong signal
    for i in range(total_catalog):
        rid = str(cat_release_ids[i])
        if rid in ruled_in or rid in ruled_out:
            continue
        mid = cat_master_ids[i]
        if mid in master_labels:
            rate = np.mean(master_labels[mid])
            n = len(master_labels[mid])
            if n >= 2 and rate >= 0.95:
                ruled_in.add(rid)
                continue
            elif n >= 2 and rate <= 0.05:
                ruled_out.add(rid)
                continue
    
    # 3. ML-based classification with calibrated thresholds
    for i in range(total_catalog):
        rid = str(cat_release_ids[i])
        if rid in ruled_in or rid in ruled_out:
            continue
        if cat_probs[i] >= pos_prob_thresh:
            ruled_in.add(rid)
        elif cat_probs[i] <= neg_prob_thresh:
            ruled_out.add(rid)
    
    # 4. Coverage management
    coverage = (len(ruled_in) + len(ruled_out)) / total_catalog
    
    if coverage < 0.75:
        # Expand negative threshold to reach 75% coverage
        # Only classify additional records as negative (safer)
        remaining_needed = int(0.76 * total_catalog) - len(ruled_in) - len(ruled_out)
        if remaining_needed > 0:
            unclassified = []
            for i in range(total_catalog):
                rid = str(cat_release_ids[i])
                if rid not in ruled_in and rid not in ruled_out:
                    unclassified.append((i, cat_probs[i]))
            # Sort unclassified by probability ascending (most likely negatives first)
            unclassified.sort(key=lambda x: x[1])
            for idx, _ in unclassified[:remaining_needed]:
                ruled_out.add(str(cat_release_ids[idx]))
    
    coverage = (len(ruled_in) + len(ruled_out)) / total_catalog

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Ruled in (positives): {len(ruled_in)}")
    logger.info(f"Ruled out (negatives): {len(ruled_out)}")
    logger.info(f"Coverage: {coverage*100:.2f}%")
    logger.info(f"Total API calls: {api_calls}/{max_api_calls}")
    logger.info(f"API results saved to database: Record table (api_enriched=True)")
    logger.info("=" * 60)

    return {
        'ruled_in': list(ruled_in),
        'ruled_out': list(ruled_out),
        'verified': verified_ids,
        'metadata': {
            'api_calls_made': api_calls,
            'coverage_ratio': coverage,
            'approach': (
                'Feature hashing LightGBM (512 artist hash + 256 label hash, '
                'no target encoding leakage). '
                '200 API calls for rank-stratified calibration to find precision thresholds. '
                '800 API calls for strategic querying of uncertain positives. '
                'Training-data propagation via master_id. '
                f'Calibrated: pos_pct={best_pos_pct:.3f}, neg_pct={best_neg_pct:.3f}.'
            )
        }
    }


if __name__ == '__main__':
    logger.info("Loading catalog...")
    with open('lp_catalog.json') as f:
        catalog = json.load(f)
    logger.info(f"Loaded {len(catalog)} catalog records")

    logger.info("Running pipeline...")
    result = classify_catalog(catalog)

    logger.info(f"\nResults:")
    logger.info(f"  Ruled in: {len(result['ruled_in'])}")
    logger.info(f"  Ruled out: {len(result['ruled_out'])}")
    logger.info(f"  Coverage: {result['metadata']['coverage_ratio']:.4f}")
    logger.info(f"  API calls: {result['metadata']['api_calls_made']}")
