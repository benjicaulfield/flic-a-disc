"""
Streaming Data Filtering Pipeline for Discogs Record Classification.

Strategy:
1. Train LightGBM with smoothed target encoding on 30k enriched training data
2. Score all catalog records
3. Use ~700 stratified API calls for calibration
4. Find optimal thresholds directly on raw scores using calibration samples
5. Use ~300 API calls for master_id group propagation
6. Coverage guarantee via fallback rules
"""

import json
import numpy as np
from collections import Counter, defaultdict
import lightgbm as lgb

import logging
import os
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_9.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Django setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ml_path = os.path.join(project_root, 'ml')
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from bandit.utils.get_user_inventory import authenticate_client
from bandit.models import Record
from django.utils import timezone

# Initialize authenticated Discogs client
logger.info("Initializing authenticated Discogs client...")
api_client_global = authenticate_client()
logger.info("Client authenticated successfully")

# API call counter and helper
api_call_counter = [0]  # Use list for mutable global

def get_release_logged(release_id):
    """Helper function to get release with logging and DB save."""
    api_call_counter[0] += 1
    logger.info(f"API call {api_call_counter[0]}: Querying release {release_id}")

    result = api_client_global.get_release(release_id)

    # Extract wants/haves from community stats
    stats = (result.data.get('stats') or {}).get('community') or {}
    wants = stats.get('in_wantlist', 0)
    haves = stats.get('in_collection', 0)

    logger.info(f"API call {api_call_counter[0]}: Release {release_id} - wants={wants}, haves={haves}")

    # Create a dict-like result for backward compatibility
    result_dict = {
        'wants': wants,
        'haves': haves,
        'data': result.data,
    }

    # Save to database
    try:
        Record.objects.get_or_create(
            discogs_id=str(release_id),
            defaults={
                'title': result.data.get('title', ''),
                'artist': ', '.join(a.get('name', '') for a in result.data.get('artists', [])),
                'year': result.data.get('year'),
                'genres': result.data.get('genres', []),
                'styles': result.data.get('styles', []),
                'label': result.data.get('labels', [{}])[0].get('name', '') if result.data.get('labels') else '',
                'country': result.data.get('country', ''),
                'format': result.data.get('formats', [{}])[0].get('name', '') if result.data.get('formats') else '',
                'master_id': result.data.get('master_id'),
                'wants': wants,
                'haves': haves,
                'added': timezone.now(),
            }
        )
    except Exception as e:
        logger.warning(f"Failed to save record {release_id} to database: {e}")

    return result_dict



def build_smoothed_te(records, labels, k=10):
    """Build Bayesian-smoothed target encoding."""
    base_rate = 0.2282
    lp, lt = Counter(), Counter()
    ap, at_ = Counter(), Counter()
    cp, ct = Counter(), Counter()
    for r, y in zip(records, labels):
        lt[r['label']] += 1
        at_[r['artist']] += 1
        ct[r['country']] += 1
        if y:
            lp[r['label']] += 1
            ap[r['artist']] += 1
            cp[r['country']] += 1
    return {
        'label_rate': {l: (lp[l]+base_rate*k)/(lt[l]+k) for l in lt if lt[l] >= 3},
        'artist_rate': {a: (ap[a]+base_rate*k)/(at_[a]+k) for a in at_ if at_[a] >= 2},
        'country_rate': {c: (cp[c]+base_rate*k)/(ct[c]+k) for c in ct if ct[c] >= 3},
    }


def extract_features(records, vocabs=None, te_stats=None):
    """Extract feature matrix from records."""
    base_rate = 0.2282
    if vocabs is None:
        gc, sc = Counter(), Counter()
        for r in records:
            gc.update(r['genre'])
            sc.update(r['style'])
        cc = Counter(r['country'] for r in records)
        vocabs = {
            'genre': list(dict(gc.most_common(30)).keys()),
            'style': list(dict(sc.most_common(200)).keys()),
            'country': list(dict(cc.most_common(60)).keys()),
        }
    if te_stats is None:
        te_stats = {'label_rate': {}, 'artist_rate': {}, 'country_rate': {}}
    
    features = []
    for r in records:
        f = []
        for g in vocabs['genre']:
            f.append(1 if g in r['genre'] else 0)
        for s in vocabs['style']:
            f.append(1 if s in r['style'] else 0)
        f.extend([len(r['genre']), len(r['style'])])
        for c in vocabs['country']:
            f.append(1 if r['country'] == c else 0)
        y_str = str(r['year'])
        if y_str.isdigit() and int(y_str) > 1900:
            f.extend([int(y_str), 1])
        else:
            f.extend([0, 0])
        f.append(1 if r['master_id'] not in ('0', '') else 0)
        f.extend([
            te_stats['label_rate'].get(r['label'], base_rate),
            te_stats['artist_rate'].get(r['artist'], base_rate),
            te_stats['country_rate'].get(r['country'], base_rate),
        ])
        features.append(f)
    return np.array(features, dtype=np.float32), vocabs


def find_optimal_threshold_from_samples(scores, labels, target_precision=0.90, direction='positive', min_samples=20):
    """
    Find the optimal threshold from calibration samples.
    
    For 'positive': lowest score threshold where top-N precision >= target
    For 'negative': highest score threshold where bottom-N negative precision >= target
    """
    n = len(scores)
    if n < min_samples:
        return (0.90 if direction == 'positive' else 0.10)
    
    if direction == 'positive':
        order = np.argsort(-scores)
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        cum_pos = np.cumsum(sorted_labels)
        cum_total = np.arange(1, n + 1)
        cum_prec = cum_pos / cum_total
        
        # Find largest N where precision >= target
        best_n = -1
        for i in range(min_samples - 1, n):
            if cum_prec[i] >= target_precision:
                best_n = i
        
        if best_n >= 0:
            return sorted_scores[best_n]
        
        # If can't hit target, find where precision is maximized with enough samples
        # Try lower precision targets
        for reduced_target in [0.88, 0.85, 0.80, 0.75]:
            best_n = -1
            for i in range(min_samples - 1, n):
                if cum_prec[i] >= reduced_target:
                    best_n = i
            if best_n >= 0:
                return sorted_scores[best_n]
        
        # Last resort: use 95th percentile
        return np.percentile(scores, 95)
    
    else:  # negative
        order = np.argsort(scores)
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        cum_neg = np.cumsum(1 - sorted_labels)
        cum_total = np.arange(1, n + 1)
        cum_prec = cum_neg / cum_total
        
        best_n = -1
        for i in range(min_samples - 1, n):
            if cum_prec[i] >= target_precision:
                best_n = i
        
        if best_n >= 0:
            return sorted_scores[best_n]
        
        for reduced_target in [0.88, 0.85, 0.80]:
            best_n = -1
            for i in range(min_samples - 1, n):
                if cum_prec[i] >= reduced_target:
                    best_n = i
            if best_n >= 0:
                return sorted_scores[best_n]
        
        return np.percentile(scores, 5)


def classify_catalog(catalog):
    """
    Classify records from catalog using up to 1,000 API calls.
    """
    # ================================================================
    # PHASE 1: Train Model
    # ================================================================
    with open('enriched_training.json', 'r') as f:
        training_data = json.load(f)
    
    y_train = np.array([1 if r['wants'] > r['haves'] else 0 for r in training_data])
    te_stats = build_smoothed_te(training_data, y_train, k=10)
    X_train, vocabs = extract_features(training_data, te_stats=te_stats)
    
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    
    # ================================================================
    # PHASE 2: Score Catalog
    # ================================================================
    X_catalog, _ = extract_features(catalog, vocabs=vocabs, te_stats=te_stats)
    scores = model.predict_proba(X_catalog)[:, 1]
    
    n_total = len(catalog)
    release_ids = [str(r['release_id']) for r in catalog]
    
    # Build master_id index
    master_to_indices = defaultdict(list)
    for i, r in enumerate(catalog):
        mid = r['master_id']
        if mid not in ('0', ''):
            master_to_indices[mid].append(i)
    
    # ================================================================
    # PHASE 3: Stratified API Sampling (~700 calls)
    # ================================================================
    api_calls = 0
    verified = set()
    api_results = {}
    
    rng = np.random.RandomState(42)
    calibration_budget = 700
    
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    cal_scores = []
    cal_labels = []
    
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bin_idx = np.where((scores >= lo) & (scores < hi))[0]
        if len(bin_idx) == 0:
            continue
        
        n_sample = min(35, len(bin_idx), calibration_budget - api_calls)
        if n_sample <= 0 or api_calls >= calibration_budget:
            break
        
        sampled = rng.choice(bin_idx, size=n_sample, replace=False)
        for idx in sampled:
            try:
                result = get_release_logged(int(release_ids[idx]))
                api_calls += 1
                is_pos = 1 if result['wants'] > result['haves'] else 0
                api_results[idx] = is_pos
                verified.add(release_ids[idx])
                cal_scores.append(scores[idx])
                cal_labels.append(is_pos)
            except Exception:
                break
    
    cal_scores = np.array(cal_scores)
    cal_labels = np.array(cal_labels)
    
    # ================================================================
    # PHASE 4: Find Optimal Thresholds
    # ================================================================
    pos_threshold = find_optimal_threshold_from_samples(
        cal_scores, cal_labels, target_precision=0.90, direction='positive', min_samples=20
    )
    neg_threshold = find_optimal_threshold_from_samples(
        cal_scores, cal_labels, target_precision=0.90, direction='negative', min_samples=20
    )
    
    # Safety check
    if neg_threshold >= pos_threshold:
        pos_threshold = np.percentile(scores, 85)
        neg_threshold = np.percentile(scores, 30)
    
    # ================================================================
    # PHASE 5: Classification
    # ================================================================
    ruled_in = set()
    ruled_out = set()
    
    # API-verified records (100% precision)
    for idx, is_pos in api_results.items():
        rid = release_ids[idx]
        if is_pos:
            ruled_in.add(rid)
        else:
            ruled_out.add(rid)
    
    # Threshold-based classification
    for i in range(n_total):
        rid = release_ids[i]
        if rid in ruled_in or rid in ruled_out:
            continue
        if scores[i] >= pos_threshold:
            ruled_in.add(rid)
        elif scores[i] <= neg_threshold:
            ruled_out.add(rid)
    
    # ================================================================
    # PHASE 6: Master_id Propagation (~300 API calls)
    # ================================================================
    remaining_budget = 1000 - api_calls
    classified_set = ruled_in | ruled_out
    
    # 6a: Propagate from confident predictions
    for mid, indices in master_to_indices.items():
        if len(indices) < 2:
            continue
        
        n_pos = sum(1 for i in indices if release_ids[i] in ruled_in)
        n_neg = sum(1 for i in indices if release_ids[i] in ruled_out)
        uncertain = [i for i in indices if release_ids[i] not in classified_set]
        
        if not uncertain:
            continue
        
        if n_pos >= 2 and n_neg == 0:
            for i in uncertain:
                if scores[i] >= 0.30:
                    ruled_in.add(release_ids[i])
        elif n_neg >= 2 and n_pos == 0:
            for i in uncertain:
                if scores[i] <= 0.70:
                    ruled_out.add(release_ids[i])
    
    # 6b: API-verified master group propagation
    classified_set = ruled_in | ruled_out
    uncertain_groups = {}
    for mid, indices in master_to_indices.items():
        uncertain = [i for i in indices if release_ids[i] not in classified_set]
        if len(uncertain) >= 2:
            uncertain_groups[mid] = uncertain
    
    sorted_groups = sorted(uncertain_groups.items(), key=lambda x: -len(x[1]))
    
    for mid, uncertain_indices in sorted_groups:
        if remaining_budget <= 0:
            break
        
        group_verdict = None
        for idx in master_to_indices[mid]:
            if idx in api_results:
                group_verdict = api_results[idx]
                break
        
        if group_verdict is None:
            best_idx = uncertain_indices[0]
            try:
                result = get_release_logged(int(release_ids[best_idx]))
                api_calls += 1
                remaining_budget -= 1
                is_pos = 1 if result['wants'] > result['haves'] else 0
                api_results[best_idx] = is_pos
                verified.add(release_ids[best_idx])
                group_verdict = is_pos
                
                if is_pos:
                    ruled_in.add(release_ids[best_idx])
                else:
                    ruled_out.add(release_ids[best_idx])
            except Exception:
                break
        
        if group_verdict is not None:
            for idx in uncertain_indices:
                rid = release_ids[idx]
                if rid in ruled_in or rid in ruled_out:
                    continue
                if group_verdict and scores[idx] >= 0.30:
                    ruled_in.add(rid)
                elif not group_verdict and scores[idx] <= 0.70:
                    ruled_out.add(rid)
    
    # ================================================================
    # PHASE 7: Coverage Guarantee
    # ================================================================
    coverage = (len(ruled_in) + len(ruled_out)) / n_total
    
    if coverage < 0.75:
        classified_set = ruled_in | ruled_out
        unclassified = [(i, scores[i]) for i in range(n_total)
                       if release_ids[i] not in classified_set]
        unclassified.sort(key=lambda x: x[1])
        
        needed = int(0.76 * n_total) - len(classified_set)
        for idx, s in unclassified[:max(0, needed)]:
            ruled_out.add(release_ids[idx])
    
    coverage = (len(ruled_in) + len(ruled_out)) / n_total
    
    return {
        'ruled_in': list(ruled_in),
        'ruled_out': list(ruled_out),
        'verified': list(verified),
        'metadata': {
            'api_calls_made': api_calls,
            'coverage_ratio': coverage,
            'approach': (
                f'LightGBM + API calibration ({len(cal_scores)} samples) + '
                f'master_id propagation. pos_t={pos_threshold:.4f}, neg_t={neg_threshold:.4f}.'
            ),
        }
    }
