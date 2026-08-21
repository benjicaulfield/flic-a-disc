"""
Streaming Data Filtering Pipeline for Discogs Records Classification
====================================================================

Classifies vinyl LP releases as "desirable" (wants > haves) or not,
using ML-based classification trained on 30k enriched records,
supplemented by strategic API calls with active learning.

Approach:
1. Train LightGBM on 30k enriched records with target-encoded features
2. Apply to 395k catalog to get probability estimates
3. Use ~200 API calls to calibrate precision at various thresholds
4. Set optimal thresholds based on calibration results
5. Use remaining ~800 API calls for master_id propagation
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
        logging.FileHandler('pipeline_3.log'),
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



def load_training_data(path='enriched_training.json'):
    """Load the 30k enriched training records."""
    with open(path) as f:
        return json.load(f)


def compute_target_encoding(key_fn, records, labels, alpha=10):
    """Compute smoothed target encoding for a categorical feature."""
    global_mean = float(np.mean(labels))
    stats = defaultdict(lambda: [0, 0])
    for r, y in zip(records, labels):
        k = key_fn(r)
        if isinstance(k, list):
            for kk in k:
                stats[kk][0] += 1
                stats[kk][1] += y
        else:
            stats[k][0] += 1
            stats[k][1] += y
    encoding = {}
    for k, (total, pos) in stats.items():
        encoding[k] = (pos + alpha * global_mean) / (total + alpha)
    encoding['__default__'] = global_mean
    return encoding


class FeatureExtractor:
    """Extract features from release records for ML classification."""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, records, labels):
        """Fit feature extraction on training data."""
        self.global_mean = float(np.mean(labels))
        
        genre_counter = Counter(g for r in records for g in r.get('genre', []))
        style_counter = Counter(s for r in records for s in r.get('style', []))
        country_counter = Counter(
            r['country'] if r.get('country') else 'MISSING' for r in records
        )
        
        self.genre_vocab = [g for g, c in genre_counter.most_common() if c >= 30]
        self.style_vocab = [s for s, c in style_counter.most_common() if c >= 30]
        self.country_vocab = [c for c, cnt in country_counter.most_common() if cnt >= 10]
        
        self.genre_te = compute_target_encoding(
            lambda r: list(r.get('genre', [])), records, labels, alpha=10)
        self.style_te = compute_target_encoding(
            lambda r: list(r.get('style', [])), records, labels, alpha=10)
        self.country_te = compute_target_encoding(
            lambda r: r['country'] if r.get('country') else 'MISSING', records, labels, alpha=10)
        self.label_te = compute_target_encoding(
            lambda r: r.get('label') or 'UNKNOWN', records, labels, alpha=10)
        self.artist_te = compute_target_encoding(
            lambda r: r.get('artist') or 'UNKNOWN', records, labels, alpha=5)
        
        self.fitted = True
        return self
    
    def transform(self, records):
        """Transform records into feature matrix."""
        assert self.fitted, "Must call fit() first"
        gm = self.global_mean
        
        n_base = 7  # year(2) + year_empty + country_empty + country_te + has_master + label_te + artist_te + has_catno
        n_country = len(self.country_vocab)
        n_genre = len(self.genre_vocab) + 3  # one-hot + count + mean_te + max_te
        n_style = len(self.style_vocab) + 3
        n_total = 3 + 1 + n_country + 1 + n_genre + n_style + 1 + 1 + 1 + 1
        
        features = np.zeros((len(records), n_total), dtype=np.float32)
        
        genre_set_cache = {}
        style_set_cache = {}
        
        for i, r in enumerate(records):
            col = 0
            
            # Year features (3)
            year = r.get('year')
            if year and year != '':
                try:
                    features[i, col] = int(year)
                    features[i, col+1] = 1
                except (ValueError, TypeError):
                    pass
            features[i, col+2] = 1 if year == '' else 0
            col += 3
            
            # Country features
            country = r.get('country') or ''
            features[i, col] = 1 if country == '' else 0
            col += 1
            
            for j, c in enumerate(self.country_vocab):
                if country == c:
                    features[i, col + j] = 1
            col += n_country
            
            c_key = country if country else 'MISSING'
            features[i, col] = self.country_te.get(c_key, self.country_te['__default__'])
            col += 1
            
            # Genre features
            genres = set(r.get('genre', []))
            for j, g in enumerate(self.genre_vocab):
                if g in genres:
                    features[i, col + j] = 1
            col += len(self.genre_vocab)
            
            features[i, col] = len(genres)
            col += 1
            
            if genres:
                g_tes = [self.genre_te.get(g, gm) for g in genres]
                features[i, col] = np.mean(g_tes)
                features[i, col+1] = max(g_tes)
            else:
                features[i, col] = gm
                features[i, col+1] = gm
            col += 2
            
            # Style features
            styles = set(r.get('style', []))
            for j, s in enumerate(self.style_vocab):
                if s in styles:
                    features[i, col + j] = 1
            col += len(self.style_vocab)
            
            features[i, col] = len(styles)
            col += 1
            
            if styles:
                s_tes = [self.style_te.get(s, gm) for s in styles]
                features[i, col] = np.mean(s_tes)
                features[i, col+1] = max(s_tes)
            else:
                features[i, col] = gm
                features[i, col+1] = gm
            col += 2
            
            # Master ID flag
            mid = str(r.get('master_id', '0'))
            features[i, col] = 1 if mid and mid != '0' else 0
            col += 1
            
            # Label target encoding
            lab = r.get('label') or 'UNKNOWN'
            features[i, col] = self.label_te.get(lab, self.label_te['__default__'])
            col += 1
            
            # Artist target encoding
            art = r.get('artist') or 'UNKNOWN'
            features[i, col] = self.artist_te.get(art, self.artist_te['__default__'])
            col += 1
            
            # Has catalog number
            features[i, col] = 1 if r.get('catalog_number') else 0
            col += 1
        
        return features[:, :col]


def train_model(training_records):
    """Train the classification model on enriched training data."""
    labels = np.array([
        1 if int(r['wants']) > int(r['haves']) else 0 
        for r in training_records
    ])
    
    extractor = FeatureExtractor()
    extractor.fit(training_records, labels)
    X = extractor.transform(training_records)
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X, labels)
    
    return model, extractor


def build_master_index(catalog):
    """Build index from master_id to list of catalog record indices."""
    master_to_indices = defaultdict(list)
    for i, r in enumerate(catalog):
        mid = str(r.get('master_id', '0'))
        if mid and mid != '0':
            master_to_indices[mid].append(i)
    return master_to_indices


def _propagate_master(mid, is_positive, master_index, probs, idx_to_rid,
                      ruled_in_set, ruled_out_set, pos_threshold):
    """Propagate an API result to sibling records in the same master group."""
    if mid not in master_index:
        return
    
    for sib_idx in master_index[mid]:
        sib_rid = idx_to_rid[sib_idx]
        sib_prob = probs[sib_idx]
        
        if is_positive:
            # If queried record is positive, siblings with moderately high
            # predicted probability are likely also positive
            if sib_prob >= pos_threshold * 0.70:
                ruled_in_set.add(sib_rid)
                ruled_out_set.discard(sib_rid)
        else:
            # If queried record is negative, siblings with low-to-moderate 
            # predicted probability are very likely also negative
            if sib_prob < pos_threshold:
                ruled_out_set.add(sib_rid)
                ruled_in_set.discard(sib_rid)


def classify_catalog(catalog):
    """
    Classify records from catalog using up to 1,000 API calls.
    
    Uses adaptive active learning:
    1. Train ML model on enriched training data
    2. Apply to catalog for initial probability estimates
    3. Use API calls to calibrate thresholds (Phase 1: ~200 calls)
    4. Use API calls for master_id propagation (Phase 2: ~800 calls)
    
    Args:
        catalog: List of records from lp_catalog.json (without wants/haves)
        api_client: API client with get_release(release_id) method
    
    Returns:
        {
            'ruled_in': [release_ids],
            'ruled_out': [release_ids],
            'verified': [release_ids],
            'metadata': {
                'api_calls_made': int,
                'coverage_ratio': float,
                'approach': str
            }
        }
    """
    
    # ===== PHASE 0: Model Training and Prediction =====
    training = load_training_data()
    model, extractor = train_model(training)
    
    X_catalog = extractor.transform(catalog)
    probs = model.predict_proba(X_catalog)[:, 1]
    
    # Build indices
    master_index = build_master_index(catalog)
    idx_to_rid = {}
    rid_to_idx = {}
    for i in range(len(catalog)):
        rid = str(catalog[i]['release_id'])
        idx_to_rid[i] = rid
        rid_to_idx[rid] = i
    
    # ===== PHASE 1: Calibration via API =====
    api_calls = 0
    verified_set = set()
    api_results = {}  # idx -> {'wants', 'haves', 'positive'}
    
    CALIBRATION_BUDGET = 200
    
    # Stratified sample across probability ranges for calibration
    np.random.seed(42)
    prob_ranges = [
        (0.35, 0.50, 40),
        (0.50, 0.60, 35),
        (0.60, 0.70, 30),
        (0.70, 0.80, 25),
        (0.80, 1.01, 25),
        (0.15, 0.35, 25),
        (0.00, 0.15, 20),
    ]
    
    for low, high, n_samples in prob_ranges:
        if api_calls >= CALIBRATION_BUDGET:
            break
        
        mask = (probs >= low) & (probs < high)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        
        sample_size = min(n_samples, len(indices), CALIBRATION_BUDGET - api_calls)
        sample_indices = np.random.choice(indices, size=sample_size, replace=False)
        
        for idx in sample_indices:
            if api_calls >= CALIBRATION_BUDGET:
                break
            
            rid = idx_to_rid[idx]
            try:
                result = get_release_logged(int(rid))
                api_calls += 1
                verified_set.add(rid)
                
                wants = result.get('wants', 0)
                haves = result.get('haves', 0)
                is_positive = wants > haves
                api_results[idx] = {
                    'wants': wants, 'haves': haves, 'positive': is_positive
                }
            except Exception:
                break
    
    # ===== Compute calibrated thresholds =====
    DEFAULT_POS_THRESHOLD = 0.50
    DEFAULT_NEG_THRESHOLD = 0.15
    
    best_pos_threshold = DEFAULT_POS_THRESHOLD
    best_neg_threshold = DEFAULT_NEG_THRESHOLD
    
    # Find optimal positive threshold
    for tp in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        verified_above = [(idx, res) for idx, res in api_results.items() 
                         if probs[idx] >= tp]
        if len(verified_above) >= 15:
            n_positive = sum(1 for _, res in verified_above if res['positive'])
            precision = n_positive / len(verified_above)
            
            # Use 90% precision target (with small margin for sampling error)
            if precision >= 0.90:
                best_pos_threshold = tp
                break
    
    # Find optimal negative threshold  
    for tn in [0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05]:
        verified_below = [(idx, res) for idx, res in api_results.items() 
                         if probs[idx] < tn]
        if len(verified_below) >= 10:
            n_negative = sum(1 for _, res in verified_below if not res['positive'])
            neg_precision = n_negative / len(verified_below)
            if neg_precision >= 0.90:
                best_neg_threshold = tn
                break
    
    # Safety: check coverage with chosen thresholds
    n_pos = int((probs >= best_pos_threshold).sum())
    n_neg = int((probs < best_neg_threshold).sum())
    coverage = (n_pos + n_neg) / len(catalog)
    
    # If coverage is too low, relax thresholds slightly
    if coverage < 0.75:
        # Try raising neg threshold first (usually safe)
        for tn in [best_neg_threshold + 0.05 * i for i in range(1, 10)]:
            n_neg_new = int((probs < tn).sum())
            if (n_pos + n_neg_new) / len(catalog) >= 0.75:
                best_neg_threshold = tn
                break
    
    # ===== Apply initial ML classification =====
    ruled_in_set = set()
    ruled_out_set = set()
    
    for i in range(len(catalog)):
        rid = idx_to_rid[i]
        if probs[i] >= best_pos_threshold:
            ruled_in_set.add(rid)
        elif probs[i] < best_neg_threshold:
            ruled_out_set.add(rid)
    
    # Override with API-verified records
    for idx, res in api_results.items():
        rid = idx_to_rid[idx]
        if res['positive']:
            ruled_in_set.add(rid)
            ruled_out_set.discard(rid)
        else:
            ruled_out_set.add(rid)
            ruled_in_set.discard(rid)
    
    # ===== PHASE 2: Strategic API calls with propagation =====
    
    # Find best master groups to query for maximum propagation impact
    master_candidates = []
    for mid, indices in master_index.items():
        if len(indices) < 2:
            continue
        
        group_probs = probs[indices]
        # Count records that could be reclassified
        n_uncertain = sum(1 for p in group_probs 
                         if best_neg_threshold <= p < best_pos_threshold)
        n_borderline = sum(1 for p in group_probs 
                          if best_pos_threshold * 0.70 <= p < best_pos_threshold)
        
        # Already have a verified member?
        already_queried = any(idx_to_rid[idx] in verified_set for idx in indices)
        
        if (n_uncertain > 0 or n_borderline > 0) and not already_queried:
            score = len(indices) * (n_uncertain + 0.5 * n_borderline)
            # Pick the record closest to the decision boundary
            best_query_idx = indices[int(np.argmin(np.abs(group_probs - best_pos_threshold)))]
            master_candidates.append((mid, best_query_idx, score))
    
    master_candidates.sort(key=lambda x: -x[2])
    
    for mid, query_idx, score in master_candidates:
        if api_calls >= 1000:
            break
        
        rid = idx_to_rid[query_idx]
        if rid in verified_set:
            continue
        
        try:
            result = get_release_logged(int(rid))
            api_calls += 1
            verified_set.add(rid)
            
            wants = result.get('wants', 0)
            haves = result.get('haves', 0)
            is_positive = wants > haves
            api_results[query_idx] = {
                'wants': wants, 'haves': haves, 'positive': is_positive
            }
            
            # Classify queried record
            if is_positive:
                ruled_in_set.add(rid)
                ruled_out_set.discard(rid)
            else:
                ruled_out_set.add(rid)
                ruled_in_set.discard(rid)
            
            # Propagate to master group siblings
            _propagate_master(mid, is_positive, master_index, probs,
                             idx_to_rid, ruled_in_set, ruled_out_set,
                             best_pos_threshold)
            
        except Exception:
            break
    
    # ===== PHASE 3: Fill remaining gaps with individual queries =====
    if api_calls < 1000:
        unclassified = []
        for i in range(len(catalog)):
            rid = idx_to_rid[i]
            if rid not in ruled_in_set and rid not in ruled_out_set and rid not in verified_set:
                unclassified.append((i, abs(probs[i] - 0.5)))
        
        # Sort by uncertainty (closest to 0.5 first)
        unclassified.sort(key=lambda x: x[1])
        
        for idx, _ in unclassified:
            if api_calls >= 1000:
                break
            
            rid = idx_to_rid[idx]
            try:
                result = get_release_logged(int(rid))
                api_calls += 1
                verified_set.add(rid)
                
                wants = result.get('wants', 0)
                haves = result.get('haves', 0)
                is_positive = wants > haves
                
                if is_positive:
                    ruled_in_set.add(rid)
                else:
                    ruled_out_set.add(rid)
                
                # Also propagate via master_id if applicable
                mid = str(catalog[idx].get('master_id', '0'))
                if mid and mid != '0' and mid in master_index:
                    _propagate_master(mid, is_positive, master_index, probs,
                                     idx_to_rid, ruled_in_set, ruled_out_set,
                                     best_pos_threshold)
                
            except Exception:
                break
    
    # ===== Final cleanup =====
    overlap = ruled_in_set & ruled_out_set
    if overlap:
        # Resolve conflicts using model probability
        for rid in overlap:
            idx = rid_to_idx.get(rid)
            if idx is not None and probs[idx] >= 0.5:
                ruled_out_set.discard(rid)
            else:
                ruled_in_set.discard(rid)
    
    ruled_in = sorted(list(ruled_in_set))
    ruled_out = sorted(list(ruled_out_set))
    final_coverage = (len(ruled_in) + len(ruled_out)) / len(catalog)
    
    return {
        'ruled_in': ruled_in,
        'ruled_out': ruled_out,
        'verified': sorted(list(verified_set)),
        'metadata': {
            'api_calls_made': api_calls,
            'coverage_ratio': final_coverage,
            'approach': (
                f'LightGBM with target-encoded features (genre/style/country/label/artist). '
                f'Adaptive thresholds calibrated via {len(api_results)} API calls. '
                f'pos_threshold={best_pos_threshold:.2f}, neg_threshold={best_neg_threshold:.2f}. '
                f'Master_id propagation for coverage expansion. '
                f'{len(ruled_in)} positives, {len(ruled_out)} negatives classified. '
                f'Coverage: {final_coverage:.1%}.'
            )
        }
    }


if __name__ == '__main__':
    import sys
    import time
    
    print("Loading catalog...")
    with open('lp_catalog.json') as f:
        catalog = json.load(f)
    print(f"Loaded {len(catalog)} records")
    
    class MockAPIClient:
        """Mock API that returns training data for known IDs, 0/0 for unknown."""
        def __init__(self, data_path='enriched_training.json'):
            with open(data_path) as f:
                records = json.load(f)
            self.data = {}
            for r in records:
                rid = str(r.get('release_id'))
                self.data[rid] = {'wants': int(r['wants']), 'haves': int(r['haves'])}
            self.calls = 0
        
        def get_release(self, release_id):
            if self.calls >= 1000:
                raise Exception("API budget exceeded")
            self.calls += 1
            rid = str(release_id)
            if rid in self.data:
                return {'release_id': int(rid), **self.data[rid]}
            return {'release_id': int(rid), 'wants': 0, 'haves': 0}
    
    start = time.time()
    client = MockAPIClient()
    result = classify_catalog(catalog, client)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  Ruled in:  {len(result['ruled_in']):,}")
    print(f"  Ruled out: {len(result['ruled_out']):,}")
    print(f"  Verified:  {len(result['verified']):,}")
    print(f"  API calls: {result['metadata']['api_calls_made']}")
    print(f"  Coverage:  {result['metadata']['coverage_ratio']:.1%}")
