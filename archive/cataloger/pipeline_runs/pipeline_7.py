"""
Vinyl LP classification pipeline using ML + active learning.

Approach:
1. Train LightGBM classifier on 30k enriched training records
2. Engineer features from artist, title, label, genre, style, year, country
3. Use CalibratedClassifierCV for well-calibrated probabilities
4. Apply conservative thresholds from cross-validation
5. Use 1000 API calls for iterative cluster propagation on uncertain records
6. Combine model probability with cluster-level API evidence for propagation
"""

import json
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import warnings
import os

import logging
import os
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_7.log'),
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


warnings.filterwarnings('ignore')


def parse_year(y):
    if y is None: return 0
    if isinstance(y, (int, float)): return int(y)
    try: return int(y)
    except: return 0


def clean_list_field(fl):
    if not isinstance(fl, list): return []
    cleaned = []
    for item in fl:
        if isinstance(item, str) and item.startswith('['):
            try:
                p = json.loads(item)
                if isinstance(p, list): cleaned.extend(p)
                else: cleaned.append(str(p))
            except: cleaned.append(item)
        else: cleaned.append(item)
    return cleaned


def compute_priors(training_data):
    y = np.array([1 if r['wants'] > r['haves'] else 0 for r in training_data])
    global_rate = y.mean()
    priors = {}
    for kn, kf in [('label', lambda r: r.get('label','') or ''),
                   ('artist', lambda r: r.get('artist','') or ''),
                   ('country', lambda r: r.get('country') or 'Unknown')]:
        d = defaultdict(lambda: [0,0])
        for r in training_data:
            k = kf(r); d[k][1] += 1
            if r['wants'] > r['haves']: d[k][0] += 1
        priors[kn] = dict(d)
    for kn, kf in [('genre', lambda r: clean_list_field(r.get('genre',[]))),
                   ('style', lambda r: clean_list_field(r.get('style',[])))]:
        d = defaultdict(lambda: [0,0])
        for r in training_data:
            ip = r['wants'] > r['haves']
            for k in kf(r): d[k][1] += 1; d[k][0] += ip
        priors[kn] = dict(d)
    d = defaultdict(lambda: [0,0])
    for r in training_data:
        k = (r.get('label','') or '', r.get('country') or 'Unknown')
        d[k][1] += 1
        if r['wants'] > r['haves']: d[k][0] += 1
    priors['lc'] = dict(d)
    return priors, global_rate


def smoothed_rate(prior_dict, key, global_rate, min_count=5):
    stats = prior_dict.get(key)
    if stats is None or stats[1] < min_count: return global_rate
    return (stats[0] + global_rate * min_count) / (stats[1] + min_count)


def build_features(records, priors, global_rate, fit=False, transformers=None):
    if transformers is None: transformers = {}
    sr = lambda pd, k: smoothed_rate(pd, k, global_rate)
    
    a = [r.get('artist','') or '' for r in records]
    t = [r.get('title','') or '' for r in records]
    l = [r.get('label','') or '' for r in records]
    cn = [r.get('catalog_number','') or '' for r in records]
    g = [clean_list_field(r.get('genre',[])) for r in records]
    s = [clean_list_field(r.get('style',[])) for r in records]
    yr = np.array([parse_year(r.get('year')) for r in records])
    c = [r.get('country') or 'Unknown' for r in records]
    
    # Target-encoded priors
    lra = np.array([sr(priors['label'],x) for x in l]).reshape(-1,1)
    ara = np.array([sr(priors['artist'],x) for x in a]).reshape(-1,1)
    cra = np.array([sr(priors['country'],x) for x in c]).reshape(-1,1)
    lcra = np.array([sr(priors['lc'],(li,ci)) for li,ci in zip(l,c)]).reshape(-1,1)
    grf = np.array([[np.mean([sr(priors['genre'],x) for x in gl]) if gl else global_rate,
                     max([sr(priors['genre'],x) for x in gl]) if gl else global_rate,
                     min([sr(priors['genre'],x) for x in gl]) if gl else global_rate] for gl in g])
    srf = np.array([[np.mean([sr(priors['style'],x) for x in sl]) if sl else global_rate,
                     max([sr(priors['style'],x) for x in sl]) if sl else global_rate,
                     min([sr(priors['style'],x) for x in sl]) if sl else global_rate] for sl in s])
    
    hy = (yr>0).astype(float).reshape(-1,1)
    hc = np.array([1.0 if r.get('country') else 0.0 for r in records]).reshape(-1,1)
    ng = np.array([len(x) for x in g]).reshape(-1,1)
    ns = np.array([len(x) for x in s]).reshape(-1,1)
    
    parts = []
    if fit:
        transformers['av'] = TfidfVectorizer(max_features=300, ngram_range=(1,2), min_df=2, sublinear_tf=True)
        parts.append(transformers['av'].fit_transform(a))
        transformers['tv'] = TfidfVectorizer(max_features=200, ngram_range=(1,2), min_df=2, sublinear_tf=True)
        parts.append(transformers['tv'].fit_transform(t))
        transformers['lv'] = TfidfVectorizer(max_features=200, ngram_range=(1,2), min_df=2, sublinear_tf=True)
        parts.append(transformers['lv'].fit_transform(l))
        transformers['cnv'] = TfidfVectorizer(max_features=100, analyzer='char_wb', ngram_range=(2,4), min_df=3)
        parts.append(transformers['cnv'].fit_transform(cn))
        transformers['gm'] = MultiLabelBinarizer()
        parts.append(csr_matrix(transformers['gm'].fit_transform(g)))
        transformers['sm'] = MultiLabelBinarizer()
        parts.append(csr_matrix(transformers['sm'].fit_transform(s)))
        transformers['ys'] = StandardScaler()
        parts.append(csr_matrix(transformers['ys'].fit_transform(yr.reshape(-1,1))))
        tc = {x for x,cnt in Counter(c).items() if cnt>=5}
        transformers['tc'] = tc
        cc = [x if x in tc else 'Other' for x in c]
        transformers['ccv'] = TfidfVectorizer(analyzer='word')
        parts.append(transformers['ccv'].fit_transform(cc))
    else:
        parts.append(transformers['av'].transform(a))
        parts.append(transformers['tv'].transform(t))
        parts.append(transformers['lv'].transform(l))
        parts.append(transformers['cnv'].transform(cn))
        parts.append(csr_matrix(transformers['gm'].transform(g)))
        parts.append(csr_matrix(transformers['sm'].transform(s)))
        parts.append(csr_matrix(transformers['ys'].transform(yr.reshape(-1,1))))
        cc = [x if x in transformers['tc'] else 'Other' for x in c]
        parts.append(transformers['ccv'].transform(cc))
    
    parts.extend([csr_matrix(hy), csr_matrix(hc), csr_matrix(ng), csr_matrix(ns),
                  csr_matrix(lra), csr_matrix(ara), csr_matrix(cra), csr_matrix(lcra),
                  csr_matrix(grf), csr_matrix(srf)])
    return hstack(parts), transformers


def find_thresholds_cv(cv_probs, y, target_precision=0.90):
    """Find thresholds that achieve target precision on CV data."""
    # Positive threshold: lowest threshold with >= target precision, maximize recall
    best_pos_t, best_pos_rec = 0.95, 0
    for t in np.arange(0.30, 0.99, 0.005):
        mask = cv_probs >= t
        if mask.sum() < 10: continue
        prec = precision_score(y, mask)
        rec = recall_score(y, mask)
        if prec >= target_precision and rec > best_pos_rec:
            best_pos_rec = rec; best_pos_t = t
    
    # Negative threshold: highest threshold with >= target precision for negatives
    best_neg_t, best_neg_rec = 0.05, 0
    for t in np.arange(0.50, 0.01, -0.005):
        mask = cv_probs < t
        if mask.sum() < 10: continue
        prec = precision_score(1 - y, mask)
        rec = recall_score(1 - y, mask)
        if prec >= target_precision and rec > best_neg_rec:
            best_neg_rec = rec; best_neg_t = t
    
    return best_pos_t, best_neg_t, best_pos_rec, best_neg_rec


def classify_catalog(catalog):
    """
    Classify records from catalog using up to 1,000 API calls.
    """
    # =========================================================================
    # PHASE 0: Load training data
    # =========================================================================
    training_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'enriched_training.json')
    with open(training_path) as f:
        training = json.load(f)
    
    y_train = np.array([1 if r['wants'] > r['haves'] else 0 for r in training])
    priors, global_rate = compute_priors(training)
    print(f"Training: {len(training)} records, {global_rate:.4f} positive rate")
    
    # =========================================================================
    # PHASE 1: Feature engineering + model training
    # =========================================================================
    print("Building features...")
    X_train, transformers = build_features(training, priors, global_rate, fit=True)
    
    # Cross-validated probabilities for threshold selection
    print("Cross-validation for thresholds...")
    cv_model = LGBMClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(cv_model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    
    pos_t, neg_t, pos_rec, neg_rec = find_thresholds_cv(cv_probs, y_train, target_precision=0.90)
    print(f"CV thresholds: pos={pos_t:.3f} (rec={pos_rec:.3f}), neg={neg_t:.3f} (rec={neg_rec:.3f})")
    
    # Train calibrated final model
    print("Training calibrated model...")
    base = LGBMClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1
    )
    calibrated = CalibratedClassifierCV(base, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)
    
    # =========================================================================
    # PHASE 2: Score catalog
    # =========================================================================
    print(f"Scoring {len(catalog)} catalog records...")
    X_catalog, _ = build_features(catalog, priors, global_rate, fit=False, transformers=transformers)
    catalog_probs = calibrated.predict_proba(X_catalog)[:, 1]
    catalog_ids = [r['release_id'] for r in catalog]
    rid_to_idx = {rid: i for i, rid in enumerate(catalog_ids)}
    
    # =========================================================================
    # PHASE 3: API Calibration (200 calls to check thresholds on actual catalog)
    # =========================================================================
    print("API calibration phase (200 calls)...")
    api_calls = 0
    verified = []
    api_results = {}
    rng = np.random.RandomState(42)
    
    CAL_BUDGET = 200
    # Sample from critical bins around the threshold regions
    cal_bins = [
        (0.00, 0.10, 15), (0.10, 0.20, 15), (0.20, 0.30, 20),
        (0.30, 0.40, 20), (0.40, 0.50, 20), (0.50, 0.55, 15),
        (0.55, 0.60, 15), (0.60, 0.65, 15), (0.65, 0.70, 15),
        (0.70, 0.75, 15), (0.75, 0.80, 15), (0.80, 0.90, 15),
        (0.90, 1.01, 5),
    ]
    
    cal_probs_list = []
    cal_labels_list = []
    
    for lo, hi, n_target in cal_bins:
        bin_mask = (catalog_probs >= lo) & (catalog_probs < hi)
        bin_indices = np.where(bin_mask)[0]
        if len(bin_indices) == 0: continue
        n = min(n_target, len(bin_indices))
        chosen = rng.choice(bin_indices, n, replace=False)
        for idx in chosen:
            if api_calls >= CAL_BUDGET: break
            rid = catalog_ids[idx]
            try:
                result = get_release_logged(rid)
                api_calls += 1
                verified.append(rid)
                is_pos = int(result['wants'] > result['haves'])
                api_results[rid] = is_pos
                cal_probs_list.append(catalog_probs[idx])
                cal_labels_list.append(is_pos)
            except: break
    
    cal_probs_arr = np.array(cal_probs_list)
    cal_labels_arr = np.array(cal_labels_list)
    
    print(f"  Calibration samples: {len(cal_probs_arr)}, {cal_labels_arr.sum()} pos ({cal_labels_arr.mean():.3f})")
    
    # Find empirical thresholds from calibration data
    if len(cal_probs_arr) >= 30 and cal_labels_arr.sum() >= 5:
        # Positive threshold: find where precision >= 90% on calibration data
        emp_pos_t = pos_t  # fallback
        best_emp_pos_rec = 0
        for t in np.arange(0.30, 0.98, 0.01):
            above = cal_probs_arr >= t
            if above.sum() < 3: continue
            p = cal_labels_arr[above].mean()
            r = cal_labels_arr[above].sum() / max(cal_labels_arr.sum(), 1)
            if p >= 0.90 and r > best_emp_pos_rec:
                best_emp_pos_rec = r
                emp_pos_t = t
        
        # Negative threshold
        emp_neg_t = neg_t  # fallback
        best_emp_neg_rec = 0
        for t in np.arange(0.50, 0.01, -0.01):
            below = cal_probs_arr < t
            if below.sum() < 3: continue
            neg_p = (1 - cal_labels_arr[below]).mean()
            neg_r = (1-cal_labels_arr)[below].sum() / max((1-cal_labels_arr).sum(), 1)
            if neg_p >= 0.90 and neg_r > best_emp_neg_rec:
                best_emp_neg_rec = neg_r
                emp_neg_t = t
        
        # Use the MORE AGGRESSIVE threshold (lower pos, higher neg) 
        # but sanity check against CV
        final_pos_t = min(pos_t, emp_pos_t)
        final_neg_t = max(neg_t, emp_neg_t)
        
        # Sanity bounds
        final_pos_t = max(final_pos_t, 0.40)  # don't go below 40% 
        final_neg_t = min(final_neg_t, 0.50)  # don't go above 50%
        
        print(f"  Empirical thresholds: pos={emp_pos_t:.3f}, neg={emp_neg_t:.3f}")
    else:
        final_pos_t = pos_t
        final_neg_t = neg_t
    
    print(f"  Final thresholds: pos={final_pos_t:.3f}, neg={final_neg_t:.3f}")
    
    # =========================================================================
    # PHASE 4: ML Classification
    # =========================================================================
    ruled_in = set()
    ruled_out = set()
    classified = set()
    
    # Classify API-verified records first
    for rid, is_pos in api_results.items():
        idx = rid_to_idx.get(rid)
        if idx is not None:
            if is_pos: ruled_in.add(rid)
            else: ruled_out.add(rid)
            classified.add(idx)
    
    # ML threshold classification
    for idx in range(len(catalog)):
        if idx in classified: continue
        p = catalog_probs[idx]
        rid = catalog_ids[idx]
        if p >= final_pos_t:
            ruled_in.add(rid)
            classified.add(idx)
        elif p < final_neg_t:
            ruled_out.add(rid)
            classified.add(idx)
    
    cov = (len(ruled_in) + len(ruled_out)) / len(catalog)
    print(f"\nAfter ML: ri={len(ruled_in)}, ro={len(ruled_out)}, cov={cov:.4f}, calls={api_calls}")
    
    # =========================================================================
    # PHASE 5: Iterative cluster propagation (remaining budget)
    # =========================================================================
    remaining = 1000 - api_calls
    uncertain_indices = set(range(len(catalog))) - classified
    
    print(f"\nCluster propagation: {remaining} calls budget, {len(uncertain_indices)} uncertain...")
    
    # Build label clusters
    label_clusters = defaultdict(list)
    for idx in uncertain_indices:
        lab = catalog[idx].get('label', '') or ''
        label_clusters[lab].append(idx)
    
    # Track per-label API observations
    label_obs = defaultdict(lambda: [0, 0])  # [pos_count, total_count]
    label_prior = {}
    for lab in label_clusters:
        stats = priors['label'].get(lab)
        if stats and stats[1] >= 3:
            label_prior[lab] = stats[0] / stats[1]
        else:
            label_prior[lab] = global_rate
    
    propagated_pos = 0
    propagated_neg = 0
    
    # Propagation thresholds (more aggressive than ML thresholds)
    prop_pos_t = max(0.45, final_pos_t - 0.15)
    prop_neg_t = min(0.30, final_neg_t + 0.10)
    
    for iteration in range(remaining):
        if not uncertain_indices: break
        
        # Find best cluster to query
        best_label = None
        best_score = -1
        for lab, indices in label_clusters.items():
            active = [i for i in indices if i in uncertain_indices]
            if len(active) < 2: continue
            avg_p = np.mean([catalog_probs[i] for i in active])
            score = len(active) * avg_p  # prioritize large positive-leaning clusters
            if score > best_score:
                best_score = score
                best_label = lab
        
        if best_label is None: break
        active = [i for i in label_clusters[best_label] if i in uncertain_indices]
        if not active: break
        
        # Query most informative record (closest to 0.5)
        query_idx = min(active, key=lambda i: abs(catalog_probs[i] - 0.5))
        rid = catalog_ids[query_idx]
        
        try:
            result = get_release_logged(rid)
            api_calls += 1
            verified.append(rid)
            is_pos = int(result['wants'] > result['haves'])
            api_results[rid] = is_pos
        except: break
        
        if is_pos: ruled_in.add(rid)
        else: ruled_out.add(rid)
        classified.add(query_idx)
        uncertain_indices.discard(query_idx)
        
        # Update label stats
        label_obs[best_label][0] += is_pos
        label_obs[best_label][1] += 1
        
        # Posterior rate
        obs = label_obs[best_label]
        prior = label_prior.get(best_label, global_rate)
        alpha = 2
        posterior = (obs[0] + prior * alpha) / (obs[1] + alpha)
        
        # Propagate to cluster members
        for idx in active:
            if idx not in uncertain_indices: continue
            p = catalog_probs[idx]
            combined = 0.5 * p + 0.5 * posterior
            
            if combined >= prop_pos_t and p >= 0.30:
                ruled_in.add(catalog_ids[idx])
                classified.add(idx)
                uncertain_indices.discard(idx)
                propagated_pos += 1
            elif combined < prop_neg_t and p < 0.40:
                ruled_out.add(catalog_ids[idx])
                classified.add(idx)
                uncertain_indices.discard(idx)
                propagated_neg += 1
        
        label_clusters[best_label] = [i for i in label_clusters[best_label]
                                       if i in uncertain_indices]
        
        if (iteration+1) % 200 == 0:
            c2 = (len(ruled_in)+len(ruled_out))/len(catalog)
            print(f"  iter={iteration+1}: calls={api_calls}, cov={c2:.4f}, "
                  f"unc={len(uncertain_indices)}, prop(+{propagated_pos}/-{propagated_neg})")
    
    cov = (len(ruled_in)+len(ruled_out))/len(catalog)
    print(f"\nAfter propagation: calls={api_calls}, cov={cov:.4f}, "
          f"prop(+{propagated_pos}/-{propagated_neg})")
    
    # =========================================================================
    # PHASE 6: Coverage boost if needed
    # =========================================================================
    if cov < 0.76:
        print(f"Boosting coverage from {cov:.4f}...")
        needed = int(0.76 * len(catalog)) - len(ruled_in) - len(ruled_out)
        remaining_list = sorted(uncertain_indices,
                               key=lambda i: abs(catalog_probs[i] - 0.5), reverse=True)
        for idx in remaining_list:
            if needed <= 0: break
            p = catalog_probs[idx]
            rid = catalog_ids[idx]
            boost_pos_t = max(0.40, final_pos_t - 0.20)
            boost_neg_t = min(0.25, final_neg_t + 0.10)
            if p >= boost_pos_t:
                ruled_in.add(rid)
                needed -= 1
            elif p < boost_neg_t:
                ruled_out.add(rid)
                needed -= 1
    
    final_cov = (len(ruled_in)+len(ruled_out))/len(catalog)
    print(f"\nFINAL: ri={len(ruled_in)}, ro={len(ruled_out)}, cov={final_cov:.4f}, calls={api_calls}")
    
    return {
        'ruled_in': list(ruled_in),
        'ruled_out': list(ruled_out),
        'verified': verified,
        'metadata': {
            'api_calls_made': api_calls,
            'coverage_ratio': final_cov,
            'approach': ('Calibrated LightGBM + API threshold calibration + '
                        'iterative label cluster propagation + coverage boost')
        }
    }
