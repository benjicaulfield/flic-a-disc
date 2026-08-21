"""
Streaming Data Filtering Pipeline for Discogs Vinyl LP Classification.
"""

import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import FeatureHasher
import lightgbm as lgb
import logging
from datetime import datetime
import os
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_t11.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Django setup for database access
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
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
api_client = authenticate_client()
logger.info("Client authenticated successfully")


def extract_features_base(record):
    features = {}
    country = str(record.get('country', ''))
    features[f'country={country}'] = 1
    features['country_empty'] = 1 if country == '' else 0
    try:
        year = int(record['year'])
        features['year'] = year
        features['decade'] = (year // 10) * 10
    except (ValueError, TypeError):
        features['year'] = 0; features['decade'] = 0; features['year_unknown'] = 1
    features['has_master'] = 0 if str(record.get('master_id', '0')) == '0' else 1
    genres = record.get('genre', [])
    if isinstance(genres, str): genres = [genres]
    for g in genres: features[f'genre={g}'] = 1
    features['n_genres'] = len(genres)
    styles = record.get('style', [])
    if isinstance(styles, str): styles = [styles]
    for s in styles: features[f'style={s}'] = 1
    features['n_styles'] = len(styles)
    features[f'label={record.get("label", "")}'] = 1
    features[f'artist={record.get("artist", "")}'] = 1
    features['title_len'] = len(str(record.get('title', '')))
    cat_num = str(record.get('catalog_number', ''))
    if cat_num: features[f'cat_prefix3={cat_num[:3]}'] = 1
    for g in genres: features[f'gxc={g}|{country}'] = 1
    try:
        decade = (int(record['year']) // 10) * 10
        for g in genres: features[f'gxd={g}|{decade}'] = 1
    except: pass
    return features


def extract_features_with_te(record, ts, gm):
    features = extract_features_base(record)
    artist = str(record.get('artist', ''))
    label_name = str(record.get('label', ''))
    country = str(record.get('country', ''))
    mid = str(record.get('master_id', '0'))
    s = 10
    for f, k in [('artist', artist), ('label', label_name), ('country', country)]:
        if k in ts.get(f, {}):
            m, c = ts[f][k]
            features[f'{f}_te'] = (m * c + gm * s) / (c + s)
        else:
            features[f'{f}_te'] = gm
    genres = record.get('genre', [])
    if isinstance(genres, str): genres = [genres]
    gs = []
    for g in genres:
        if g in ts.get('genre', {}):
            m, c = ts['genre'][g]
            gs.append((m * c + gm * s) / (c + s))
    if gs:
        features['genre_te_mean'] = np.mean(gs)
        features['genre_te_max'] = max(gs)
    styles = record.get('style', [])
    if isinstance(styles, str): styles = [styles]
    ss = []
    for st in styles:
        if st in ts.get('style', {}):
            m, c = ts['style'][st]
            ss.append((m * c + gm * s) / (c + s))
    if ss:
        features['style_te_mean'] = np.mean(ss)
        features['style_te_max'] = max(ss)
    if mid != '0' and mid in ts.get('master_id', {}):
        m, c = ts['master_id'][mid]
        features['master_te'] = (m * c + gm * s) / (c + s)
        features['master_te_count'] = min(c, 20)
    return features


def compute_target_stats(records, labels):
    stats = {}
    for f in ['artist', 'label', 'country']:
        g = defaultdict(list)
        for i, r in enumerate(records):
            g[str(r.get(f, ''))].append(labels[i])
        stats[f] = {k: (np.mean(v), len(v)) for k, v in g.items()}
    for af in ['genre', 'style']:
        g = defaultdict(list)
        for i, r in enumerate(records):
            vals = r.get(af, [])
            if isinstance(vals, str): vals = [vals]
            for v in vals: g[v].append(labels[i])
        stats[af] = {k: (np.mean(v), len(v)) for k, v in g.items()}
    mg = defaultdict(list)
    for i, r in enumerate(records):
        mid = str(r.get('master_id', '0'))
        if mid != '0': mg[mid].append(labels[i])
    stats['master_id'] = {k: (np.mean(v), len(v)) for k, v in mg.items()}
    return stats


def classify_catalog(catalog: list) -> dict:
    logger.info("Loading catalog...")
    logger.info(f"Loaded {len(catalog)} catalog records")
    logger.info("Running pipeline...")
    logger.info("=" * 60)
    logger.info("PHASE 0: Loading training data and building model")
    logger.info("=" * 60)

    with open('enriched_training.json') as f:
        training_data = json.load(f)
    logger.info(f"Loaded {len(training_data)} training records")

    logger.info("Building training labels...")
    train_labels = np.array([1 if r['wants'] > r['haves'] else 0 for r in training_data])
    gm = train_labels.mean()
    logger.info(f"Global mean (positive rate): {gm:.4f}")
    logger.info("Computing target statistics...")
    ts = compute_target_stats(training_data, train_labels)

    tms = {}
    for mid, (rate, count) in ts.get('master_id', {}).items():
        tms[mid] = {'rate': rate, 'count': count}

    logger.info("Extracting features...")
    hb = FeatureHasher(n_features=2**16, alternate_sign=False)
    ht = FeatureHasher(n_features=2**17, alternate_sign=False)
    logger.info("Extracting base features for training data...")
    Xb = hb.transform([extract_features_base(r) for r in training_data])
    logger.info(f"Base features shape: {Xb.shape}")
    logger.info("Extracting TE features for training data...")
    Xt = ht.transform([extract_features_with_te(r, ts, gm) for r in training_data])
    logger.info(f"TE features shape: {Xt.shape}")

    logger.info("Training base models...")
    models_base = []
    for i, (seed, d, l) in enumerate([(42, 7, 63), (123, 8, 127)]):
        logger.info(f"  Training base model {i+1}/2 (seed={seed}, depth={d}, leaves={l})...")
        m = lgb.LGBMClassifier(
            objective='binary', n_estimators=500, learning_rate=0.05,
            max_depth=d, num_leaves=l, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=20, random_state=seed, verbosity=-1, is_unbalance=True)
        m.fit(Xb, train_labels); models_base.append(m)

    logger.info("Training TE model...")
    mt = lgb.LGBMClassifier(
        objective='binary', n_estimators=500, learning_rate=0.05,
        max_depth=8, num_leaves=127, subsample=0.7, colsample_bytree=0.6,
        min_child_samples=20, random_state=42, verbosity=-1, is_unbalance=True,
        reg_alpha=0.1, reg_lambda=0.1)
    mt.fit(Xt, train_labels)
    logger.info("Models trained successfully")

    logger.info("Scoring catalog...")
    logger.info("Extracting base features for catalog...")
    Xcb = hb.transform([extract_features_base(r) for r in catalog])
    logger.info(f"Catalog base features shape: {Xcb.shape}")
    logger.info("Extracting TE features for catalog...")
    Xct = ht.transform([extract_features_with_te(r, ts, gm) for r in catalog])
    logger.info(f"Catalog TE features shape: {Xct.shape}")
    logger.info("Generating predictions from base models...")
    pb = np.mean([m.predict_proba(Xcb)[:, 1] for m in models_base], axis=0)
    logger.info("Generating predictions from TE model...")
    pt = mt.predict_proba(Xct)[:, 1]
    logger.info("Computing ensemble predictions...")
    pf = (pb + pt) / 2.0
    logger.info("Scoring complete")

    n = len(catalog)
    status = np.zeros(n, dtype=int)
    source = [''] * n

    cmi = defaultdict(list)
    for i, r in enumerate(catalog):
        mid = str(r.get('master_id', '0'))
        if mid != '0': cmi[mid].append(i)

    logger.info("=" * 60)
    logger.info("PHASE A: ML Classification with country-aware thresholds")
    logger.info("=" * 60)
    # PHASE A: ML with country-aware thresholds
    for i in range(n):
        c = str(catalog[i].get('country', ''))
        if c == '':
            if pf[i] >= 0.65 and pb[i] >= 0.55:
                status[i] = 1; source[i] = 'ml_ep'
            elif pf[i] < 0.30:
                status[i] = -1; source[i] = 'ml_en'
        else:
            if pf[i] >= 0.80 and pb[i] >= 0.75:
                status[i] = 1; source[i] = 'ml_np'
            elif pf[i] < 0.35:
                status[i] = -1; source[i] = 'ml_nn'

    ml_in = (status == 1).sum()
    ml_out = (status == -1).sum()
    logger.info(f"Phase A complete: {ml_in} ruled in, {ml_out} ruled out")

    logger.info("=" * 60)
    logger.info("PHASE B: Training master propagation")
    logger.info("=" * 60)
    # PHASE B: Training master propagation (96% precision)
    mp_in = mp_out = 0
    for mid, indices in cmi.items():
        if mid not in tms: continue
        r = tms[mid]['rate']; cnt = tms[mid]['count']
        for idx in indices:
            if status[idx] != 0: continue
            if cnt >= 2 and r >= 0.80:
                status[idx] = 1; source[idx] = 'mp'; mp_in += 1
            elif cnt >= 1 and r == 1.0:
                status[idx] = 1; source[idx] = 'mps'; mp_in += 1
            if status[idx] == 0:
                if cnt >= 2 and r == 0.0 and pf[idx] < 0.65:
                    status[idx] = -1; source[idx] = 'mn'; mp_out += 1
                elif cnt >= 1 and r == 0.0 and pf[idx] < 0.40:
                    status[idx] = -1; source[idx] = 'mns'; mp_out += 1

    logger.info(f"Phase B complete: +{mp_in} ruled in, +{mp_out} ruled out")

    logger.info("=" * 60)
    logger.info("PHASE C: API active learning")
    logger.info("=" * 60)
    # PHASE C: API active learning
    ac = 0; verified = []; api_res = {}
    ubm = defaultdict(list)
    for i in range(n):
        if status[i] == 0:
            mid = str(catalog[i].get('master_id', '0'))
            if mid != '0': ubm[mid].append(i)

    cq = [(mid, len(idx), len(cmi.get(mid, [])), idx)
          for mid, idx in ubm.items()
          if len(idx) >= 2 or len(cmi.get(mid, [])) >= 3]
    cq.sort(key=lambda x: (-x[1], -x[2]))

    ap_in = ap_out = 0
    for mid, nu, tot, ui in cq:
        if ac >= 5000: break
        bi = min(ui, key=lambda i: abs(pf[i] - 0.5))
        rid = int(catalog[bi]['release_id'])
        try:
            logger.info(f"API call {ac+1}: Querying release {rid} for master_id {mid}")
            res = api_client.release(rid)
            ac += 1; verified.append(str(rid))

            # Extract wants/haves from community stats
            stats = (res.data.get('stats') or {}).get('community') or {}
            wants = stats.get('in_wantlist', 0)
            haves = stats.get('in_collection', 0)
            ip = wants > haves
            logger.info(f"API call {ac}: Release {rid} - wants={wants}, haves={haves}, positive={ip}")

            # Save to database
            record, created = Record.objects.get_or_create(
                discogs_id=str(rid),
                defaults={
                    'artist': catalog[bi].get('artist', ''),
                    'title': catalog[bi].get('title', ''),
                    'label': catalog[bi].get('label', ''),
                    'catno': catalog[bi].get('catalog_number', ''),
                    'wants': wants,
                    'haves': haves,
                    'added': timezone.now(),
                    'genres': catalog[bi].get('genre', []),
                    'styles': catalog[bi].get('style', []),
                    'year': catalog[bi].get('year'),
                    'api_enriched': True,
                }
            )
            if not created:
                record.wants = wants
                record.haves = haves
                record.api_enriched = True
                record.save()

            api_res[mid] = ip
            status[bi] = 1 if ip else -1; source[bi] = 'ad'
            for idx in ui:
                if status[idx] != 0: continue
                if ip:
                    status[idx] = 1; source[idx] = 'app'; ap_in += 1
                elif pf[idx] < 0.60:
                    status[idx] = -1; source[idx] = 'apn'; ap_out += 1
            if ip:
                for idx in cmi.get(mid, []):
                    if idx in set(ui): continue
                    if status[idx] == -1 and source[idx].startswith('ml_'):
                        status[idx] = 1; source[idx] = 'af'; ap_in += 1
        except Exception as e:
            logger.warning(f"API call failed for release {rid}: {e}")
            break

    logger.info(f"Phase C complete: {ac} API calls made, +{ap_in} ruled in, +{ap_out} ruled out")

    logger.info("=" * 60)
    logger.info("PHASE D: Controlled relaxation")
    logger.info("=" * 60)
    # PHASE D: Controlled relaxation
    # Add empty-country positives where TE model is highly confident
    for i in range(n):
        if status[i] == 0:
            c = str(catalog[i].get('country', ''))
            if c == '':
                # TE model highly confident + moderate ensemble
                if pt[i] >= 0.80 and pf[i] >= 0.58:
                    status[i] = 1; source[i] = 'rl_ep'
                elif pf[i] < 0.35:
                    status[i] = -1; source[i] = 'rl_en'
            else:
                if pf[i] < 0.40:
                    status[i] = -1; source[i] = 'rl_nn'

    ri = [str(catalog[i]['release_id']) for i in range(n) if status[i] == 1]
    ro = [str(catalog[i]['release_id']) for i in range(n) if status[i] == -1]
    cov = (len(ri) + len(ro)) / n

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)
    logger.info(f"Total ruled in: {len(ri)}")
    logger.info(f"Total ruled out: {len(ro)}")
    logger.info(f"Coverage: {cov:.4f} ({cov*100:.2f}%)")
    logger.info(f"API calls made: {ac}")

    sc = defaultdict(int)
    for s in source:
        if s: sc[s] += 1

    logger.info("Source breakdown:")
    for source_type, count in sorted(sc.items(), key=lambda x: -x[1]):
        logger.info(f"  {source_type}: {count}")

    return {
        'ruled_in': ri, 'ruled_out': ro, 'verified': verified,
        'metadata': {
            'api_calls_made': ac, 'coverage_ratio': cov,
            'approach': (f'ML({ml_in}in,{ml_out}out)+master(+{mp_in}in,+{mp_out}out)+'
                        f'API({ac}calls,+{ap_in}in,+{ap_out}out)|'
                        f'Total:{len(ri)}in,{len(ro)}out,cov={cov:.4f}'),
            'source_counts': dict(sc)
        }
    }
