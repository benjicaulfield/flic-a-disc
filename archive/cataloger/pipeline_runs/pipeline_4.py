"""
Streaming Active Learning Pipeline for Discogs Classification - v2
Improved propagation strategy
"""

import json
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from typing import Dict, List, Set
import heapq

import logging
import os
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_4.log'),
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



class DiscogsClassificationPipeline:
    def __init__(self, training_path='enriched_training.json'):
        """Initialize pipeline and train base model"""
        self.training_path = training_path
        self.tfidf = None
        self.model = None
        self.feature_dim = None
        
        # Cache structures
        self.master_to_records = defaultdict(list)
        self.verified_records = {}  # release_id -> label
        self.predictions = {}  # release_id -> probability
        
        # Training data structures
        self.training_masters = {}  # master_id -> majority label
        
        print("Training base model...")
        self._train_base_model()
        
    def _train_base_model(self):
        """Train LightGBM classifier on training data"""
        with open(self.training_path, 'r') as f:
            training_data = json.load(f)
        
        # Create labels
        y = np.array([1 if r['wants'] > r['haves'] else 0 for r in training_data])
        
        # Extract features
        X = self._extract_features(training_data, fit_tfidf=True)
        
        # Build master lookup from training data
        for record in training_data:
            master_id = record['master_id']
            if master_id != '0':
                label = 1 if record['wants'] > record['haves'] else 0
                if master_id not in self.training_masters:
                    self.training_masters[master_id] = []
                self.training_masters[master_id].append(label)
        
        # Aggregate to majority vote and confidence
        for master_id in self.training_masters:
            labels = self.training_masters[master_id]
            majority = 1 if sum(labels) / len(labels) > 0.5 else 0
            confidence = max(sum(labels), len(labels) - sum(labels)) / len(labels)
            self.training_masters[master_id] = (majority, confidence, len(labels))
        
        # Train model
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1,
            'seed': 42
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=200)
        print(f"Model trained on {len(training_data)} records")
        print(f"Training masters cached: {len(self.training_masters)}")
        
    def _extract_features(self, records, fit_tfidf=False):
        """Extract features from records"""
        # Text features
        texts = [self._create_text_field(r) for r in records]
        
        if fit_tfidf:
            self.tfidf = TfidfVectorizer(max_features=500, min_df=3, ngram_range=(1, 2))
            X_text = self.tfidf.fit_transform(texts).toarray()
        else:
            X_text = self.tfidf.transform(texts).toarray()
        
        # Numeric features
        X_numeric = np.array([self._extract_numeric_features(r) for r in records])
        
        # Combine
        X = np.hstack([X_numeric, X_text])
        
        if fit_tfidf:
            self.feature_dim = X.shape[1]
        
        return X
    
    def _create_text_field(self, record):
        """Combine text fields for TF-IDF"""
        parts = [
            record.get('artist', ''),
            record.get('label', ''),
            ' '.join(record.get('genre', [])),
            ' '.join(record.get('style', []))
        ]
        return ' '.join(parts).lower()
    
    def _extract_numeric_features(self, record):
        """Extract numeric features"""
        features = []
        
        # Year features
        try:
            year = int(record['year'])
            features.append(year)
            features.append(2024 - year)
            features.append(1 if year < 1970 else 0)
            features.append(1 if 1970 <= year < 1980 else 0)
            features.append(1 if 1980 <= year < 1990 else 0)
            features.append(1 if 1990 <= year < 2000 else 0)
            features.append(1 if year >= 2000 else 0)
        except:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # Has master
        features.append(1 if record['master_id'] != '0' else 0)
        
        # Country features
        features.append(1 if record['country'] == 'US' else 0)
        features.append(1 if record['country'] == 'UK' else 0)
        features.append(1 if record['country'] == 'Japan' else 0)
        features.append(1 if record['country'] == 'Germany' else 0)
        features.append(1 if record['country'] == 'France' else 0)
        
        # Text length features
        features.append(len(record.get('artist', '')))
        features.append(len(record.get('title', '')))
        features.append(len(record.get('label', '')))
        
        # Count features
        features.append(len(record.get('genre', [])))
        features.append(len(record.get('style', [])))
        
        return features
    
    def _build_master_index(self, catalog):
        """Build index of master_id -> release_ids"""
        self.master_to_records = defaultdict(list)
        for record in catalog:
            master_id = record['master_id']
            if master_id != '0':
                self.master_to_records[master_id].append(record['release_id'])
    
    def _initial_predictions(self, catalog):
        """Generate initial predictions for all records"""
        print("Generating initial predictions...")
        X = self._extract_features(catalog, fit_tfidf=False)
        probas = self.model.predict(X)
        
        for i, record in enumerate(catalog):
            self.predictions[record['release_id']] = probas[i]
        
        print(f"Predictions generated for {len(catalog)} records")
    
    def _calculate_propagation_value(self, record, catalog_dict):
        """Calculate value of querying this record for propagation"""
        release_id = record['release_id']
        master_id = record['master_id']
        
        value = 0
        
        # Value from master siblings
        if master_id != '0' and master_id in self.master_to_records:
            siblings = self.master_to_records[master_id]
            unverified_siblings = [s for s in siblings if s not in self.verified_records]
            
            # Bonus for masters with many siblings (high propagation potential)
            if len(unverified_siblings) > 1:
                value += min(len(unverified_siblings), 20) * 3
        
        # Value from uncertainty (prefer uncertain records)
        prob = self.predictions[release_id]
        uncertainty = 1 - abs(prob - 0.5) * 2  # Max at 0.5, min at 0/1
        value += uncertainty * 5
        
        # Bonus if master not in training (explore new masters)
        if master_id != '0' and master_id not in self.training_masters:
            value += 2
        
        return value
    
    def _propagate_from_verified(self, release_id, label, catalog_dict):
        """Propagate label from verified record to similar records"""
        record = catalog_dict[release_id]
        master_id = record['master_id']
        prob = self.predictions[release_id]
        
        propagated = []
        
        # Master-based propagation with relaxed criteria
        if master_id != '0' and master_id in self.master_to_records:
            siblings = self.master_to_records[master_id]
            
            # Don't propagate to very large master groups (likely compilations/VA)
            if len(siblings) <= 50:
                for sibling_id in siblings:
                    if sibling_id == release_id or sibling_id in self.verified_records:
                        continue
                    
                    sibling = catalog_dict[sibling_id]
                    sibling_prob = self.predictions[sibling_id]
                    
                    # Relaxed propagation criteria
                    if self._should_propagate_relaxed(record, sibling, prob, sibling_prob, label):
                        self.verified_records[sibling_id] = label
                        propagated.append(sibling_id)
        
        return propagated
    
    def _should_propagate_relaxed(self, source, target, source_prob, target_prob, label):
        """Relaxed propagation criteria for better coverage"""
        # Model predictions should be consistent (same side of 0.5)
        source_pred = 1 if source_prob >= 0.5 else 0
        target_pred = 1 if target_prob >= 0.5 else 0
        
        if source_pred != target_pred:
            return False
        
        # Verified label should match model prediction
        if label != source_pred:
            # Model was wrong, don't propagate
            return False
        
        # Predictions should be reasonably close (within 0.3)
        if abs(source_prob - target_prob) > 0.3:
            return False
        
        # Target shouldn't be too uncertain (propagate to moderately confident predictions)
        if 0.4 < target_prob < 0.6:
            return False
        
        # Optional: same country or similar year (relaxed)
        same_country = source['country'] == target['country']
        try:
            year_diff = abs(int(source['year']) - int(target['year']))
            similar_year = year_diff <= 5
        except:
            similar_year = False
        
        # At least one similarity indicator
        if same_country or similar_year:
            return True
        
        # If predictions are very close, propagate anyway
        if abs(source_prob - target_prob) < 0.15:
            return True
        
        return False
    
    def classify_catalog(self, catalog):
        """
        Main classification pipeline
        
        Args:
            catalog: List of records from lp_catalog.json
            api_client: API client with get_release(release_id) method
            
        Returns:
            Dictionary with ruled_in, ruled_out, verified, and metadata
        """
        # Build catalog lookup
        catalog_dict = {r['release_id']: r for r in catalog}
        
        # Build master index
        self._build_master_index(catalog)
        
        # Generate initial predictions
        self._initial_predictions(catalog)
        
        # Check if we can use training masters for initial propagation
        initial_from_training = 0
        for master_id, (label, confidence, count) in self.training_masters.items():
            if master_id in self.master_to_records and confidence >= 0.9:
                # High confidence training master - pre-populate
                for release_id in self.master_to_records[master_id]:
                    if release_id not in self.verified_records:
                        # Only if model agrees
                        model_pred = 1 if self.predictions[release_id] >= 0.5 else 0
                        if model_pred == label:
                            self.verified_records[release_id] = label
                            initial_from_training += 1
        
        print(f"Pre-populated {initial_from_training} records from high-confidence training masters")
        
        # Stage 1: Classify high-confidence predictions
        high_conf_threshold = 0.85
        low_conf_threshold = 0.15
        
        initial_positives = set()
        initial_negatives = set()
        uncertain = []
        
        for release_id, prob in self.predictions.items():
            if release_id in self.verified_records:
                continue
            
            if prob >= high_conf_threshold:
                initial_positives.add(release_id)
            elif prob <= low_conf_threshold:
                initial_negatives.add(release_id)
            else:
                uncertain.append(release_id)
        
        print(f"\nInitial classification:")
        print(f"  High-conf positives: {len(initial_positives)}")
        print(f"  High-conf negatives: {len(initial_negatives)}")
        print(f"  Uncertain: {len(uncertain)}")
        print(f"  Pre-verified: {len(self.verified_records)}")
        
        # Stage 2: Active learning with API budget
        api_budget = 1000
        api_calls_made = 0
        
        # Build priority queue for uncertain records
        priority_queue = []
        for release_id in uncertain:
            if release_id not in self.verified_records:
                record = catalog_dict[release_id]
                value = self._calculate_propagation_value(record, catalog_dict)
                heapq.heappush(priority_queue, (-value, release_id))
        
        print(f"\nStarting active learning with {api_budget} API budget...")
        
        propagation_stats = []
        
        while api_calls_made < api_budget and priority_queue:
            _, release_id = heapq.heappop(priority_queue)
            
            # Skip if already verified
            if release_id in self.verified_records:
                continue
            
            # Query API
            try:
                api_response = get_release_logged(int(release_id))
                api_calls_made += 1
                
                wants = api_response['wants']
                haves = api_response['haves']
                label = 1 if wants > haves else 0
                
                self.verified_records[release_id] = label
                
                # Propagate to similar records
                propagated = self._propagate_from_verified(release_id, label, catalog_dict)
                propagation_stats.append(len(propagated))
                
                if api_calls_made % 100 == 0:
                    avg_prop = np.mean(propagation_stats[-100:]) if propagation_stats else 0
                    print(f"  API calls: {api_calls_made}, Verified: {len(self.verified_records)}, Avg propagation: {avg_prop:.1f}")
                
            except Exception as e:
                print(f"API error: {e}")
                break
        
        print(f"\nActive learning complete:")
        print(f"  API calls made: {api_calls_made}")
        print(f"  Total verified: {len(self.verified_records)}")
        if propagation_stats:
            print(f"  Avg propagation per API call: {np.mean(propagation_stats):.1f}")
        
        # Stage 3: Final classification
        ruled_in = set()
        ruled_out = set()
        
        # Add high-confidence predictions
        ruled_in.update(initial_positives)
        ruled_out.update(initial_negatives)
        
        # Add verified records (overrides initial predictions)
        for release_id, label in self.verified_records.items():
            if label == 1:
                ruled_in.add(release_id)
                ruled_out.discard(release_id)
            else:
                ruled_out.add(release_id)
                ruled_in.discard(release_id)
        
        # For remaining uncertain records, use more lenient threshold
        final_threshold_high = 0.65
        final_threshold_low = 0.35
        
        for release_id in uncertain:
            if release_id not in self.verified_records:
                prob = self.predictions[release_id]
                if prob >= final_threshold_high:
                    ruled_in.add(release_id)
                elif prob <= final_threshold_low:
                    ruled_out.add(release_id)
        
        coverage = (len(ruled_in) + len(ruled_out)) / len(catalog)
        
        print(f"\nFinal Results:")
        print(f"  Ruled in: {len(ruled_in)}")
        print(f"  Ruled out: {len(ruled_out)}")
        print(f"  Coverage: {coverage:.2%}")
        print(f"  Verified via API: {api_calls_made}")
        print(f"  Total verified (incl. propagation): {len(self.verified_records)}")
        
        return {
            'ruled_in': list(ruled_in),
            'ruled_out': list(ruled_out),
            'verified': list(self.verified_records.keys()),
            'metadata': {
                'api_calls_made': api_calls_made,
                'coverage_ratio': coverage,
                'approach': 'LightGBM + Active Learning + Relaxed Master Propagation'
            }
        }


def classify_catalog(catalog):
    """
    Required interface for evaluation
    """
    pipeline = DiscogsClassificationPipeline()
    return pipeline.classify_catalog(catalog, api_client)
