#!/usr/bin/env python3
"""
ML-based prediction for desirable Discogs records.
Trains on existing database records (with is_keeper as strong signal)
to predict which new LPs will pass wants > haves filter.
"""

import json
import os
import django
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from bandit.models import Record

class DesirabilityPredictor:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.scaler = StandardScaler()

        # Feature vocabularies (built from training data)
        self.genre_vocab = set()
        self.style_vocab = set()
        self.label_freq = Counter()
        self.artist_freq = Counter()

        # Statistics
        self.year_mean = 0
        self.year_std = 1

    def extract_features(self, record_dict, is_training=False):
        """
        Extract feature vector from a record.

        Features:
        - Genre one-hot (top 20 genres)
        - Style one-hot (top 30 styles)
        - Year (normalized)
        - Label frequency (how common is this label in training data)
        - Artist frequency
        - Has master_id (boolean)
        """
        features = []

        # Genre features (one-hot for top genres)
        genres = record_dict.get('genres', record_dict.get('genre', []))
        if isinstance(genres, str):
            genres = [genres]

        if is_training:
            self.genre_vocab.update(genres)

        genre_features = [1 if g in genres else 0 for g in sorted(self.genre_vocab)[:20]]
        features.extend(genre_features)

        # Style features (one-hot for top styles)
        styles = record_dict.get('styles', record_dict.get('style', []))
        if isinstance(styles, str):
            styles = [styles]

        if is_training:
            self.style_vocab.update(styles)

        style_features = [1 if s in styles else 0 for s in sorted(self.style_vocab)[:30]]
        features.extend(style_features)

        # Year feature (normalized)
        year = record_dict.get('year')
        if year:
            try:
                year_int = int(year) if isinstance(year, str) else year
                if is_training:
                    features.append(year_int)  # Will normalize later
                else:
                    # Normalize using training statistics
                    year_normalized = (year_int - self.year_mean) / (self.year_std + 1e-6)
                    features.append(year_normalized)
            except (ValueError, TypeError):
                features.append(0)
        else:
            features.append(0)

        # Label frequency (how often does this label appear in training data?)
        label = record_dict.get('label', '')
        if is_training:
            self.label_freq[label] += 1
        label_freq_score = np.log1p(self.label_freq.get(label, 0))
        features.append(label_freq_score)

        # Artist frequency
        artist = record_dict.get('artist', '')
        if is_training:
            self.artist_freq[artist] += 1
        artist_freq_score = np.log1p(self.artist_freq.get(artist, 0))
        features.append(artist_freq_score)

        # Has master_id (boolean)
        has_master = 1 if record_dict.get('master_id') else 0
        features.append(has_master)

        # Wants/haves ratio (only for training data)
        # New records won't have this, so we set it to 0
        wants = record_dict.get('wants', 0)
        haves = record_dict.get('haves', 0)
        if wants > 0 and haves > 0:
            wants_haves_ratio = np.log1p(wants / haves)
        else:
            wants_haves_ratio = 0
        features.append(wants_haves_ratio)

        return features

    def train(self, training_records):
        """
        Train model on existing database records.

        Training data:
        - Positive examples: is_keeper=True (weight 3x)
        - Positive examples: passed wants > haves (weight 1x)

        We treat all DB records as positive (they passed filter),
        but keepers get higher weight.
        """
        print(f"\nTraining on {len(training_records)} database records...")

        # First pass: build vocabularies
        print("Building feature vocabularies...")
        for record in training_records:
            self.extract_features(record, is_training=True)

        # Compute year statistics
        years = [int(r.get('year', 0)) for r in training_records if r.get('year')]
        self.year_mean = np.mean(years) if years else 1980
        self.year_std = np.std(years) if years else 20

        print(f"  - {len(self.genre_vocab)} unique genres (using top 20)")
        print(f"  - {len(self.style_vocab)} unique styles (using top 30)")
        print(f"  - {len(self.label_freq)} unique labels")
        print(f"  - Year: mean={self.year_mean:.0f}, std={self.year_std:.1f}")

        # Second pass: extract features
        print("Extracting features...")
        X = []
        y = []
        sample_weights = []

        keeper_count = 0
        for record in training_records:
            features = self.extract_features(record, is_training=False)
            X.append(features)
            y.append(1)  # All DB records are positive (passed filter)

            # Keepers get 3x weight
            is_keeper = record.get('is_keeper', False)
            weight = 3.0 if is_keeper else 1.0
            sample_weights.append(weight)

            if is_keeper:
                keeper_count += 1

        X = np.array(X)
        y = np.array(y)
        sample_weights = np.array(sample_weights)

        print(f"  - {keeper_count} keepers (3x weight)")
        print(f"  - {len(X) - keeper_count} non-keepers (1x weight)")
        print(f"  - Feature vector size: {X.shape[1]}")

        # Normalize features
        X = self.scaler.fit_transform(X)

        # Train model
        print("\nTraining logistic regression...")
        self.model.fit(X, y, sample_weight=sample_weights)

        # Cross-validation score (just for reporting)
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        print(f"  - Cross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        return self

    def predict(self, new_records):
        """
        Predict desirability for new records.

        Returns: List of (record, probability) tuples
        """
        print(f"\nPredicting on {len(new_records)} new records...")

        X = []
        for record in new_records:
            features = self.extract_features(record, is_training=False)
            X.append(features)

        X = np.array(X)
        X = self.scaler.transform(X)

        # Get probability of class 1 (desirable)
        probabilities = self.model.predict_proba(X)[:, 1]

        return list(zip(new_records, probabilities))


def main():
    # Load training data from database
    print("="*60)
    print("DISCOGS DESIRABILITY PREDICTION")
    print("="*60)

    print("\nLoading training data from database...")
    db_records = Record.objects.all()

    training_data = []
    for record in db_records:
        record_dict = {
            'discogs_id': record.discogs_id,
            'artist': record.artist or '',
            'title': record.title or '',
            'label': record.label or '',
            'year': record.year,
            'genres': record.genres or [],
            'styles': record.styles or [],
            'master_id': None,  # Not in current Record model
            'is_keeper': record.wanted,  # wanted=True indicates keeper
            'wants': record.wants,
            'haves': record.haves,
        }
        training_data.append(record_dict)

    print(f"Loaded {len(training_data)} records from database")

    # Train model
    predictor = DesirabilityPredictor()
    predictor.train(training_data)

    # Load new LP catalog
    catalog_path = Path(__file__).parent / 'data' / 'lp_catalog.json'
    print(f"\nLoading new LP catalog from {catalog_path}...")

    new_lps = []
    with open(catalog_path) as f:
        for line in f:
            if line.strip():
                new_lps.append(json.loads(line))

    print(f"Loaded {len(new_lps)} new LPs from shard")

    # Check for duplicates with existing DB
    existing_ids = {r['discogs_id'] for r in training_data}
    new_lps_filtered = [lp for lp in new_lps if lp['release_id'] not in existing_ids]

    print(f"After deduplication: {len(new_lps_filtered)} new LPs to predict")

    # Predict
    predictions = predictor.predict(new_lps_filtered)

    # Categorize by confidence
    high_confidence = []  # P > 0.85
    medium_confidence = []  # 0.15 <= P <= 0.85
    low_confidence = []  # P < 0.15

    for record, prob in predictions:
        record_with_prob = {**record, 'predicted_probability': float(prob)}

        if prob > 0.85:
            high_confidence.append(record_with_prob)
        elif prob < 0.15:
            low_confidence.append(record_with_prob)
        else:
            medium_confidence.append(record_with_prob)

    # Sort by probability
    high_confidence.sort(key=lambda x: x['predicted_probability'], reverse=True)
    medium_confidence.sort(key=lambda x: x['predicted_probability'], reverse=True)

    # Output results
    output_dir = Path(__file__).parent / 'data'

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nHigh confidence (P > 0.85): {len(high_confidence)} records")
    print(f"  → Add to database immediately with api_verified=False")

    print(f"\nMedium confidence (0.15-0.85): {len(medium_confidence)} records")
    print(f"  → Queue for API verification")

    print(f"\nLow confidence (P < 0.15): {len(low_confidence)} records")
    print(f"  → Skip (unlikely to pass wants > haves)")

    print(f"\nAPI calls saved: {len(high_confidence) + len(low_confidence)}")
    print(f"API calls needed: {len(medium_confidence)}")
    print(f"Reduction: {(len(high_confidence) + len(low_confidence)) / len(new_lps_filtered) * 100:.1f}%")

    # Write output files
    with open(output_dir / 'predictions_high_confidence.json', 'w') as f:
        json.dump(high_confidence, f, indent=2)

    with open(output_dir / 'predictions_medium_confidence.json', 'w') as f:
        json.dump(medium_confidence, f, indent=2)

    with open(output_dir / 'predictions_low_confidence.json', 'w') as f:
        json.dump(low_confidence, f, indent=2)

    print(f"\nOutput files written to {output_dir}/")
    print("  - predictions_high_confidence.json")
    print("  - predictions_medium_confidence.json")
    print("  - predictions_low_confidence.json")

    # Show some examples
    print("\n" + "="*60)
    print("TOP 10 HIGH CONFIDENCE PREDICTIONS")
    print("="*60)
    for i, record in enumerate(high_confidence[:10], 1):
        print(f"\n{i}. {record['artist']} - {record['title']}")
        print(f"   Label: {record['label']} ({record['year']})")
        print(f"   Genre: {', '.join(record.get('genre', []))}")
        print(f"   Style: {', '.join(record.get('style', []))}")
        print(f"   Probability: {record['predicted_probability']:.3f}")


if __name__ == '__main__':
    main()
