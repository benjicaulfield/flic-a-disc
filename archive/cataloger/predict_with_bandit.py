#!/usr/bin/env python3
import json
import os
import sys
import django
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from bandit.models import Record
from bandit.training import BanditTrainer
import torch        


def predict_with_bandit(new_lps, trainer):
    """Use bandit trainer to predict on new LPs"""

    print(f"\n📊 Preparing {len(new_lps)} records for prediction...")

    # Convert LPs to dict format expected by feature extractor
    records = []
    for i, lp in enumerate(new_lps):
        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i+1}/{len(new_lps)}")

        # Feature extractor expects dicts with these keys
        record_dict = {
            'discogs_id': lp['release_id'],
            'artist': lp.get('artist', ''),
            'title': lp.get('title', ''),
            'label': lp.get('label', ''),
            'year': int(lp['year'][:4]) if lp.get('year') and lp['year'][:4].isdigit() else None,            'genres': lp.get('genre', []),
            'styles': lp.get('style', []),
        }
        records.append(record_dict)

    # Extract features using bandit's feature extractor
    print("\n🔄 Extracting features...")
    features = trainer.feature_extractor.extract_batch_features(records)
    features_tensor = torch.FloatTensor(features)

    # Get predictions from model
    print("🔄 Running predictions...")
    mean_probs, variances = trainer.model.predict_with_uncertainty(features_tensor)
    mean_probs = mean_probs.cpu().numpy()
    uncertainties = np.sqrt(variances.cpu().numpy())

    # Return list of (lp, probability) tuples
    predictions = [(lp, float(prob)) for lp, prob in zip(new_lps, mean_probs)]

    return predictions
    



def main():
    print("="*60)
    print("BANDIT-BASED PREDICTION")
    print("="*60)

    # Load bandit model
    print("\n🔄 Loading bandit model...")
    trainer = BanditTrainer()
    if not trainer.load_latest_model():
        print("❌ ERROR: No trained model found!")
        print("Train a model first with: python manage.py train_bandit")
        return

    print(f"✅ Loaded model successfully")

    # Load new LP catalog
    catalog_path = Path(__file__).parent / 'data' / 'lp_catalog_filtered_thrice.json'
    print(f"\nLoading LP catalog from {catalog_path}...")

    new_lps = []
    with open(catalog_path) as f:
        for line in f:
            if line.strip():
                new_lps.append(json.loads(line))

    print(f"Loaded {len(new_lps)} LPs")

    existing_ids = set(Record.objects.values_list('discogs_id', flat=True))
    new_lps_filtered = [lp for lp in new_lps if lp['release_id'] not in existing_ids]

    print(f"After deduplication: {len(new_lps_filtered)} new LPs")

    # Predict
    predictions = predict_with_bandit(new_lps_filtered, trainer)

    # Categorize by confidence
    high_confidence = []  # P > 0.85
    medium_confidence = []  # 0.15 <= P <= 0.85
    low_confidence = []  # P < 0.15

    for record, prob in predictions:
        record_with_prob = {
            **record,
            'predicted_probability': prob,
            'model': 'bandit'
        }

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
    print("RESULTS (BANDIT MODEL)")
    print("="*60)
    print(f"\nHigh confidence (P > 0.85): {len(high_confidence)} records")
    print(f"Medium confidence (0.15-0.85): {len(medium_confidence)} records")
    print(f"Low confidence (P < 0.15): {len(low_confidence)} records")

    print(f"\nAPI calls saved: {len(high_confidence) + len(low_confidence)}")
    print(f"API calls needed: {len(medium_confidence)}")
    print(f"Reduction: {(len(high_confidence) + len(low_confidence)) / len(new_lps_filtered) * 100:.1f}%")

    # Write output files
    with open(output_dir / 'bandit_predictions_high.json', 'w') as f:
        json.dump(high_confidence, f, indent=2)

    with open(output_dir / 'bandit_predictions_medium.json', 'w') as f:
        json.dump(medium_confidence, f, indent=2)

    with open(output_dir / 'bandit_predictions_low.json', 'w') as f:
        json.dump(low_confidence, f, indent=2)

    print(f"\nOutput files written to {output_dir}/")
    print("  - bandit_predictions_high.json")
    print("  - bandit_predictions_medium.json")
    print("  - bandit_predictions_low.json")

    # Show top predictions
    print("\n" + "="*60)
    print("TOP 10 BANDIT PREDICTIONS")
    print("="*60)
    for i, record in enumerate(high_confidence[:10], 1):
        print(f"\n{i}. {record['artist']} - {record['title']}")
        print(f"   Label: {record['label']} ({record.get('year', 'N/A')})")
        print(f"   Genre: {', '.join(record.get('genre', []))}")
        print(f"   Probability: {record['predicted_probability']:.3f}")


if __name__ == '__main__':
    main()
