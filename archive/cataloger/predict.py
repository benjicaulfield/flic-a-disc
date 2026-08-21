import os
import sys
import json
import django
import torch
import numpy as np

# Add parent directory to path for Django imports
sys.path.insert(0, os.path.abspath('..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from bandit.training import BanditTrainer

def load_catalog():
    """Load enriched training data"""
    print("Loading enriched training data...")
    with open('data/enriched_training.json') as f:
        catalog = json.load(f)

    # Convert year strings to integers
    for record in catalog:
        if 'year' in record and record['year']:
            try:
                record['year'] = int(record['year'])
            except (ValueError, TypeError):
                record['year'] = None

    print(f"Loaded {len(catalog):,} records")
    return catalog

def predict_catalog(catalog, batch_size=10000, checkpoint_file='data/predictions_checkpoint.npz'):
    """Get predictions for all catalog records"""
    print("\nLoading model...")
    trainer = BanditTrainer()
    trainer.load_latest_model()

    # Check for existing checkpoint
    start_idx = 0
    all_probs = []
    all_uncertainties = []

    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file, loading...")
        checkpoint = np.load(checkpoint_file)
        all_probs = checkpoint['probs'].tolist()
        all_uncertainties = checkpoint['uncertainties'].tolist()
        start_idx = len(all_probs)
        print(f"Resuming from record {start_idx:,}")

    print("Generating predictions...")

    # Process in batches to avoid memory issues
    for i in range(start_idx, len(catalog), batch_size):
        batch = catalog[i:i+batch_size]

        # Ensure years are integers (defensive)
        for record in batch:
            if 'year' in record and record['year'] is not None:
                if isinstance(record['year'], str):
                    try:
                        record['year'] = int(record['year'])
                    except (ValueError, TypeError):
                        record['year'] = None

        # Extract features
        features = trainer.feature_extractor.extract_batch_features(batch)
        features_tensor = torch.FloatTensor(features)

        # Get predictions with uncertainty
        mean_probs, variances = trainer.model.predict_with_uncertainty(features_tensor)
        mean_probs = mean_probs.cpu().numpy()
        uncertainties = np.sqrt(variances.cpu().numpy())

        all_probs.extend(mean_probs)
        all_uncertainties.extend(uncertainties)

        # Save checkpoint every 100k records
        if (i + batch_size) % 100000 == 0:
            print(f"  Processed {i + batch_size:,} / {len(catalog):,} records (saving checkpoint...)")
            np.savez(checkpoint_file,
                     probs=np.array(all_probs),
                     uncertainties=np.array(all_uncertainties))

    # Remove checkpoint file when done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"  Completed {len(catalog):,} predictions")
    return np.array(all_probs), np.array(all_uncertainties)

def analyze_distribution(catalog, probs, uncertainties):
    """Analyze prediction distribution"""
    print("\n" + "="*60)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*60)

    # Calculate actual labels (since this is enriched data)
    actual_labels = np.array([1 if r.get('wants', 0) > r.get('haves', 0) else 0 for r in catalog])
    actual_positive_rate = actual_labels.mean()

    print(f"\nActual positive rate: {actual_positive_rate:.4f} ({actual_labels.sum():,} / {len(catalog):,})")

    # Basic stats
    print(f"\nPredicted Probability (wants > haves):")
    print(f"  Mean:   {probs.mean():.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    print(f"  Std:    {probs.std():.4f}")
    print(f"  Min:    {probs.min():.4f}")
    print(f"  Max:    {probs.max():.4f}")

    print(f"\nUncertainty:")
    print(f"  Mean:   {uncertainties.mean():.4f}")
    print(f"  Median: {np.median(uncertainties):.4f}")
    print(f"  Std:    {uncertainties.std():.4f}")

    # Percentiles
    print(f"\nPercentiles:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(probs, p)
        print(f"  {p:2d}th: {val:.4f}")

    # Distribution buckets
    print(f"\nDistribution by probability:")
    buckets = [
        (0.0, 0.1, "Very low (0.0-0.1)"),
        (0.1, 0.3, "Low (0.1-0.3)"),
        (0.3, 0.5, "Medium-low (0.3-0.5)"),
        (0.5, 0.7, "Medium-high (0.5-0.7)"),
        (0.7, 0.9, "High (0.7-0.9)"),
        (0.9, 1.0, "Very high (0.9-1.0)"),
    ]

    for low, high, label in buckets:
        count = ((probs >= low) & (probs < high)).sum()
        pct = count / len(probs) * 100
        print(f"  {label:20s}: {count:8,} ({pct:5.2f}%)")

    # High confidence predictions
    print(f"\nHigh confidence predictions (≥90%):")
    high_conf_pos = (probs >= 0.9).sum()
    high_conf_neg = (probs <= 0.1).sum()
    total_high_conf = high_conf_pos + high_conf_neg
    print(f"  Positives (≥0.9): {high_conf_pos:,} ({high_conf_pos/len(probs)*100:.2f}%)")
    print(f"  Negatives (≤0.1): {high_conf_neg:,} ({high_conf_neg/len(probs)*100:.2f}%)")
    print(f"  Total high conf:  {total_high_conf:,} ({total_high_conf/len(probs)*100:.2f}%)")

    # Uncertain predictions
    print(f"\nUncertain predictions (0.4-0.6):")
    uncertain = ((probs >= 0.4) & (probs <= 0.6)).sum()
    print(f"  Count: {uncertain:,} ({uncertain/len(probs)*100:.2f}%)")

    # Calibration analysis
    print(f"\nCalibration (predicted prob vs actual rate):")
    calibration_buckets = [
        (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
    ]
    for low, high in calibration_buckets:
        mask = (probs >= low) & (probs < high)
        count = mask.sum()
        if count > 0:
            actual_rate = actual_labels[mask].mean()
            predicted_avg = probs[mask].mean()
            print(f"  [{low:.1f}-{high:.1f}): n={count:6,} | pred={predicted_avg:.3f} actual={actual_rate:.3f} | diff={abs(predicted_avg - actual_rate):.3f}")

    # Overall accuracy at different thresholds
    print(f"\nAccuracy at different thresholds:")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        predicted_labels = (probs >= threshold).astype(int)
        accuracy = (predicted_labels == actual_labels).mean()
        precision = (actual_labels[predicted_labels == 1].sum() / predicted_labels.sum()) if predicted_labels.sum() > 0 else 0
        recall = (predicted_labels[actual_labels == 1].sum() / actual_labels.sum()) if actual_labels.sum() > 0 else 0
        print(f"  {threshold:.1f}: acc={accuracy:.3f} prec={precision:.3f} recall={recall:.3f} (n={predicted_labels.sum():,})")

    return {
        'mean_prob': float(probs.mean()),
        'median_prob': float(np.median(probs)),
        'std_prob': float(probs.std()),
        'high_conf_positive': int(high_conf_pos),
        'high_conf_negative': int(high_conf_neg),
        'uncertain': int(uncertain),
        'actual_positive_rate': float(actual_positive_rate)
    }

def save_predictions(catalog, probs, uncertainties, output_file='data/predictions.json'):
    """Save predictions with catalog data"""
    print(f"\nSaving predictions to {output_file}...")

    predictions = []
    for i, record in enumerate(catalog):
        # Handle both release_id and discogs_id
        release_id = record.get('release_id') or record.get('discogs_id')

        predictions.append({
            'release_id': release_id,
            'master_id': record.get('master_id'),
            'artist': record.get('artist', 'Unknown'),
            'title': record.get('title', 'Unknown'),
            'prob': float(probs[i]),
            'uncertainty': float(uncertainties[i]),
            'actual_positive': record.get('wants', 0) > record.get('haves', 0) if 'wants' in record else None
        })

    with open(output_file, 'w') as f:
        json.dump(predictions, f)

    print(f"Saved {len(predictions):,} predictions")

if __name__ == '__main__':
    # Load catalog
    catalog = load_catalog()

    # Get predictions
    probs, uncertainties = predict_catalog(catalog)

    # Analyze distribution
    stats = analyze_distribution(catalog, probs, uncertainties)

    # Save predictions
    save_predictions(catalog, probs, uncertainties)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
