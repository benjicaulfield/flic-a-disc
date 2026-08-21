import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder

# Load enriched set
with open("enriched.json") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Basic feature engineering - just enough to run the experiment
# You'd do this more carefully for the real pipeline
def featurize(r, include_price=True):
    features = {
        "year": r.get("year") or 0,
        "has_master_id": int(bool(r.get("master_id"))),
        "country_encoded": hash(r.get("country", "")) % 1000,
        "label_encoded": hash(r.get("label", "")) % 5000,
        "genre_count": len(r.get("genre", [])),
        "style_count": len(r.get("style", [])),
        # add more features here as you see fit
    }
    if include_price:
        features["suggested_price"] = r.get("suggested_price") or 0
    return list(features.values())

labels = [int(r["wants"] > r["haves"]) for r in records]

X_with = np.array([featurize(r, include_price=True) for r in records])
X_without = np.array([featurize(r, include_price=False) for r in records])

X_with_train, X_with_val, y_train, y_val = train_test_split(
    X_with, labels, test_size=0.2, random_state=42, stratify=labels
)
X_without_train, X_without_val, _, _ = train_test_split(
    X_without, labels, test_size=0.2, random_state=42, stratify=labels
)

# Train both models
model_with = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_with.fit(X_with_train, y_train)

model_without = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_without.fit(X_without_train, y_train)

# Precision-recall curves
def recall_at_90_precision(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    # find recall values where precision >= 0.90
    mask = precision >= 0.90
    if not mask.any():
        return None, None
    best_recall = recall[mask].max()
    best_threshold = thresholds[mask[:-1]].min()
    return best_recall, best_threshold

recall_with, thresh_with = recall_at_90_precision(model_with, X_with_val, y_val)
recall_without, thresh_without = recall_at_90_precision(model_without, X_without_val, y_val)

print(f"With suggested price:    recall @ 90% precision = {recall_with:.3f} (threshold={thresh_with:.3f})")
print(f"Without suggested price: recall @ 90% precision = {recall_without:.3f} (threshold={thresh_without:.3f})")
print(f"Signal loss: {recall_with - recall_without:.3f}")