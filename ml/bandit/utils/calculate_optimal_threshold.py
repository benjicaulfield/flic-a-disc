import numpy as np

from ..models import BanditTrainingInstance, ThresholdConfig

def calculate_optimal_threshold(window_size=500):
    recent_instances = BanditTrainingInstance.objects.order_by('-timestamp')[:window_size]

    if len(recent_instances) < 50:
        return 0.5, 0.0
    
    predictions = []
    actuals = []

    for instance in recent_instances:
        predictions.append(instance.predicted_prob)
        actuals.append(1 if instance.actual else 0)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    PRECISION_FLOOR = 0.75

    best_threshold = 0.5
    best_recall = 0.0
    best_precision = 0.0

    for threshold in np.arange(0.05, 0.95, 0.05):
        pred_binary = (predictions > threshold).astype(int)
        tp = ((pred_binary == 1) & (actuals == 1)).sum()
        fp = ((pred_binary == 1) & (actuals == 0)).sum()
        fn = ((pred_binary == 0) & (actuals == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision >= PRECISION_FLOOR and recall > best_recall:
            best_recall = recall
            best_precision = precision
            best_threshold = threshold

    if best_recall == 0.0:
        print(f"⚠️  No threshold achieved precision≥{PRECISION_FLOOR} in training window — threshold held at {best_threshold:.2f}")

    config, _ = ThresholdConfig.objects.get_or_create(id=1)
    config.threshold = best_threshold
    config.f1_score = best_precision  # stores precision, field name is legacy
    config.window_size = window_size
    config.save()

    return best_threshold, best_precision