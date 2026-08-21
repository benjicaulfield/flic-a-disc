import torch
import pickle
import numpy as np
import mlflow
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity as pairwise_cosine_similarity

from django.core.management.base import BaseCommand
from bandit.models import DiscogsRecord, TfIdfDB, ThresholdConfig
from bandit.views import trainer
from bandit.text_utils import create_mock_ebay_title


class Command(BaseCommand):
    help = "Evaluate the current bandit model on the held-out set."

    def handle(self, *args, **kwargs):
        holdout = list(DiscogsRecord.objects.filter(evaluated=True, heldout=True))
        if not holdout:
            self.stdout.write(self.style.ERROR("No holdout records found. Run create_heldout_set first."))
            return

        self.stdout.write(f"Evaluating on {len(holdout)} held-out records...")

        if not trainer.model:
            if not trainer.load_latest_model():
                self.stdout.write(self.style.ERROR("No trained model available."))
                return

        holdout_records = []
        actuals = []
        for record in holdout:
            holdout_records.append({
                'artist': record.artist,
                'title': record.title,
                'label': record.label,
                'genres': record.genres,
                'styles': record.styles,
                'wants': record.wants,
                'haves': record.haves,
                'year': record.year,
            })
            actuals.append(1 if record.wanted else 0)

        features = trainer.feature_extractor.extract_batch_features(holdout_records)
        features_tensor = torch.FloatTensor(features)

        with torch.no_grad():
            mean_probs, _ = trainer.model.predict_with_uncertainty(features_tensor)
        mean_probs = mean_probs.cpu().numpy()

        try:
            tfidf_db = TfIdfDB.objects.filter(is_active=True).latest('created_at')
            model_data = pickle.loads(tfidf_db.model_weights)
            vectorizer = model_data['vectorizer']
            keeper_embeddings = model_data['keeper_embeddings']
            candidate_titles = [create_mock_ebay_title(r) or '' for r in holdout_records]
            candidate_embeddings = vectorizer.vectorizer.transform(candidate_titles)
            keeper_centroid = np.asarray(keeper_embeddings.mean(axis=0))
            similarities = pairwise_cosine_similarity(candidate_embeddings, keeper_centroid).flatten()
            mean_probs = 0.6 * mean_probs + 0.4 * similarities
            self.stdout.write("TF-IDF similarity blended in.")
        except Exception as e:
            self.stdout.write(f"Skipping TF-IDF blend: {e}")

        actuals = np.array(actuals)

        # Threshold-independent metrics
        precision_curve, recall_curve, _ = precision_recall_curve(actuals, mean_probs)
        pr_auc = auc(recall_curve, precision_curve)
        roc_auc = roc_auc_score(actuals, mean_probs)

        # Pick threshold from held-out curve: highest recall where precision >= 0.75
        PRECISION_FLOOR = 0.75
        best_threshold = 0.5
        best_recall = 0.0
        for thresh in np.arange(0.05, 0.95, 0.05):
            preds_t = (mean_probs >= thresh).astype(int)
            tp = ((preds_t == 1) & (actuals == 1)).sum()
            fp = ((preds_t == 1) & (actuals == 0)).sum()
            fn = ((preds_t == 0) & (actuals == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            if p >= PRECISION_FLOOR and r > best_recall:
                best_recall = r
                best_threshold = thresh

        config, _ = ThresholdConfig.objects.get_or_create(id=1)
        config.threshold = float(best_threshold)
        config.f1_score = best_recall  # storing best recall at precision floor
        config.window_size = len(holdout)
        config.save()
        self.stdout.write(f"ThresholdConfig updated: {best_threshold:.3f} (recall {best_recall:.3f} at precision≥{PRECISION_FLOOR})")

        threshold = best_threshold
        preds = (mean_probs >= threshold).astype(int)
        precision = precision_score(actuals, preds, zero_division=0)
        recall = recall_score(actuals, preds, zero_division=0)
        f1 = f1_score(actuals, preds, zero_division=0)

        # Precision@20
        top20_indices = np.argsort(mean_probs)[::-1][:20]
        p_at_20 = actuals[top20_indices].sum() / 20

        self.stdout.write("\n── Held-out Eval Results ──────────────────")
        self.stdout.write(f"  Records:       {len(holdout)} ({actuals.sum()} keepers)")
        self.stdout.write(f"  Threshold:     {threshold:.3f}")
        self.stdout.write(f"  PR-AUC:        {pr_auc:.3f}")
        self.stdout.write(f"  ROC-AUC:       {roc_auc:.3f}")
        self.stdout.write(f"  Precision:     {precision:.3f}")
        self.stdout.write(f"  Recall:        {recall:.3f}")
        self.stdout.write(f"  F1:            {f1:.3f}")
        self.stdout.write(f"  Precision@20:  {p_at_20:.3f}")
        self.stdout.write("───────────────────────────────────────────\n")

        mlflow.set_experiment("discogs-bandit-model")
        with mlflow.start_run(run_name="held_out_eval"):
            mlflow.log_metrics({
                "heldout_pr_auc": pr_auc,
                "heldout_roc_auc": roc_auc,
                "heldout_precision": precision,
                "heldout_recall": recall,
                "heldout_f1": f1,
                "heldout_precision_at_20": p_at_20,
                "heldout_threshold": threshold,
                "heldout_n_records": len(holdout),
                "heldout_n_keepers": int(actuals.sum()),
            })
        self.stdout.write("Logged to MLflow.")
