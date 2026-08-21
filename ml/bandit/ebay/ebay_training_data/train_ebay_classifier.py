import json
import re
import mlflow
import pickle
import numpy as np

from django.conf import settings
from django.utils import timezone as django_timezone

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ...models import EbayListing, EbayFirstPassModel

class EbayFirstPassClassifier:
    def __init__(self):
        project_root = settings.BASE_DIR
        tracking_uri = f"file://{project_root}/mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("ebay-first-classifier")
        self.vectorizer = None
        self.price_scaler = None
        self.clf = None
        self.training_history = []

    def normalize_title(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())
    
    def prepare_training_data(self):
        records = EbayListing.objects.filter(evaluated=True)
        if not records.exists():
            raise ValueError("no annotated listings found")
        titles, prices, wanted = [], [], []
        for r in records:
            titles.append(self.normalize_title(r.ebay_title))
            prices.append(float(r.price) if r.price else 0)
            wanted.append(1 if r.wanted else 0)
        print(f"Sample titles: {titles[:5]}")
        print(f"Empty titles: {sum(1 for t in titles if not t)}")
        print(f"Total titles: {len(titles)}")
        return titles, prices, wanted

    def train_new_model(self):
        with mlflow.start_run(run_name='initial_training'):
            titles, prices, wanted = self.prepare_training_data()
            labels = np.array(wanted)

            mlflow.log_params({
                "ngram_range": "(1, 2)",
                "max_features": 20000,
                "C": 1.0,
                "class_weight": "balanced",
                "n_total": len(labels),
                "n_positive": int(labels.sum())
            })

            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=20000,
                min_df=2,
                max_df=0.95,
            )

            X_title = self.vectorizer.fit_transform(titles)

            self.price_scaler = MinMaxScaler()
            X_price = self.price_scaler.fit_transform(np.array(prices).reshape(-1,1))

            X = hstack([X_title, X_price]).tocsr()

            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=labels
            )

            self.clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
            self.clf.fit(X_train, y_train)

            y_pred = self.clf.predict(X_test)
            y_proba = self.clf.predict_proba(X_test)[:, 1]

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            avg_precision = average_precision_score(y_test, y_proba)

            print(f"\nPrecision: {precision:.3f}")
            print(f"Recall:    {recall:.3f}")
            print(f"ROC AUC:   {auc:.3f}")
            print(f"PR AUC:    {avg_precision:.3f}")

            print("\nThreshold table:")
            for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                pred = (y_proba >= thresh).astype(int)
                if pred.sum() == 0:
                    print(f"  {thresh:.1f}: no predictions")
                    continue
                p = precision_score(y_test, pred, zero_division=0)
                r = recall_score(y_test, pred, zero_division=0)
                print(f"  {thresh:.1f}: precision={p:.3f}  recall={r:.3f}  flagged={pred.sum()}")

            mlflow.log_metrics({
                "precision": precision, 
                "recall": recall,
                "auc": auc,
                "avg_precision": avg_precision
            })

            history = {
                "n_total": len(labels),
                "n_positive": int(labels.sum()),
                "precision": float(precision),
                "recall": float(recall),
                "roc_auc": float(auc),
                "pr_auc": float(avg_precision),
            }

            self.save_model_to_db(history)
            return history
    
    def load_latest_model(self):
        try:
            latest = EbayFirstPassModel.objects.filter(is_active=True).latest('created_at')
            model_data = pickle.loads(latest.model_weights)
            self.vectorizer = model_data['vectorizer']
            self.price_scaler = model_data['price_scaler']
            self.clf = model_data['classifier']
            print(f"Loaded classifier version {latest.version}")
            return True
        except EbayFirstPassModel.DoesNotExist:
            print("No active classifier found")
            return False
        except Exception as e:
            print(f"Error loading classifier: {type(e).__name__}: {e}")
            return False


    def save_model_to_db(self, history):
        model_weights = pickle.dumps({
            "vectorizer": self.vectorizer,
            "price_scaler": self.price_scaler,
            "classifier": self.clf,
            "trained_at": django_timezone.now().isoformat(),
        })
        
        classifier = EbayFirstPassModel.objects.create(
            version=f"v{django_timezone.now().strftime('%Y%m%d_%H%M%S')}",
            model_weights=model_weights,
            hyperparams={
                "ngram_range": [1, 2],
                "max_features": 20000,
                "C": 1.0,
                "class_weight": "balanced",
                "operating_threshold": 0.3,
            },
            training_stats=history,
            is_active=True,
        )

        EbayFirstPassModel.objects.filter(is_active=True).exclude(id=classifier.id).update(is_active=False)
    
                
    def classify(self, listings):
        if not self.clf:
            if not self.load_latest_model():
                raise ValueError("no classifier available. train a new one")
        
        threshold = 0.3
        try:
            latest = EbayFirstPassModel.objects.filter(is_active=True).latest("created_at")
            threshold = latest.hyperparams.get("operating_threshold", 0.3)
        except EbayFirstPassModel.DoesNotExist:
            pass

        titles = [l.get("ebay_title", "") for l in listings]
        prices = [float(l.get("price", 0.0)) for l in listings]

        X_title = self.vectorizer.transform(titles)
        X_price = self.price_scaler.transform(np.array(prices).reshape(-1, 1))
        X = hstack([X_title, X_price]).tocsr()

        probas = self.clf.predict_proba(X)[:, 1]

        results = []
        for listing, score in zip(listings, probas):
            results.append({
                **listing,
                "keeper_score": float(score),
                "passes_threshold": float(score) >= threshold,
            })

        return results