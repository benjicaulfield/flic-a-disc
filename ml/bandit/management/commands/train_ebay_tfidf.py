import pickle
from datetime import datetime

from django.core.management.base import BaseCommand
from bandit.models import DiscogsRecord, TfIdfDB
from bandit.text_utils import create_mock_ebay_title
from bandit.title_vectorizer import TitleVectorizer

class Command(BaseCommand):
    help = "Train Tf-IDF vectorizer on mock Ebay titles"

    def handle(self, *args, **kwargs):
        keepers = DiscogsRecord.objects.filter(wanted=True, evaluated=True)
        self.stdout.write(f"Found {keepers.count()} keepers")

        titles = []
        for record in keepers:
            record_dict = {
                'artist': record.artist,
                'title': record.title,
                'label': record.label,
                'year': record.year,
                'genres': record.genres or [],
                'styles': record.styles or []
            }
            title = create_mock_ebay_title(record_dict)
            if title:
                titles.append(title)

        self.stdout.write(f"Generated {len(titles)} titles")

        vectorizer = TitleVectorizer(max_features=1000)
        vectorizer.fit(titles)
        keeper_embeddings = vectorizer.vectorizer.transform(titles)

        model_data = {
            'vectorizer': vectorizer,
            'keeper_embeddings': keeper_embeddings
        }
        
        weights = pickle.dumps(model_data)
        
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tfidf_model = TfIdfDB.objects.create(
            version=version,
            model_weights=weights,
            hyperparams={
                'max_features': 1000,
                'min_df': 2,
                'max_df': 0.95
            },
            training_stats={
                'num_titles': len(titles),
                'vocab_size': len(vectorizer.vectorizer.vocabulary_),
                'num_keepers': keepers.count()
            },
            is_active=True
        )

        TfIdfDB.objects.filter(is_active=True).exclude(id=tfidf_model.id).update(is_active=False)
        
        self.stdout.write(self.style.SUCCESS(
            f"Model saved to database with version {version}"
        ))




