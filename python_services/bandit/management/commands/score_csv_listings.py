# python_services/bandit/management/commands/score_csv_listings.py

from django.core.management.base import BaseCommand
from bandit.models import TfIdfDB
from bandit.text_utils import normalize_title
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import csv

class Command(BaseCommand):
    help = 'Score listings from CSV with TF-IDF'

    def handle(self, *args, **options):
        # Load TF-IDF model
        tfidf_model = TfIdfDB.objects.get(is_active=True)
        model_data = pickle.loads(tfidf_model.model_weights)
        vectorizer = model_data['vectorizer']
        keeper_embeddings = model_data['keeper_embeddings']
        
        # Read listings CSV
        results = []
        with open('data/listings.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row['Title']
                ebay_id = row['EbayID']
                
                normalized = normalize_title(title)
                if not normalized:
                    results.append({'ebay_id': ebay_id, 'title': title, 'score': 0.0})
                    continue
                
                # Transform and score
                embedding = vectorizer.vectorizer.transform([normalized])
                similarities = cosine_similarity(embedding, keeper_embeddings)
                score = float(similarities.max())
                
                results.append({'ebay_id': ebay_id, 'title': title, 'score': score})
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Write scored output
        with open('tfidf_scores.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ebay_id', 'title', 'score'])
            writer.writeheader()
            writer.writerows(results)
        
        self.stdout.write(f"Scored {len(results)} listings")
        self.stdout.write(f"Top 25 scores: {[r['score'] for r in results[:25]]}")
        self.stdout.write(f"Bottom 25 scores: {[r['score'] for r in results[-25:]]}")