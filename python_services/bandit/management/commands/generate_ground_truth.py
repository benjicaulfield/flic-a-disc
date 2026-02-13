# python_services/bandit/management/commands/generate_ground_truth.py

from django.core.management.base import BaseCommand
from bandit.models import TfIdfDB
from bandit.text_utils import normalize_title
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import csv
import json

class Command(BaseCommand):
    help = 'Generate ground truth JSON from TF-IDF scores'

    def handle(self, *args, **options):
        # Load TF-IDF model
        tfidf_model = TfIdfDB.objects.get(is_active=True)
        vectorizer = pickle.loads(tfidf_model.model_weights)  # It's the vectorizer directly
        
        # Read and score listings
        results = []
        with open('data/listings.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row['Title']
                ebay_id = row['EbayID']
                
                normalized = normalize_title(title)
                if not normalized:
                    continue
                
                # Transform with vectorizer
                embedding = vectorizer.transform(normalized)
                # Score is the magnitude or a simple heuristic
                score = float(embedding.sum())  # Simple scoring
                
                results.append({'ebay_id': ebay_id, 'title': title, 'score': score})
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 10 and bottom 10
        ground_truth = []
        
        for listing in results[:25]:
            ground_truth.append({
                'ebay_id': listing['ebay_id'],
                'title': listing['title'],
                'expected_bm25_score': round(listing['score'], 3),
                'type': 'keeper'
            })
        
        for listing in results[-25:]:
            ground_truth.append({
                'ebay_id': listing['ebay_id'],
                'title': listing['title'],
                'expected_bm25_score': round(listing['score'], 3),
                'type': 'anti-keeper'
            })
        
        # Write
        with open('data/ground_truth.json', 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS('Generated ground_truth.json'))