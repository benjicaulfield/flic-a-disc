# python_services/bandit/management/commands/generate_bm25_ground_truth.py

from django.core.management.base import BaseCommand
from bandit.text_utils import normalize_title, create_mock_ebay_title
import json
import csv
import math
from collections import Counter

class Command(BaseCommand):
    help = 'Generate BM25 ground truth scores'

    def handle(self, *args, **options):
        # Load keepers
        with open('data/keepers.json', 'r') as f:
            keepers = json.load(f)
        
        self.stdout.write(f"Loaded {len(keepers)} keepers")
        
        # Build corpus from mock eBay titles
        corpus = []
        for keeper in keepers:
            mock_title = create_mock_ebay_title(keeper)
            normalized = normalize_title(mock_title)
            tokens = normalized.split() if normalized else []
            corpus.append(tokens)
        
        # BM25 parameters
        k1 = 1.5
        b = 0.75
        N = len(corpus)
        avgdl = sum(len(doc) for doc in corpus) / N if N > 0 else 0
        
        # Document frequencies
        df = Counter()
        for doc in corpus:
            for term in set(doc):
                df[term] += 1
        
        # IDF scores
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        
        self.stdout.write(f"Corpus: {N} docs, avgdl={avgdl:.2f}")
        
        # Score all listings
        results = []
        with open('data/listings.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row['Title']
                ebay_id = row['EbayID']
                
                normalized = normalize_title(title)
                query = normalized.split() if normalized else []
                
                # Score against all keeper docs
                scores = []
                for doc in corpus:
                    doc_len = len(doc)
                    doc_freqs = Counter(doc)
                    score = 0
                    
                    for term in query:
                        if term not in df:
                            continue
                        
                        freq = doc_freqs.get(term, 0)
                        term_idf = idf[term]
                        
                        numerator = freq * (k1 + 1)
                        denominator = freq + k1 * (1 - b + b * (doc_len / avgdl))
                        
                        score += term_idf * (numerator / denominator)
                    
                    scores.append(score)
                
                # Mean score across corpus
                mean_score = sum(scores) / len(scores) if scores else 0
                
                results.append({
                    'ebay_id': ebay_id,
                    'title': title,
                    'score': mean_score
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 10 and bottom 10
        ground_truth = []
        
        for item in results[:25]:
            ground_truth.append({
                'ebay_id': item['ebay_id'],
                'title': item['title'],
                'expected_bm25_score': round(item['score'], 3),
                'type': 'keeper'
            })
        
        for item in results[-25:]:
            ground_truth.append({
                'ebay_id': item['ebay_id'],
                'title': item['title'],
                'expected_bm25_score': round(item['score'], 3),
                'type': 'anti-keeper'
            })
        
        # Save
        with open('data/ground_truth.json', 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS(
            f'Generated ground_truth.json with 20 test cases'
        ))
        self.stdout.write(f'Top score: {results[0]["score"]:.3f}')
        self.stdout.write(f'Bottom score: {results[-1]["score"]:.3f}')