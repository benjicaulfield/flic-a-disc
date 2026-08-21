from django.core.management.base import BaseCommand
from bandit.models import EbayListing
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import numpy as np

class Command(BaseCommand):
    help = 'Train eBay bandit model'

    def handle(self, *args, **options):
        listings = list(EbayListing.objects.filter(evaluated=True))
        
        titles = [l.title for l in listings]
        prices = np.array([float(l.price) for l in listings]).reshape(-1, 1)
        labels = np.array([1.0 if l.wanted else 0.0 for l in listings])
        
        self.stdout.write(f"Total: {len(listings)}, Keepers: {labels.sum()}, Keeper rate: {labels.mean()*100:.2f}%")
        
        # TF-IDF on titles
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        title_features = vectorizer.fit_transform(titles).toarray()
        
        # Combine title + price
        features = np.hstack([title_features, prices])
        
        self.stdout.write(f"Feature shape: {features.shape}")
        
        # Train simple model
        # TODO: Add actual training loop here
        
        self.stdout.write(self.style.SUCCESS('✅ Features extracted'))