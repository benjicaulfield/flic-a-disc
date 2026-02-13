from django.core.management.base import BaseCommand
from bandit.training import BanditTrainer
import json
import torch

class Command(BaseCommand):
    help = 'Add prediction scores to listings.json'

    def handle(self, *args, **options):
        with open('bandit/listings.json', 'r') as f:
            listings = json.load(f)

        trainer = BanditTrainer()
        if not trainer.model:
            trainer.load_latest_model()

        records = [{
            'artist': l['artist'],
            'title': l['title'],
            'label': l['label'],
            'genres': l['genres'],
            'styles': l['styles'],
            'wants': l['wants'],
            'haves': l['haves'],
            'year': l['year']
        } for l in listings]
        
        features = trainer.feature_extractor.extract_batch_features(records)
        features_tensor = torch.FloatTensor(features)
        
        with torch.no_grad():
            mean_probs, _ = trainer.model.predict_with_uncertainty(features_tensor)
            predictions = mean_probs.cpu().numpy()
        
        for i, listing in enumerate(listings):
            listing['score'] = float(predictions[i])

        with open('bandit/listings.json', 'w') as f:
            json.dump(listings, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f'Added scores to {len(listings)} listings'))