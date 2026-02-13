# python_services/bandit/management/commands/export_model.py

from django.core.management.base import BaseCommand
from bandit.models import BanditModel
import pickle
import os

class Command(BaseCommand):
    help = 'Export trained model to pickle file'

    def handle(self, *args, **options):
        model = BanditModel.objects.get(is_active=True)
        
        # Create directory if needed
        os.makedirs('ml', exist_ok=True)
        
        # Model weights are already pickled in DB
        with open('ml/trained_model.pkl', 'wb') as f:
            f.write(model.model_weights)
        
        self.stdout.write(self.style.SUCCESS('Exported model to ml/trained_model.pkl'))