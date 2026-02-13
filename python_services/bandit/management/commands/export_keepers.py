# python_services/bandit/management/commands/export_keepers.py

from django.core.management.base import BaseCommand
from bandit.models import Record
import json

class Command(BaseCommand):
    help = 'Export keeper records to JSON'

    def handle(self, *args, **options):
        keepers = Record.objects.filter(wanted=True, evaluated=True)
        
        data = []
        for record in keepers:
            data.append({
                'artist': record.artist or '',
                'title': record.title or '',
                'label': record.label or '',
                'year': record.year or 0
            })
        
        with open('data/keepers.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS(
            f'Exported {len(data)} keepers to data/keepers.json'
        ))