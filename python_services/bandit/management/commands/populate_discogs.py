# bandit/management/commands/populate_discogs.py

from django.core.management.base import BaseCommand
from ...utils.populate_database import Populator

class Command(BaseCommand):
    help = "Populate Discogs listings"

    def handle(self, *args, **options):
        Populator(self).run()
