from collections import defaultdict

from django.core.management.base import BaseCommand
from bandit.models import BatchPerformance, BanditTrainingInstance

class Command(BaseCommand):
    help = 'Backfill batch performances with pre-sliding window results'

    def handle(self, *args, **options):
        self.stdout.write("Backfilling batch performances...")
        instances = BanditTrainingInstance.objects.order_by('timestamp')

        batch_data = defaultdict(lambda: {'correct': 0, 'total': 0})
        batch_num = 0

        for i, instance in enumerate(instances):
            if i > 0 and i % 20 == 0:
                batch_num += 1

            batch_data[batch_num]['total'] += 1
            if instance.predicted == instance.actual:
                batch_data[batch_num]['correct'] += 1

        for batch_num, data in batch_data.items():
            accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0

            BatchPerformance.objects.get_or_create(
                batch_number = batch_num,
                defaults = {
                    'correct': data['correct'],
                    'total': data['total'],
                    'accuracy': accuracy
                }
            )

            self.stdout.write(self.style.SUCCESS(
                f'✅ Backfilled {len(batch_data)} batches'
            ))