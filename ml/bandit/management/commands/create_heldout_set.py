from django.core.management.base import BaseCommand
from django.db.models import Min
from bandit.models import DiscogsRecord, BanditTrainingInstance


class Command(BaseCommand):
    help = "Flag the most recent 20% of evaluated records as the held-out eval set. Run once."

    def handle(self, *args, **kwargs):
        if DiscogsRecord.objects.filter(heldout=True).exists():
            self.stdout.write(self.style.ERROR("Holdout set already exists. Aborting."))
            return

        # Earliest training instance timestamp per record = proxy for annotation time
        earliest = (
            BanditTrainingInstance.objects
            .values('record_id')
            .annotate(first_seen=Min('timestamp'))
        )
        timestamp_by_record = {row['record_id']: row['first_seen'] for row in earliest}

        # Evaluated records that have a training instance (i.e. have been through a batch)
        evaluated = DiscogsRecord.objects.filter(evaluated=True, id__in=timestamp_by_record.keys())
        sorted_records = sorted(evaluated, key=lambda r: timestamp_by_record[r.id])

        n_holdout = max(1, len(sorted_records) // 5)  # 20%
        holdout_records = sorted_records[-n_holdout:]
        holdout_ids = [r.id for r in holdout_records]

        DiscogsRecord.objects.filter(id__in=holdout_ids).update(heldout=True)

        keepers = sum(1 for r in holdout_records if r.wanted)
        cutoff = timestamp_by_record[holdout_records[0].id]

        self.stdout.write(self.style.SUCCESS(
            f"Flagged {n_holdout} records as holdout (out of {len(sorted_records)} evaluated with training history)\n"
            f"  Keepers: {keepers} / Non-keepers: {n_holdout - keepers}\n"
            f"  Annotation cutoff: {cutoff.strftime('%Y-%m-%d')}"
        ))
