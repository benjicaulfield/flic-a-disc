import lxml.etree as ET

from django.core.management.base import BaseCommand

from bandit.models import DiscogsRecord

BATCH_SIZE = 5000


class Command(BaseCommand):

    def handle(self, *args, **kwargs):
        record_ids_by_discogs_id = {
            discogs_id: pk
            for pk, discogs_id in DiscogsRecord.objects.values_list('id', 'discogs_id')
        }

        context = ET.iterparse('bandit/zines/masters.xml', events=('end',), tag='master')

        batch = []
        total = 0
        matched = 0

        for _, master in context:
            total += 1

            master_id = master.get('id')
            main_release = master.find('main_release')

            if master_id is not None and main_release is not None and main_release.text:
                pk = record_ids_by_discogs_id.get(main_release.text)
                if pk is not None:
                    batch.append(DiscogsRecord(pk=pk, master_id=int(master_id), is_master=True))
                    matched += 1

            master.clear()
            while master.getprevious() is not None:
                del master.getparent()[0]

            if len(batch) >= BATCH_SIZE:
                DiscogsRecord.objects.bulk_update(batch, ['master_id', 'is_master'])
                batch.clear()

            if total % 100000 == 0:
                print(f"processed {total}, matched {matched}")

        if batch:
            DiscogsRecord.objects.bulk_update(batch, ['master_id', 'is_master'])

        print(f"done: processed {total}, matched {matched}")
