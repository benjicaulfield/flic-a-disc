import json
from collections import Counter
from django.core.management.base import BaseCommand
from django.forms.models import model_to_dict

from bandit.models import Record
from bandit.utils.get_user_inventory import authenticate_client

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        d = authenticate_client()
        keepers = list(Record.objects.filter(wanted=True)[:1000])
        non_keepers = list(Record.objects.filter(evaluated=True, wanted=False, haves__gt=20).order_by('?')[:1000])

        for record in keepers + non_keepers:
            try:
                release = d.release(record.discogs_id)
                formats = release.formats
                if formats:
                    fmt = formats[0]
                    parts = []
                    if fmt.get('name'):
                        parts.append(fmt['name'])
                    parts.extend(fmt.get('descriptions', []))
                    record.format = parts
                else:
                    record_format = []
                record.save()
                self.stdout.write(f"Updated {record.discogs_id}: {record.format}")
            except Exception as e:
                self.stdout.write(f"Failed on {record.discogs_id}: {e}")

        keeper_formats = Counter(tuple(r.format) for r in keepers)
        self.stdout.write("\nKeeper format counts:")
        for fmt, count in keeper_formats.most_common(20):
            self.stdout.write(f"  {list(fmt)}: {count}")

        all_formats = Counter(tuple(r.format) for r in keepers + non_keepers)
        self.stdout.write("\nAll format counts:")
        for fmt, count in all_formats.most_common(20):
            self.stdout.write(f"  {list(fmt)}: {count}")

        from django.forms.models import model_to_dict

        with open('keepers.json', 'w') as f:
            json.dump([model_to_dict(r) for r in keepers], f, indent=2, default=str)

        with open('mock_api_data.json', 'w') as f:
            all_records = {str(r.discogs_id): model_to_dict(r) for r in keepers + non_keepers}
            json.dump(all_records, f, indent=2, default=str)

        self.stdout.write(f"\nSaved {len(keepers)} keepers to keepers.json")
        self.stdout.write(f"Saved {len(keepers) + len(non_keepers)} records to mock_api_data.json")
        self.stdout.write("donezo")
