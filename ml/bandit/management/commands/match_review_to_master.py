import csv
import re
from difflib import SequenceMatcher

from django.core.management.base import BaseCommand

from bandit.models import DiscogsRecord

CSV_PATH = 'bandit/zines/answer_keys/agent-review-results.csv'


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\s*\(\d+\)$', '', text)  # strip discogs disambiguation suffix e.g. "(2)"
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def score(row, candidate):
    artist_score = SequenceMatcher(None, normalize(row['artist']), normalize(candidate.artist)).ratio()
    title_score = SequenceMatcher(None, normalize(row['title']), normalize(candidate.title)).ratio()
    return (artist_score + title_score) / 2


class Command(BaseCommand):

    def handle(self, *args, **kwargs):
        with open(CSV_PATH, newline='') as f:
            rows = list(csv.DictReader(f))

        matched = 0
        unmatched = []

        for row in rows:
            title_norm = normalize(row['title'])
            candidates = list(DiscogsRecord.objects.filter(
                is_master=True,
                title__iregex=r'^\s*' + re.escape(row['title'].strip()) + r'\s*$',
            ))
            if not candidates:
                candidates = list(DiscogsRecord.objects.filter(is_master=True, title__icontains=title_norm.split(' ')[0]))

            best, best_score = None, 0.0
            for candidate in candidates:
                s = score(row, candidate)
                if s > best_score:
                    best, best_score = candidate, s

            if best and best_score >= 0.85:
                row['match_id'] = best.discogs_id
                matched += 1
            else:
                unmatched.append((row['zine'], row['artist'], row['title'], round(best_score, 2) if best else 0))

        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['zine', 'artist', 'title', 'page', 'match_id'])
            writer.writeheader()
            writer.writerows(rows)

        print(f'matched {matched}/{len(rows)}')
        if unmatched:
            print('unmatched:')
            for zine, artist, title, s in unmatched:
                print(f'  [{zine}] {artist} - {title} (best score {s})')
