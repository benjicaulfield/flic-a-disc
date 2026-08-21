from pathlib import Path
from collections import defaultdict
import os
import sys
import django
import gzip
import json
import lxml.etree as ET

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from bandit.models import Record

EXCLUDED_GENRES = {'Childrens', 'Brass & Military'}
EXCLUDED_STYLES = {'Broken Beat', 'Tech House', 'Tribal House',
                        'Drum n Bass', 'Downtempo', 'Dub Techno',
                        'IDM', 'Garage House', 'Schlager', 
                        'Black Metal', 'Death Metal', 'Thrash',
                        'Holiday', 'Opera', 'Doom Metal', 'Rockabilly',
                        'Comedy', 'Grindcore', 'Spoken Word', 'Dubstep',
                        'J-pop', 'K-pop', 'Choral', 'Hard House', 'Trip Hop',
                        'Volksmusik', 'EBN', 'Jungle', 'Story', 'Trap', 'Oi',
                        'Radioplay', 'Celtic', 'Bluegrass', 'Metalcore', 'Laiko',
                        'Happy Hardcore', 'Polka', 'UK Garage', 'Novelty',
                        'Progressive Metal', 'Nu Metal', 'Dixieland', 'Parody',
                        'Power Metal', 'Audiobook', 'Gabber', 'Speed Metal',
                        'Hi NRG', 'Poetry', 'Acid House', 'Marches', 'Renaissance',
                        'Chiptune', 'Interview', 'Education', 'Crust', 'Operetta',
                        'Video Game Music', 'Goregrind', 'Deep House', 'Techno',
                        'Acid', 'Progressive Trance', 'Hip-House', 'Bleep', 'Hardcore',
                        'Electro', 'Breaks', 'Tribal', 'Ghettotech', 'House', 'Trance',
                        'Tech Trance', 'Stage & Screen', 'La\u00efk\u00f3', 'Romantic',
                        'House'
                    }

class XMLParsingPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.output_dir = data_dir
        self.master_to_releases = defaultdict(set)

    def is_lp(self, formats):
        if not formats: return False

        for fmt in formats:
            name = fmt.get('name', '').lower()
            descriptions = [desc.text.lower() for desc in fmt.findall('descriptions/description') if desc.text]

            if 'vinyl' not in name:
                continue
            if any(excl in descriptions for excl in ['single', '7"', '10"', 'ep', 'maxi-single', 'mini-album']):
                continue
            if '33 ⅓ rpm' in descriptions or 'lp' in descriptions:
                return True

        return False
    
    
    def extract_text(self, element, default = ""):
        if element is not None and element.text:
            return element.text.strip()
        return default

    def extract_artist(self, release):
        artists = release.find('artists')
        if artists is not None:
            artist_names = []
            for artist in artists.findall('artist'):
                name_elem = artist.find('name')
                if name_elem is not None and name_elem.text:
                    artist_names.append(name_elem.text.strip())
            if artist_names:
                return ', '.join(artist_names)
        return "Unknown Artist"

    def extract_genres_styles(self, release):
        genres = []
        styles = []

        genres_elem = release.find('genres')
        if genres_elem is not None:
            genres = [g.text.strip() for g in genres_elem.findall('genre') if g.text]

        styles_elem = release.find('styles')
        if styles_elem is not None:
            styles = [s.text.strip() for s in styles_elem.findall('style') if s.text]

        return genres, styles
    
    def build_catalog(self, releases_file="releases.xml.gz"):
        releases_path = self.data_dir / releases_file
        output_path = self.output_dir / "lp_catalog.json"
        
        lp_count = 0
        total_count = 0

        with gzip.open(releases_path, 'rb') as gz_file:
            with open(output_path, 'w') as out_file:
                context = ET.iterparse(gz_file, events=('end',), tag='release')

                for _, release in context:
                    total_count += 1

                    if total_count % 100000 == 0:
                        print(f"Processed {total_count} releases, found {lp_count} LPs...")

                    release_id = release.get('id', '')
                    if Record.objects.filter(discogs_id=release_id).exists():
                        release.clear()
                        continue

                    formats = release.findall('formats/format')
                    if not self.is_lp(formats):
                        release.clear()
                        continue

                    master_id = self.extract_text(release.find('master_id'), None)
                    artist = self.extract_artist(release)
                    title = self.extract_text(release.find('title'), "Unknown Title")

                    labels = release.find('labels')
                    label = "Unknown Label"
                    catalog_number = None
                    if labels is not None:
                        label_elem = labels.find('label')
                        if label_elem is not None:
                            label = label_elem.get('name', 'Unknown Label')
                            catalog_number = label_elem.get('catno') or None
                            if catalog_number and catalog_number.lower() == 'none':
                                catalog_number = None

                    year = self.extract_text(release.find('released'), None)
                    if year:
                        # Extract just the year (format: YYYY-MM-DD or YYYY)
                        year = year.split('-')[0] if '-' in year else year

                    country = self.extract_text(release.find('country'), None)
                    genres, styles = self.extract_genres_styles(release)
                    if genres & EXCLUDED_GENRES or styles & EXCLUDED_STYLES:
                        continue

                    # Build record
                    record = {
                        'release_id': release_id,
                        'master_id': master_id,
                        'artist': artist,
                        'title': title,
                        'label': label,
                        'catalog_number': catalog_number,
                        'year': year,
                        'country': country,
                        'genre': genres,
                        'style': styles
                    }

                    # Track master_id mapping
                    if master_id:
                        self.master_to_releases[master_id].add(release_id)

                    # Write as newline-delimited JSON
                    out_file.write(json.dumps(record) + '\n')
                    lp_count += 1

                    # Clear element to free memory
                    release.clear()
                    while release.getprevious() is not None:
                        del release.getparent()[0]

        print(f"\n✓ Extracted {lp_count} LPs from {total_count} total releases")
        print(f"✓ Output: {output_path}")

        return output_path

def filter_styles_and_genres():
    excluded_genres = {'Childrens', 'Brass & Military'}
    excluded_styles = {'Broken Beat', 'Tech House', 'Tribal House',
                    'Drum n Bass', 'Downtempo', 'Dub Techno',
                    'IDM', 'Garage House', 'Schlager', 
                    'Black Metal', 'Death Metal', 'Thrash',
                    'Holiday', 'Opera', 'Doom Metal', 'Rockabilly',
                    'Comedy', 'Grindcore', 'Spoken Word', 'Dubstep',
                    'J-pop', 'K-pop', 'Choral', 'Hard House', 'Trip Hop',
                    'Volksmusik', 'EBN', 'Jungle', 'Story', 'Trap', 'Oi',
                    'Radioplay', 'Celtic', 'Bluegrass', 'Metalcore', 'Laiko',
                    'Happy Hardcore', 'Polka', 'UK Garage', 'Novelty',
                    'Progressive Metal', 'Nu Metal', 'Dixieland', 'Parody',
                    'Power Metal', 'Audiobook', 'Gabber', 'Speed Metal',
                    'Hi NRG', 'Poetry', 'Acid House', 'Marches', 'Renaissance',
                    'Chiptune', 'Interview', 'Education', 'Crust', 'Operetta',
                    'Video Game Music', 'Goregrind', 'Deep House', 'Techno',
                    'Acid', 'Progressive Trance', 'Hip-House', 'Bleep', 'Hardcore',
                    'Electro', 'Breaks', 'Tribal', 'Ghettotech', 'House', 'Trance',
                    'Tech Trance', 'Stage & Screen', 'La\u00efk\u00f3', 'Romantic',
                    'House'
                    
            }

    kept = 0
    dropped = 0

    input_path = Path("data/lp_catalog.json")
    output_path = Path("data/lp_catalog_filtered_four.json")

    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            record = json.loads(line)
            genres = set(record.get('genre', []))
            styles = set(record.get('style', []))

            if genres & excluded_genres or styles & excluded_styles:
                dropped += 1
                if dropped % 100000 == 0:
                    print(f"{dropped} records dropped")
                continue

            f_out.write(line)
            kept += 1
            if kept % 100000 == 0:
                print(f"{kept} records kept")

    print(f"Kept: {kept}")

if __name__ == "__main__":
    filter_styles_and_genres()
