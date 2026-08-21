import json
import gzip
import time
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
import lxml.etree as ET
from pybloom_live import BloomFilter
import discogs_client


class DiscogsLPPipeline:
    def __init__(self, data_dir, output_dir, use_mock: bool = False, mock_url: str = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.use_mock = use_mock
        self.mock_url = mock_url

        # Lazy load Discogs client (only needed for Step 2)
        self._client = None

        # Bloom filters for step 2
        self.cleared_filter = BloomFilter(capacity=500000, error_rate=0.001)
        self.blocked_filter = BloomFilter(capacity=500000, error_rate=0.001)

        # Track master_id to release_ids mapping for sibling propagation
        self.master_to_releases: Dict[str, Set[str]] = defaultdict(set)

        # Decision log
        self.decisions = []

    @property
    def client(self):
        """Lazy load Discogs client when first accessed"""
        if self._client is None:
            if self.use_mock and self.mock_url:
                self._client = None  # Mock API uses requests directly
            else:
                self._client = discogs_client.Client("DiscogsDump/1.0")
        return self._client

    def log_decision(self, decision: str):
        """Log a normalization decision"""
        self.decisions.append(decision)
        print(f"[DECISION] {decision}")

    def is_lp_release(self, formats):
        if not formats:
            return False

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

    def step1_build_catalog(self, releases_file="releases.xml.gz"):
        """
        Step 1: Stream-parse releases.xml.gz and extract LP releases
        Returns path to output file
        """
        print("\n" + "="*60)
        print("STEP 1: BUILDING CLEAN LP CATALOG")
        print("="*60)

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

                    formats = release.findall('formats/format')
                    if not self.is_lp_release(formats):
                        release.clear()
                        continue

                    release_id = release.get('id', '')
                    master_id = self.extract_text(release.find('master_id'), None)
                    artist = self.extract_artist(release)
                    title = self.extract_text(release.find('title'), "Unknown Title")

                    labels = release.find('labels')
                    label = "Unknown Label"
                    catalog_number = None
                    if labels is not None:
                        label_elem = labels.find('label')
                        if label_elem is not None:
                            label = self.extract_text(label_elem.find('.'), label_elem.get('name', 'Unknown Label'))
                            catalog_number = label_elem.get('catno', None)

                    year = self.extract_text(release.find('released'), None)
                    if year:
                        # Extract just the year (format: YYYY-MM-DD or YYYY)
                        year = year.split('-')[0] if '-' in year else year

                    country = self.extract_text(release.find('country'), None)
                    genres, styles = self.extract_genres_styles(release)

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

    def validate_keepers(self, catalog_path: Path):
        """Validate that all keeper release_ids are in the catalog"""
        print("\nValidating against keepers.json...")

        keepers_path = self.data_dir / "keepers.json"
        with open(keepers_path) as f:
            keepers = json.load(f)

        keeper_ids = {k.get('discogs_id', k.get('release_id', '')) for k in keepers}
        keeper_ids = {str(kid) for kid in keeper_ids if kid}

        # Load catalog IDs
        catalog_ids = set()
        with open(catalog_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    catalog_ids.add(record['release_id'])

        missing = keeper_ids - catalog_ids

        if missing:
            print(f"✗ WARNING: {len(missing)} keeper IDs missing from catalog!")
            print(f"  Missing IDs: {list(missing)[:10]}...")
        else:
            print(f"✓ All {len(keeper_ids)} keeper records found in catalog")

    # ==================== STEP 2: ENRICH WITH API DATA ====================

    def check_bloom_filters(self, release_id: str, master_id: Optional[str]) -> Optional[str]:
        """
        Check if release or master is in bloom filters.
        Returns: 'cleared', 'blocked', or None if not found
        """
        # Check release_id
        if release_id in self.cleared_filter:
            return 'cleared'
        if release_id in self.blocked_filter:
            return 'blocked'

        # Check master_id
        if master_id:
            if master_id in self.cleared_filter:
                return 'cleared'
            if master_id in self.blocked_filter:
                return 'blocked'

        return None

    def propagate_to_siblings(self, master_id: str, status: str):
        """Propagate bloom filter status to all sibling releases"""
        if not master_id or master_id not in self.master_to_releases:
            return

        sibling_ids = self.master_to_releases[master_id]
        target_filter = self.cleared_filter if status == 'cleared' else self.blocked_filter

        for sibling_id in sibling_ids:
            target_filter.add(sibling_id)

    def fetch_release_data(self, release_id: str) -> Optional[Dict]:
        """Fetch release data from Discogs API or mock API"""
        if self.use_mock and self.mock_url:
            # Use mock API with session for connection pooling
            if not hasattr(self, '_http_session'):
                import requests
                self._http_session = requests.Session()

            try:
                response = self._http_session.get(f"{self.mock_url}/releases/{release_id}", timeout=5)
                if response.status_code == 200:
                    return response.json()
                return None
            except Exception as e:
                # Don't print every error - too many 404s
                return None
        else:
            # Use real Discogs API
            try:
                release = self.client.release(int(release_id))
                data = {
                    'id': release.id,
                    'title': release.title,
                    'artists': [{'name': a.name} for a in release.artists] if hasattr(release, 'artists') else [],
                    'labels': [{'name': l.name, 'catno': l.catno} for l in release.labels] if hasattr(release, 'labels') else [],
                    'formats': [{'name': f['name'], 'descriptions': f.get('descriptions', [])} for f in release.formats] if hasattr(release, 'formats') else [],
                    'genres': release.genres if hasattr(release, 'genres') else [],
                    'styles': release.styles if hasattr(release, 'styles') else [],
                    'year': release.year if hasattr(release, 'year') else None,
                    'master_id': release.master.id if hasattr(release, 'master') and release.master else None,
                    'community': {
                        'want': release.marketplace_stats.want if hasattr(release, 'marketplace_stats') else 0,
                        'have': release.marketplace_stats.have if hasattr(release, 'marketplace_stats') else 0,
                    },
                    'lowest_price': release.marketplace_stats.lowest_price if hasattr(release, 'marketplace_stats') else None,
                    'images': [{'uri': img['uri']} for img in release.images] if hasattr(release, 'images') else [],
                }
                return data
            except Exception as e:
                print(f"Error fetching {release_id}: {e}")
                return None

    def step2_enrich_catalog(self, catalog_path: Path, max_requests: int = None) -> Path:
        """
        Step 2: Enrich catalog with API data using bloom filters
        Returns path to enriched output
        """
        print("\n" + "="*60)
        print("STEP 2: ENRICHING WITH API DATA")
        print("="*60)

        output_path = self.output_dir / ("lp_catalog_verified.json" if self.use_mock else "lp_catalog_enriched.json")

        # Load catalog
        catalog = []
        with open(catalog_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    catalog.append(record)
                    # Rebuild master_to_releases mapping
                    if record.get('master_id'):
                        self.master_to_releases[record['master_id']].add(record['release_id'])

        print(f"Loaded {len(catalog)} LP releases")
        print(f"Rebuilt {len(self.master_to_releases)} master ID mappings")

        # Prioritization: releases with master_ids first, then by genre diversity
        def priority_score(record):
            has_master = 1 if record.get('master_id') else 0
            genre_count = len(record.get('genre', []))
            style_count = len(record.get('style', []))
            return (has_master, genre_count + style_count)

        catalog.sort(key=priority_score, reverse=True)
        print("Sorted catalog by priority (master_id, metadata richness)")

        # Enrich
        enriched = []
        api_calls = 0
        request_count = 0
        start_time = time.time()

        for i, record in enumerate(catalog):
            if max_requests and api_calls >= max_requests:
                print(f"\nReached max API requests ({max_requests})")
                break

            release_id = record['release_id']
            master_id = record.get('master_id')

            # Check bloom filters
            cached_status = self.check_bloom_filters(release_id, master_id)

            if cached_status == 'cleared':
                # Already cleared (inferred from sibling)
                enriched_record = self.build_enriched_record(record, None, 'inferred_cleared')
                enriched.append(enriched_record)
                continue
            elif cached_status == 'blocked':
                # Already blocked (inferred from sibling)
                # Skip this record (don't include in output per requirements)
                continue

            # Need to fetch from API
            api_calls += 1
            request_count += 1

            # Rate limiting: 60 requests per minute
            if not self.use_mock and request_count >= 60:
                elapsed = time.time() - start_time
                if elapsed < 60:
                    sleep_time = 60 - elapsed
                    print(f"\nRate limit: sleeping {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                request_count = 0
                start_time = time.time()

            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{len(catalog)} ({api_calls} API calls, {len(enriched)} enriched)")

            # Fetch data
            api_data = self.fetch_release_data(release_id)

            if not api_data:
                continue

            # Check wants > haves
            wants = api_data.get('community', {}).get('want', 0)
            haves = api_data.get('community', {}).get('have', 0)

            if wants > haves:
                # Cleared
                status = 'enriched'
                self.cleared_filter.add(release_id)
                if master_id:
                    self.cleared_filter.add(master_id)
                    self.propagate_to_siblings(master_id, 'cleared')

                enriched_record = self.build_enriched_record(record, api_data, status)
                enriched.append(enriched_record)
            else:
                # Blocked
                self.blocked_filter.add(release_id)
                if master_id:
                    self.blocked_filter.add(master_id)
                    self.propagate_to_siblings(master_id, 'blocked')
                # Don't add to output

        # Sort enriched records deterministically by discogs_id
        enriched.sort(key=lambda x: int(x['discogs_id']))

        # Write enriched catalog
        with open(output_path, 'w') as f:
            json.dump(enriched, f, indent=2, sort_keys=True)

        print(f"\n✓ Enriched {len(enriched)} releases with {api_calls} API calls")
        print(f"✓ Output: {output_path}")

        return output_path

    def build_enriched_record(self, record: Dict, api_data: Optional[Dict], status: str) -> Dict:
        """Build enriched record according to schema"""
        # Use deterministic timestamp for mock API verification
        timestamp = "2025-01-01T00:00:00Z" if self.use_mock else datetime.utcnow().isoformat() + 'Z'

        if api_data:
            # Full enrichment from API
            artists = api_data.get('artists', [])
            artist = artists[0]['name'] if artists else record.get('artist', 'Unknown Artist')

            labels = api_data.get('labels', [])
            label = labels[0]['name'] if labels else record.get('label', 'Unknown Label')
            catno = labels[0].get('catno') if labels else record.get('catalog_number')

            formats = api_data.get('formats', [])
            format_list = [f['name'] for f in formats] if formats else ['Vinyl']

            images = api_data.get('images', [])
            record_image = images[0]['uri'] if images else None

            wants = api_data.get('community', {}).get('want', 0)
            haves = api_data.get('community', {}).get('have', 0)
            suggested_price = str(api_data.get('lowest_price', '')) if api_data.get('lowest_price') else ''

            return {
                'discogs_id': record['release_id'],
                'artist': artist,
                'title': record['title'],
                'format': format_list,
                'label': label,
                'catno': catno,
                'wants': wants,
                'haves': haves,
                'added': timestamp,
                'genres': api_data.get('genres', record.get('genre', [])),
                'styles': api_data.get('styles', record.get('style', [])),
                'suggested_price': suggested_price,
                'year': api_data.get('year', record.get('year')),
                'record_image': record_image,
                'description': None,
                'wanted': False,
                'evaluated': True,
                'status': status
            }
        else:
            # Inferred from bloom filter
            return {
                'discogs_id': record['release_id'],
                'artist': record.get('artist', 'Unknown Artist'),
                'title': record['title'],
                'format': ['Vinyl'],
                'label': record.get('label', 'Unknown Label'),
                'catno': record.get('catalog_number'),
                'wants': 0,
                'haves': 0,
                'added': timestamp,
                'genres': record.get('genre', []),
                'styles': record.get('style', []),
                'suggested_price': '',
                'year': record.get('year'),
                'record_image': None,
                'description': None,
                'wanted': False,
                'evaluated': False,
                'status': status
            }

    # ==================== STEP 3: VALIDATION ====================

    def step3_compute_checksum(self, verified_path: Path) -> str:
        """Compute SHA-256 checksum of verified output"""
        print("\n" + "="*60)
        print("STEP 3: COMPUTING CHECKSUM")
        print("="*60)

        with open(verified_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        checksum_path = self.output_dir / "checksum.txt"
        with open(checksum_path, 'w') as f:
            f.write(checksum + '\n')

        print(f"✓ SHA-256: {checksum}")
        print(f"✓ Written to: {checksum_path}")

        # Compare with target
        target_checksum_path = Path(__file__).parent / "verification" / "checksum.txt"
        if target_checksum_path.exists():
            with open(target_checksum_path) as f:
                target = f.read().strip()

            if checksum == target:
                print("✓ CHECKSUM MATCH! Validation successful.")
            else:
                print(f"✗ CHECKSUM MISMATCH")
                print(f"  Expected: {target}")
                print(f"  Got:      {checksum}")

        return checksum

    def run_full_pipeline(self, skip_step1: bool = False, releases_file: str = "releases.xml.gz"):
        """Run the complete pipeline"""
        print("\n" + "="*60)
        print("DISCOGS LP CATALOG PIPELINE")
        print("="*60)

        # Step 1: Build catalog
        if skip_step1:
            catalog_path = self.output_dir / "lp_catalog.json"
            print(f"\nSkipping Step 1, using existing: {catalog_path}")
        else:
            catalog_path = self.step1_build_catalog(releases_file=releases_file)

        # Step 2: Enrich
        enriched_path = self.step2_enrich_catalog(catalog_path)

        # Step 3: Checksum
        if self.use_mock:
            self.step3_compute_checksum(enriched_path)

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discogs LP Catalog Pipeline")
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Data directory')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='Output directory')
    parser.add_argument('--releases-file', type=str, default='releases.xml.gz', help='Releases filename (default: releases.xml.gz)')
    parser.add_argument('--mock', action='store_true', help='Use mock API for validation')
    parser.add_argument('--mock-url', type=str, default='http://localhost:5000', help='Mock API URL')
    parser.add_argument('--skip-step1', action='store_true', help='Skip step 1 if catalog exists')
    parser.add_argument('--max-requests', type=int, help='Max API requests (for testing)')

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    pipeline = DiscogsLPPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_mock=args.mock,
        mock_url=args.mock_url if args.mock else None
    )

    pipeline.run_full_pipeline(skip_step1=args.skip_step1, releases_file=args.releases_file)


if __name__ == '__main__':
    main()
