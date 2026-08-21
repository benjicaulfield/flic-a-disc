#!/usr/bin/env python3
"""
Enrich lp_catalog.json with wants/haves data by crawling seller inventories.

Uses the hack: inventory listings include wants/haves for free (250 records per API call).

Usage:
    python enrich_catalog_from_inventories.py --sellers ../ml/sellers.json --max-sellers 100
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import discogs_client
from decouple import config


class InventoryCrawler:
    """Crawl seller inventories to collect wants/haves for catalog records."""

    def __init__(self, catalog_path, sellers_path, output_path='lp_catalog_enriched.json', token_file='discogs_token.json', skip_used=True):
        self.catalog_path = Path(catalog_path)
        self.sellers_path = Path(sellers_path)
        self.output_path = Path(output_path)
        self.token_file = Path(token_file)
        self.skip_used = skip_used
        self.used_sellers_file = Path('enriched_sellers_used.json')

        # Load catalog
        print(f"Loading catalog from {self.catalog_path}...")
        with open(self.catalog_path) as f:
            self.catalog = json.load(f)
        print(f"  Loaded {len(self.catalog):,} records")

        # Load sellers
        print(f"Loading sellers from {self.sellers_path}...")
        with open(self.sellers_path) as f:
            sellers_data = json.load(f)
            # Handle both list and dict formats
            if isinstance(sellers_data, list):
                self.sellers = [s['username'] if isinstance(s, dict) else s for s in sellers_data]
            elif isinstance(sellers_data, dict):
                self.sellers = list(sellers_data.keys())
            else:
                raise ValueError(f"Unexpected sellers format: {type(sellers_data)}")
        print(f"  Loaded {len(self.sellers):,} sellers")

        # Load/filter used sellers
        self.used_sellers = self._load_used_sellers()
        if self.skip_used and self.used_sellers:
            original_count = len(self.sellers)
            self.sellers = [s for s in self.sellers if s not in self.used_sellers]
            skipped = original_count - len(self.sellers)
            if skipped > 0:
                print(f"  Skipping {skipped} already-crawled sellers ({len(self.sellers)} remaining)")
        elif self.used_sellers:
            print(f"  Found {len(self.used_sellers)} previously-crawled sellers (not skipping)")

        # Create lookup: release_id -> catalog record
        self.catalog_by_id = {str(r['release_id']): r for r in self.catalog}

        # Track which records we've enriched
        self.enriched_ids = set()

        # Stats
        self.stats = {
            'sellers_crawled': 0,
            'api_calls': 0,
            'records_enriched': 0,
            'records_skipped': 0,  # Already enriched
            'records_not_in_catalog': 0,  # In inventory but not in our catalog
        }

        # Authenticate Discogs client
        self.client = self._authenticate()

    def _load_used_sellers(self):
        """Load list of already-crawled sellers."""
        if self.used_sellers_file.exists():
            with open(self.used_sellers_file) as f:
                data = json.load(f)
                return set(data.get('sellers', []))
        return set()

    def _save_used_seller(self, username):
        """Mark a seller as crawled."""
        self.used_sellers.add(username)
        with open(self.used_sellers_file, 'w') as f:
            json.dump({
                'sellers': sorted(list(self.used_sellers)),
                'count': len(self.used_sellers),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)

    def _authenticate(self):
        """Authenticate with Discogs API."""
        consumer_key = config('DISCOGS_CONSUMER_KEY')
        consumer_secret = config('DISCOGS_CONSUMER_SECRET')

        client = discogs_client.Client('catalog-enricher/1.0')
        client.set_consumer_key(consumer_key, consumer_secret)

        # Load saved tokens
        if self.token_file.exists():
            with open(self.token_file) as f:
                tokens = json.load(f)
            client.set_token(tokens['token'], tokens['secret'])
            print("✓ Authenticated with saved tokens")
        else:
            # Need to authenticate
            token, secret, url = client.get_authorize_url()
            print(f"\nPlease visit this URL to authorize:\n{url}\n")
            verifier = input("Enter the verifier code: ")
            access_token, access_secret = client.get_access_token(verifier)
            client.set_token(access_token, access_secret)

            # Save tokens
            with open(self.token_file, 'w') as f:
                json.dump({'token': access_token, 'secret': access_secret}, f)
            print("✓ Authenticated and saved tokens")

        return client

    def crawl_seller(self, username, max_pages=100):
        """
        Crawl a single seller's inventory and enrich catalog records.

        Returns:
            dict with counts: {enriched, skipped, not_in_catalog}
        """
        print(f"\n{'='*70}")
        print(f"SELLER: {username}")
        print(f"{'='*70}")

        enriched = 0
        skipped = 0
        not_in_catalog = 0
        seller_start_time = datetime.now()

        try:
            # Get user and inventory
            user = self.client.user(username)
            inventory = user.inventory
            inventory.per_page = 250

            # Get first page to determine total pages
            self.stats['api_calls'] += 1
            first_page = inventory.page(1)
            total_pages = min(inventory.pages if hasattr(inventory, 'pages') else 100, max_pages)

            print(f"Total pages: {total_pages} (capped at {max_pages})")

            # Process all pages
            for page_num in range(1, total_pages + 1):
                print(f"  Page {page_num}/{total_pages}...", end=' ', flush=True)

                if page_num > 1:  # Already fetched page 1
                    self.stats['api_calls'] += 1
                    page = inventory.page(page_num)
                else:
                    page = first_page

                page_enriched = 0
                page_skipped = 0
                page_not_in_catalog = 0

                for listing in page:
                    try:
                        # Get release_id
                        release_id = str(listing.release.id)

                        # Skip if not in our catalog
                        if release_id not in self.catalog_by_id:
                            not_in_catalog += 1
                            page_not_in_catalog += 1
                            continue

                        # Skip if already enriched
                        if release_id in self.enriched_ids:
                            skipped += 1
                            page_skipped += 1
                            continue

                        # Extract wants/haves from listing.data (FREE!)
                        data = listing.data or {}
                        release = data.get('release') or {}
                        stats = (release.get('stats') or {}).get('community') or {}
                        wants = stats.get('in_wantlist', 0)
                        haves = stats.get('in_collection', 0)

                        # Enrich catalog record
                        self.catalog_by_id[release_id]['wants'] = wants
                        self.catalog_by_id[release_id]['haves'] = haves

                        # Mark as enriched
                        self.enriched_ids.add(release_id)
                        enriched += 1
                        page_enriched += 1

                    except Exception as e:
                        print(f"\n    Error processing listing: {e}")
                        continue

                # Calculate running stats
                total_processed = enriched + skipped + not_in_catalog
                enrichment_rate = (enriched / total_processed * 100) if total_processed > 0 else 0

                print(f"+{page_enriched} enriched, {page_skipped} skipped, {page_not_in_catalog} not in catalog")
                print(f"    Running totals: {enriched:,} enriched ({enrichment_rate:.1f}% hit rate), {len(self.enriched_ids):,} unique records")

            print(f"\n✓ Seller complete: {enriched:,} records enriched")

            # Mark seller as used
            self._save_used_seller(username)

        except Exception as e:
            print(f"\n✗ Error crawling seller {username}: {e}")
            # Still mark as used to avoid retrying failed sellers
            self._save_used_seller(username)

        return {
            'enriched': enriched,
            'skipped': skipped,
            'not_in_catalog': not_in_catalog
        }

    def crawl_sellers(self, max_sellers=None, max_pages_per_seller=100):
        """Crawl multiple sellers' inventories."""
        sellers_to_crawl = self.sellers[:max_sellers] if max_sellers else self.sellers

        print(f"\n{'='*70}")
        print(f"CRAWLING {len(sellers_to_crawl):,} SELLERS")
        print(f"{'='*70}")
        print(f"Target: Enrich {len(self.catalog):,} catalog records with wants/haves")
        print(f"Max pages per seller: {max_pages_per_seller}")
        print()

        start_time = datetime.now()

        for i, username in enumerate(sellers_to_crawl, 1):
            result = self.crawl_seller(username, max_pages=max_pages_per_seller)

            self.stats['sellers_crawled'] += 1
            self.stats['records_enriched'] += result['enriched']
            self.stats['records_skipped'] += result['skipped']
            self.stats['records_not_in_catalog'] += result['not_in_catalog']

            # Progress update
            elapsed = (datetime.now() - start_time).total_seconds()
            coverage = len(self.enriched_ids) / len(self.catalog) * 100

            print(f"\n{'='*70}")
            print(f"PROGRESS: {i}/{len(sellers_to_crawl)} sellers crawled")
            print(f"{'='*70}")
            print(f"  Coverage:      {len(self.enriched_ids):,} / {len(self.catalog):,} ({coverage:.1f}%)")
            print(f"  API calls:     {self.stats['api_calls']:,}")
            print(f"  Records/call:  {len(self.enriched_ids) / max(self.stats['api_calls'], 1):.1f}")
            print(f"  Elapsed:       {elapsed/60:.1f} minutes")

            # Save progress every 10 sellers
            if i % 10 == 0:
                self.save_progress(f"{self.output_path}.progress")
                print(f"  ✓ Progress saved to {self.output_path}.progress")

        # Final save
        self.save_final()

        # Final stats
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n{'='*70}")
        print(f"CRAWL COMPLETE")
        print(f"{'='*70}")
        print(f"Sellers crawled:        {self.stats['sellers_crawled']:,}")
        print(f"API calls made:         {self.stats['api_calls']:,}")
        print(f"Records enriched:       {self.stats['records_enriched']:,}")
        print(f"Coverage:               {len(self.enriched_ids):,} / {len(self.catalog):,} ({len(self.enriched_ids)/len(self.catalog)*100:.1f}%)")
        print(f"Records per API call:   {len(self.enriched_ids) / max(self.stats['api_calls'], 1):.1f}")
        print(f"Time:                   {elapsed/60:.1f} minutes")
        print(f"Output:                 {self.output_path}")
        print(f"{'='*70}")

    def save_progress(self, filepath):
        """Save enriched catalog to file (for checkpointing)."""
        enriched_catalog = [
            r for r in self.catalog
            if str(r['release_id']) in self.enriched_ids
        ]

        with open(filepath, 'w') as f:
            json.dump({
                'enriched_count': len(enriched_catalog),
                'total_catalog': len(self.catalog),
                'coverage_pct': len(self.enriched_ids) / len(self.catalog) * 100,
                'stats': self.stats,
                'enriched_records': enriched_catalog
            }, f, indent=2)

    def save_final(self):
        """Save final enriched catalog."""
        # Separate enriched and unenriched
        enriched = []
        unenriched = []

        for record in self.catalog:
            if str(record['release_id']) in self.enriched_ids:
                enriched.append(record)
            else:
                unenriched.append(record)

        # Save enriched records
        with open(self.output_path, 'w') as f:
            json.dump(enriched, f, indent=2)

        print(f"\n✓ Saved {len(enriched):,} enriched records to {self.output_path}")

        # Save unenriched separately (for analysis)
        unenriched_path = self.output_path.with_name(f"{self.output_path.stem}_unenriched.json")
        with open(unenriched_path, 'w') as f:
            json.dump(unenriched, f, indent=2)

        print(f"✓ Saved {len(unenriched):,} unenriched records to {unenriched_path}")

        # Save metadata
        metadata_path = self.output_path.with_name(f"{self.output_path.stem}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'catalog_path': str(self.catalog_path),
                'sellers_path': str(self.sellers_path),
                'enriched_count': len(enriched),
                'unenriched_count': len(unenriched),
                'coverage_pct': len(enriched) / len(self.catalog) * 100,
                'stats': self.stats
            }, f, indent=2)

        print(f"✓ Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Enrich lp_catalog.json with wants/haves from seller inventories'
    )
    parser.add_argument(
        '--catalog',
        default='lp_catalog.json',
        help='Path to lp_catalog.json (default: lp_catalog.json)'
    )
    parser.add_argument(
        '--sellers',
        default='../ml/sellers.json',
        help='Path to sellers.json (default: ../ml/sellers.json)'
    )
    parser.add_argument(
        '--output',
        default='lp_catalog_enriched.json',
        help='Output file for enriched catalog (default: lp_catalog_enriched.json)'
    )
    parser.add_argument(
        '--max-sellers',
        type=int,
        help='Maximum number of sellers to crawl (default: all)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=100,
        help='Maximum pages per seller (default: 100)'
    )
    parser.add_argument(
        '--no-skip-used',
        action='store_true',
        help='Do not skip previously-crawled sellers (default: skip used sellers)'
    )

    args = parser.parse_args()

    crawler = InventoryCrawler(
        catalog_path=args.catalog,
        sellers_path=args.sellers,
        output_path=args.output,
        skip_used=not args.no_skip_used
    )

    crawler.crawl_sellers(
        max_sellers=args.max_sellers,
        max_pages_per_seller=args.max_pages
    )


if __name__ == '__main__':
    main()

