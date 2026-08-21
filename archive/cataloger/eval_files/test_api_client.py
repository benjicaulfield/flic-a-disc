"""
Test API Client backed by lp_catalog.json ground truth.

Provides the same interface as the real Discogs API but returns
pre-fetched wants/haves from the catalog, enabling deterministic evaluation.
"""

import json
from pathlib import Path


class TestAPIClient:
    """
    API client backed by catalog ground truth.

    Enforces 1,000 API call budget and returns real wants/haves
    for records in the catalog.
    """

    def __init__(self, catalog, max_calls=1000):
        """
        Initialize client with catalog data.

        Args:
            catalog: List of catalog records (with wants/haves) or path to JSON file
            max_calls: Maximum API calls allowed (default: 1000)
        """
        self.max_calls = max_calls
        self.calls_made = 0
        self.data = {}

        # Load catalog
        if isinstance(catalog, (str, Path)):
            with open(catalog) as f:
                catalog = json.load(f)

        # Index by release_id
        for record in catalog:
            rid = str(record.get('release_id'))
            self.data[rid] = {
                'release_id': int(rid),
                'wants': record['wants'],
                'haves': record['haves']
            }

        print(f"[TestAPIClient] Loaded {len(self.data):,} records")
        print(f"[TestAPIClient] Budget: {self.max_calls} API calls")

    def get_release(self, release_id):
        """
        Get release data (mimics Discogs API).

        Args:
            release_id: Discogs release ID (int or str)

        Returns:
            dict with 'release_id', 'wants', 'haves'

        Raises:
            Exception: If API budget exceeded
            KeyError: If release not in test set
        """
        self.calls_made += 1

        if self.calls_made > self.max_calls:
            raise Exception(
                f"API budget exceeded: {self.calls_made} calls made, "
                f"max allowed is {self.max_calls}"
            )

        rid = str(release_id)

        if rid not in self.data:
            raise KeyError(
                f"Release {rid} not found in test set. "
                f"Pipeline should only query records from test set."
            )

        return self.data[rid]

    def reset(self):
        """Reset call counter (for testing)."""
        self.calls_made = 0

    def get_stats(self):
        """Get client statistics."""
        return {
            'calls_made': self.calls_made,
            'max_calls': self.max_calls,
            'remaining': self.max_calls - self.calls_made,
            'test_set_size': len(self.data)
        }
