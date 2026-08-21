"""
Integration tests for the by-seller feature.

Tests the full pipeline:
  ML (get_inventory → condition filter → non_embedding_scoring) → response shape
  Type consistency between ML response, Go struct, and TypeScript interface

Run with: uv run python manage.py test bandit.tests.test_by_seller
"""
import json
from unittest.mock import patch, MagicMock
from django.test import TestCase
from rest_framework.test import APIClient

from bandit.utils.non_embedding_scoring import non_embedding_scoring, demands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_listing(record_price="20.0, USD", wants=40, haves=20,
                 condition="Very Good Plus (VG+)", fmt=None):
    return {
        'discogs_id': '123456',
        'media_condition': condition,
        'record_price': record_price,
        'seller': 'TestSeller',
        'artist': 'Test Artist',
        'title': 'Test Album',
        'label': 'Test Label',
        'catno': 'TST-001',
        'wants': wants,
        'haves': haves,
        'genres': ['Jazz'],
        'styles': ['Soul-Jazz'],
        'year': 1973,
        'suggested_price': 25.0,
        'format': fmt or ['LP'],
    }


# ---------------------------------------------------------------------------
# non_embedding_scoring
# ---------------------------------------------------------------------------

class NonEmbeddingScoringTests(TestCase):

    def test_basic_scoring(self):
        inventory = [make_listing()]
        result = non_embedding_scoring(inventory)
        self.assertEqual(len(result), 1)
        self.assertIn('score', result[0])
        self.assertIn('price', result[0])
        self.assertIn('currency', result[0])
        self.assertIsInstance(result[0]['score'], float)
        self.assertIsInstance(result[0]['price'], float)
        self.assertEqual(result[0]['currency'], 'USD')

    def test_record_price_with_currency(self):
        """Prices like '20.0, EUR' must be parsed and converted."""
        inventory = [make_listing(record_price="20.0, EUR")]
        result = non_embedding_scoring(inventory)
        self.assertGreater(result[0]['price'], 0)

    def test_record_price_without_currency(self):
        """Prices like '65.00' (no currency) must default to USD."""
        inventory = [make_listing(record_price="65.00")]
        result = non_embedding_scoring(inventory)
        self.assertGreater(result[0]['price'], 0)
        self.assertEqual(result[0]['currency'], 'USD')

    def test_record_price_space_separated(self):
        """Prices like '14.99 USD' (space, no comma) must be handled."""
        inventory = [make_listing(record_price="14.99 USD")]
        result = non_embedding_scoring(inventory)
        self.assertGreater(result[0]['price'], 0)

    def test_zero_wants(self):
        """Records with 0 wants should score 0."""
        inventory = [make_listing(wants=0, haves=100)]
        result = non_embedding_scoring(inventory)
        self.assertEqual(result[0]['score'], 0.0)

    def test_higher_volume_scores_better(self):
        """40/20 should score much better than 4/2 (same ratio, more volume)."""
        high_vol = make_listing(wants=40, haves=20)
        low_vol = make_listing(wants=4, haves=2)
        scored = non_embedding_scoring([high_vol, low_vol])
        scores = {s['wants']: s['score'] for s in scored}
        self.assertGreater(scores[40], scores[4],
                           "Higher volume should score better than equal ratio with lower volume")

    def test_empty_inventory(self):
        result = non_embedding_scoring([])
        self.assertEqual(result, [])

    def test_scores_normalised_to_one(self):
        """Max score in a batch should be 1.0."""
        inventory = [make_listing(wants=100, haves=10), make_listing(wants=5, haves=50)]
        result = non_embedding_scoring(inventory)
        max_score = max(r['score'] for r in result)
        self.assertAlmostEqual(max_score, 1.0, places=6)

    def test_all_currencies(self):
        """All known currencies should be convertible without error."""
        for currency in ['USD', 'EUR', 'GBP', 'CAD', 'BRL', 'SEK', 'AUD', 'JPY']:
            inventory = [make_listing(record_price=f"10.0, {currency}")]
            result = non_embedding_scoring(inventory)
            self.assertGreater(result[0]['price'], 0,
                               f"currency {currency} should convert to positive USD price")


# ---------------------------------------------------------------------------
# Response shape — what ML sends vs what Go/frontend expect
# ---------------------------------------------------------------------------

EXPECTED_FIELDS = {
    'discogs_id': str,
    'media_condition': str,
    'record_price': str,
    'artist': str,
    'title': str,
    'label': str,
    'wants': int,
    'haves': int,
    'genres': list,
    'styles': list,
    'format': list,
    'score': float,
    'price': float,
    'currency': str,
}

NULLABLE_FIELDS = {'year', 'suggested_price', 'catno'}


class ResponseShapeTests(TestCase):

    def _scored_listing(self, **kwargs):
        inventory = [make_listing(**kwargs)]
        return non_embedding_scoring(inventory)[0]

    def test_all_required_fields_present(self):
        result = self._scored_listing()
        for field, expected_type in EXPECTED_FIELDS.items():
            self.assertIn(field, result, f"Missing required field: {field}")
            if result[field] is not None:
                self.assertIsInstance(result[field], expected_type,
                                      f"Field {field}: expected {expected_type}, got {type(result[field])}")

    def test_nullable_fields_may_be_none(self):
        listing = make_listing()
        listing['year'] = None
        listing['suggested_price'] = None
        listing['catno'] = None
        result = non_embedding_scoring([listing])[0]
        for field in NULLABLE_FIELDS:
            self.assertIn(field, result, f"Nullable field missing entirely: {field}")

    def test_format_is_always_list(self):
        """format must always be a list, never a string or None."""
        result = self._scored_listing(fmt=['LP'])
        self.assertIsInstance(result['format'], list)

    def test_suggested_price_is_numeric_or_none(self):
        result = self._scored_listing()
        sp = result.get('suggested_price')
        if sp is not None:
            self.assertIsInstance(sp, (int, float),
                                  f"suggested_price must be numeric, got {type(sp)}")

    def test_record_price_is_string(self):
        """record_price must stay as a string — Go decodes it as string."""
        result = self._scored_listing(record_price="20.0, EUR")
        self.assertIsInstance(result['record_price'], str)

    def test_price_is_float(self):
        """price (USD-converted) must be float — Go decodes as float64."""
        result = self._scored_listing()
        self.assertIsInstance(result['price'], float)

    def test_score_between_zero_and_one(self):
        inventory = [make_listing(wants=50, haves=10), make_listing(wants=5, haves=100)]
        scored = non_embedding_scoring(inventory)
        for item in scored:
            self.assertGreaterEqual(item['score'], 0.0)
            self.assertLessEqual(item['score'], 1.0)


# ---------------------------------------------------------------------------
# by_seller endpoint
# ---------------------------------------------------------------------------

MOCK_LISTING = make_listing()

class BySellerEndpointTests(TestCase):

    def setUp(self):
        self.client = APIClient()

    @patch('bandit.views.get_inventory')
    def test_missing_seller_returns_400(self, _mock):
        resp = self.client.post('/ml/discogs/by-seller/', {}, format='json')
        self.assertEqual(resp.status_code, 400)

    @patch('bandit.views.get_inventory')
    def test_empty_inventory_returns_empty_results(self, mock_get):
        mock_get.return_value = []
        resp = self.client.post('/ml/discogs/by-seller/',
                                {'seller': 'NoOne'}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['total'], 0)
        self.assertEqual(resp.data['results'], [])

    @patch('bandit.views.get_inventory')
    def test_vg_records_excluded(self, mock_get):
        """VG condition records must be filtered out."""
        mock_get.return_value = [make_listing(condition='Very Good (VG)')]
        resp = self.client.post('/ml/discogs/by-seller/',
                                {'seller': 'TestSeller'}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['total'], 0)

    @patch('bandit.views.get_inventory')
    def test_vg_plus_records_included(self, mock_get):
        mock_get.return_value = [make_listing(condition='Very Good Plus (VG+)')]
        resp = self.client.post('/ml/discogs/by-seller/',
                                {'seller': 'TestSeller'}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['total'], 1)

    @patch('bandit.views.get_inventory')
    def test_nm_records_included(self, mock_get):
        mock_get.return_value = [make_listing(condition='Near Mint (NM or M-)')]
        resp = self.client.post('/ml/discogs/by-seller/',
                                {'seller': 'TestSeller'}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['total'], 1)

    @patch('bandit.views.get_inventory')
    def test_response_shape(self, mock_get):
        """Response must have seller, total, results keys."""
        mock_get.return_value = [make_listing()]
        resp = self.client.post('/ml/discogs/by-seller/',
                                {'seller': 'TestSeller'}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('seller', resp.data)
        self.assertIn('total', resp.data)
        self.assertIn('results', resp.data)
        self.assertEqual(resp.data['seller'], 'TestSeller')

    @patch('bandit.views.get_inventory')
    def test_results_sorted_by_score_desc(self, mock_get):
        low = make_listing(wants=2, haves=100)
        high = make_listing(wants=400, haves=50)
        mock_get.return_value = [low, high]
        resp = self.client.post('/ml/discogs/by-seller/',
                                {'seller': 'TestSeller'}, format='json')
        scores = [r['score'] for r in resp.data['results']]
        self.assertEqual(scores, sorted(scores, reverse=True))

    @patch('bandit.views.get_inventory')
    def test_result_fields_match_go_struct(self, mock_get):
        """Every field Go's SellerRecord expects must be present and correctly typed."""
        mock_get.return_value = [make_listing()]
        resp = self.client.post('/ml/discogs/by-seller/',
                                {'seller': 'TestSeller'}, format='json')
        self.assertEqual(resp.status_code, 200)
        result = resp.data['results'][0]

        # Fields Go decodes as string
        for field in ['discogs_id', 'artist', 'title', 'label', 'media_condition',
                      'record_price', 'currency']:
            self.assertIsInstance(result[field], str, f"{field} must be str")

        # Fields Go decodes as float64
        for field in ['score', 'price']:
            self.assertIsInstance(result[field], (int, float), f"{field} must be numeric")

        # Fields Go decodes as int
        for field in ['wants', 'haves']:
            self.assertIsInstance(result[field], int, f"{field} must be int")

        # Fields Go decodes as []string
        for field in ['genres', 'styles', 'format']:
            self.assertIsInstance(result[field], list, f"{field} must be list")

    @patch('bandit.views.get_inventory')
    def test_price_without_currency_does_not_crash(self, mock_get):
        """record_price='65.00' (no currency) must not cause a 500."""
        mock_get.return_value = [make_listing(record_price='65.00')]
        resp = self.client.post('/ml/discogs/by-seller/',
                                {'seller': 'TestSeller'}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['total'], 1)
