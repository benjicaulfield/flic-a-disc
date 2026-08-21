import unittest
import numpy as np
from unittest.mock import Mock, patch
from bandit.knapsack import (
    knapsack,
    demands,
    demand_normalizer,
    price_diffs,
    price_diff_normalizer,
    get_embeddings,
    score_and_filter_seller_listings
)

class TestKnapsack(unittest.TestCase):
    
    def test_knapsack_respects_budget(self):
        """Test that knapsack doesn't exceed budget"""
        items = [
            {'id': 1, 'score': 10, 'price': 50, 'currency': 'USD'},
            {'id': 2, 'score': 8, 'price': 40, 'currency': 'USD'},
            {'id': 3, 'score': 6, 'price': 30, 'currency': 'USD'},
            {'id': 4, 'score': 5, 'price': 25, 'currency': 'USD'},
            {'id': 5, 'score': 3, 'price': 20, 'currency': 'USD'},
        ]
        budget = 100
        
        selected = knapsack(items, budget)
        total_cost = sum(item['price'] for item in selected)
        
        self.assertLessEqual(total_cost, budget, 
            f"Knapsack exceeded budget: ${total_cost} > ${budget}")
    
    def test_knapsack_selects_best_items(self):
        """Test that knapsack selects optimal items"""
        items = [
            {'id': 1, 'score': 100, 'price': 10, 'currency': 'USD'},  # Best value
            {'id': 2, 'score': 50, 'price': 90, 'currency': 'USD'},   # Fills budget
            {'id': 3, 'score': 1, 'price': 5, 'currency': 'USD'},     # Low value
        ]
        budget = 100
        
        selected = knapsack(items, budget)
        selected_ids = {item['id'] for item in selected}
        
        self.assertIn(1, selected_ids, "Should select high-value item 1")
        self.assertIn(2, selected_ids, "Should select item 2 to maximize score")
        
        total_cost = sum(item['price'] for item in selected)
        self.assertLessEqual(total_cost, budget)
    
    def test_knapsack_with_exact_budget(self):
        """Test knapsack when items exactly match budget"""
        items = [
            {'id': 1, 'score': 10, 'price': 50, 'currency': 'USD'},
            {'id': 2, 'score': 10, 'price': 50, 'currency': 'USD'},
        ]
        budget = 100
        
        selected = knapsack(items, budget)
        total_cost = sum(item['price'] for item in selected)
        
        self.assertEqual(total_cost, 100, "Should use entire budget when possible")
        self.assertEqual(len(selected), 2, "Should select both items")
    
    def test_knapsack_with_zero_budget(self):
        """Test knapsack with zero budget"""
        items = [
            {'id': 1, 'score': 10, 'price': 50, 'currency': 'USD'},
        ]
        budget = 0
        
        selected = knapsack(items, budget)
        
        self.assertEqual(len(selected), 0, "Should select nothing with zero budget")
    
    def test_real_world_bug(self):
        """Reproduce the bug: 23 items costing $402 with $300 budget"""
        items = []
        for i in range(50):
            items.append({
                'id': i,
                'score': 10.0 - (i * 0.1),
                'price': 15.0 + (i * 0.5),
                'currency': 'USD'
            })

        budget = 300
        selected = knapsack(items, budget)
        total_cost = sum(item['price'] for item in selected)

        self.assertLessEqual(total_cost, budget,
            f"BUG REPRODUCED: Selected {len(selected)} items costing ${total_cost:.2f} with budget ${budget}")


class TestDemandFunctions(unittest.TestCase):
    """Test demand calculation and normalization"""

    def test_demands_with_zero_haves(self):
        """Test demands returns 0 when haves is 0"""
        listing = {'wants': 100, 'haves': 0}
        result = demands(listing)
        self.assertEqual(result, 0, "Should return 0 when haves is 0")

    def test_demands_normal_case(self):
        """Test demands with normal values"""
        listing = {'wants': 100, 'haves': 50}
        result = demands(listing)
        self.assertGreater(result, 0, "Should return positive value")

    def test_demand_normalizer_all_zero(self):
        """Test normalizer returns 1 when all demands are 0"""
        inventory = [
            {'wants': 0, 'haves': 0},
            {'wants': 10, 'haves': 0},
            {'wants': 20, 'haves': 0}
        ]
        result = demand_normalizer(inventory)
        self.assertEqual(result, 1, "Should return 1 when all demands are 0")

    def test_demand_normalizer_normal_case(self):
        """Test normalizer with normal values"""
        inventory = [
            {'wants': 100, 'haves': 50},
            {'wants': 200, 'haves': 100}
        ]
        result = demand_normalizer(inventory)
        self.assertGreater(result, 0, "Should return positive normalizer")


class TestPriceFunctions(unittest.TestCase):
    """Test price difference calculation"""

    def test_price_diffs_basic(self):
        """Test basic price difference calculation"""
        listing = {
            'record_price': '10.00, USD',
            'suggested_price': 20.0
        }
        with patch('bandit.knapsack.convert_to_usd', return_value=10.0):
            result = price_diffs(listing)
            self.assertEqual(result, 10.0, "Should return difference")

    def test_price_diffs_no_suggested(self):
        """Test price diff when no suggested price"""
        listing = {
            'record_price': '10.00, USD',
            'suggested_price': None
        }
        with patch('bandit.knapsack.convert_to_usd', return_value=10.0):
            result = price_diffs(listing)
            self.assertEqual(result, 0, "Should return 0 when no suggested price")


class TestEmbeddings(unittest.TestCase):
    """Test embedding generation"""

    def test_embeddings_single_item(self):
        """Test embeddings with single item (0-d array bug)"""
        inventory = [{'artist': 'Test', 'title': 'Test'}]

        mock_trainer = Mock()
        mock_trainer.feature_extractor.extract_batch_features.return_value = [[0.1, 0.2, 0.3]]

        # Simulate model returning 0-d array (the bug)
        mock_probs = Mock()
        mock_probs.cpu.return_value.numpy.return_value = np.array(0.5)  # 0-d scalar
        mock_trainer.model.predict_with_uncertainty.return_value = (mock_probs, None)

        result = get_embeddings(inventory, mock_trainer)

        self.assertEqual(result.ndim, 1, "Should convert 0-d to 1-d array")
        self.assertEqual(len(result), 1, "Should have 1 element")

    def test_embeddings_multiple_items(self):
        """Test embeddings with multiple items"""
        inventory = [
            {'artist': 'Test1', 'title': 'Test1'},
            {'artist': 'Test2', 'title': 'Test2'}
        ]

        mock_trainer = Mock()
        mock_trainer.feature_extractor.extract_batch_features.return_value = [[0.1, 0.2], [0.3, 0.4]]

        mock_probs = Mock()
        mock_probs.cpu.return_value.numpy.return_value = np.array([0.5, 0.6])
        mock_trainer.model.predict_with_uncertainty.return_value = (mock_probs, None)

        result = get_embeddings(inventory, mock_trainer)

        self.assertEqual(result.ndim, 1, "Should be 1-d array")
        self.assertEqual(len(result), 2, "Should have 2 elements")


class TestScoreAndFilter(unittest.TestCase):
    """Integration tests for score_and_filter_seller_listings"""

    @patch('bandit.knapsack.save_listings')
    @patch('bandit.knapsack.get_embeddings')
    @patch('bandit.knapsack.BanditTrainer')
    @patch('bandit.knapsack.KnapsackWeights.objects.first')
    def test_filter_by_condition(self, mock_weights, mock_trainer, mock_embeddings, mock_save):
        """Test filtering by media condition"""
        inventory = [
            {'media_condition': 'Near Mint (NM or M-)', 'wants': 10, 'haves': 5, 'record_price': '10, USD', 'suggested_price': 15},
            {'media_condition': 'Poor (P)', 'wants': 10, 'haves': 5, 'record_price': '10, USD', 'suggested_price': 15},
            {'media_condition': 'Very Good Plus (VG+)', 'wants': 10, 'haves': 5, 'record_price': '10, USD', 'suggested_price': 15}
        ]

        # Mock weights
        mock_w = Mock()
        mock_w.embedding = 1.0
        mock_w.price_diff = 1.0
        mock_w.demand = 1.0
        mock_weights.return_value = mock_w

        # Mock embeddings
        mock_embeddings.return_value = np.array([0.5, 0.6])

        with patch('bandit.knapsack.convert_to_usd', return_value=10.0):
            result = score_and_filter_seller_listings(inventory)

        # Should only keep NM and VG+ (2 items)
        self.assertEqual(len(result), 2, f"Should filter to 2 items, got {len(result)}")

    @patch('bandit.knapsack.save_listings')
    @patch('bandit.knapsack.get_embeddings')
    @patch('bandit.knapsack.BanditTrainer')
    @patch('bandit.knapsack.KnapsackWeights.objects.first')
    def test_empty_after_filter(self, mock_weights, mock_trainer, mock_embeddings, mock_save):
        """Test when all items filtered out"""
        inventory = [
            {'media_condition': 'Poor (P)', 'wants': 10, 'haves': 5}
        ]

        result = score_and_filter_seller_listings(inventory)

        self.assertEqual(len(result), 0, "Should return empty list when all filtered")


if __name__ == '__main__':
    unittest.main()