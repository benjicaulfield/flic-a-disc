# python_services/bandit/tests/test_knapsack.py

import unittest
from bandit.knapsack import knapsack

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

if __name__ == '__main__':
    unittest.main()