import os
import traceback
from pathlib import Path
import json
from django.conf import settings
from django.core.management.base import BaseCommand
from bandit.knapsack import score_and_filter_seller_listings, knapsack
from bandit.utils.get_exchange_rates import get_exchange_rates, convert_to_usd

class Command(BaseCommand):
    help = 'Test knapsack optimization for all sellers'

    def add_arguments(self, parser):
        parser.add_argument('--budget', type=float, default=500, help='Budget in USD')

    def handle(self, *args, **options):
        budget = options['budget']
        sellers = [
            {"username": "kim_melody", "amount": 250, "currency": "EUR"},
        ]

        results_path = 'knapsack_results.json'
        
        results = []
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                self.stdout.write(f"Loaded {len(results)} existing results")


        for seller_data in sellers:
            seller = seller_data['username']

            self.stdout.write(f"Processing {seller}...")
            
            try:
                scored_inventory = score_and_filter_seller_listings(seller)
                scored_inventory.sort(key=lambda x: x['score'], reverse=True)
                
                selected = knapsack(scored_inventory, budget)
                
                selected_ids = {id(item) for item in selected}
                contenders = [item for item in scored_inventory if id(item) not in selected_ids][:40]
                
                for item in selected + contenders:
                    item['score'] = float(item['score'])
                    item['price'] = float(item['price'])
                
                results.append({
                    "seller": seller,
                    "knapsack": selected,
                    "contenders": contenders,
                    "total_cost": float(sum(item['price'] for item in selected)),
                    "total_score": float(sum(item['score'] for item in selected)),
                })
                
                self.stdout.write(self.style.SUCCESS(
                    f"  ✓ Selected {len(selected)} items, ${sum(item['price'] for item in selected):.2f}"
                ))
                
            except Exception as e:
                tb = traceback.format_exc()
                self.stdout.write(self.style.ERROR(f"  ✗ Error: {e}"))
                self.stdout.write(tb)  # Print to console
                results.append({"seller": seller, "error": str(e), "traceback": tb})
            # Save results
            with open('knapsack_results.json', 'w') as f:
                json.dump(results, f, indent=2)

        self.stdout.write(self.style.SUCCESS("\n✅ Done! Results saved to knapsack_results.json"))