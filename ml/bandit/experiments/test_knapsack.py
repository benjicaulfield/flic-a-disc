

import json
from ..knapsack import score_and_filter_seller_listings, knapsack

# Read sellers file
with open('sellers.json', 'r') as f:
    sellers = json.load(f)

budget = 500

results = []

for seller_data in sellers:
    seller = seller_data['username']
    print(f"Processing {seller}...")
    
    try:
        scored_inventory = score_and_filter_seller_listings(seller)
        scored_inventory.sort(key=lambda x: x.score, reverse=True)
        top_40 = scored_inventory[:40]
        
        selected = knapsack(top_40, budget)
        
        selected_ids = {id(item) for item in selected}
        contenders = [item for item in top_40 if id(item) not in selected_ids]
        
        results.append({
            "seller": seller,
            "knapsack_count": len(selected),
            "total_cost": sum(item.price for item in selected),
            "total_score": sum(item.score for item in selected),
            "contenders_count": len(contenders)
        })
        
        print(f"  ✓ Selected {len(selected)} items, ${sum(item.price for item in selected):.2f}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append({"seller": seller, "error": str(e)})

# Save results
with open('knapsack_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Done! Results saved to knapsack_results.json")