import math

import torch
import numpy as np
from ortools.algorithms.python import knapsack_solver

from .models import KnapsackWeights, DiscogsListing, DiscogsSeller, DiscogsRecord
from .utils.get_user_inventory import get_inventory
from .utils.get_exchange_rates import get_exchange_rates, convert_to_usd
from .training import trainer

RATES = get_exchange_rates()

def save_listings(inventory):
    for item in inventory:
        seller, _ = DiscogsSeller.objects.get_or_create(name=item['seller'])
        price_str, currency = item['record_price'].split(', ')
        price = float(price_str)

        record, _ = DiscogsRecord.objects.get_or_create(
            discogs_id=item['discogs_id'],
            defaults={
                'artist': item['artist'],
                'title': item['title'],
                'label': item['label'],
                'catno': item['catno'],
                'wants': item['wants'],
                'haves': item['haves'],
                'genres': item['genres'],
                'styles': item['styles'],
                'year': item['year'],
                'suggested_price': str(item['suggested_price']) if item['suggested_price'] else '',
            }
        )
        
        # Create listing (allow duplicates since same record can be listed multiple times)
        DiscogsListing.objects.create(
            seller=seller,
            record=record,
            record_price=price,
            currency=currency,
            media_condition=item['media_condition']
        )


def score_and_filter_seller_listings(inventory):

    ALLOWED_CONDITIONS = [
        'Near Mint (NM or M-)',
        'Very Good Plus (VG+)',
        'Very Good (VG)'
    ]

    inventory = [
        item for item in inventory
        if item.get('media_condition') in ALLOWED_CONDITIONS
    ]

    if not inventory:
        return []

    return weighted_score(inventory)

def weighted_score(inventory):
    weights = KnapsackWeights.objects.first()
    if weights is None:
        class DefaultWeights:
            demand = 0.34
            price_diff = 0.33
            embedding = 0.33
        weights = DefaultWeights()

    max_demand = demand_normalizer(inventory)
    max_price_diff = price_diff_normalizer(inventory)

    model_ready = trainer.model is not None or trainer.load_latest_model()
    if model_ready:
        probs, variances = get_embeddings(inventory, trainer)
    else:
        probs, variances = [0] * len(inventory), [None] * len(inventory)

    rates = get_exchange_rates()
    for listing, prob, var, in zip(inventory, probs, variances):
        d = demands(listing) / max_demand
        p = price_diffs(listing) / max_price_diff
        e = float(prob)
        listing['score'] = weights.demand * d + weights.price_diff * p + weights.embedding * e
        listing['probability'] = e
        listing['uncertainty'] = float(var) if var is not None else None
        price, currency = parse_record_price(listing['record_price'])
        listing['price'] = convert_to_usd(float(price), currency, rates)
        listing['currency'] = 'USD'

    return inventory

    

def knapsack(inventory, budget):
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        'DiscogsKnapsack'
    )
    values = [int(listing['score'] * 1000) for listing in inventory]
    # Price is already in USD from score_and_filter_seller_listings()
    weights = [[int(listing['price'] * 100) for listing in inventory]]
    capacities = [int(budget * 100)]

    # Debug logging
    print(f"\n🎒 Knapsack Debug:")
    print(f"  Inventory size: {len(inventory)}")
    print(f"  Budget: ${budget} (capacity: {capacities[0]} cents)")
    if inventory:
        print(f"  Score range: {min(listing['score'] for listing in inventory):.3f} - {max(listing['score'] for listing in inventory):.3f}")
        print(f"  Price range: ${min(listing['price'] for listing in inventory):.2f} - ${max(listing['price'] for listing in inventory):.2f}")
        print(f"  Values (int scores*1000) range: {min(values)} - {max(values)}")
        print(f"  Weights (int prices*100) range: {min(weights[0])} - {max(weights[0])}")
        zero_scores = sum(1 for v in values if v == 0)
        if zero_scores > 0:
            print(f"  ⚠️  WARNING: {zero_scores}/{len(values)} items have score=0 after int conversion!")

    solver.init(values, weights, capacities)
    computed_value = solver.solve()

    print(f"  Computed optimal value: {computed_value}")

    selected = []
    for i in range(len(inventory)):
        if solver.best_solution_contains(i):
            selected.append(inventory[i])

    if selected:
        total_cost = sum(item['price'] for item in selected)
        total_score = sum(item['score'] for item in selected)
        print(f"  Selected {len(selected)} items: ${total_cost:.2f} / ${budget}, total score: {total_score:.2f}\n")
    else:
        print(f"  Selected 0 items\n")

    return selected

def demands(listing):
    haves = listing['haves']
    wants = listing['wants']
    if haves == 0:
        return 0
    ratio = wants / haves
    confidence = math.log(wants + 1)
    return ratio * confidence

def demand_normalizer(inventory):
    demand_scores = [demands(listing) for listing in inventory]
    max_demand = max(demand_scores) if demand_scores else 0
    return max_demand if max_demand > 0 else 1

def price_diffs(listing):
    price, currency = parse_record_price(listing['record_price'])
    dollar_price = convert_to_usd(price, currency, RATES)
    # Use listing price as default if no suggested_price (no false signal)
    sugg_price = listing.get('suggested_price')
    if not sugg_price:
        sugg_price = dollar_price
    return max(0, (float(sugg_price) - dollar_price))

def price_diff_normalizer(inventory):
    diffs = [price_diffs(listing) for listing in inventory]
    max_diff = max(diffs) if diffs else 0
    return max_diff if max_diff > 0 else 1

def get_embeddings(inventory, trainer):
    features = trainer.feature_extractor.extract_batch_features(inventory)
    features_tensor = torch.FloatTensor(features)
    probs, variances = trainer.model.predict_with_uncertainty(features_tensor)
    embeddings = probs.cpu().numpy()
    uncertainties = variances.cpu().numpy()

    # Ensure at least 1-dimensional (handles single-item case)
    if embeddings.ndim == 0:
        embeddings = np.array([embeddings])
        uncertainties = np.array([uncertainties])

    return embeddings, uncertainties

def parse_record_price(record_price):
    price_str, *rest = record_price.replace(', ', ' ').split()
    currency = rest[0] if rest else 'USD'
    return float(price_str), currency



