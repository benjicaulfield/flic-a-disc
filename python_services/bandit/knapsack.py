import math

import torch
import numpy as np
from ortools.algorithms.python import knapsack_solver

from django.utils import timezone
from .models import KnapsackWeights, DiscogsListing, DiscogsSeller, Record
from .utils.get_user_inventory import get_inventory
from .utils.get_exchange_rates import get_exchange_rates, convert_to_usd
from .training import BanditTrainer

RATES = get_exchange_rates()

def save_listings(inventory):
    for item in inventory:
        seller, _ = DiscogsSeller.objects.get_or_create(name=item['seller'])

        # Create/get Record
        record, _ = Record.objects.get_or_create(
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
                'added': timezone.now(),
            }
        )
        
        # Create listing (allow duplicates since same record can be listed multiple times)
        DiscogsListing.objects.create(
            seller=seller,
            record=record,
            record_price=f"{item['price']}, {item['currency']}",
            media_condition=item['media_condition']
        )


def score_and_filter_seller_listings(seller):
    trainer = BanditTrainer()
    if not trainer.model:
        trainer.load_latest_model()
    
    weights = KnapsackWeights.objects.first()
    
    inventory = get_inventory(seller)
    save_listings(inventory)

    demand_norm = demand_normalizer(inventory)
    diff_norm = price_diff_normalizer(inventory)

    embeddings = get_embeddings(inventory, trainer)

    for i, listing in enumerate(inventory):
        demand = demands(listing) / demand_norm
        price_diff = price_diffs(listing) / diff_norm
        score = (weights.embedding * embeddings[i] + 
                 weights.price_diff * price_diff +
                 weights.demand * demand)
        listing['score'] = score

    return inventory

def knapsack(inventory, budget):
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        'DiscogsKnapsack'
    )

    values = [int(listing['score'] * 1000) for listing in inventory]  
    weights = [[int(convert_to_usd(listing['price'], listing['currency'], RATES) * 100) 
            for listing in inventory]]    
    capacities = [int(budget * 100)]  
    
    solver.init(values, weights, capacities)
    solver.solve()
    
    selected = []
    for i in range(len(inventory)):
        if solver.best_solution_contains(i):
            selected.append(inventory[i])
    
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
    return max(demand_scores) if demand_scores else 1

def price_diffs(listing):
    price = listing['price']
    currency = listing['currency']
    dollar_price = convert_to_usd(price, currency, RATES)
    sugg_price = listing.get('suggested_price', 0)
    return max(0, (sugg_price - dollar_price))

def price_diff_normalizer(inventory):
    diffs = [price_diffs(listing) for listing in inventory]
    return max(diffs) if diffs else 1

def get_embeddings(inventory, trainer):
    features = trainer.feature_extractor.extract_batch_features(inventory)
    features_tensor = torch.FloatTensor(features)
    probs, _ = trainer.model.predict_with_uncertainty(features_tensor)
    return probs.cpu().numpy()






