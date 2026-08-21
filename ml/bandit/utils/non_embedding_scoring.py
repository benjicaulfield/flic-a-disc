import math

from .get_exchange_rates import get_exchange_rates, convert_to_usd
from ..knapsack import price_diffs, price_diff_normalizer


def demands(listing):
    haves = listing['haves']
    wants = listing['wants']
    if wants == 0:
        return 0
    return (wants / (wants + haves)) * math.log(wants + 1)


def demand_normalizer(inventory):
    scores = [demands(item) for item in inventory]
    max_score = max(scores) if scores else 0
    return max_score if max_score > 0 else 1


def non_embedding_scoring(inventory):
    if not inventory:
        return []

    max_demand = demand_normalizer(inventory)
    max_price_dff = price_diff_normalizer(inventory)

    rates = get_exchange_rates()
    for listing in inventory:
        listing['score'] = demands(listing) / max_demand
        raw = listing['record_price'].replace(', ', ' ').split()
        price = raw[0]
        currency = raw[1] if len(raw) > 1 else 'USD'
        listing['price'] = convert_to_usd(float(price), currency, rates)
        listing['currency'] = 'USD'

    return inventory
