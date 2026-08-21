import math

def required_margin(buy_price, floor=0.15, ceiling=0.50, decay=0.01):
    margin = floor + (ceiling - floor) * math.exp(-decay * buy_price)
    return margin

def max_bid(suggested_sell_price, shipping_estimate=6.00, fee_rate=0.15):
    net_proceeds = suggested_sell_price * (1 - fee_rate)
    target_margin = required_margin(suggested_sell_price)  # keyed on sell price, not buy price
    max_buy = net_proceeds * (1 - target_margin) - shipping_estimate
    return max_buy