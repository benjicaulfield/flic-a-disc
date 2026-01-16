import requests

def get_exchange_rates():
    response = requests.get("https://open.er-api.com/v6/latest/USD")
    data = response.json()
    return data['rates']

def convert_to_usd(amount, currency, rates):
    return float(amount) / rates[currency]