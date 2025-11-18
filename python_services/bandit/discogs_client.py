import os
import json
import discogs_client
from decouple import config

from .rate_limiter import RateLimiter, rate_limit_client

LIMITER = RateLimiter(60)


consumer_key = config('DISCOGS_CONSUMER_KEY')
consumer_secret = config('DISCOGS_CONSUMER_SECRET')
TOKEN_FILE = 'discogs_token.json'

def load_tokens():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return json.load(f)
    return None

def save_tokens(token, secret):
    with open(TOKEN_FILE, 'w') as f:
        json.dump({'token': token, 'secret': secret}, f)

def authenticate_client():
    d = discogs_client.Client('flic-a-disc/1.0')
    d.set_consumer_key(consumer_key, consumer_secret)
    rate_limit_client(d, LIMITER)
    tokens = load_tokens()

    if tokens:
        d.set_token(tokens['token'], tokens['secret'])
    else:
        token, secret, url = d.get_authorize_url()
        print(f"Please visit this URL to authorize: {url}")
        verifier = input("Enter the verifier code: ")
        access_token, access_secret = d.get_access_token(verifier)
        d.set_token(access_token, access_secret)
        save_tokens(access_token, access_secret)
    
    return d