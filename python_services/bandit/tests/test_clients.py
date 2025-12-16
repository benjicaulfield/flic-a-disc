from bandit.discogs_client import authenticate_client
from bandit.ebay_client import EbayApi

def test_discogs_client():
    assert authenticate_client()

def test_ebay_client():
    assert EbayApi()