import unittest

from bandit.discogs_client import authenticate_client
from bandit.models import DiscogsRecord

d = authenticate_client
r = d.release(7385384)
print(r)

class TestInventoryFilters(unittest.TestCase):
    
    def test_data(self):
        pass