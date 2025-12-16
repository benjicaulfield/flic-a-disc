import pytest
from bandit.enhance_listings import LookupByID

def test_ebay_api():
    lu = LookupByID()
    assert lu.api.get_access_token()

@pytest.mark.django_db
def test_lookup():
    id = "v1|277518406131|0"
    lu = LookupByID()
    listing = lu.lookup_by_item_id(id)
    assert listing