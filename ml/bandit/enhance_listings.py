import requests
from urllib.parse import quote
from .ebay_client import EbayApi
from .models import EbayListing

class LookupByID:
    def __init__(self):
        self.api = EbayApi()
        self.api.get_access_token()
        self.threshold = 0.75
        
    def lookup_by_item_id(self, item_id):
        url = f'{self.api.base_url}/buy/browse/v1/item/{item_id}'
        
        headers = {
            'Authorization': f'Bearer {self.api.access_token}',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US',
            'Accept-Language': 'en-US',  
            }
        
        item_id = str(item_id)
        
        encoded = quote(item_id, safe='')
        url = f'{self.api.base_url}/buy/browse/v1/item/{encoded}'
        params = {'fieldgroups': 'PRODUCT'}

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        listing = EbayListing.objects.get(ebay_id=item_id)
        if 'localizedAspects' in response:
            aspects = {aspect['name']: aspect['value'] 
                                    for aspect in response['localizedAspects']}     
            listing.artist = aspects.get('Artist')
            listing.album = aspects.get('Release Title')
            listing.record_condition = aspects.get('Record Grading')
            listing.sleeve_condition = aspects.get('Sleeve Grading')
            listing.format = aspects.get('Record Size')
            listing.style = aspects.get('Style')
            listing.genre = aspects.get('Genre')
            listing.year = aspects.get('Release Year')
        
        listing.enriched = True
        listing.save()

        return listing






        
        
        


