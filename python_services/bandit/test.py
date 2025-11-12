from .models import DiscogsListing
from .training import BanditTrainer

print(DiscogsListing.objects.order_by('-id')[:500])
