from django.db import models

class DiscogsRecord(models.Model):
    discogs_id = models.CharField(max_length=255, unique=True)
    artist = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    format = models.JSONField(max_length=255, blank=True)
    label = models.TextField()
    catno = models.CharField(max_length=255, null=True, blank=True)
    wants = models.IntegerField(default=0)
    haves = models.IntegerField(default=0)
    genres = models.JSONField(default=list)
    styles = models.JSONField(default=list)
    suggested_price = models.CharField(max_length=255, default='')
    year = models.IntegerField(null=True, blank=True)
    country = models.CharField(null=True, blank=True)
    record_image = models.URLField(max_length=500, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    evaluated = models.BooleanField(default=False)
    wanted = models.BooleanField(default=False)
    wantlist = models.BooleanField(default=False)
    wantlist_evaluated = models.BooleanField(default=False)
    heldout = models.BooleanField(default=False)
    is_master = models.BooleanField(default=False)
    master_id = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = 'discogs_record'

class DiscogsSeller(models.Model):
    name = models.CharField(max_length=255)
    currency = models.CharField(max_length=8)
    # Country/region string from the scrape's seller.shipsFrom, captured on
    # import. Not a reliable domestic/foreign signal on its own (a seller
    # can price in USD and still ship from abroad -- currency != origin).
    ships_from = models.CharField(max_length=100, null=True, blank=True)
    # Curated from sellers_sorted.json via the sync_seller_shipping command,
    # not derived from the scrape -- Discogs doesn't expose these via API.
    free_shipping_min_amount = models.FloatField(null=True, blank=True)
    free_shipping_min_currency = models.CharField(max_length=8, null=True, blank=True)
    free_shipping_min_usd = models.FloatField(null=True, blank=True)
    # Annotated cost of shipping 5 LPs together in one order -- a single
    # reference-order-size number instead of a base+per-item model, since
    # sellers don't publish a clean marginal rate.
    flat_rate_amount = models.FloatField(null=True, blank=True)
    flat_rate_currency = models.CharField(max_length=8, null=True, blank=True)
    flat_rate_usd = models.FloatField(null=True, blank=True)
    shipping_notes = models.TextField(blank=True, default='')

    class Meta:
        db_table = 'discogs_discogsseller'

class DiscogsListing(models.Model):
    seller = models.ForeignKey(DiscogsSeller, on_delete=models.CASCADE, related_name='listings')
    record = models.ForeignKey(DiscogsRecord, on_delete=models.CASCADE, related_name='listings')
    record_price = models.FloatField()
    currency = models.CharField(max_length=255, default="")
    media_condition = models.CharField(max_length=255)
    sleeve_condition = models.CharField(max_length=255, blank=True, default="")
    # Discogs marketplace item ID (their itemId) -- lets us upsert a specific
    # listing idempotently. Null for older rows saved via the by-seller
    # inventory scrape, which never captured this.
    discogs_listing_id = models.CharField(max_length=255, null=True, blank=True, unique=True)
    # When the seller listed it for sale on Discogs (not when we scraped it).
    listed_date = models.DateTimeField(null=True, blank=True)
    # Denormalized copy of the seller's shipping fields, propagated in bulk
    # by sync_seller_shipping so per-listing queries don't need a join.
    ships_from = models.CharField(max_length=100, null=True, blank=True)
    free_shipping_min_amount = models.FloatField(null=True, blank=True)
    free_shipping_min_currency = models.CharField(max_length=8, null=True, blank=True)
    free_shipping_min_usd = models.FloatField(null=True, blank=True)
    flat_rate_amount = models.FloatField(null=True, blank=True)  # cost of shipping 5 LPs together
    flat_rate_currency = models.CharField(max_length=8, null=True, blank=True)
    flat_rate_usd = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = 'discogs_discogslisting'


class EbayListing(models.Model):
    id = models.AutoField(primary_key=True)
    ebay_id = models.CharField(max_length=255, unique=True)
    ebay_title = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)    
    artist = models.CharField(max_length=255, blank=True)
    title = models.CharField(max_length=255, blank=True)
    label = models.CharField(max_length=255, blank=True)
    format = models.JSONField(max_length=255, blank=True)
    year = models.CharField(max_length=10, blank=True)
    media_condition = models.CharField(max_length=50, blank=True)
    genres = models.JSONField(default=list)
    styles = models.JSONField(default=list)
    wanted = models.BooleanField(null=True, blank=True)
    evaluated = models.BooleanField(default=False)
    source = models.CharField(max_length=64, blank=True)
    keeper_score = models.FloatField(default=0.0)
    
    class Meta:
        db_table = 'ebay_listings'  # Match Go's table name

class BanditModel(models.Model):
    version = models.CharField(max_length=255)
    model_weights = models.BinaryField()
    hyperparams = models.JSONField(default=dict)
    training_stats = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    batch_count = models.IntegerField(default=0)
    
    class Meta:
        db_table = 'bandit_model'

class EbayFirstPassModel(models.Model):
    version = models.CharField(max_length=255)
    model_weights = models.BinaryField()
    hyperparams = models.JSONField(default=dict)
    training_stats = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)


    class Meta:
        db_table = 'ebay_first_pass'

class BanditTrainingInstance(models.Model):
    record = models.ForeignKey(DiscogsRecord, on_delete=models.CASCADE, null=True)
    context = models.JSONField()
    predicted = models.BooleanField()
    predicted_prob = models.FloatField(null=True, blank=True)
    predicted_uncertainty = models.FloatField(null=True, blank=True)
    actual = models.BooleanField()
    reward = models.FloatField() 
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'bandit_training_example'

class ThresholdConfig(models.Model):
    threshold = models.FloatField(default=0.5)
    f1_score = models.FloatField(null=True)
    window_size = models.IntegerField(default=500)

    class Meta:
        db_table = 'bandit_threshold_config'

class BatchPerformance(models.Model):
    batch_number = models.IntegerField()
    correct = models.IntegerField()
    total = models.IntegerField()
    accuracy = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'batch_performance'
        ordering = ['-batch_number']

class TfIdfDB(models.Model):
    version = models.CharField(max_length=100, unique=True)
    model_weights = models.BinaryField()  # Pickled vectorizer + embeddings
    hyperparams = models.JSONField()
    training_stats = models.JSONField()
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'similarity_index'
        ordering = ['-created_at']

class Todo(models.Model):
    user_id = models.CharField(max_length=255)
    text = models.TextField()
    status = models.CharField(max_length=20, choices=[('in-progress', 'In Progress'), ('backlog', 'Backlog')])
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['order', 'created_at']
        db_table = 'todos'

    def __str__(self):
        return f"User {self.user_id}: {self.text[:50]}"
    
# In bandit/models.py, add:

class EbayBatchPerformance(models.Model):
    batch_number = models.IntegerField()
    correct = models.IntegerField()
    total = models.IntegerField()
    accuracy = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-batch_number']
        db_table = 'ebay_batch_performance'
    
    def __str__(self):
        return f"eBay Batch {self.batch_number}: {self.accuracy*100:.1f}% ({self.correct}/{self.total})"
    
class KnapsackWeights(models.Model):
    embedding = models.FloatField(default=0.33)
    price_diff = models.FloatField(default=0.33)
    demand = models.FloatField(default=0.34)
    updated_at = models.DateTimeField(auto_now=True)

class KnapsackSession(models.Model):
    seller_name = models.CharField(max_length=255)
    budget = models.FloatField()
    total_cost = models.FloatField()
    total_score = models.FloatField()
    selected_count = models.IntegerField()
    selected_items = models.JSONField()  # Store full item details
    contenders = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    saved_for_comparison = models.BooleanField(default=False)
    notes = models.TextField(blank=True, default='')

    class Meta:
        db_table = 'knapsack_sessions'
        ordering = ['-created_at']