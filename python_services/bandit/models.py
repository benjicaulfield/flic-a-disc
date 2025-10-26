from django.db import models

class Record(models.Model):
    discogs_id = models.CharField(max_length=255, unique=True)
    artist = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    format = models.CharField(max_length=255, blank=True)
    label = models.TextField()
    catno = models.CharField(max_length=255, null=True, blank=True)
    wants = models.IntegerField(default=0)
    haves = models.IntegerField(default=0)
    added = models.DateTimeField()
    genres = models.JSONField(default=list)
    styles = models.JSONField(default=list)
    suggested_price = models.CharField(max_length=255, default='')
    year = models.IntegerField(null=True, blank=True)
    record_image = models.URLField(max_length=500, blank=True, null=True)
    wanted = models.BooleanField(default=False)
    evaluated = models.BooleanField(default=False)

    class Meta:
        db_table = 'discogs_discogsrecord'

class DiscogsListing(models.Model):
    seller_id = models.IntegerField()
    record = models.ForeignKey(Record, on_delete=models.CASCADE)
    record_price = models.CharField(max_length=255)
    media_condition = models.CharField(max_length=255)

    class Meta:
        db_table = 'discogs_discogslisting'
        managed = False

class EbayListing(models.Model):
    id = models.AutoField(primary_key=True)
    ebay_id = models.CharField(max_length=255, unique=True)
    ebay_title = models.TextField()
    price = models.CharField(max_length=50)
    currency = models.CharField(max_length=10, blank=True)
    current_bid = models.CharField(max_length=50)
    bid_count = models.IntegerField(default=0)
    end_date = models.DateTimeField()
    creation_date = models.DateTimeField()
    
    # Parsed metadata
    artist = models.CharField(max_length=255, blank=True)
    album = models.CharField(max_length=255, blank=True)
    label = models.CharField(max_length=255, blank=True)
    format = models.CharField(max_length=100, blank=True)
    year = models.CharField(max_length=10, blank=True)
    record_condition = models.CharField(max_length=50, blank=True)
    sleeve_condition = models.CharField(max_length=50, blank=True)
    genre = models.CharField(max_length=100, blank=True)
    style = models.CharField(max_length=100, blank=True)
    
    wanted = models.BooleanField(default=False)
    evaluated = models.BooleanField(default=False)
    saved = models.BooleanField(default=False)
    enriched = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
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

class BanditTrainingInstance(models.Model):
    record = models.ForeignKey(Record, on_delete=models.CASCADE, null=True)
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