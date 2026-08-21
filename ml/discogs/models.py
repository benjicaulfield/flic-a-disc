from django.db import models


class CatalogBatch(models.Model):
    """Pre-computed batch of catalog records, already enriched with API data"""
    records = models.JSONField()  # 20 enriched records
    predictions = models.JSONField(default=list)  # Model predictions for each record
    mean_predictions = models.JSONField(default=list)  # Mean predictions
    uncertainties = models.JSONField(default=list)  # Uncertainty values
    created_at = models.DateTimeField(auto_now_add=True)
    used = models.BooleanField(default=False)

    class Meta:
        ordering = ['created_at']
        verbose_name = 'Catalog Batch'
        verbose_name_plural = 'Catalog Batches'

    def __str__(self):
        return f"Batch {self.id} ({'used' if self.used else 'available'})"
