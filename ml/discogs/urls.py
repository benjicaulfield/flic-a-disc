from django.urls import path
from . import views
from bandit.views import discogs_knapsack

urlpatterns = [
    path('select-batch/', views.select_batch, name='select_catalog_batch'),
    path('enrich/', views.enrich_records, name='enrich_records'),
    path('knapsack/', discogs_knapsack, name='knapsack_solve'),
    path('oof/', views.oof_batch, name='oof_batch'),
]