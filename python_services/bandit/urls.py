from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.train, name='train'),
    path('predict/', views.predict, name='predict'),
    path('metrics/', views.metrics, name='metrics'),
    path('health/', views.health, name='health'),
    path('retrain/', views.retrain, name='retrain'),
    path('feedback/', views.receive_feedback, name="feedback"),
    path('performance/', views.record_batch_performance, name='performance'),
    path('select_batch/', views.select_batch, name="select_batch"),
    path('rebuild_tfidf/', views.rebuild_tfidf_vocab, name='rebuild_tfidf'),
    path('ebay/annotate/', views.ebay_annotate, name='ebay_annotate'),
    path('recommend/rotd/', views.record_of_the_day, name='record-of-the-day'),
    path('ebay_first_pass/', views.ebay_title_similarity_filter, name='ebay_title_similarity_filter'),
    path('process_annotations', views.process_annotations, name="process_annotations"),
    path('stats/', views.get_stats, name='stats'),
]