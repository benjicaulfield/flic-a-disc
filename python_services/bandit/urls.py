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
    path('performance_history/', views.performance_history, name='performance_history'),
    path('select_batch/', views.select_batch, name="select_batch"),
    path('rebuild_tfidf/', views.rebuild_tfidf_vocab, name='rebuild_tfidf'),
    path('ebay/annotated/', views.ebay_annotate, name='ebay_annotate'),
    path('discogs/knapsack/', views.discogs_knapsack, name='discogs_knapsack'),
    path('recommend/rotd/', views.record_of_the_day, name='record-of-the-day'),
    path('ebay_title_similarity_filter/', views.ebay_title_similarity_filter, name='ebay_title_similarity_filter'),
    path('stats/', views.get_stats, name='stats'),
    path('todos/', views.todos, name='todos'),
    path('todos/<int:todo_id>/', views.todo_detail, name='todo_detail'),
    path('ebay/stats/', views.ebay_stats, name='ebay_stats'),
    path('ebay/batch_performance/', views.record_ebay_batch_performance, name='record_ebay_batch_performance'),
    path('test', views.test, name='test'),
]