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
    path('ebay/annotate/', views.ebay_annotate, name='ebay_annotate'),
    path('recommend/rotd/', views.record_of_the_day, name='record_of_the_day'),
    path('ebay_title_similarity_filter/', views.ebay_title_similarity_filter, name='ebay_title_similarity_filter'),
    path('process_annotations', views.process_annotations, name="process_annotations"),
    path('discogs/stats/', views.discogs_stats, name='discogs_stats'),
    path('todos/', views.todos, name='todos'),
    path('todos/<int:todo_id>/', views.todo_detail, name='todo_detail'),
    path('ebay/stats/', views.ebay_stats, name='ebay_stats'),
    path('ebay/unenriched_keepers/', views.get_unenriched_keepers),
    path('ebay/update_keeper/', views.update_keeper),   
    path('ebay/batch_performance/', views.record_ebay_batch_performance, name='record_ebay_batch_performance'),
    #path('feature_importance/', views.feature_importance_analysis, name='feature_importance'),
    path('title_similarity_analysis_v2', views.title_similarity_analysis_v2, name="title_similarity"),
    path('test_flow', views.test_flow, name="test_flow"),
]