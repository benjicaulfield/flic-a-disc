import json
import time
import torch 
import pickle
import random
import numpy as np 
import anthropic
from decouple import config

from django.db.models import Count, Avg, Max
from django.utils import timezone

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from sklearn.metrics.pairwise import cosine_similarity
from .models import (Record, DiscogsListing, EbayListing, BanditModel as BanditModelDB, BanditTrainingInstance, 
                     ThresholdConfig, BatchPerformance, TfIdfDB, Todo, EbayBatchPerformance,
                     KnapsackWeights, DiscogsSeller)
from .training import BanditTrainer
from .knapsack import knapsack, score_and_filter_seller_listings
from .features import RecordFeatureExtractor
from .bandit_selection import adaptive_batch_selection
from .enhance_listings import LookupByID
from .text_utils import create_mock_ebay_title, normalize_title
from .title_vectorizer import TitleVectorizer
from .utils.get_user_inventory import get_inventory
from .discogs_client import authenticate_client

trainer = BanditTrainer()
feedback_buffer = []

@api_view(['POST'])
def predict(request):
    try:
        print("📥 Predict endpoint called")
        records = request.data.get('records', [])
        print(f"📊 Received {len(records)} records")
        
        if not records:
            return Response({'error': 'No records provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        print("🔄 Loading model...")
        if not trainer.model:
            if not trainer.load_latest_model():
                return Response({
                    'error': 'No trained model available',
                    'predictions': [0.5] * len(records),
                    'model_version': 'none'
                }, status=status.HTTP_200_OK)
        
        print("🔄 Extracting features...")
        features = trainer.feature_extractor.extract_batch_features(records)
        features_tensor = torch.FloatTensor(features)
        
        print("🔄 Getting predictions...")
        sampled_probs = trainer.model.thompson_sample(features_tensor)
        mean_probs, variances = trainer.model.predict_with_uncertainty(features_tensor)
        mean_probs = mean_probs.cpu().numpy()
        uncertainties = np.sqrt(variances.cpu().numpy())

        print("✅ Predictions successful")
        return Response({
            'predictions': sampled_probs.tolist(),
            'mean_predictions': mean_probs.tolist(),
            'uncertainties': uncertainties.tolist(),
            'model_version': 'latest'
        })
        
    except Exception as e:
        print(f"❌ ERROR in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response({
            'error': f'Prediction failed: {str(e)}',
            'predictions': [0.5] * len(request.data.get('records', [])),
            'model_version': 'error'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST']) 
def train(request):
    try:
        instances = request.data.get('instances', [])
        
        if not instances:
            return Response({'error': 'No training instances provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate instance format
        for instance in instances:
            if not all(key in instance for key in ['id', 'predicted', 'actual']):
                return Response({
                    'error': 'Invalid instance format. Required: id, predicted, actual'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Update model with new instances
        result = trainer.update_model_online(instances)
        
        if 'error' in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(result)
        
    except Exception as e:
        return Response({
            'error': f'Training failed: {str(e)}',
            'model_updated': False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['GET'])
def metrics(request):
    try:
        try:
            latest_model = BanditModelDB.objects.filter(is_active=True).latest('created_at')
            training_stats = json.loads(latest_model.training_stats)
            total_instances = BanditTrainingInstance.objects.count()
            
            # ✅ Use last 100 batches for rolling window
            recent_batches = BatchPerformance.objects.all()[:100]
            
            if recent_batches:
                total_correct = sum(b.correct for b in recent_batches)
                total_records = sum(b.total for b in recent_batches)
                rolling_accuracy = total_correct / total_records if total_records > 0 else 0.0
                
                # Get last 10 batches for recent trend
                last_10 = recent_batches[:10]
                recent_accuracy = sum(b.accuracy for b in last_10) / len(last_10) if last_10 else 0.0
            else:
                rolling_accuracy = 0.0
                recent_accuracy = 0.0
            
            return Response({
                'model_version': latest_model.version,
                'training_accuracy': training_stats.get('val_accuracy', [0])[-1] if training_stats.get('val_accuracy') else 0,
                'recent_accuracy': recent_accuracy,
                'rolling_100_accuracy': rolling_accuracy,  # ✅ NEW
                'total_training_instances': total_instances,
                'total_batches': BatchPerformance.objects.count(),  # ✅ NEW
                'model_created': latest_model.created_at,
                'explore_rate': 0.1
            })
            
        except BanditModelDB.DoesNotExist:
            return Response({
                'model_version': 'none',
                'training_accuracy': 0.0,
                'recent_accuracy': 0.0,
                'rolling_100_accuracy': 0.0,
                'total_training_instances': 0,
                'total_batches': 0,
                'explore_rate': 1.0,
                'message': 'No trained model available'
            })
            
    except Exception as e:
        return Response({
            'error': f'Metrics retrieval failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def health(request):
    return Response({
        'status': 'healthy', 
        'service': 'bandit-ml',
        'model_loaded': trainer.model is not None,
        'feature_extractor_loaded': trainer.feature_extractor is not None
    })


@api_view(['POST', 'GET'])
@permission_classes([AllowAny])
def retrain(request):
    """Force complete retraining from scratch"""
    try:
        epochs = request.data.get('epochs', 50)
        batch_size = request.data.get('batch_size', 64)
        learning_rate = request.data.get('learning_rate', 0.001)
        
        history = trainer.train_new_model(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return Response({
            'message': 'Model retrained successfully',
            'final_accuracy': history['val_accuracy'][-1],
            'final_loss': history['val_loss'][-1],
            'epochs_completed': len(history['train_loss'])
        })
        
    except Exception as e:
        return Response({
            'error': f'Retraining failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
@permission_classes([AllowAny])
def receive_feedback(request):
    print("=" * 80)
    print("🚀 RECEIVE_FEEDBACK CALLED!")
    print(f"📍 Request method: {request.method}")
    print(f"📍 Request path: {request.path}")
    print(f"📍 Request data keys: {request.data.keys() if request.data else 'NO DATA'}")
    print("=" * 80)
    
    print(f"📥 Feedback received: {len(request.data.get('records', []))} records")
    print(f"📥 Feedback received: {len(request.data.get('records', []))} records")

    records = request.data.get('records', [])
    labels = request.data.get('labels', [])
    predictions = request.data.get('predictions', [])
    mean_predictions = request.data.get('mean_predictions', [])  # ✅ NEW
    uncertainties = request.data.get('uncertainties', [])        # ✅ NEW

    print(f"📊 Labels: {len(labels)}, Predictions: {len(predictions)}")

    if records:
        print(f"🔍 First record keys: {records[0].keys()}")
        print(f"🔍 First record: {records[0]}")
    
    if not records or not labels or not predictions:
        return Response(
            {'error': 'Missing required fields'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # ✅ NEW: Get current batch number
    try:
        active_model = BanditModelDB.objects.get(is_active=True)
        batch_num = active_model.batch_count
    except BanditModelDB.DoesNotExist:
        batch_num = 0
    
    # ✅ NEW: Calculate and save batch performance
    labels_array = np.array(labels)
    predictions_array = np.array(predictions)
    
    # Use threshold of 0.5 for binary classification
    predicted_labels = (predictions_array > 0.5).astype(int)
    correct = np.sum(predicted_labels == labels_array)
    accuracy = correct / len(labels) if len(labels) > 0 else 0.0
    
    BatchPerformance.objects.create(
        batch_number=batch_num,
        correct=int(correct),
        total=len(labels),
        accuracy=accuracy
    )
    
    print(f"📊 Batch {batch_num} Performance: {correct}/{len(labels)} = {accuracy:.2%}")
    
    # Increment batch counter
    try:
        active_model = BanditModelDB.objects.get(is_active=True)
        active_model.batch_count += 1
        active_model.save()
    except BanditModelDB.DoesNotExist:
        pass
    
    # Original buffer logic
    instances = []
    for i, record in enumerate(records):
        listing_id = record.get('listing_id')
        print(f"🔍 Processing listing_id: {listing_id}")
        instances.append({
            'id': listing_id,
            'predicted': predictions[i],
            'actual': labels[i]
        })

    print(f"📦 Created {len(instances)} instances")
    print(f"📦 First instance: {instances[0] if instances else 'None'}")

    feedback_buffer.append({
        'instances': instances,
        'timestamp': time.time()
    })
    RETRAIN_THRESHOLD = 5  # 5 pages = 200 records
    print(f"📊 Buffer size: {len(feedback_buffer)}/{RETRAIN_THRESHOLD}")

    # Trigger retraining if buffer is large enough
    if len(feedback_buffer) >= RETRAIN_THRESHOLD:
        # Flatten all instances from buffer
        all_instances = []
        for batch in feedback_buffer:
            all_instances.extend(batch['instances'])
        
        # Use incremental learning, not full retrain
        print(f"🔄 Triggering incremental update with {len(all_instances)} instances")
        result = trainer.update_model_online(all_instances)
        print(f"✅ Update result: {result}")

        feedback_buffer.clear()
        new_threshold, f1_score = calculate_optimal_threshold()
        print(f"📊 Optimal threshold: {new_threshold:.3f} (F1: {f1_score:.3f})")

        return Response({
            'status': 'model updated',
            'result': result,
            'new_threshold': new_threshold,
            'f1_score': f1_score,
            'batch_performance': {  # ✅ NEW
                'batch_num': batch_num,
                'accuracy': accuracy,
                'correct': correct,
                'total': len(labels)
            }
        })
    
    return Response({
        'status': 'feedback stored',
        'buffer_size': len(feedback_buffer),
        'batch_performance': {  # ✅ NEW
            'batch_num': batch_num,
            'accuracy': accuracy,
            'correct': correct,
            'total': len(labels)
        }
    })

def calculate_optimal_threshold(window_size=500):
    recent_instances = BanditTrainingInstance.objects.order_by('-timestamp')[:window_size]

    if len(recent_instances) < 50:
        return 0.5, 0.0
    
    predictions = []
    actuals = []

    for instance in recent_instances:
        predictions.append(instance.predicted_prob)
        actuals.append(1 if instance.actual else 0)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.05, 0.95, 0.05):
        pred_binary = (predictions > threshold).astype(int)
        tp = ((pred_binary == 1) & (actuals == 1)).sum()
        fp = ((pred_binary == 1) & (actuals == 0)).sum()
        fn = ((pred_binary == 0) & (actuals == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    config, _ = ThresholdConfig.objects.get_or_create(id=1)
    config.threshold = best_threshold
    config.f1_score = best_f1
    config.window_size = window_size
    config.save()

    return best_threshold, best_f1

@api_view(['GET'])
def optimize_threshold(request):
    window_size = int(request.query_params.get('window_size'))
    threshold, f1 = calculate_optimal_threshold(window_size)
    print(threshold)

    return Response({
        'optimal_threshold': float(threshold),
        'best_f1': float(f1),
        'window_size': window_size
    })

@api_view(['POST'])
def select_batch(request):
    try:
        active_model = BanditModelDB.objects.get(is_active=True)
        batch_num = active_model.batch_count
    except BanditModelDB.DoesNotExist:
        batch_num = 0

    records = request.data.get('records', [])
    mean_predictions = request.data.get('mean_predictions', [])
    uncertainties = request.data.get('uncertainties', [])

    if not records or not mean_predictions or not uncertainties:
        return Response(
            {'error': 'Missing required fields'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    
    # Create candidate indices (0 to len-1)
    candidate_indices = list(range(len(records)))
    
    selected_indices = adaptive_batch_selection(
        candidates=candidate_indices,
        predictions=mean_predictions,
        uncertainties=uncertainties,
        batch_num=batch_num,
        total_batch_size=20,
        total_batches=100,
        random_count=3
    )
    
    return Response({
        'selected_indices': selected_indices,
        'batch_num': batch_num
    })

@api_view(['POST'])
def record_batch_performance(request):
    print("📊 record_batch_performance called!")
    print(f"📊 Request data: {request.data}")
    
    recent_batches = BatchPerformance.objects.all()[:100]

    if recent_batches:
        window_correct = sum(batch.correct for batch in recent_batches)
        window_total = sum(batch.total for batch in recent_batches)
        window_accuracy = (window_correct / window_total * 100) if window_total > 0 else 0
        window_size = len(recent_batches)
        
        # Get the most recent batch for "this batch" stats
        latest_batch = recent_batches[0]
        batch_accuracy = latest_batch.accuracy * 100
        batch_correct = latest_batch.correct
        batch_total = latest_batch.total
    else:
        # No batches yet - return zeros
        window_accuracy = 0
        window_size = 0
        batch_accuracy = 0
        batch_correct = 0
        batch_total = 0

    print(f"📊 Returning stats: {window_size} batches, {window_accuracy:.1f}% accuracy")
    
    return Response({
        'batch_accuracy': batch_accuracy,
        'batch_correct': batch_correct,
        'batch_total': batch_total,
        'cumulative_accuracy': window_accuracy,
        'total_batches': window_size
    })

@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def ebay_annotate(request):
    annotations = request.data.get('annotations', [])
    if not annotations: return Response({'error': 'No annotations'}, status=400)

    keeper_ids = []
    non_keeper_ids = []

    errors = []

    for annotation in annotations:
        ebay_id = annotation.get('ebay_id')
        label = annotation.get('label')  # True = keeper, False = non-keeper
        
        if label: keeper_ids.append(ebay_id)
        else: non_keeper_ids.append(ebay_id)
        
        try:
            # Update the listing with annotation
            listing = EbayListing.objects.get(ebay_id=ebay_id)
            listing.wanted = label
            listing.evaluated = True
            listing.save()
            
        except EbayListing.DoesNotExist:
            errors.append(f"Listing {ebay_id} not found")
            continue
    
    
    return Response({
        'success': True,
        'keepers': len(keeper_ids),
        'non_keepers': len(non_keeper_ids),
        'keeper_ids': keeper_ids,
        'errors': errors if errors else None
    })

@api_view(['GET'])
def test(request):
    recent_listings = DiscogsListing.objects.order_by('-id')[:500]
    bandit = BanditTrainer()
    success = bandit.train_new_model()
    print(success)
    return Response("ok")

def select_record_of_the_day():
    recent_listings = DiscogsListing.objects.order_by('-id')[:500]
    bandit = BanditTrainer()
    success = bandit.load_latest_model()

    if not success: raise Exception("failed to load model")

    scored = []
    for listing in recent_listings:
        record = listing.record

        record_data = {
            'artist': record.artist or '',
            'title': record.title or '',
            'label': record.label or '',
            'genres': record.genres or [],
            'styles': record.styles or [],
            'wants': record.wants or 0,
            'haves': record.haves or 0,
            'price': listing.price or '0',
            'currency': listing.currency or '',  # From listing, not record
            'year': record.year,
            'media_condition': listing.media_condition or '',  # From listing
            '_is_ebay': False  # This is Discogs data, not eBay
        }

        features = bandit.feature_extractor.extract_features(record_data)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        mean, _ = bandit.model.predict_with_uncertainty(features_tensor)
        scored.append((listing, mean.item()))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top_50 = scored[:50]

    selected_listing, score = random.choice(top_50)
    return selected_listing.record

@api_view(['GET'])
def record_of_the_day(request):
    try:
        record = select_record_of_the_day()

        if not record.record_image and record.discogs_id:
            try:
                d = authenticate_client()
                release = d.release(record.discogs_id)
                images = release.images
                if images:
                    record.record_image = images[0]['uri']
                    record.save()
            except Exception as e:
                print(f"Error fetching image: {e}")

        description = getattr(record, 'description', None)

        if not description:
            try:
                client = anthropic.Anthropic(api_key=config("ANTHROPIC_KEY"))
                prompt = f"""Write a 2-3 sentence description in the style of Byron
                            Coley writing for Forced Exposure magazine circa 1990. 
                            Use his characteristic mix of underground music knowledge, 
                            obscure references, passionate enthusiasm, and  irreverent tone.
                            Artist: {record.artist}
                            Album: {record.title}
                            Year: {record.year}
                            Label: {record.label}
                            Genres: {', '.join(record.genres) if record.genres else 'N/A'}
                            Styles: {', '.join(record.styles) if record.styles else 'N/A'}

                            Write ONLY the review text, nothing else."""
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    system="You are Byron Coley. Write only the review text, no meta-commentary.",
                    messages=[{"role": "user", "content": prompt}])
                description = message.content[0].text
                record.description = description
                record.save()

            except Exception as e:
                print(f"Error generating description: {e}")
                description = "A hidden gem worth discovering"

        return Response({
            'artist': record.artist,
            'title': record.title,
            'year': record.year,
            'record_image': record.record_image,
            'description': description,
        })
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
@api_view(['GET', 'POST'])
def rebuild_tfidf_vocab(request):
    vocab_size = 10000

    keepers = Record.objects.filter(evaluated=True, wanted=True)
    keeper_titles = []
    for record in keepers:
        mock_title = create_mock_ebay_title({
            'artist': record.artist,
            'title': record.title,
            'label': record.label,
            'year': record.year
        })
        if mock_title:
            keeper_titles.append(mock_title)

    

    keeper_titles

    vectorizer = TitleVectorizer(vocab_size)
    vectorizer.fit(keeper_titles)

    keeper_embeddings = vectorizer.vectorizer.transform(keeper_titles)

    model_weights = pickle.dumps({
        'vectorizer': vectorizer,
        'keeper_embeddings': keeper_embeddings,
        'vocab_size': len(vectorizer.vectorizer.vocabulary_)
        })

    similarity_model = TfIdfDB.objects.create(
        version=f"v{timezone.now().strftime('%Y%m%d_%H%M%S')}",  # ← Fixed
        model_weights=model_weights,
        hyperparams={
            'vocab_size': vocab_size,
        },
        training_stats={
            'keeper_count': keeper_embeddings.shape[0],
        },
        is_active=True
    )

    TfIdfDB.objects.filter(is_active=True).exclude(id=similarity_model.id).update(is_active=False)

    return Response({
        'success': True,
        'version': similarity_model.version,
        'vocab_size': len(vectorizer.vectorizer.vocabulary_),
        'keeper_count': keeper_embeddings.shape[0],
    })

@api_view(['POST'])
def ebay_title_similarity_filter(request):
    listings = request.data.get('listings', [])
    top_n = request.data.get('top_n', 500)
    if not listings: return Response({'error': 'no listings'}, status=400)
    
    try:
        similarity_index = TfIdfDB.objects.filter(is_active=True).latest('created_at')
        model_data = pickle.loads(similarity_index.model_weights)
        vectorizer = model_data['vectorizer']
        keeper_embeddings = model_data['keeper_embeddings']
    except TfIdfDB.DoesNotExist:
        return Response("no tf-idf model found")
    
    ebay_titles = []
    ebay_ids = []

    for listing in listings:
        title = listing.get('ebay_title', '')
        ebay_id = listing.get('ebay_id', '')
        normalized = normalize_title(title)
        if normalized and ebay_id:
            ebay_titles.append(normalized)
            ebay_ids.append(ebay_id)
    
    if not ebay_titles:
        return Response({'results': [], 'total': 0, 'message': 'No unevaluated listings'})

    ebay_embeddings = vectorizer.vectorizer.transform(ebay_titles)
    keeper_similarities = cosine_similarity(ebay_embeddings, keeper_embeddings)
    final_scores = keeper_similarities.max(axis=1)

    sorted_items = sorted(
        zip(listings, final_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    top_listings = [item for item, _ in sorted_items]
    random.shuffle(top_listings)
    
    return Response({'top_listings': top_listings})

@api_view(['GET'])
@permission_classes([AllowAny])
def get_stats(request):
    total_discogs = Record.objects.count()
    evaluated_discogs = Record.objects.filter(evaluated=True).count()
    keepers = Record.objects.filter(evaluated=True, wanted=True).count()
    non_keepers = Record.objects.filter(evaluated=True, wanted=False).count()
    keeper_rate = (keepers / evaluated_discogs * 100) if evaluated_discogs > 0 else 0

    recent_batches = BatchPerformance.objects.all()[:100]
    if recent_batches:
        total_correct = sum(b.correct for b in recent_batches)
        total_evaluated = sum(b.total for b in recent_batches)
        discogs_accuracy = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
    else:
        discogs_accuracy = 0

    total_ebay = EbayListing.objects.count()

    training_instances = BanditTrainingInstance.objects.count()

    try:
        active_model = BanditModelDB.objects.get(is_active=True)
        model_stats = active_model.training_stats
        batch_count = active_model.batch_count
        model_version = active_model.version
    except BanditModelDB.DoesNotExist:
        model_stats = {}
        batch_count = 0
        model_version = "none"

    try:
        tfidf_model = TfIdfDB.objects.get(is_active=True)
        vocab_size = tfidf_model.training_stats.get('keeper_count', 0) + tfidf_model.training_stats.get('non_keeper_count', 0)
    except TfIdfDB.DoesNotExist:
        vocab_size = 0

    last_10_batches = BatchPerformance.objects.all()[:10]
    avg_accuracy = recent_batches.aggregate(Avg('accuracy'))['accuracy__avg'] if recent_batches.exists() else 0

    return Response({
        # For dashboard compatibility
        'total_records': total_discogs,
        'evaluated_records': evaluated_discogs,
        'keeper_count': keepers,
        'keeper_rate': keeper_rate,
        'discogs_accuracy': discogs_accuracy,
        'ebay_accuracy': 0,  # TODO: implement when eBay batch tracking exists
        'model_version': model_version,
        'total_batches': batch_count,
        
        # Detailed breakdown
        'discogs': {
            'total': total_discogs,
            'evaluated': evaluated_discogs,
            'keepers': keepers,
            'non_keepers': non_keepers
        },
        'ebay': {
            'total': total_ebay,
        },
        'training': {
            'instances': training_instances,
            'batch_count': batch_count
        },
        'model': {
            'stats': model_stats,
            'vocab_size': vocab_size,
            'avg_accuracy': round(avg_accuracy * 100, 1) if avg_accuracy else 0  # Convert to percentage
        }
    })

@api_view(['GET'])
@permission_classes([AllowAny])
def performance_history(request):
    batches = BatchPerformance.objects.all()[:100]

    return Response({
        'batches': [
            {
                'batch_number': b.batch_number,
                'accuracy': b.accuracy,
                'correct': b.correct,
                'total': b.total
            }
            for b in reversed(list(batches))
        ]
    })

@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def todos(request):
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return Response({'error': 'no user id'}, status=400)
    
    if request.method == 'GET':
        user_todos = Todo.objects.filter(user_id = user_id)
        return Response([{
            'id': str(todo.id),
            'text': todo.text,
            'status': todo.status,
            'order': todo.order
        } for todo in user_todos])
    
    elif request.method == 'POST':
        todo = Todo.objects.create(
            user_id=user_id,
            text=request.data['text'],
            status=request.data.get('status', 'backlog'),
            order=request.data.get('order', 0)
        )
        return Response({
            'id': str(todo.id),
            'text': todo.text,
            'status': todo.status,
            'order': todo.order
        }, status=201)

@api_view(['PATCH', 'DELETE'])
@permission_classes([AllowAny])
def todo_detail(request, todo_id):
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return Response({'error': 'Missing user ID'}, status=400)
    
    try:
        todo = Todo.objects.get(id=todo_id, user_id=user_id)
    except Todo.DoesNotExist:
        return Response({'error': 'Not found'}, status=404)
    
    if request.method == 'PATCH':
        if 'text' in request.data:
            todo.text = request.data['text']
        if 'status' in request.data:
            todo.status = request.data['status']
        if 'order' in request.data:
            todo.order = request.data['order']
        todo.save()
        return Response({
            'id': str(todo.id),
            'text': todo.text,
            'status': todo.status,
            'order': todo.order
        })
    
    elif request.method == 'DELETE':
        todo.delete()
        return Response(status=204)
    
# In bandit/views.py, add:

@api_view(['GET'])
@permission_classes([AllowAny])
def ebay_stats(request):
    """Get eBay annotation statistics"""
    total_listings = EbayListing.objects.count()
    
    return Response({
        'total_listings': total_listings
    })

@api_view(['POST'])
@permission_classes([AllowAny])
def record_ebay_batch_performance(request):
    """Record eBay batch performance after annotations"""
    correct = request.data.get('correct')
    total = request.data.get('total')
    
    if correct is None or total is None:
        return Response({'error': 'Missing data'}, status=400)
    
    # Get next batch number
    last_batch = EbayBatchPerformance.objects.first()
    batch_num = (last_batch.batch_number + 1) if last_batch else 1
    
    accuracy = correct / total if total > 0 else 0
    
    EbayBatchPerformance.objects.create(
        batch_number=batch_num,
        correct=correct,
        total=total,
        accuracy=accuracy
    )
    
    # Calculate rolling stats
    recent_batches = EbayBatchPerformance.objects.all()[:100]
    total_correct = sum(b.correct for b in recent_batches)
    total_evaluated = sum(b.total for b in recent_batches)
    rolling_accuracy = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
    
    return Response({
        'batch_number': batch_num,
        'accuracy': accuracy * 100,
        'rolling_accuracy': rolling_accuracy,
        'total_batches': EbayBatchPerformance.objects.count()
    })

@api_view(['POST'])
@permission_classes([AllowAny])
def discogs_knapsack(request):
    sellers = request.data.get("sellers", [])
    budget = request.data.get("budget", 0)
    knapsacks = []

    for seller_data in sellers:
        seller = seller_data['name']
        scored_inventory = score_and_filter_seller_listings(seller)
        selected = knapsack(scored_inventory, budget)
        selected_ids = {id(item) for item in selected}
        contenders = [item for item in scored_inventory if id(item) not in selected_ids][:40]
        for item in selected + contenders:
            item['score'] = float(item['score'])
            item['price'] = float(item['price'])

        knapsacks.append({
            "seller": seller,
            "knapsack": selected,
            "contenders": contenders,
            "total_selected": len(selected),
            "total_cost": float(sum(item['price'] for item in selected)),
            "total_score": float(sum(item['score'] for item in selected)),
        })

    return Response({
        "knapsacks": knapsacks,
        "budget": budget
    })

    



