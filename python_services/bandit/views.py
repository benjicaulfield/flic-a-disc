import os
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
from rest_framework.decorators import api_view
from rest_framework.response import Response

from sklearn.metrics.pairwise import cosine_similarity

from .models import Record, DiscogsListing, EbayListing, BanditModel as BanditModelDB, BanditTrainingInstance, ThresholdConfig, BatchPerformance, TfIdfDB
from .training import BanditTrainer
from .features import RecordFeatureExtractor
from .bandit_selection import adaptive_batch_selection
from .enhance_listings import LookupByID
from .text_utils import create_mock_ebay_title, normalize_title
from .title_vectorizer import TitleVectorizer
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
    """
    Output: Current model performance metrics
    """
    try:
        try:
            latest_model = BanditModelDB.objects.filter(is_active=True).latest('created_at')
            training_stats = json.loads(latest_model.training_stats)
            total_instances = BanditTrainingInstance.objects.count()
            
            recent_instances = BanditTrainingInstance.objects.order_by('-timestamp')[:100]
            if recent_instances:
                correct_predictions = sum(1 for inst in recent_instances 
                                        if inst.predicted == inst.actual)
                recent_accuracy = correct_predictions / len(recent_instances)
            else:
                recent_accuracy = 0.0
            
            return Response({
                'model_version': latest_model.version,
                'training_accuracy': training_stats.get('val_accuracy', [0])[-1] if training_stats.get('val_accuracy') else 0,
                'recent_accuracy': recent_accuracy,
                'total_training_instances': total_instances,
                'model_created': latest_model.created_at,
                'explore_rate': 0.1  # Could be dynamic based on uncertainty
            })
            
        except BanditModelDB.DoesNotExist:
            return Response({
                'model_version': 'none',
                'training_accuracy': 0.0,
                'recent_accuracy': 0.0,
                'total_training_instances': 0,
                'explore_rate': 1.0,  # Full exploration without model
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

@api_view(['POST'])
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
def receive_feedback(request):
    print(f"📥 Feedback received: {len(request.data.get('records', []))} records")

    records = request.data.get('records', [])
    labels = request.data.get('labels', [])
    predictions = request.data.get('predictions', [])

    print(f"📊 Labels: {len(labels)}, Predictions: {len(predictions)}")

    if records:
        print(f"🔍 First record keys: {records[0].keys()}")
        print(f"🔍 First record: {records[0]}")
    
    if not records or not labels or not predictions:
        return Response(
            {'error': 'Missing required fields'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
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
            'f1_score': f1_score
        })
        
    
    return Response({
        'status': 'feedback stored',
        'buffer_size': len(feedback_buffer)
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
    
    correct = request.data.get('correct')
    total = request.data.get('total')
        
    if correct is None or total is None:
        return Response({'error': 'Missing data'}, status=400)
    
    accuracy = correct / total if total > 0 else 0

    try:
        active_model = BanditModelDB.objects.get(is_active=True)
        batch_num = active_model.batch_count
        print(f"📊 Batch number: {batch_num}")

    except BanditModelDB.DoesNotExist:
        batch_num = 0
        print(f"📊 No active model, using batch 0")


    BatchPerformance.objects.create(
        batch_number = batch_num,
        correct = correct,
        total = total,
        accuracy = accuracy
    )

    recent_batches = BatchPerformance.objects.all()[:10]

    if recent_batches:
        window_correct = sum(batch.correct for batch in recent_batches)
        window_total = sum(batch.total for batch in recent_batches)
        window_accuracy = (window_correct / window_total * 100) if window_total > 0 else 0
        window_size = len(recent_batches)
    else:
        window_correct = correct
        window_total = total
        window_accuracy = accuracy * 100
        window_size = 1

    return Response({
        'batch_accuracy': accuracy * 100,
        'batch_correct': correct,
        'batch_total': total,
        'cumulative_accuracy': window_accuracy,  # This is now sliding window
        'total_batches': window_size
    })

@api_view(['GET', 'POST'])
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
            'record_price': listing.record_price or '0',  # From listing, not record
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

        description = ""
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
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                system="You are Byron Coley. Write only the review text, no meta-commentary.",
                messages=[{"role": "user", "content": prompt}])
            description = message.content[0].text
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

@api_view(['GET', 'POST'])
def ebay_title_similarity_filter(request):
    threshold = 0.75
    lookup = LookupByID()
    try:
        similarity_index = TfIdfDB.objects.filter(is_active=True).latest('created_at')
        model_data = pickle.loads(similarity_index.model_weights)
        vectorizer = model_data['vectorizer']
        keeper_embeddings = model_data['keeper_embeddings']
    except TfIdfDB.DoesNotExist:
        return Response("FUCK NO INDEX GODDAMNIT")
    
    ebay_listings = EbayListing.objects.filter(evaluated=False)
    ebay_titles = []
    ebay_ids = []

    for listing in ebay_listings:
        normalized = normalize_title(listing.ebay_title)
        if normalized:
            ebay_titles.append(normalized)
            ebay_ids.append(listing.ebay_id)

    ebay_embeddings = vectorizer.vectorizer.transform(ebay_titles)

    keeper_similarities = cosine_similarity(ebay_embeddings, keeper_embeddings)
    final_scores = keeper_similarities.max(axis=1)

    results = []
    for i, score in enumerate(final_scores):
        if score >= threshold:
            listing = EbayListing.objects.get(ebay_id=ebay_ids[i])
            results.append({
                'id': listing.id,
                'ebay_id': ebay_ids[i],
                'score': float(score),
                'ebay_title': listing.ebay_title,
                'bid': listing.current_bid,
            })

    results.sort(key=lambda x: x['score'], reverse=True)

    return Response({
        'results': results,
        'total': len(results)
    })

@api_view(['GET'])
def process_annotations(request):
    pass

@api_view(['GET'])
def get_stats(request):
    total_discogs = Record.objects.count()
    evaluated_discogs = Record.objects.filter(evaluated=True).count()
    keepers = Record.objects.filter(evaluated=True, wanted=True).count()
    non_keepers = Record.objects.filter(evaluated=True, wanted=False).count()

    total_ebay = EbayListing.objects.count()
    evaluated_ebay = EbayListing.objects.filter(evaluated=True).count()
    enriched_ebay = EbayListing.objects.filter(enriched=True).count()

    training_instances = BanditTrainingInstance.objects.count()

    try:
        active_model = BanditModelDB.objects.get(is_active=True)
        model_stats = active_model.training_stats
        batch_count = active_model.batch_count
    except BanditModelDB.DoesNotExist:
        model_stats = {}
        batch_count = 0

    try:
        tfidf_model = TfIdfDB.objects.get(is_active=True)
        vocab_size = tfidf_model.training_stats.get('keeper_count', 0) + tfidf_model.training_stats.get('non_keeper_count', 0)
    except TfIdfDB.DoesNotExist:
        vocab_size = 0

    recent_batches = BatchPerformance.objects.all()[:10]
    avg_accuracy = recent_batches.aggregate(Avg('accuracy'))['accuracy__avg'] if recent_batches.exists() else 0

    return Response({
        'discogs': {
            'total': total_discogs,
            'evaluated': evaluated_discogs,
            'keepers': keepers,
            'non_keepers': non_keepers
        },
        'ebay': {
            'total': total_ebay,
            'evaluated': evaluated_ebay,
            'enriched': enriched_ebay
        },
        'training': {
            'instances': training_instances,
            'batch_count': batch_count
        },
        'model': {
            'stats': model_stats,
            'vocab_size': vocab_size,
            'avg_accuracy': round(avg_accuracy, 3) if avg_accuracy else None
        }
    })