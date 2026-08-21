import torch
import numpy as np
import pickle
import json
import mlflow
import random

from django.utils import timezone as django_timezone
from django.db.models import F, Max
from django.conf import settings

from .utils.calculate_optimal_threshold import calculate_optimal_threshold
from .models import (DiscogsRecord, DiscogsListing, BanditModel as BanditModelDB, BanditTrainingInstance,
                     ThresholdConfig, BatchPerformance)

from .features import RecordFeatureExtractor
from .neural_bandit import NeuralContextualBandit
from .triplet_generation import generate_triplets, generate_triplets_from_batch

class BanditTrainer:
    def __init__(self):
        project_root = settings.BASE_DIR
        tracking_uri = f"file://{project_root}/mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("discogs-bandit-model")
        self.feature_extractor = None
        self.model = None 
        self.training_history = []
    
    def prepare_training_data(self):
        # Query listings where the associated record has been evaluated
        evaluated_records = list(DiscogsRecord.objects.filter(evaluated=True))

        if not evaluated_records:
            raise ValueError("No evaluated listings found for training")

        # Most recent DiscogsListing per record, in two queries (not one per record) —
        # DiscogsRecord has no price/condition of its own, that's per-listing.
        latest_listing_ids = (
            DiscogsListing.objects
            .filter(record_id__in=[r.id for r in evaluated_records])
            .values('record_id')
            .annotate(latest_id=Max('id'))
        )
        latest_listings = {
            l.id: l for l in DiscogsListing.objects.filter(
                id__in=[row['latest_id'] for row in latest_listing_ids]
            )
        }
        listing_by_record = {
            row['record_id']: latest_listings[row['latest_id']]
            for row in latest_listing_ids
        }

        records = []
        labels = []

        for record in evaluated_records:
            listing = listing_by_record.get(record.id)
            record_dict = {
                'artist': record.artist,
                'title': record.title,
                'label': record.label,
                'genres': record.genres,
                'styles': record.styles,
                'wants': record.wants,
                'haves': record.haves,
                'year': record.year,
                'record_price': f"{listing.record_price}, {listing.currency}" if listing else '',
                'media_condition': listing.media_condition if listing else '',
            }
            records.append(record_dict)
            labels.append(record.wanted)  # The evaluation decision
        
        print(f"Training targets sample: {labels[:10]}")
        print(f"Training targets type: {type(labels[0])}")
        print(f"Unique values: {set(labels)}")
        
        return records, labels
    
    def train_new_model(self, epochs=100, batch_size=32, learning_rate=0.01, use_tfidf=True, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print("=" * 60)
        print("🚀 Starting new model training")
        print("=" * 60)

        mlflow.set_experiment("discogs-bandit-model")

        with mlflow.start_run(run_name="initial_training"):
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_dims": [128, 64, 32],
                "dropout_rate": 0.2,
                "use_tfidf": use_tfidf
            })
        
            records, labels = self.prepare_training_data()
            combined = list(zip(records, labels))
            random.shuffle(combined)
            records, labels = zip(*combined)
            records, labels = list(records), list(labels)

            mlflow.log_param("num_training_samples", len(records))
            mlflow.log_param("num_keepers", sum(labels))
            print(f"📊 Loaded {len(records)} records ({sum(labels)} keepers, {len(labels) - sum(labels)} non-keepers)")
        
            self.feature_extractor = RecordFeatureExtractor(
                artist_vocab_size=1000,
                label_vocab_size=500,
                genre_vocab_size=100,
                style_vocab_size=200,
                title_tfidf_features=1000,
            )
            
            print("🔧 Fitting feature extractor...")
            self.feature_extractor.fit(records)
            vocab_sizes = self.feature_extractor.get_vocab_sizes()
            embedding_dims = self.feature_extractor.get_embedding_dims()
            vocab_sizes_converted = {
                'artist_vocab_size': vocab_sizes['artist'],
                'label_vocab_size': vocab_sizes['label'],
                'genre_vocab_size': vocab_sizes['genre'],
                'style_vocab_size': vocab_sizes['style']
            }

            embedding_dims_converted = {
                'artist_embedding_dim': embedding_dims['artist'],
                'label_embedding_dim': embedding_dims['label'],
                'genre_embedding_dim': embedding_dims['genre'],
                'style_embedding_dim': embedding_dims['style']
            }
            print(f"📐 Vocab sizes: {vocab_sizes}")
            print(f"📐 Embedding dims: {embedding_dims}")
            
            print("🏗️ Building model architecture...")
            self.model = NeuralContextualBandit(
                vocab_sizes=vocab_sizes_converted,
                embedding_dims=embedding_dims_converted,
                hidden_dims=[128, 64, 32],
                embedding_dim=64,
                tfidf_dim=len(self.feature_extractor.title_vectorizer.vectorizer.vocabulary_) if self.feature_extractor.use_tfidf else 0, 
                dropout_rate=0.2
            )
            print(f"✅ Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
            
            # Separate keepers and non-keepers for triplet generation
            keeper_records = [r for r, l in zip(records, labels) if l]
            non_keeper_records = [r for r, l in zip(records, labels) if not l]
            
            print(f"🔗 Generating triplets from {len(keeper_records)} keepers and {len(non_keeper_records)} non-keepers...")
            
            # Generate triplets
            triplet_data = None
            if len(keeper_records) >= 2 and len(non_keeper_records) > 0:
                triplets = generate_triplets(
                    keeper_records,
                    non_keeper_records,
                    num_triplets=min(len(keeper_records) * 10, 10000),
                    hard_mining=True,
                    feature_extractor=self.feature_extractor
                )
                triplet_data = triplets
                print(f"📦 Generated {len(triplets['anchors'])} triplets for contrastive learning")
            else:
                print("⚠️ Not enough data for triplets, training without contrastive loss")
            
            # Train with contrastive + supervised learning
            print(f"\n🎯 Starting training for {epochs} epochs...")
            print("-" * 60)
    
            history = self.model.fit(
                feature_extractor=self.feature_extractor,
                training_records=records,
                labels=labels,
                triplet_records=triplet_data,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            for epoch in range(len(history['val_accuracy'])):
                mlflow.log_metrics({
                    "val_accuracy": history['val_accuracy'][epoch],
                    "val_loss": history['val_loss'][epoch],
                    "train_loss": history['train_loss'][epoch],
                    "val_f1": history['val_f1'][epoch],
                    "val_precision": history['val_precision'][epoch],
                    "val_recall": history['val_recall'][epoch],
                }, step=epoch)

            mlflow.log_metrics({
                "final_val_accuracy": history['val_accuracy'][-1],
                "final_val_loss": history['val_loss'][-1]
            })

            mlflow.pytorch.log_model(self.model, "bandit_model")
            print(f"✅ Training complete! Logged to MLflow")
        
            self.save_model_to_db(history)
            print("✅ Model saved successfully!")
            print("=" * 60)
        
            return history
        
    def record_to_dict(self, record):
        listing = DiscogsListing.objects.filter(record_id=record.id).order_by('-id').first()
        return {
            'artist': record.artist,
            'title': record.title,
            'label': record.label,
            'genres': record.genres,
            'styles': record.styles,
            'wants': record.wants,
            'haves': record.haves,
            'year': record.year,
            'record_price': f"{listing.record_price}, {listing.currency}" if listing else '',
            'media_condition': listing.media_condition if listing else '',
        }
    
    def update_model_online(self, instances):
        if not self.model or not self.feature_extractor: return self.train_new_model()
        
        mlflow.set_experiment("discogs-bandit-model")
        with mlflow.start_run(run_name=f"batch_update", nested=True):
        
            threshold_config = ThresholdConfig.objects.first()
            threshold = threshold_config.threshold if threshold_config else 0.5
            
            # Get current batch count for tagging
            try:
                old_model = BanditModelDB.objects.get(is_active=True)
                batch_num = old_model.batch_count + 1
            except BanditModelDB.DoesNotExist:
                batch_num = 1
            
            # Log batch info
            mlflow.log_params({
                "batch_number": batch_num,
                "batch_size": len(instances),
                "learning_rate": 0.0001,
                "num_epochs": 10
            })
        
        
            records = []
            labels = []
            record_ids = []
            print(f"🎯 Starting online update with {len(instances)} instances")

            for instance in instances:
                try:
                    record = DiscogsRecord.objects.get(id=instance['id'])
                    record_dict = self.record_to_dict(record)
                    records.append(record_dict)
                    labels.append(instance['actual'])
                    record_ids.append(record.id)
                    
                except DiscogsRecord.DoesNotExist:
                    print(f"Warning: Record {instance['id']} not found, skipping")
                    continue
            
            if not records:
                return {'error': 'No valid records found for training'}
            
            print(f"📊 Found {len(records)} valid records for training")

            features = torch.FloatTensor(self.feature_extractor.extract_batch_features(records))
            labels_tensor = torch.FloatTensor(labels)

            self.model.eval()
            with torch.no_grad():
                mean_preds, variance_preds = self.model.forward(features)
                uncertainties = torch.sqrt(variance_preds).cpu().numpy()
        
            # Map listing ID to uncertainty
            instance_uncertainties = {}
            for i, instance in enumerate(instances):
                if i < len(uncertainties):
                    instance_uncertainties[instance['id']] = float(uncertainties[i])
            
            keeper_history = []
            non_keeper_history = []

            historical_records = DiscogsRecord.objects.filter(evaluated=True)
            
            for record in historical_records:  
                record_dict = self.record_to_dict(record)  
                if record.wanted:  
                    keeper_history.append(record_dict)
                else:
                    non_keeper_history.append(record_dict)

            triplets = generate_triplets_from_batch(
                current_batch=records,
                current_labels=labels,
                keeper_history=keeper_history,
                non_keeper_history=non_keeper_history,
                num_triplets_per_keeper=10,
                hard_mining_ratio=0.5
            )

            # Extract triplet features if available
            triplet_features = None
            if triplets:
                anchor_features = self.feature_extractor.extract_batch_features(triplets['anchors'])
                positive_features = self.feature_extractor.extract_batch_features(triplets['positives'])
                negative_features = self.feature_extractor.extract_batch_features(triplets['negatives'])
                
                triplet_features = {
                    'anchor': torch.FloatTensor(anchor_features),
                    'positive': torch.FloatTensor(positive_features),
                    'negative': torch.FloatTensor(negative_features)
                }
                print(f"📦 Generated {len(triplets['anchors'])} triplets for contrastive learning")
            
            # Online learning update with smaller learning rate
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            
            self.model.train()
            batch_losses = []

            # Multiple passes over the new data
            print(f"🔄 Running 10 training epochs...")

            for epoch in range(10):
                optimizer.zero_grad()
                
                losses = self.model.combined_loss(
                            features,
                            labels_tensor,
                            triplet_data=triplet_features)
                losses['total'].backward()
                optimizer.step()

                batch_losses.append(losses['total'].item())
                
                mlflow.log_metrics({
                    "total_loss": losses['total'].item(),
                    "classification_loss": losses['classification'].item(),
                    "triplet_loss": losses['triplet'].item()
                }, step=epoch)

            avg_total_loss = sum(batch_losses) / len(batch_losses)
                      
            self.model.eval()
            with torch.no_grad():
                mean_pred, _ = self.model.forward(features)
                predictions = (mean_pred > threshold).float()
                accuracy = (predictions == labels_tensor).float().mean().item()
                print(f"📈 Training accuracy on this batch: {accuracy*100:.1f}%")        

                keeper_count = int(labels_tensor.sum().item())
                total_count = len(labels_tensor)
                print(f"📊 Batch keeper rate: {keeper_count}/{total_count} ({keeper_count/total_count:.1%})")
                mlflow.log_metrics({
                    "batch_accuracy": accuracy,
                    "batch_keeper_rate": keeper_count / total_count
                })

                BatchPerformance.objects.create(
                    batch_number=batch_num, 
                    correct=int(accuracy * total_count),
                    total=total_count,
                    accuracy=accuracy
                )
                
                if batch_num % 5 == 0:
                    mlflow.pytorch.log_model(self.model, f"bandit_model_batch_{batch_num}")
                # Store training instances in database for record keeping
                for i, instance in enumerate(instances):  # ✅ Use enumerate to get index
                    try:
                        if i >= len(record_ids):  # ✅ Safety check
                            continue
                
                        record_id = record_ids[i]  # ✅ Get from our list
                        predicted_prob = float(instance['predicted'])
                        predicted_bool = predicted_prob >= threshold 
                        uncertainty = instance_uncertainties.get(instance['id'])
                                    
                        BanditTrainingInstance.objects.create(
                            record_id=record_id,
                            context=json.dumps(instance.get('context', {})),
                            predicted=predicted_bool,
                            predicted_prob=predicted_prob,
                            predicted_uncertainty=uncertainty,
                            actual=instance['actual'],
                            reward=1.0 if instance['predicted'] == instance['actual'] else -1.0
                        )
                    except Exception as e:
                        print(f"Warning: Could not save training instance: {e}")

                new_threshold, new_precision = calculate_optimal_threshold()
                mlflow.log_metrics({
                    "updated_threshold": new_threshold,
                    "updated_precision": new_precision,
                })

            
                # Update model in database (simple approach: just save new version)
                self.save_model_to_db({'online_update_loss': avg_total_loss})
                BanditModelDB.objects.filter(is_active=True).update(batch_count=F('batch_count') + 1)        

            
            return {
                'instances_processed': len(records),
                'average_loss': avg_total_loss,
                'batch_keeper_rate': keeper_count / total_count,
                'updated_threshold': new_threshold,
                'model_updated': True,
                'accuracy': accuracy,
                'message': f'Updated model with {len(records)} new instances'
            }
        
    def save_model_to_db(self, history):
        """Save trained model and feature extractor to database"""
        try:
            old_model = BanditModelDB.objects.get(is_active=True)
            current_batch_count = old_model.batch_count
        except BanditModelDB.DoesNotExist:
            current_batch_count = 0
        
        model_weights = pickle.dumps({
            'model_state_dict': self.model.state_dict(),
            'feature_extractor': self.feature_extractor,
            'vocab_sizes': self.feature_extractor.get_vocab_sizes(),
            'embedding_dims': self.feature_extractor.get_embedding_dims()
        })
        
        bandit_model = BanditModelDB.objects.create(
            version = f"v{django_timezone.now().strftime('%Y%m%d_%H%M%S')}",
            model_weights = model_weights,
            hyperparams = json.dumps({
                'hidden_dims': [128, 64, 32],
                'dropout_rate': 0.2,
                'vocab_sizes': self.feature_extractor.get_vocab_sizes()
            }),
            training_stats = json.dumps(history),
            is_active=True,
            batch_count=current_batch_count
        )
        
        # Deactivate previous models
        BanditModelDB.objects.filter(is_active=True).exclude(id=bandit_model.id).update(is_active=False)
        
        print(f"Model saved to database with version {bandit_model.version}")
    
    def load_latest_model(self):
        try:
            active_models = BanditModelDB.objects.filter(is_active=True)
            print(f"Active models count: {active_models.count()}")
            print(f"Active models: {list(active_models.values_list('id', flat=True))}")
        
            latest_model = BanditModelDB.objects.filter(is_active=True).latest('created_at')
            print(f"Found model: {latest_model.id}")

            model_data = pickle.loads(latest_model.model_weights)
            print(f"Pickle loaded, keys: {model_data.keys()}")

            self.feature_extractor = model_data['feature_extractor']
            print(f"Feature extractor loaded: {self.feature_extractor}")

            
            vocab_sizes = model_data['vocab_sizes']
            embedding_dims = model_data['embedding_dims']
            actual_tfidf_dim = len(self.feature_extractor.title_vectorizer.vectorizer.vocabulary_) if self.feature_extractor.use_tfidf else 0

            
            # CONVERT KEY NAMES (same as in train_new_model)
            vocab_sizes_converted = {
                'artist_vocab_size': vocab_sizes['artist'],
                'label_vocab_size': vocab_sizes['label'],
                'genre_vocab_size': vocab_sizes['genre'],
                'style_vocab_size': vocab_sizes['style']
            }
            
            embedding_dims_converted = {
                'artist_embedding_dim': embedding_dims['artist'],
                'label_embedding_dim': embedding_dims['label'],
                'genre_embedding_dim': embedding_dims['genre'],
                'style_embedding_dim': embedding_dims['style']
            }
            
            self.model = NeuralContextualBandit(
                vocab_sizes=vocab_sizes_converted,      # Use converted
                embedding_dims=embedding_dims_converted, # Use converted
                hidden_dims=[128, 64, 32],
                tfidf_dim = actual_tfidf_dim,
                dropout_rate=0.2
            )
            
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.eval()
            
            print(f"Loaded model version {latest_model.version}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {type(e).__name__}: {e}")
            return False

trainer = BanditTrainer()