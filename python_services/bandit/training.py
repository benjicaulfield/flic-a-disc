import torch
import numpy as np
import pickle
import json

from django.utils import timezone
from django.db.models import F

from .models import Record, BanditModel as BanditModelDB, BanditTrainingInstance, ThresholdConfig
from .features import RecordFeatureExtractor
from .neural_bandit import NeuralContextualBandit
from .triplet_generation import generate_triplets, generate_triplets_from_batch


class BanditTrainer:
    def __init__(self):
        self.feature_extractor = None
        self.model = None 
        self.training_history = []
    
    def prepare_training_data(self):
        # Query listings where the associated record has been evaluated
        evaluated_records = Record.objects.filter(evaluated=True)
        
        if not evaluated_records.exists():
            raise ValueError("No evaluated listings found for training")
        
        records = []
        labels = []
        
        for record in evaluated_records:
            record_dict = {
                'artist': record.artist,
                'title': record.title,
                'label': record.label,
                'genres': record.genres,
                'styles': record.styles,
                'wants': record.wants,
                'haves': record.haves,
                'year': record.year,  
            }
            records.append(record_dict)
            labels.append(record.wanted)  # The evaluation decision
        
        print(f"Training targets sample: {labels[:10]}")
        print(f"Training targets type: {type(labels[0])}")
        print(f"Unique values: {set(labels)}")
        
        return records, labels
    
    def train_new_model(self, epochs=100, batch_size=32, learning_rate=0.0001):
        print("=" * 60)
        print("🚀 Starting new model training")
        print("=" * 60)
        
        records, labels = self.prepare_training_data()
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
            tfidf_dim=len(self.feature_extractor.title_vectorizer.vectorizer.vocabulary_), 
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
                num_triplets=min(len(keeper_records), 100),
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
        
        try: 
            history = self.model.fit(
                feature_extractor=self.feature_extractor,
                training_records=records,
                labels=labels,
                triplet_records=triplet_data,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
        except Exception as e:
            import traceback
            print("=" * 60)
            print("FULL TRACEBACK:")
            print(traceback.format_exc())
            print("=" * 60)
            raise
        
        print("-" * 60)
        print(f"✅ Training complete!")
        print(f"📈 Final validation accuracy: {history['val_accuracy'][-1]:.2%}")
        print(f"📉 Final validation loss: {history['val_loss'][-1]:.4f}")
        
        print("\n💾 Saving model to database...")
        self.save_model_to_db(history)
        print("✅ Model saved successfully!")
        print("=" * 60)
        
        return history
        
    def record_to_dict(self, record):
        return {
            'artist': record.artist,
            'title': record.title,
            'label': record.label,
            'genres': record.genres,
            'styles': record.styles,
            'wants': record.wants,
            'haves': record.haves,
            'year': record.year,
        }
    
    def update_model_online(self, instances):
        if not self.model or not self.feature_extractor: return self.train_new_model()
        
        threshold_config = ThresholdConfig.objects.first()
        threshold = threshold_config.threshold if threshold_config else 0.5
        
        records = []
        labels = []
        record_ids = []
        print(f"🎯 Starting online update with {len(instances)} instances")

        for instance in instances:
            try:
                record = Record.objects.get(id=instance['id'])
                record_dict = self.record_to_dict(record)
                records.append(record_dict)
                labels.append(instance['actual'])
                record_ids.append(record.id)
                
            except Record.DoesNotExist:
                print(f"Warning: Record {instance['record_id']} not found, skipping")
                continue
        
        if not records:
            return {'error': 'No valid records found for training'}
        
        print(f"📊 Found {len(records)} valid records for training")

        features = torch.FloatTensor(self.feature_extractor.extract_batch_features(records))
        labels = torch.FloatTensor(labels)

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

        historical_records = Record.objects.filter(evaluated=True)[:500]
        
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
            non_keeper_history=non_keeper_history
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
        total_losses = {'total': 0, 'classification': 0, 'triplet': 0, 'uncertainty': 0}
        
        # Multiple passes over the new data
        print(f"🔄 Running 10 training epochs...")

        for epoch in range(10):
            optimizer.zero_grad()
            
            losses = self.model.combined_loss(
                        features,
                        labels,
                        triplet_data=triplet_features)
            losses['total'].backward()
            optimizer.step()

            for key in total_losses:
                total_losses[key] += losses[key].item()

            if epoch % 3 == 0:
                print(f"  Epoch {epoch}: total={losses['total'].item():.4f}, "
                    f"classification={losses['classification'].item():.4f}, "
                    f"triplet={losses['triplet'].item():.4f}")
    
        avg_total_loss = total_losses['total'] / 10
                      
        self.model.eval()
        with torch.no_grad():
            mean_pred, _ = self.model.forward(features)
            predictions = (mean_pred > 0.5).float()
            accuracy = (predictions == labels).float().mean().item()
            print(f"📈 Training accuracy on this batch: {accuracy*100:.1f}%")        
            
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
        
        # Update model in database (simple approach: just save new version)
        self.save_model_to_db({'online_update_loss': avg_total_loss})
        BanditModelDB.objects.filter(is_active=True).update(batch_count=F('batch_count') + 1)        

        
        return {
            'instances_processed': len(records),
            'average_loss': avg_total_loss,
            'model_updated': True,
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
            version = f"v{timezone.now().strftime('%Y%m%d_%H%M%S')}",
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
            latest_model = BanditModelDB.objects.filter(is_active=True).latest('created_at')
            
            model_data = pickle.loads(latest_model.model_weights)
            
            self.feature_extractor = model_data['feature_extractor']
            
            vocab_sizes = model_data['vocab_sizes']
            embedding_dims = model_data['embedding_dims']
            actual_tfidf_dim = len(self.feature_extractor.title_vectorizer.vectorizer.vocabulary_)

            
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
            
        except BanditModelDB.DoesNotExist:
            print("No trained model found in database")
            return False