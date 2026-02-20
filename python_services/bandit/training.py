import os
import torch
import numpy as np
import pickle
import json
import mlflow

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
        self.training_learning_rate = 0.001
        self.online_learning_rate = 0.0001
        self.epochs = 10
    
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
                
        return records, labels
    
    def train_new_model(self, epochs=100, batch_size=32):
        print("=" * 60)
        print("🚀 Starting new model training")
        print("=" * 60)

        mlflow.set_tracking_uri('file:///Users/benjamincaulfield/Documents/flic-a-disc/python_services/mlruns')        
        mlflow.set_experiment("discogs-bandit-model")

        with mlflow.start_run(run_name="5000 triplets"):
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": self.training_learning_rate,
                "hidden_dims": [128, 64, 32],
                "dropout_rate": 0.2
            })

            records, labels = self.prepare_training_data()
            
        
            self.feature_extractor = RecordFeatureExtractor(
                artist_vocab_size=1000,
                label_vocab_size=500,
                genre_vocab_size=100,
                style_vocab_size=200,
                title_tfidf_features=1000
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
                    num_triplets = min(len(keeper_records) * 2, 5000),
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
                    learning_rate=self.training_learning_rate
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
        
        try:
            old_model = BanditModelDB.objects.get(is_active=True)
            batch_num = old_model.batch_count + 1
        except BanditModelDB.DoesNotExist:
            batch_num = 1

        mlflow.set_tracking_uri('file:///Users/benjamincaulfield/Documents/flic-a-disc/python_services/mlruns')        
        mlflow.set_experiment("discogs-bandit-model")
        
        with mlflow.start_run(run_name=f"batch_{batch_num}"):
            mlflow.log_params({
                "batch_number": batch_num,
                "num_instances": len(instances),
                "learning_rate": self.online_learning_rate,
                "num_epochs": self.epochs
            })

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

            features = torch.FloatTensor(self.feature_extractor.extract_batch_features(records))
            labels_tensor = torch.FloatTensor(labels)

            n_samples = len(features)
            indices = torch.randperm(n_samples)

            train_size = int(0.8 * n_samples)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:]

            train_features = features[train_idx]
            train_labels = labels_tensor[train_idx]
            val_features = features[val_idx]
            val_labels = labels_tensor[val_idx]

            keeper_records = [records[idx] for idx, label in zip(train_idx, train_labels) if label == 1]
            non_keeper_records = [records[idx] for idx, label in zip(train_idx, train_labels) if label == 0]
            
            triplet_features = None
            if len(keeper_records) >= 2 and len(non_keeper_records) >= 1:
                num_triplets = min(len(keeper_records), 20)
                triplets = generate_triplets(keeper_records, non_keeper_records, num_triplets=num_triplets)
                
                if triplets:
                    anchor_features = self.feature_extractor.extract_batch_features(triplets['anchors'])
                    positive_features = self.feature_extractor.extract_batch_features(triplets['positives'])
                    negative_features = self.feature_extractor.extract_batch_features(triplets['negatives'])
                    
                    triplet_features = {
                        'anchor': torch.FloatTensor(anchor_features),
                        'positive': torch.FloatTensor(positive_features),
                        'negative': torch.FloatTensor(negative_features)
                    }
                    print(f"📦 Generated {len(triplets['anchors'])} triplets")
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.online_learning_rate)
            
            # Early stopping setup
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            best_epoch = 0
            patience = 10

            print(f"🔄 Running up to {self.epochs} training epochs with early stopping...")
            
            # Training loop with early stopping
            for epoch in range(self.epochs):
                # Training
                self.model.train()
                optimizer.zero_grad()
                
                losses = self.model.combined_loss(
                    train_features, 
                    train_labels, 
                    triplet_data=triplet_features
                )
                losses['total'].backward()
                optimizer.step()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_losses = self.model.combined_loss(
                        val_features,
                        val_labels,
                        triplet_data=None  # Don't use triplets for validation
                    )
                    val_loss = val_losses['total'].item()
                    
                    # Calculate validation accuracy
                    val_mean, _ = self.model.forward(val_features)
                    val_pred = (torch.sigmoid(val_mean) > threshold).float()
                    val_acc = (val_pred == val_labels).float().mean().item()
                
                # Log to MLflow
                mlflow.log_metrics({
                    "train_loss": losses['total'].item(),
                    "train_classification_loss": losses['classification'].item(),
                    "train_triplet_loss": losses['triplet'].item(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                }, step=epoch)
                
                # Print progress every 5 epochs
                if epoch % 5 == 0 or epoch == self.epochs - 1:
                    print(f"  Epoch {epoch}: train_loss={losses['total'].item():.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"  🛑 Early stopping at epoch {epoch} (best: epoch {best_epoch})")
                    break
            
            # Restore best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                print(f"  ✅ Restored model from epoch {best_epoch}")
            
            # Final evaluation on full batch
            self.model.eval()
            with torch.no_grad():
                all_mean, _ = self.model.forward(features)
                all_pred = (torch.sigmoid(all_mean) > threshold).float()
                final_acc = (all_pred == labels_tensor).float().mean().item()
            
            print(f"📈 Final accuracy on full batch: {final_acc*100:.1f}%")
            
            # Log final metrics
            mlflow.log_metrics({
                "final_batch_accuracy": final_acc,
                "best_epoch": best_epoch,
                "epochs_trained": epoch + 1
            })
            
            # Save training instances to database
            for i, record_id in enumerate(record_ids):
                BanditTrainingInstance.objects.create(
                    record_id=record_id,
                    predicted_prob=float(torch.sigmoid(all_mean[i]).item()),
                    actual=bool(labels[i])
                )
            
            # Save model
            self.save_model_to_db({'online_update_loss': best_val_loss})
            
            # Increment batch counter
            BanditModelDB.objects.filter(is_active=True).update(batch_count=F('batch_count') + 1)
            
            return {
                'instances_processed': len(records),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'final_accuracy': final_acc,
                'model_updated': True,
                'message': f'Updated model with {len(records)} instances (early stopped at epoch {epoch})'
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