import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .contrastive_encoder import ContrastiveEncoder

class NeuralContextualBandit(nn.Module):
    def __init__(self, vocab_sizes, embedding_dims, 
                 hidden_dims = [128, 64, 32], 
                 embedding_dim = 64,
                 tfidf_dim = 1000,
                 dropout_rate = 0.4):
        super().__init__()
        
        self.tfidf_dim = tfidf_dim

        self.encoder = ContrastiveEncoder(
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            tfidf_dim=tfidf_dim,
            hidden_dims=[256, 128],
            output_dim=embedding_dim,
            dropout_rate=dropout_rate
        )
        
        head_layers = []
        prev_dim = embedding_dim + tfidf_dim

        for hidden_dim in hidden_dims:
            head_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim 

        self.prediction_head = nn.Sequential(*head_layers)
        self.mean_layer = nn.Linear(prev_dim, 1)
        self.log_var_layer = nn.Linear(prev_dim, 1)

    def encode(self, features):
        return self.encoder(features)

    def forward(self, features):

        embeddings = self.encoder(features)

        tfidf_features = features[:, -self.tfidf_dim:]
        combined = torch.cat([embeddings, tfidf_features], dim=1)
        hidden = self.prediction_head(combined)
        mean = torch.sigmoid(self.mean_layer(hidden))
        log_var = self.log_var_layer(hidden)
        log_var = torch.clamp(log_var, min=-10, max=2)
        variance = torch.exp(log_var) + 1e-6

        return mean.squeeze(), variance.squeeze()
        
    def predict_with_uncertainty(self, features):
        self.eval()
        with torch.no_grad():
            mean, variance = self.forward(features)
            return mean, variance
            
    def thompson_sample(self, features, n_samples = 10):
        self.eval()

        samples = []
        for _ in range(n_samples):
            mean, variance = self.forward(features)
            noise = torch.randn_like(mean) * torch.sqrt(variance)
            sample = mean + noise
            sample = torch.clamp(sample, 0.0, 1.0)
            samples.append(sample)
        
        sampled_rewards = torch.stack(samples).mean(dim=0)
        self.eval()
        return sampled_rewards
    
    def combined_loss(self, features, labels, triplet_data, loss_weights=None):
        if loss_weights is None:
            loss_weights = {
                'triplet': 0.2,
                'classification': 0.8
            }

        mean_pred, variance_pred = self.forward(features)

        num_pos = labels.sum()
        num_neg = len(labels) - num_pos
        weight_for_1 = num_neg / (num_pos + 1e-6)

        sample_weights = labels * min(weight_for_1, 10.0) + (1 - labels) 

        classification_loss = F.binary_cross_entropy(
            mean_pred, 
            labels.float(), 
            weight=sample_weights,
            reduction='mean'
        )
        
        triplet_loss = torch.tensor(0.0).to(features.device)
        
        if triplet_data is not None:
            # Pass full features to encoder - it handles TF-IDF internally
            anchor_emb = self.encoder(triplet_data['anchor'])
            positive_emb = self.encoder(triplet_data['positive'])
            negative_emb = self.encoder(triplet_data['negative'])
            
            triplet_loss = self.encoder.triplet_loss(
                anchor_emb, positive_emb, negative_emb,
                margin=1.0
            )
        
        # Combined loss
        total_loss = (
            loss_weights['classification'] * classification_loss +
            loss_weights['triplet'] * triplet_loss
        )
        
        return {
            'total': total_loss,
            'classification': classification_loss,
            'uncertainty': variance_pred.mean(),
            'triplet': triplet_loss
        }
    
    def fit(self, feature_extractor, training_records, labels, triplet_records,
            epochs = 100, batch_size = 32, learning_rate = 0.005):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.001)

        features = feature_extractor.extract_batch_features(training_records)
        features_tensor = torch.FloatTensor(features)
        print(f"🔍 Main features shape: {features_tensor.shape}")

        labels_tensor = torch.FloatTensor(labels)
        print(f"🔍 Labels shape: {labels_tensor.shape}")


        triplet_features = None
        if triplet_records is not None:
            anchor_features = feature_extractor.extract_batch_features(triplet_records['anchors'])
            print(f"🔍 Anchor features shape: {anchor_features.shape}")

            positive_features = feature_extractor.extract_batch_features(triplet_records['positives'])
            negative_features = feature_extractor.extract_batch_features(triplet_records['negatives'])
            
            triplet_features = {
                'anchor': torch.FloatTensor(anchor_features),
                'positive': torch.FloatTensor(positive_features),
                'negative': torch.FloatTensor(negative_features)
            }
            print(f"🔍 Triplet anchor tensor shape: {triplet_features['anchor'].shape}")


        n_train = int(0.8 * len(features_tensor))
        train_features = features_tensor[:n_train]
        train_labels = labels_tensor[:n_train]
        val_features = features_tensor[n_train:]
        val_labels = labels_tensor[n_train:]
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 
                   'val_f1': [], 'val_precision': [], 'val_recall': []}
        
        # Early stopping setup
        best_f1 = 0.0
        best_model_state = None
        patience = 15  # Wait 15 epochs before giving up
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            epoch_losses = []
            
            for i in range(0, len(train_features), batch_size):
                batch_features = train_features[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                batch_triplets = None
                if triplet_features is not None and len(triplet_features['anchor']) > 0:
                    n_samples = min(len(batch_features), len(triplet_features['anchor']))
                    if n_samples > 0:
                        indices = torch.randperm(len(triplet_features['anchor']))[:n_samples]
                        batch_triplets = {
                            'anchor': triplet_features['anchor'][indices],
                            'positive': triplet_features['positive'][indices],
                            'negative': triplet_features['negative'][indices]
                        }
                        # Ensure 2D shape even for single sample
                        if batch_triplets['anchor'].dim() == 1:
                            batch_triplets['anchor'] = batch_triplets['anchor'].unsqueeze(0)
                            batch_triplets['positive'] = batch_triplets['positive'].unsqueeze(0)
                            batch_triplets['negative'] = batch_triplets['negative'].unsqueeze(0)

                optimizer.zero_grad()
                
                losses = self.combined_loss(
                    batch_features, batch_labels, 
                    triplet_data=batch_triplets)
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(losses['total'].item())
            
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            self.eval()
            with torch.no_grad():
                val_mean, _ = self.forward(val_features)
                val_loss = F.binary_cross_entropy(val_mean, val_labels).item()
                
                threshold, f1 = self.find_optimal_threshold(val_mean, val_labels)
                val_pred = (val_mean > threshold).float()
                val_acc = (val_pred == val_labels).float().mean().item()

                val_pred_np = val_pred.cpu().numpy()
                val_labels_np = val_labels.cpu().numpy()

                tp = ((val_pred_np == 1) & (val_labels_np == 1)).sum()
                fp = ((val_pred_np == 1) & (val_labels_np == 0)).sum()
                fn = ((val_pred_np == 0) & (val_labels_np == 1)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                history['val_f1'].append(f1)
                history['val_precision'].append(precision)
                history['val_recall'].append(recall)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)

            # Early stopping logic
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
                print("✓ NEW BEST")
            else:
                patience_counter += 1
                print(f"({patience_counter}/{patience})")
            
            print(f"Epoch {epoch}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}, "
                    f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f} ")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                print(f"   Best val acc: {f1:.4f} at epoch {epoch - patience}")
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"✅ Restored best model with f1: {f1:.4f}")
        
        return history
    
    def find_optimal_threshold(self, val_mean, val_labels):
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(0.05, 0.95, 0.05):
            pred = (val_mean > threshold).float()
            tp = ((pred == 1) & (val_labels == 1)).sum().item()
            fp = ((pred == 1) & (val_labels == 0)).sum().item()
            fn = ((pred == 0) & (val_labels == 1)).sum().item()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold, best_f1