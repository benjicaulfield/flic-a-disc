import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import average_precision_score

from .contrastive_encoder import ContrastiveEncoder
from .models import ThresholdConfig
from .utils.best_f1_threshold import best_f1_threshold

class NeuralContextualBandit(nn.Module):
    def __init__(self, vocab_sizes, embedding_dims, 
                 hidden_dims = [128, 64, 32], 
                 embedding_dim = 64,
                 tfidf_dim = 1000,
                 dropout_rate = 0.4):
        super().__init__()
        
        self.tfidf_dim = tfidf_dim
        self.loss_weights = {'classification': 0.5, 'triplet': 0.5}

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
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim 

        self.prediction_head = nn.Sequential(*head_layers)
        self.mean_layer = nn.Linear(prev_dim, 1)
        self.log_var_layer = nn.Linear(prev_dim, 1)


    def forward(self, features):
        embeddings = self.encoder(features)

        tfidf_features = features[:, -self.tfidf_dim:]
        combined = torch.cat([embeddings, tfidf_features], dim=1)

        hidden = self.prediction_head(combined)
        logits = self.mean_layer(hidden).view(-1)
        log_var = self.log_var_layer(hidden)
        log_var = torch.clamp(log_var, min=-10, max=2)
        variance = (torch.exp(log_var) + 1e-6).view(-1)

        return logits, variance.squeeze()
        
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
    
    @staticmethod
    def focal_bce_with_logits(logits, y, pos_weight, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        loss = ((1 - pt) ** gamma) * bce
        return loss.mean()
    
    def combined_loss(self, features, labels, triplet_data, loss_weights=None, pos_weight=None):
        w = self.loss_weights

        logits, variance_pred = self.forward(features)
        labels = labels.float().view(-1)

        if pos_weight is None:
            pos_weight = torch.tensor(1.0, device=features.device)

        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        classification_loss = bce(logits, labels)
        
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
            w['classification'] * classification_loss +
            w['triplet'] * triplet_loss
        )
        
        return {
            'total': total_loss,
            'classification': classification_loss,
            'uncertainty': variance_pred.mean(),
            'triplet': triplet_loss
        }
    
    def fit(self, feature_extractor, training_records, labels, triplet_records,
            epochs=100, batch_size=32, learning_rate=3e-4, loss_weights=None, 
            patience=15, focal_gamma=2.0, eval_threshold=0.4):
        
        if loss_weights is None:
            loss_weights = {"classification": 0.5, "triplet": 0.5}
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.001)

        features = feature_extractor.extract_batch_features(training_records)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = torch.tensor(features, dtype=torch.float32)
        y_all = torch.tensor(labels, dtype=torch.float32).view(-1)

        triplet_features = None
        if triplet_records is not None:
            anchor_features = feature_extractor.extract_batch_features(triplet_records['anchors'])
            positive_features = feature_extractor.extract_batch_features(triplet_records['positives'])
            negative_features = feature_extractor.extract_batch_features(triplet_records['negatives'])
            
            triplet_features = {
                'anchor': torch.FloatTensor(anchor_features),
                'positive': torch.FloatTensor(positive_features),
                'negative': torch.FloatTensor(negative_features)
            }
            print(f"🔍 Triplet anchor tensor shape: {triplet_features['anchor'].shape}")

        pos_idx = torch.where(y_all > 0.5)[0]
        neg_idx = torch.where(y_all <= 0.5)[0]

        pos_idx = pos_idx[torch.randperm(len(pos_idx))]
        neg_idx = neg_idx[torch.randperm(len(neg_idx))]

        n_pos_train = int(0.8 * len(pos_idx))
        n_neg_train = int(0.8 * len(neg_idx))

        train_idx = torch.cat([pos_idx[:n_pos_train], neg_idx[:n_neg_train]])
        val_idx   = torch.cat([pos_idx[n_pos_train:], neg_idx[n_neg_train:]])

        # shuffle within train/val
        train_idx = train_idx[torch.randperm(len(train_idx))]
        val_idx   = val_idx[torch.randperm(len(val_idx))]

        X_train = X[train_idx]
        y_train = y_all[train_idx].view(-1)
        X_val = X[val_idx]
        y_val = y_all[val_idx].view(-1)

        n_pos = float(y_train.sum().item())
        n_neg = float(len(y_train) - y_train.sum().item())
        raw_pos_weight = n_neg / (n_pos + 1e-6)
        pos_weight = torch.tensor(raw_pos_weight, dtype=torch.float32).clamp(1.0, 20.0)

        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_ap": []}
        
        # Early stopping setup
        best_val_ap = -1.0
        best_model_state = None
        patience_counter = 0
        best_epoch = -1

        for epoch in range(epochs):
            ramp = min(1.0, epoch / 10.0)
            w_cls = loss_weights.get("classification", 0.7)
            w_trip = loss_weights.get("triplet", 0.3) * ramp

            self.train()
            perm = torch.randperm(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            batch_losses = []
            
            for i in range(0, len(X_train), batch_size):
                xb = X_train[i : i + batch_size]
                yb = y_train[i : i + batch_size]

                if xb.size(0) < 2:
                    continue
                
                batch_triplets = None
                if triplet_features is not None and triplet_features["anchor"].size(0) > 0:
                    k = min(max(32, xb.size(0)), triplet_features["anchor"].size(0))
                    idx = torch.randperm(triplet_features["anchor"].size(0))[:k]
                    batch_triplets = {
                        'anchor': triplet_features['anchor'][idx],
                        'positive': triplet_features['positive'][idx],
                        'negative': triplet_features['negative'][idx],
                        }
        
                optimizer.zero_grad(set_to_none=True)

                logits, _ = self.forward(xb)
                logits = logits.view(-1)

                cls_loss = self.focal_bce_with_logits(logits, yb, pos_weight=pos_weight, gamma=focal_gamma)

                triplet_loss = torch.tensor(0.0)
                if batch_triplets is not None:
                    anchor_emb = F.normalize(self.encoder(batch_triplets["anchor"]), dim=1)
                    positive_emb = F.normalize(self.encoder(batch_triplets["positive"]), dim=1)
                    negative_emb = F.normalize(self.encoder(batch_triplets["negative"]), dim=1)
                    triplet_loss = self.encoder.triplet_loss(anchor_emb, positive_emb, negative_emb, margin=1.0)
                 
                    total_loss = w_cls * cls_loss + w_trip * triplet_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)    
                optimizer.step()
                
                batch_losses.append(total_loss.item())
            
            avg_train_loss = float(sum(batch_losses) / max(1, len(batch_losses)))
            history["train_loss"].append(avg_train_loss)
            
            self.eval()            
            with torch.no_grad():
                val_logits, _ = self.forward(X_val)
                val_logits = val_logits.view(-1)

                val_loss = float(F.binary_cross_entropy_with_logits(val_logits, y_val).item())
                val_probs = torch.sigmoid(val_logits)

                # PR-AUC / AP
                y_np = y_val.cpu().numpy()
                p_np = val_probs.cpu().numpy()
                val_ap = 0.0 if y_np.sum() == 0 else float(average_precision_score(y_np, p_np))

                val_pred = (val_probs > eval_threshold).float()
                val_acc = float((val_pred == y_val).float().mean().item())

            history["val_loss"].append(val_loss)
            history["val_ap"].append(val_ap)
            history["val_acc"].append(val_acc)
                    
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
                status = "✓ NEW BEST"
            else:
                patience_counter += 1
                status = f"({patience_counter}/{patience})"

            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc@{eval_threshold:.2f}: {val_acc:.4f}"
            )
            print(
                f"  Val probs: min={val_probs.min():.3f}, max={val_probs.max():.3f}, mean={val_probs.mean():.3f}"
            )
            print(f"  Val AP (PR-AUC): {val_ap:.4f} | Early-stop metric (AP): {val_ap:.4f} {status}")

            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                print(f"  Best Val AP: {best_val_ap:.4f} at epoch {best_epoch}")
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"✅ Restored best-by-AP model | Val AP={best_val_ap:.4f} (epoch {best_epoch})")

        return history
    
    def precision_at_min_recall(self, probs, labels, min_recall=0.40):
        """Find threshold that maximizes precision while maintaining min_recall"""
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0
        
        for thr in np.arange(0.01, 0.99, 0.01):
            pred = (probs > thr).float()
            tp = ((pred == 1) & (labels == 1)).sum().item()
            fp = ((pred == 1) & (labels == 0)).sum().item()
            fn = ((pred == 0) & (labels == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Only consider thresholds that meet minimum recall
            if recall >= min_recall and precision > best_precision:
                best_precision = precision
                best_threshold = thr
                best_recall = recall
        
        return best_threshold, best_precision, best_recall