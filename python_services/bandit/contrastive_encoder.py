import torch 
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveEncoder(nn.Module):

    def __init__(self, vocab_sizes, embedding_dims, hidden_dims = [256, 128],
                 output_dim = 64, dropout_rate = 0.2):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims
        self.output_dim = output_dim

        self.artist_embedding = nn.Embedding(
            vocab_sizes['artist_vocab_size'],
            embedding_dims['artist_embedding_dim'],
            padding_idx = 0
        )

        self.label_embedding = nn.Embedding(
            vocab_sizes['artist_vocab_size'],
            embedding_dims['label_embedding_dim'],
            padding_idx = 0
        )

        self.genre_embedding = nn.Embedding(
            vocab_sizes['genre_vocab_size'],
            embedding_dims['genre_embedding_dim'],
            padding_idx=0
        )
        
        self.style_embedding = nn.Embedding(
            vocab_sizes['style_vocab_size'],
            embedding_dims['style_embedding_dim'],
            padding_idx=0
        )
        
        total_input_dim = (
            embedding_dims['artist_embedding_dim'] +      # 64
            embedding_dims['label_embedding_dim'] +       # 32
            3 * embedding_dims['genre_embedding_dim'] +   # 48 (3 genres)
            3 * embedding_dims['style_embedding_dim'] +   # 48 (3 styles)
            5 +  # numerical: wants, haves, ratio, price, year
            8    # condition one-hot
        ) 

        layers = []
        prev_dim = total_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
        self.normalize = True

    def forward(self, features):
        artist_idx = features[:, 0].long()
        label_idx = features[:, 1].long()
        genre_idx = features[:, 2:5].long()
        style_idx = features[:, 5:8].long()
        numerical = features[:, 8:13]
        condition = features[:, 13:21]

        artist_embedding = self.artist_embedding(artist_idx)
        label_embedding = self.label_embedding(label_idx)
        genre_embedding_0 = self.genre_embedding(genre_idx[:, 0])
        genre_embedding_1 = self.genre_embedding(genre_idx[:, 1])
        genre_embedding_2 = self.genre_embedding(genre_idx[:, 2])
        style_embedding_0 = self.style_embedding(style_idx[:, 0])
        style_embedding_1 = self.style_embedding(style_idx[:, 1])
        style_embedding_2 = self.style_embedding(style_idx[:, 2])

        all_features = torch.cat([artist_embedding, label_embedding, 
                                  genre_embedding_0, genre_embedding_1, genre_embedding_2,
                                  style_embedding_0, style_embedding_1, style_embedding_2,
                                  numerical, condition], dim=1)
        
        embeddings = self.encoder(all_features)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

    def triplet_loss(self, anchor, positive, negative, margin = 1.0, distance_fn = 'euclidean'):
        if distance_fn == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif distance_fn == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance function: {distance_fn}")
        
        loss = F.relu(pos_dist - neg_dist + margin)
        return loss.mean()
    
    def hard_triplet_loss(self, embeddings, labels, margin = 1.0):
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float()
        negative_mask = (labels != labels.t()).float()
    
        # Remove diagonal (distance to self)
        positive_mask.fill_diagonal_(0)
        
        losses = []
        
        for i in range(embeddings.size(0)):
            # For each anchor
            anchor_positive_dists = distance_matrix[i] * positive_mask[i]
            anchor_negative_dists = distance_matrix[i] * negative_mask[i]
            
            # Skip if no positives or negatives available
            if positive_mask[i].sum() == 0 or negative_mask[i].sum() == 0:
                continue
            
            # Hardest positive: most distant positive example
            hardest_positive_dist = anchor_positive_dists.max()
            
            # Hardest negative: closest negative example  
            # Set positive distances to infinity so they're not selected
            anchor_negative_dists = torch.where(
                negative_mask[i].bool(),
                anchor_negative_dists,
                torch.tensor(float('inf')).to(embeddings.device)
            )
            hardest_negative_dist = anchor_negative_dists.min()
            
            # Triplet loss for this anchor
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0).to(embeddings.device)
        
        return torch.stack(losses).mean()








