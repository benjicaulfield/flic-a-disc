import pytest
import torch
import numpy as np
from bandit.contrastive_encoder import ContrastiveEncoder

@pytest.fixture
def vocab_sizes():
    return {
        'artist_vocab_size': 100,
        'label_vocab_size': 50,
        'genre_vocab_size': 20,
        'style_vocab_size': 30
    }

@pytest.fixture
def embedding_dims():
    return {
        'artist_embedding_dim': 64,
        'label_embedding_dim': 32,
        'genre_embedding_dim': 16,
        'style_embedding_dim': 16
    }

@pytest.fixture
def encoder(vocab_sizes, embedding_dims):
    return ContrastiveEncoder(
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        tfidf_dim=100
    )

def create_valid_features(batch_size, vocab_sizes, tfidf_dim=100):
    features = []
    
    for _ in range(batch_size):
        artist_idx = np.random.randint(0, vocab_sizes['artist_vocab_size'])
        label_idx = np.random.randint(0, vocab_sizes['label_vocab_size'])        
        genre_idx = np.random.randint(0, vocab_sizes['genre_vocab_size'], size=3)        
        style_idx = np.random.randint(0, vocab_sizes['style_vocab_size'], size=3)        
        numerical = np.random.randn(5)        
        condition = np.random.randn(8)        
        tfidf = np.random.randn(tfidf_dim)
        
        feature_vector = np.concatenate([
            [artist_idx, label_idx],
            genre_idx,
            style_idx,
            numerical,
            condition,
            tfidf
        ])

        features.append(feature_vector)
    
    features_array = np.array(features)
    return torch.FloatTensor(features_array)

def test_forward_output_is_normalized(encoder, vocab_sizes):
    features = create_valid_features(5, vocab_sizes)
    output = encoder(features)
    norms = torch.norm(output, p=2, dim=1)
    
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)

def test_forward_handles_large_batch(encoder, vocab_sizes):
    features = create_valid_features(100, vocab_sizes)    
    output = encoder(features)
    
    assert output.shape == (100, 64)


def test_embeddings_are_deterministic(encoder, vocab_sizes):
    encoder.eval()
    features = create_valid_features(5, vocab_sizes)    
    output1 = encoder(features)
    output2 = encoder(features)
    
    assert torch.allclose(output1, output2)


def test_different_inputs_give_different_outputs(encoder, vocab_sizes):
    encoder.eval()
    features1 = create_valid_features(5, vocab_sizes)
    features2 = create_valid_features(5, vocab_sizes)
    output1 = encoder(features1)
    output2 = encoder(features2)
    
    assert not torch.allclose(output1, output2)