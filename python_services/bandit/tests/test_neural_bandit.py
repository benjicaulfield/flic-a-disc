import pytest

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

