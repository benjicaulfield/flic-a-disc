# experiments/hyperparameter_search.py
import torch
import numpy as np

import os
import sys
from pathlib import Path
import django

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from bandit.neural_bandit import NeuralContextualBandit
from bandit.features import RecordFeatureExtractor
from bandit.training import BanditTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime

def evaluate_model(model, feature_extractor, test_records, test_labels):
    """Evaluate model performance on test set"""
    features = feature_extractor.extract_batch_features(test_records)
    features_tensor = torch.FloatTensor(features)
    
    model.eval()
    with torch.no_grad():
        mean, variance = model.predict_with_uncertainty(features_tensor)
        predictions = (mean > 0.5).numpy()
    
    return {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions, zero_division=0),
        'recall': recall_score(test_labels, predictions, zero_division=0),
        'f1': f1_score(test_labels, predictions, zero_division=0),
        'mean_uncertainty': variance.mean().item()
    }


def train_and_evaluate(config, train_records, train_labels, val_records, val_labels, 
                       test_records, test_labels, feature_extractor):
    """Train a model with given config and evaluate it"""
    print(f"\n{'='*60}")
    print(f"Testing config: {config['name']}")
    print(f"{'='*60}")
    
    # Create model with this config
    model = NeuralContextualBandit(
        vocab_sizes=feature_extractor.get_vocab_sizes(),
        embedding_dims={
            'artist_embedding_dim': config['artist_emb'],
            'label_embedding_dim': config['label_emb'],
            'genre_embedding_dim': config['genre_emb'],
            'style_embedding_dim': config['style_emb']
        },
        hidden_dims=config['hidden_dims'],
        tfidf_dim=len(feature_extractor.title_vectorizer.vectorizer.vocabulary_),
        dropout_rate=config['dropout']
    )
    
    # Train
    history = model.fit(
        feature_extractor=feature_extractor,
        training_records=train_records,
        labels=train_labels,
        triplet_records=None,  # Add if you want
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, feature_extractor, test_records, test_labels)
    
    return {
        'config': config,
        'history': history,
        'test_metrics': test_metrics
    }


def run_hyperparameter_search():
    """Test different hyperparameter configurations"""
    
    # Load your data
    trainer = BanditTrainer()
    trainer.load_latest_model()  # Loads feature extractor
    
    # Get all evaluated records
    records, labels = trainer.prepare_training_data()
    
    # Split: 60% train, 20% val, 20% test
    n = len(records)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    train_records = records[:train_end]
    train_labels = labels[:train_end]
    val_records = records[train_end:val_end]
    val_labels = labels[train_end:val_end]
    test_records = records[val_end:]
    test_labels = labels[val_end:]
    
    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_records)} records")
    print(f"   Val:   {len(val_records)} records")
    print(f"   Test:  {len(test_records)} records")
    
    # Define configurations to test
    configs = [
        {
            'name': 'baseline',
            'artist_emb': 64,
            'label_emb': 32,
            'genre_emb': 16,
            'style_emb': 16,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.4,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        {
            'name': 'larger_embeddings',
            'artist_emb': 128,
            'label_emb': 64,
            'genre_emb': 32,
            'style_emb': 32,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.4,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        {
            'name': 'deeper_network',
            'artist_emb': 64,
            'label_emb': 32,
            'genre_emb': 16,
            'style_emb': 16,
            'hidden_dims': [256, 128, 64, 32],
            'dropout': 0.4,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        {
            'name': 'less_dropout',
            'artist_emb': 64,
            'label_emb': 32,
            'genre_emb': 16,
            'style_emb': 16,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        {
            'name': 'higher_lr',
            'artist_emb': 64,
            'label_emb': 32,
            'genre_emb': 16,
            'style_emb': 16,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.4,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.005
        },
    ]
    
    # Run experiments
    results = []
    for config in configs:
        result = train_and_evaluate(
            config, 
            train_records, train_labels,
            val_records, val_labels,
            test_records, test_labels,
            trainer.feature_extractor
        )
        results.append(result)
        
        print(f"\n📊 Results for {config['name']}:")
        print(f"   Accuracy:  {result['test_metrics']['accuracy']:.3f}")
        print(f"   Precision: {result['test_metrics']['precision']:.3f}")
        print(f"   Recall:    {result['test_metrics']['recall']:.3f}")
        print(f"   F1:        {result['test_metrics']['f1']:.3f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'experiments/results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - Best Configurations")
    print(f"{'='*60}")
    
    sorted_by_f1 = sorted(results, key=lambda x: x['test_metrics']['f1'], reverse=True)
    
    for i, result in enumerate(sorted_by_f1[:3], 1):
        print(f"\n#{i} - {result['config']['name']}")
        print(f"   F1: {result['test_metrics']['f1']:.3f}")
        print(f"   Accuracy: {result['test_metrics']['accuracy']:.3f}")
        print(f"   Config: {result['config']}")


if __name__ == '__main__':
    run_hyperparameter_search()