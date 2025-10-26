import random

def generate_triplets(keepers, non_keepers, num_triplets = None,
                      hard_mining = False, feature_extractor = None):
    anchors = []
    positives = []
    negatives = []

    for _ in range(num_triplets):
        anchor, positive = random.sample(keepers, 2)

        if hard_mining and feature_extractor:
            negative = select_hard_negative(anchor, non_keepers, feature_extractor)
        else:
            negative = random.choice(non_keepers)
    
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)

    return {
        'anchors': anchors,
        'positives': positives,
        'negatives': negatives
    }

def select_hard_negative(anchor, negative_candidates, feature_extractor):
    
    def similarity_score(anchor_record, candidate_record):
        score = 0
        
        # Genre overlap
        anchor_genres = set(anchor_record.get('genres', []))
        candidate_genres = set(candidate_record.get('genres', []))
        if anchor_genres & candidate_genres:
            score += 3
        
        # Style overlap
        anchor_styles = set(anchor_record.get('styles', []))
        candidate_styles = set(candidate_record.get('styles', []))
        if anchor_styles & candidate_styles:
            score += 2
        
        # Same label
        if anchor_record.get('label') == candidate_record.get('label'):
            score += 2
        
        # Similar year (within 5 years)
        anchor_year = anchor_record.get('year')
        candidate_year = candidate_record.get('year')
        if anchor_year and candidate_year:
            if abs(anchor_year - candidate_year) <= 5:
                score += 1
        
        return score
    
    # Find negative with highest similarity to anchor
    scored_negatives = [
        (neg, similarity_score(anchor, neg))
        for neg in negative_candidates
    ]
    
    # Select from top 20% most similar (balance hard mining with diversity)
    scored_negatives.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_negatives[:max(1, len(scored_negatives) // 5)]
    
    return random.choice(top_candidates)[0]

def generate_triplets_from_batch(current_batch, current_labels,
                                 keeper_history, non_keeper_history):
    batch_keepers = [
        record for record, label in zip(current_batch, current_labels)
        if label
    ]
    
    if len(batch_keepers) == 0:
        # No keepers in this batch
        return None
    
    if len(keeper_history) == 0 or len(non_keeper_history) == 0:
        # Not enough history yet
        return None
    
    anchors = []
    positives = []
    negatives = []
    
    for anchor in batch_keepers:
        # Positive: random keeper from history
        positive = random.choice(keeper_history)
        
        # Negative: random non-keeper from history
        negative = random.choice(non_keeper_history)
        
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    
    return {
        'anchors': anchors,
        'positives': positives,
        'negatives': negatives
    }
        
