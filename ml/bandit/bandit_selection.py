import math
import random
import numpy as np

def calculate_exploration_rate(batch_num, start=0.7, end=0.3, total_batches=100):
    decay = math.log(end / start) / total_batches
    return start * math.exp(decay * batch_num)

def exploit_selection(candidates, predictions, uncertainties, n):
    # top by reward-uncertainty
    scores = np.array(predictions) + 2.0 * np.array(uncertainties)    
    top_indices = np.argsort(scores)[-n:][::-1]
    return [candidates[i] for i in top_indices]


def explore_selection(candidates, uncertainties, m):
    # top by uncertainty
    top_indices = np.argsort(uncertainties)[-m:][::-1]
    return [candidates[i] for i in top_indices]

def enforce_artist_diversity(selected, candidates, artists, max_per_artist=2, target_size=20):
    artist_counts = {}
    diverse = []
    overflow = []

    for idx in selected:
        artist = artists[idx] if idx < len(artists) else ''
        count = artist_counts.get(artist, 0)
        if count < max_per_artist:
            diverse.append(idx)
            artist_counts[artist] = count + 1
        else:
            overflow.append(idx)

    # Fill remaining slots from candidates not already selected
    if len(diverse) < target_size:
        remaining = [c for c in candidates if c not in diverse and c not in overflow]
        random.shuffle(remaining)
        for idx in remaining:
            if len(diverse) >= target_size:
                break
            artist = artists[idx] if idx < len(artists) else ''
            if artist_counts.get(artist, 0) < max_per_artist:
                diverse.append(idx)
                artist_counts[artist] = artist_counts.get(artist, 0) + 1

    return diverse[:target_size]


def adaptive_batch_selection(candidates, predictions, uncertainties, batch_num,
                             total_batch_size=20, total_batches=100, random_count=8,
                             artists=None, max_per_artist=2):

    bandit_size = total_batch_size - random_count

    exploration_rate = calculate_exploration_rate(batch_num)
    m = int(total_batch_size * exploration_rate)
    n = bandit_size - m

    exploit = exploit_selection(candidates, predictions, uncertainties, n)
    explore = explore_selection(candidates, uncertainties, m)

    bandit_selected = list(dict.fromkeys(explore + exploit))[:bandit_size]

    # Add completely random records for diversity
    remaining = [c for c in candidates if c not in bandit_selected]
    random_sample = random.sample(remaining, min(random_count, len(remaining)))

    final_batch = bandit_selected + random_sample

    # Apply artist diversity cap if artist list provided
    if artists:
        final_batch = enforce_artist_diversity(
            final_batch, candidates, artists,
            max_per_artist=max_per_artist,
            target_size=total_batch_size
        )
        print(f"📦 Batch: {len(final_batch)} records (artist-diverse, max {max_per_artist}/artist)")
    else:
        print(f"📦 Batch: {len(bandit_selected)} bandit ({n} exploit + {m} explore) + {len(random_sample)} random")

    return final_batch

