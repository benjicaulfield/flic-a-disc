import math
import random
import numpy as np

def calculate_exploration_rate(batch_num, start=0.6, end=0.2, total_batches=100):
    decay = math.log(end / start) / total_batches
    return start * math.exp(decay * batch_num)

def exploit_selection(candidates, predictions, uncertainties, n):
    # top by reward-uncertainty
    scores = np.array(predictions) - np.array(uncertainties)
    top_indices = np.argsort(scores)[-n:][::-1]
    return [candidates[i] for i in top_indices]


def explore_selection(candidates, uncertainties, m):
    # top by uncertainty
    top_indices = np.argsort(uncertainties)[-m:][::-1]
    return [candidates[i] for i in top_indices]

def adaptive_batch_selection(candidates, predictions, uncertainties, batch_num, 
                             total_batch_size=20, total_batches=100, random_count=3):

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
    
    print(f"📦 Batch: {len(bandit_selected)} bandit ({n} exploit + {m} explore) + {len(random_sample)} random")
    
    return final_batch

