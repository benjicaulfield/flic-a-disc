import torch

def best_f1_threshold(probs, labels, step=0.01):
    """
    probs: (N,) float tensor in [0,1]
    labels: (N,) float/bool tensor in {0,1}
    Returns: (best_threshold: float, best_f1: float, precision: float, recall: float)
    """
    probs = probs.view(-1)
    labels = labels.view(-1).float()

    # same guardrails you used
    if probs.numel() < 50:
        return 0.5, 0.0, 0.0, 0.0

    num_pos = labels.sum().item()
    if num_pos < 10:
        return 0.5, 0.0, 0.0, 0.0

    best_thr = 0.5
    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0

    thresholds = torch.arange(0.01, 0.99, step, device=probs.device)

    for t in thresholds:
        pred = (probs > t).float()
        tp = ((pred == 1) & (labels == 1)).sum().item()
        fp = ((pred == 1) & (labels == 0)).sum().item()
        fn = ((pred == 0) & (labels == 1)).sum().item()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t.item())
            best_p = p
            best_r = r

    return best_thr, best_f1, best_p, best_r
