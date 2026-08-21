"""
Evaluation metrics for record classification pipeline.

Calculates precision, recall, coverage, and confidence intervals.
"""

import numpy as np
from typing import Dict, List, Set


def calculate_metrics(
    pipeline_results: Dict,
    test_set: List[Dict]
) -> Dict:
    """
    Calculate evaluation metrics for pipeline results.

    Args:
        pipeline_results: Output from classify_catalog() containing:
            - ruled_in: list of release_ids
            - ruled_out: list of release_ids
            - verified: list of release_ids queried via API
            - metadata: dict with api_calls_made, etc.

        test_set: List of records with ground truth wants/haves

    Returns:
        dict with comprehensive metrics
    """
    # Build ground truth map
    ground_truth = {}
    for record in test_set:
        rid = str(record.get('discogs_id') or record.get('release_id'))
        ground_truth[rid] = (record['wants'] > record['haves'])

    # Convert to sets for fast lookup
    ruled_in_set = set(str(rid) for rid in pipeline_results['ruled_in'])
    ruled_out_set = set(str(rid) for rid in pipeline_results['ruled_out'])

    # Calculate precision (ruled_in)
    ri_in_test = ruled_in_set & set(ground_truth.keys())
    ri_correct = sum(1 for rid in ri_in_test if ground_truth[rid])
    precision_in = ri_correct / len(ri_in_test) if ri_in_test else 0.0

    # Calculate precision (ruled_out)
    ro_in_test = ruled_out_set & set(ground_truth.keys())
    ro_correct = sum(1 for rid in ro_in_test if not ground_truth[rid])
    precision_out = ro_correct / len(ro_in_test) if ro_in_test else 0.0

    # Calculate recall
    total_positives = sum(1 for is_pos in ground_truth.values() if is_pos)
    total_negatives = len(ground_truth) - total_positives

    recall_in = ri_correct / total_positives if total_positives else 0.0
    recall_out = ro_correct / total_negatives if total_negatives else 0.0

    # Coverage
    total_classified = len(ri_in_test) + len(ro_in_test)
    api_calls = pipeline_results['metadata'].get('api_calls_made', 0)
    records_per_call = total_classified / api_calls if api_calls > 0 else 0.0
    coverage_pct = total_classified / len(ground_truth) if len(ground_truth) > 0 else 0.0

    # Confidence intervals (95%)
    ci_in = wilson_confidence_interval(ri_correct, len(ri_in_test))
    ci_out = wilson_confidence_interval(ro_correct, len(ro_in_test))

    # F1 scores
    f1_in = 2 * precision_in * recall_in / (precision_in + recall_in) if (precision_in + recall_in) > 0 else 0.0
    f1_out = 2 * precision_out * recall_out / (precision_out + recall_out) if (precision_out + recall_out) > 0 else 0.0

    return {
        'precision': {
            'ruled_in': precision_in,
            'ruled_out': precision_out,
            'ruled_in_ci': ci_in,
            'ruled_out_ci': ci_out,
        },
        'recall': {
            'ruled_in': recall_in,
            'ruled_out': recall_out,
        },
        'f1': {
            'ruled_in': f1_in,
            'ruled_out': f1_out,
        },
        'coverage': {
            'coverage_pct': coverage_pct,
            'records_per_api_call': records_per_call,
            'total_classified': total_classified,
            'api_calls_made': api_calls,
        },
        'counts': {
            'ruled_in': len(ri_in_test),
            'ruled_out': len(ro_in_test),
            'ruled_in_correct': ri_correct,
            'ruled_out_correct': ro_correct,
            'total_positives': total_positives,
            'total_negatives': total_negatives,
            'test_set_size': len(ground_truth),
        }
    }


def wilson_confidence_interval(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """
    Calculate Wilson score confidence interval.

    More accurate than normal approximation for small samples or extreme proportions.

    Args:
        successes: Number of successes
        total: Total trials
        confidence: Confidence level (default: 0.95)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    if total == 0:
        return (0.0, 0.0)

    # Z-score for confidence level
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return (
        max(0.0, center - margin),
        min(1.0, center + margin)
    )


def format_metrics_report(metrics: Dict) -> str:
    """
    Format metrics as human-readable report.

    Args:
        metrics: Output from calculate_metrics()

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION METRICS")
    lines.append("=" * 70)

    # Precision
    lines.append("\nPRECISION:")
    prec_in = metrics['precision']['ruled_in']
    ci_in = metrics['precision']['ruled_in_ci']
    lines.append(f"  Ruled In:  {prec_in:6.1%}  (95% CI: [{ci_in[0]:.1%}, {ci_in[1]:.1%}])")

    prec_out = metrics['precision']['ruled_out']
    ci_out = metrics['precision']['ruled_out_ci']
    lines.append(f"  Ruled Out: {prec_out:6.1%}  (95% CI: [{ci_out[0]:.1%}, {ci_out[1]:.1%}])")

    # Recall
    lines.append("\nRECALL:")
    lines.append(f"  Ruled In:  {metrics['recall']['ruled_in']:6.1%}")
    lines.append(f"  Ruled Out: {metrics['recall']['ruled_out']:6.1%}")

    # F1
    lines.append("\nF1 SCORE:")
    lines.append(f"  Ruled In:  {metrics['f1']['ruled_in']:6.1%}")
    lines.append(f"  Ruled Out: {metrics['f1']['ruled_out']:6.1%}")

    # Coverage
    lines.append("\nCOVERAGE:")
    cov = metrics['coverage']
    lines.append(f"  Catalog coverage: {cov['coverage_pct']:6.1%}")
    lines.append(f"  Records/API call: {cov['records_per_api_call']:.0f}")
    lines.append(f"  Total classified: {cov['total_classified']:,}")
    lines.append(f"  API calls made:   {cov['api_calls_made']:,}")

    # Counts
    lines.append("\nCOUNTS:")
    c = metrics['counts']
    lines.append(f"  Ruled in:  {c['ruled_in']:,} ({c['ruled_in_correct']:,} correct)")
    lines.append(f"  Ruled out: {c['ruled_out']:,} ({c['ruled_out_correct']:,} correct)")
    lines.append(f"  Test set:  {c['test_set_size']:,} ({c['total_positives']:,} pos, {c['total_negatives']:,} neg)")

    lines.append("=" * 70)

    return '\n'.join(lines)
