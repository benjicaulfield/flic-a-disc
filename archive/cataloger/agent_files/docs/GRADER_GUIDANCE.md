AGENTIC GRADER GUIDANCE

You are evaluating a machine learning pipeline that classifies 395,508 vinyl LP records as "desirable" or not using active learning and propagation strategies. The pipeline has access to catalog metadata (artist, title, label, genre, year) for all records, but community demand data (wants/haves counts) is only available via API calls with a strict budget of 1,000 calls.

The key challenge: achieve 75% coverage on a 395k catalog using only 1,000 API calls while maintaining 90% precision and 70% recall on both positive and negative classifications. This requires combining ML-based classification with strategic propagation techniques.

================================================================================

EVALUATION FILES

The grader package includes 5 files (flat structure):

├── evaluate.py                   (Main evaluation script)
├── metrics.py                    (Metric calculation functions)
├── test_api_client.py            (Mock API client that enforces budget)
├── requirements.txt              (Python dependencies)
└── lp_catalog.json               (365k records WITH wants/haves - ground truth)

Ground Truth: lp_catalog.json contains 395,508 enriched records with known wants/haves values:
  - Positive class: wants > haves (records people want more than own)
  - Negative class: wants ≤ haves (records people own as much or more than want)
  - Distribution: 22.82% positive (90,243 records), 77.18% negative (305,265 records)

================================================================================

RUNNING EVALUATION

cd eval
pip install -r requirements.txt
python evaluate.py --pipeline ../cataloger/pipeline.py

This will:
  1. Load catalog WITH ground truth (395k records)
  2. Strip wants/haves to create test input
  3. Create API client backed by ground truth (enforces 1,000 call budget)
  4. Run the agent's classify_catalog() function
  5. Compare predictions to ground truth
  6. Calculate all metrics
  7. Check pass/fail criteria
  8. Save evaluation_report.json
  9. Exit with code 0 (pass) or 1 (fail)

================================================================================

PASS/FAIL CRITERIA

All six requirements below are mandatory. If any requirement fails, the entire evaluation fails.

1. Coverage ≥ 75%
   Formula: (ruled_in_count + ruled_out_count) / 395,508
   Location: metrics.coverage.coverage_pct >= 0.75
   Requirement: Must classify at least 296,631 out of 395,508 records
   Why it matters: Pipeline must provide decisions for most of catalog

2. Precision (Ruled In) ≥ 90%
   What it measures: Accuracy when predicting "desirable" (wants > haves)
   Location: metrics.precision.ruled_in >= 0.90
   Why it matters: False positives waste money buying undesirable records

3. Precision (Ruled Out) ≥ 90%
   What it measures: Accuracy when predicting "undesirable" (wants ≤ haves)
   Location: metrics.precision.ruled_out >= 0.90
   Why it matters: False negatives miss profitable buying opportunities

4. Recall (Ruled In) ≥ 70%
   What it measures: % of true positives correctly classified
   Location: metrics.recall.ruled_in >= 0.70
   True positives: 90,243 records in catalog (22.82%)
   Target: Find at least 63,170 desirable records (70%)
   Why it matters: Must find majority of the desirable records

5. Recall (Ruled Out) ≥ 70%
   What it measures: % of true negatives correctly classified
   Location: metrics.recall.ruled_out >= 0.70
   True negatives: 305,265 records in catalog (77.18%)
   Target: Correctly classify at least 213,686 undesirable records (70%)
   Why it matters: Must correctly filter out majority of undesirable records

6. API Budget ≤ 1,000 Calls
   What it measures: Total API calls made during classification
   Location: metrics.coverage.api_calls_made <= 1000
   Why it matters: Pipeline must respect strict rate limits

If any requirement is not met, the task is considered a FAILURE.

================================================================================

METRICS LOCATION IN REPORT

{
  "timestamp": "2026-04-05T12:00:00",
  "catalog_size": 395508,
  "ground_truth": {
    "positives": 90243,
    "negatives": 305265,
    "positive_rate": 0.2282
  },
  "metrics": {
    "coverage": {
      "coverage_pct": 0.76,              (Must be ≥ 0.75)
      "records_per_api_call": 323.5,     (Context only - not evaluated)
      "total_classified": 323500,
      "api_calls_made": 1000             (Must be ≤ 1000)
    },
    "precision": {
      "ruled_in": 0.91,                  (Must be ≥ 0.90)
      "ruled_out": 0.93,                 (Must be ≥ 0.90)
      "ruled_in_ci": [0.89, 0.93],
      "ruled_out_ci": [0.92, 0.94]
    },
    "recall": {
      "ruled_in": 0.72,                  (Must be ≥ 0.70)
      "ruled_out": 0.74                  (Must be ≥ 0.70)
    },
    "f1": {
      "ruled_in": 0.905,
      "ruled_out": 0.925
    },
    "counts": {
      "ruled_in": 88000,
      "ruled_out": 235500,
      "ruled_in_correct": 80080,
      "ruled_out_correct": 218415,
      "total_positives": 90243,
      "total_negatives": 305265,
      "test_set_size": 395508
    }
  },
  "pass": true                           (All requirements met)
}

================================================================================

EXPECTED PIPELINE INTERFACE

The agent's pipeline must implement:

def classify_catalog(catalog: list[dict], api_client: APIClient) -> dict:
    """
    Classify records from catalog using up to 1,000 API calls.

    Args:
        catalog: List of records from lp_catalog.json (WITHOUT wants/haves)
        api_client: API client with get_release(release_id) method

    Returns:
        {
            'ruled_in': [release_ids],      # Confident positives (wants > haves)
            'ruled_out': [release_ids],     # Confident negatives (wants ≤ haves)
            'verified': [release_ids],      # IDs queried via API
            'metadata': {
                'api_calls_made': int,
                'coverage_ratio': float,
                'approach': str
            }
        }
    """

The API client enforces the 1,000 call budget and will raise an exception if exceeded.

================================================================================

UNDERSTANDING THE CHALLENGE

The Core Constraint:
  - 395,508 records to classify
  - Only 1,000 API calls allowed
  - Must classify 75%+ (296,631+ records)
  - Must achieve 90% precision, 70% recall on both classes

How Solutions Achieve This:

1. ML-Based Classification (primary coverage driver)
   - Train model on 30k training records (stratified at 22.82% positive, matches test exactly)
   - Apply to full catalog to generate predictions
   - This is the PRIMARY coverage driver (not master grouping)
   - Must achieve 90% precision, 70% recall on metadata-only features

2. Master ID Sibling Propagation (supplemental coverage, 2-10%)
   - Records in same master_id TEND to have correlated wants/haves
   - But they do NOT share identical values (factually incorrect claim)
   - Single-sample propagation accuracy: ~82% (below 90% threshold)
   - Majority-vote propagation (2-3 queries per master): ~90% accuracy
   - Dataset characteristics:
     * master_id = '0' represents 95,877 unrelated records (sentinel value)
     * Largest real master group: 57 members (not "hundreds")
     * Average multi-member master size: ~3.4 members
     * Top 1,000 masters cover only 4.4% of catalog (18,855 records)

3. Active Learning / Strategic API Usage
   - Query high-uncertainty predictions to validate
   - Query representative records from large master groups
   - Use API results to update model predictions
   - Refine classification iteratively

Feasibility Analysis:

To achieve 75% coverage (296,631 records) with 70% recall on both classes:
  - ML predictions: ~275,000-285,000 records (69-72% coverage)
  - Master grouping: ~12,000-20,000 records (3-5%) with 1,000 API calls

Minimum correctly classified to meet recall requirements:
  - 63,170 true positives (70% of 90,243)
  - 213,686 true negatives (70% of 305,265)
  - Total: 276,856 records minimum
  - Buffer: ~19,775 records for imperfect precision

This is achievable with a well-calibrated ML model maintaining 90% precision and 70% recall.

Why All Metrics Matter:
  - Coverage: Must classify most of catalog to be useful
  - Precision: Both false positives and false negatives are costly
  - Recall: Must find most true positives AND most true negatives
  - API Budget: Real-world constraint (Discogs API is rate-limited)

================================================================================

COMMON FAILURE MODES

1. Low Coverage (<75%)
   - Not using training data as anchor labels
   - Pipeline too conservative, doesn't make enough predictions
   - Model confidence thresholds set too high

2. Low Precision
   - Overly aggressive propagation without validation
   - Poor model calibration
   - Treating master_id='0' as a real propagation group (it's a sentinel value)
   - Naive single-sample master sibling propagation (only 82% accurate)

3. Low Recall (<70%)
   - Threshold too high, missing too many true positives/negatives
   - Unbalanced approach (e.g., high recall on negatives, low on positives)
   - Too conservative: achieving 90% precision at cost of recall
   - Poor feature engineering leading to low-confidence predictions

4. Budget Exceeded
   - Querying too many individual records
   - Not leveraging training data as free labels
   - Excessive validation queries

================================================================================

WHAT MAKES A STRONG SOLUTION

A passing solution will demonstrate:

1. Strong ML Model
   - Trained on 30k enriched training data (stratified at 22.82% to match test)
   - Feature engineering on metadata (artist, title, label, genre, style, year, country)
   - Calibrated to achieve 90% precision, 70% recall
   - Handles class imbalance (22.82% positive rate)
   - Balanced to meet recall requirements on both classes
   - No distribution mismatch between train and test

2. Supplemental Propagation Strategies
   - Master sibling propagation with validation (not naive propagation)
   - Majority-vote or uncertainty-based sampling within master groups
   - Avoid treating master_id='0' as a propagation group

3. Strategic API Usage
   - Query high-uncertainty predictions
   - Query representative records from large master groups
   - Use results to validate and refine predictions
   - 1,000 calls used for validation/refinement, not primary coverage

4. Scalable Architecture
   - Efficient data structures (indexes, caches)
   - Handles 395k records without memory issues
   - Batch processing where appropriate

The evaluation is pass/fail based on meeting ALL six requirements. A solution that gets 95% precision but only 70% coverage still fails. All metrics must meet thresholds.

================================================================================

IMPORTANT DATA NOTES

Training Data Field Mapping:
  Training and catalog use consistent field names:
  - release_id, master_id, artist, title, label, catalog_number
  - genre (array), style (array), year, country
  - Training also includes: wants, haves

Training Records Separate from Evaluation:
  The 30,000 training records are NOT in the 395,508 evaluation catalog.
  Clean train/test split - zero overlap between training and evaluation sets.
  Training is stratified sample at 22.82% positive (matches test distribution exactly).
  Agents must generalize from 30k training to 395k unseen evaluation records.

Master ID Sentinel Values:
  - master_id = '0' (string): 95,877 records with no real master (unrelated releases)
  - master_id = '' (empty): varies by dataset
  These should NOT be treated as propagation groups - they are sentinel values.
