# Evaluation Files Documentation

Technical reference for the evaluation package contents.

---

## File Structure

```
eval/
├── agent_files/                # Files provided to agent for development
│   ├── enriched_training.json  # 60k training records with wants/haves
│   ├── lp_catalog.json         # 365k catalog WITHOUT wants/haves (test input)
│   ├── mock_api.py             # Mock API server (no rate limits)
│   └── mock_api_ids.json       # List of IDs available in mock API
│
└── Grader Package (5 files):   # Flat structure - sent to grading agent
    ├── evaluate.py             # Main evaluation script
    ├── metrics.py              # Metric calculation functions
    ├── test_api_client.py      # Test API client (enforces budget)
    ├── requirements.txt        # Python dependencies
    └── lp_catalog.json         # 365k catalog WITH wants/haves (ground truth)
```

---

## Agent Files (Provided for Development)

### enriched_training.json (11 MB)

**Purpose:** Training data with wants/haves

**Format:** JSON array of 30,000 records

**Schema:**
```json
{
  "discogs_id": "12345",
  "release_id": "12345",
  "artist": "Artist Name",
  "title": "Album Title",
  "label": "Label Name",
  "catalog_number": "CAT123",
  "year": 1970,
  "country": "US",
  "genre": ["Rock"],
  "style": ["Blues Rock"],
  "wants": 100,
  "haves": 50
}
```

**Distribution:**
- Positives (wants > haves): 6,845 (22.82%)
- Negatives (wants ≤ haves): 23,155 (77.18%)
- **Matches test distribution exactly** (22.82% positive)

**Usage:**
- Train your classification model
- Validate propagation strategies
- Develop active learning approach

---

### lp_catalog.json (151 MB)

**Purpose:** Test catalog WITHOUT wants/haves - this is your pipeline input

**Format:** JSON array of 395,508 records

**Schema:**
```json
{
  "release_id": "161",
  "master_id": "18519",
  "artist": "Artist Name",
  "title": "Album Title",
  "label": "Label Name",
  "catalog_number": "CAT123",
  "year": 1996,
  "country": "UK",
  "genre": ["Electronic"],
  "style": ["Abstract", "Experimental"]
}
```

**Note:**
- Does NOT include wants/haves (that's what you must predict/query)
- Your pipeline receives this as input
- Use API to selectively query wants/haves (max 1,000 calls)

---

### mock_api.py (2.3 KB)

**Purpose:** Mock API server for development (no rate limits)

**Usage:**
```bash
cd eval/agent_files
uvicorn mock_api:app --port 8001
```

**Endpoints:**
- `GET /` - API info
- `GET /releases/{release_id}` - Get wants/haves for a release
- `GET /health` - Health check

**Example:**
```python
import requests

response = requests.get('http://localhost:8001/releases/12345')
data = response.json()
# Returns: {'release_id': 12345, 'wants': 100, 'haves': 50}
```

**Features:**
- Serves 30k training records
- No rate limits (fast iteration)
- Same interface as evaluation API

---

### mock_api_ids.json (396 KB)

**Purpose:** List of release IDs available in mock API

**Format:** JSON array of 30,000 release ID strings

**Usage:**
- Know which records can be queried during development
- Test your pipeline's API query strategy
- Validate propagation works with available data

---

## Grader Package Files

### lp_catalog.json (151 MB)

**Purpose:** Full catalog WITH wants/haves - ground truth for evaluation

**Format:** JSON array of 395,508 records (same as agent version + wants/haves)

**Distribution:**
- Positives (wants > haves): 90,243 (22.82%)
- Negatives (wants ≤ haves): 305,265 (77.18%)

**Usage:**
- Evaluation script strips wants/haves before giving to pipeline
- After pipeline runs, compares predictions to this ground truth
- Calculates precision/recall/coverage metrics

**Note:** This file is in the grader package but NOT visible to the agent during development

---

## Evaluation Scripts

### evaluate.py

Main evaluation script that runs your pipeline and calculates metrics.

**Usage:**
```bash
cd eval
python evaluate.py --pipeline ../cataloger/pipeline.py
```

**What it does:**

1. Loads lp_catalog.json WITH ground truth (395k records)
2. Strips wants/haves to create test input
3. Creates API client backed by ground truth (max 1,000 calls)
4. Runs your `classify_catalog()` function
5. Compares your predictions to ground truth
6. Calculates all metrics
7. Checks pass/fail criteria
8. Saves `evaluation_report.json`

**Pass/Fail Criteria:**
- Coverage: ≥75% of catalog classified
- Precision (ruled_in): ≥90%
- Precision (ruled_out): ≥90%
- Recall (ruled_in): ≥70%
- Recall (ruled_out): ≥70%
- API Budget: ≤1,000 calls

---

### test_api_client.py

Mock API client that enforces budget and returns ground truth.

**Interface:**
```python
class TestAPIClient:
    def __init__(self, ground_truth: list, max_calls: int = 1000):
        """
        Args:
            ground_truth: List of records with wants/haves
            max_calls: Maximum API calls allowed
        """

    def get_release(self, release_id: int) -> dict:
        """
        Returns:
            {'release_id': int, 'wants': int, 'haves': int}

        Raises:
            Exception: If budget exceeded or release not found
        """
```

**Features:**
- Enforces 1,000 call budget
- Returns wants/haves from ground truth
- Raises exception if budget exceeded
- Same interface as mock API

---

### metrics.py

Metric calculation functions for evaluation.

**Main function:**
```python
metrics = calculate_metrics(pipeline_results, ground_truth)
```

**Calculates:**
- **Coverage**: % of catalog classified, records per API call
- **Precision**: Accuracy on ruled_in and ruled_out predictions
- **Recall**: % of true positives/negatives found
- **F1 scores**: Harmonic mean of precision/recall
- **Confidence intervals**: 95% Wilson intervals for precision

**Output format:**
```json
{
  "coverage": {
    "coverage_pct": 0.76,
    "records_per_api_call": 323.5,
    "total_classified": 323500,
    "api_calls_made": 1000
  },
  "precision": {
    "ruled_in": 0.91,
    "ruled_out": 0.93,
    "ruled_in_ci": [0.89, 0.93],
    "ruled_out_ci": [0.92, 0.94]
  },
  "recall": {
    "ruled_in": 0.72,
    "ruled_out": 0.74
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
    "total_positives": 97088,
    "total_negatives": 328420,
    "test_set_size": 425508
  }
}
```

---

## Pipeline Interface

Your pipeline must implement:

```python
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
```

**Requirements:**
- Must classify ≥75% of catalog (296,631+ records)
- Must maintain ≥90% precision on both classes
- Must achieve ≥70% recall on both classes
- Must use ≤1,000 API calls

---

## Output Format

**evaluation_report.json:**
```json
{
  "timestamp": "2026-04-04T12:00:00",
  "catalog_size": 395508,
  "ground_truth": {
    "positives": 90243,
    "negatives": 305265,
    "positive_rate": 0.2282
  },
  "metrics": {
    "coverage": { ... },
    "precision": { ... },
    "recall": { ... },
    "f1": { ... },
    "counts": { ... }
  }
}
```

---

## Development Workflow

**1. Local Development:**
```bash
# Start mock API
cd eval/agent_files
uvicorn mock_api:app --port 8001

# Develop your pipeline using:
# - enriched_training.json (train model)
# - lp_catalog.json (test input)
# - mock API (query wants/haves)
```

**2. Test Locally:**
```bash
# Run evaluation on your pipeline
cd eval
python evaluate.py --pipeline your_pipeline.py
```

**3. Check Results:**
```bash
# View metrics
cat evaluation_report.json

# Check pass/fail
echo $?  # 0 = pass, 1 = fail
```

---

## Key Differences from Agent's View

**What Agent Has:**
- `lp_catalog.json` WITHOUT wants/haves
- Mock API with 60k training records
- No ground truth visibility

**What Grader Has:**
- `lp_catalog.json` WITH wants/haves (full ground truth) - 5 files in flat structure
- API backed by all 395k records
- Can calculate exact precision/recall
- Pass/fail thresholds: 75% coverage, 90% precision, 70% recall both classes

**Why This Matters:**
- Agent must use model + propagation to classify 75%+ of catalog
- Agent can only query 1,000 records via API (strategic selection critical)
- Evaluation measures exact accuracy on all predictions
- Simulates real-world constraint: limited API budget, need smart strategies
- Training and test have identical distributions (22.82% positive) - fair, clean split

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy (metrics calculations)
- fastapi (mock API)
- uvicorn (API server)
