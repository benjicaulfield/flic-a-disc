# Handover Document: Discogs Active Learning Pipeline Evaluation

**Date:** April 2, 2026
**Status:** Test set collection in progress, evaluation infrastructure needed

---

## Project Overview

Building an evaluation framework for an active learning pipeline that classifies ~3 million vinyl LP records as "desirable" (wants > haves) using limited API calls.

**The Challenge:**
- 3M records in lp_catalog.json (filtered from 31M releases)
- Community data (wants/haves) only available via API (60 requests/min rate limit)
- Pipeline must classify records using max 5,000 API calls
- Target: 90%+ precision, 100+ records per API call (via propagation strategies)

**Key Constraint:** Evaluation must be deterministic and verifiable.

---

## Current State

### ✅ Completed

1. **Agent Pipeline** (`cataloger/agent_pipeline.py`)
   - TF-IDF + Logistic Regression model
   - Master_id group propagation strategy
   - OOF-validated thresholds (90%+ precision)
   - Successfully runs on 2.9M catalog in ~3.5 minutes (mock API)
   - Achieves ~208 records/API call coverage

2. **Mock API** (`cataloger/mock_api.py`)
   - FastAPI server serving enriched_training.json
   - GET /releases/{release_id} endpoint
   - No rate limits (fast iteration)

3. **Data Files** (in `cataloger/data/ready/`)
   - `lp_catalog.json` (1GB, 2.9M records) - Full catalog
   - `enriched_training.json` (22MB, 60k records) - Training data with wants/haves
   - Base positive rate: 29.4%

4. **Test Set Builder** (`ml/bandit/management/commands/build_test_set.py`)
   - Currently running in background
   - Collecting negatives from seller inventories
   - Target: ~10k balanced test set (positives + negatives)
   - Using seller inventory "loophole" (wants/haves from listing.data, no extra API calls)

### 🔄 In Progress

**Test Set Collection:**
- Script is running: `uv run python manage.py build_test_set`
- Randomly selects sellers from `ml/sellers.json`
- Collects negatives with full metadata (skips positives to avoid API calls)
- Saves to Django database (Record model)
- Target: 21,000 negatives total (currently have 3,021 baseline)

**Location:**
- Command: `ml/bandit/management/commands/build_test_set.py`
- Function: `ml/bandit/utils/get_user_inventory.py::build_test_set_negatives()`
- Database: PostgreSQL (Record table)

### ❌ Not Started

1. **Test set export** - Export 10k records from database to JSON
2. **Evaluation script** - Run pipeline and grade against test set
3. **Metrics calculation** - Precision/recall on test set
4. **Documentation** - Final evaluation report

---

## Key Files & Directories

```
cataloger/
├── agent_pipeline.py          # The pipeline being evaluated
├── mock_api.py                # Mock Discogs API (FastAPI)
├── data/ready/
│   ├── lp_catalog.json        # 2.9M catalog (input)
│   ├── enriched_training.json # 60k training data
│   └── test_set.json          # ← NEEDS TO BE CREATED
└── HANDOVER.md                # Additional context

ml/
├── bandit/
│   ├── models.py              # Django models (Record, etc.)
│   ├── management/commands/
│   │   ├── build_test_set.py      # Test set builder (currently running)
│   │   └── count_available_records.py  # Check DB record counts
│   └── utils/
│       └── get_user_inventory.py  # Seller inventory fetcher
├── sellers.json               # List of Discogs sellers
└── discogs/data/
    └── enriched_training.json # 60k training data (copy)
```

---

## Database Schema

**Record Model** (`ml/bandit/models.py`):
```python
class Record(models.Model):
    discogs_id = CharField(unique=True)
    artist = CharField
    title = CharField
    label = TextField
    catno = CharField
    wants = IntegerField
    haves = IntegerField
    genres = JSONField
    styles = JSONField
    year = IntegerField
    format = JSONField
    suggested_price = CharField  # Not used for test set
    skipped = BooleanField       # True if evaluated and rejected
```

**Current counts:**
- Total records NOT in enriched_training.json: 55,212
- Positives (wants > haves): 52,191 (94.5%)
- Negatives (wants ≤ haves): 3,021 (5.5%)

---

## What Needs to Happen Next

### 1. Export Test Set from Database

Once the test set builder finishes collecting negatives:

**Goal:** Create `cataloger/data/ready/test_set.json` with ~10k records

**Approach:**
```python
# Pseudocode
from bandit.models import Record
import json

# Load enriched_training IDs to exclude
with open('cataloger/data/ready/enriched_training.json') as f:
    training_ids = {r['discogs_id'] for r in json.load(f)}

# Sample from database
available = Record.objects.filter(
    skipped=False
).exclude(
    discogs_id__in=training_ids
)

# Get balanced sample (~6k negatives + ~4k positives = 10k total)
negatives = available.filter(wants__lte=F('haves')).values()[:6000]
positives = available.filter(wants__gt=F('haves')).values()[:4000]

# Combine and export
test_set = list(negatives) + list(positives)
with open('cataloger/data/ready/test_set.json', 'w') as f:
    json.dump(test_set, f, indent=2)
```

**Format:** Same as enriched_training.json:
```json
[
  {
    "discogs_id": "12345",
    "artist": "Artist Name",
    "title": "Album Title",
    "label": "Label Name",
    "catno": "CAT123",
    "genres": ["Rock"],
    "styles": ["Blues Rock"],
    "year": 1970,
    "format": ["Vinyl", "LP"],
    "wants": 100,
    "haves": 50
  }
]
```

### 2. Create Evaluation Script

**Goal:** Deterministic grading of pipeline against test set

**Location:** `cataloger/evaluate_pipeline.py`

**Requirements:**
```python
def evaluate_pipeline(pipeline_results, test_set):
    """
    Grade pipeline predictions against ground truth.

    Args:
        pipeline_results: Output from classify_catalog()
            {
                'ruled_in': [release_ids],
                'ruled_out': [release_ids],
                'verified': [release_ids],
                'metadata': {...}
            }
        test_set: List of records with ground truth wants/haves

    Returns:
        {
            'precision_in': float,     # % of ruled_in that are true positives
            'precision_out': float,    # % of ruled_out that are true negatives
            'recall': float,           # % of true positives found
            'coverage': int,           # records classified per API call
            'total_classified': int,
            'api_calls_used': int
        }
    """
    # Build ground truth map
    ground_truth = {
        str(r['discogs_id']): (r['wants'] > r['haves'])
        for r in test_set
    }

    # Calculate metrics
    ruled_in_set = set(pipeline_results['ruled_in'])
    ruled_out_set = set(pipeline_results['ruled_out'])

    # Precision (ruled_in)
    ri_correct = sum(1 for rid in ruled_in_set if ground_truth.get(rid, False))
    precision_in = ri_correct / len(ruled_in_set) if ruled_in_set else 0

    # Precision (ruled_out)
    ro_correct = sum(1 for rid in ruled_out_set if not ground_truth.get(rid, True))
    precision_out = ro_correct / len(ruled_out_set) if ruled_out_set else 0

    # Recall
    total_positives = sum(1 for is_pos in ground_truth.values() if is_pos)
    recall = ri_correct / total_positives if total_positives else 0

    # Coverage
    total_classified = len(ruled_in_set) + len(ruled_out_set)
    api_calls = pipeline_results['metadata']['api_calls_made']
    coverage = total_classified / api_calls if api_calls else 0

    return {
        'precision_in': precision_in,
        'precision_out': precision_out,
        'recall': recall,
        'coverage': coverage,
        'total_classified': total_classified,
        'api_calls_used': api_calls
    }
```

**Usage:**
```python
# Load test set
with open('cataloger/data/ready/test_set.json') as f:
    test_set = json.load(f)

# Create test API client
test_api = TestAPIClient(test_set)  # Returns real wants/haves

# Run pipeline on test set only
test_catalog = [r for r in catalog if r['release_id'] in test_ids]
results = classify_catalog(test_catalog, test_api)

# Evaluate
metrics = evaluate_pipeline(results, test_set)
print(f"Precision (in): {metrics['precision_in']:.1%}")
print(f"Precision (out): {metrics['precision_out']:.1%}")
print(f"Recall: {metrics['recall']:.1%}")
print(f"Coverage: {metrics['coverage']:.0f} records/call")
```

### 3. Create TestAPIClient

Similar to MockAPIClient but uses test_set.json:

```python
class TestAPIClient:
    """API client backed by test set ground truth."""
    def __init__(self, test_set):
        self.data = {
            str(r['discogs_id']): {
                'wants': r['wants'],
                'haves': r['haves']
            }
            for r in test_set
        }
        self.calls_made = 0
        self.max_calls = 5000

    def get_release(self, release_id):
        self.calls_made += 1
        if self.calls_made > self.max_calls:
            raise Exception("API budget exceeded")

        rid = str(release_id)
        if rid not in self.data:
            raise KeyError(f"Release {rid} not in test set")

        return {
            'release_id': release_id,
            'wants': self.data[rid]['wants'],
            'haves': self.data[rid]['haves']
        }
```

---

## Expected Metrics (from OOF validation)

The pipeline author claims these metrics based on out-of-fold validation:

- **Ruled_in precision:** 92.2% (threshold: P ≥ 0.890)
- **Ruled_out precision:** 90.1% (threshold: P ≤ 0.170)
- **Propagated ruled_in precision:** 92.6% (sibling + P ≥ 0.75)
- **Coverage:** ~210 records per API call
- **Total classified:** ~35% of catalog (~1M records)

**Your job:** Verify these claims on the held-out test set.

---

## How to Run Things

### Check test set builder progress:
```bash
cd ml
tail -f /private/tmp/claude-501/-Users-benjamincaulfield-Documents-flic-a-disc/tasks/b65c78f.output
```

### Count current database records:
```bash
cd ml
uv run python manage.py count_available_records
```

### Run agent pipeline on full catalog:
```bash
cd cataloger
python agent_pipeline.py
# Runtime: ~3.5 minutes with mock API
```

### Start mock API server:
```bash
cd cataloger
uvicorn mock_api:app --reload --port 8001
# Access at http://localhost:8001
# Endpoints: /releases/{id}, /stats, /health
```

---

## Important Context

### Why test_set.json is deterministic:
- All wants/haves pre-fetched (no API calls during evaluation)
- Fixed set of records (same input every time)
- Ground truth is objective (wants > haves is binary)
- Same pipeline → same classifications → same score

### Why we need a separate test set:
- enriched_training.json was used to train the model (in-sample bias)
- "In-sample validation" showed 98%+ precision (too optimistic!)
- Need held-out data that the model has never seen
- Test set provides unbiased estimate of real-world performance

### The seller inventory "loophole":
- Fetching seller inventory pages gives listing.data for free
- listing.data includes wants/haves without extra API calls
- Only expensive call is suggested_price (not needed for test set)
- This is how we can build a large test set efficiently

---

## Deliverables for Evaluation

1. **test_set.json** - 10k balanced records with ground truth
2. **evaluate_pipeline.py** - Evaluation script with TestAPIClient
3. **TestAPIClient class** - API client backed by test set
4. **Evaluation report** - Metrics on test set vs. OOF claims

---

## Questions to Answer

1. Does the pipeline achieve 90%+ precision on both ruled_in and ruled_out?
2. Is the coverage really ~210 records/API call?
3. How does test set performance compare to OOF validation claims?
4. What's the precision/recall tradeoff at different thresholds?

---

## Notes

- The pipeline is already well-calibrated with conservative thresholds
- Main risk: distribution shift between enriched_training and test set
- Test set should have similar positive rate (~29%) for fair comparison
- Focus on deterministic, reproducible evaluation

---

## Contact

Original session context preserved at:
`/Users/benjamincaulfield/.claude/projects/-Users-benjamincaulfield-Documents-flic-a-disc/`

Good luck! 🎯
