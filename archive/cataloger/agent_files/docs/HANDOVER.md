# Cataloger Mock API - Handover Document

**Date**: 2026-03-30
**Context**: Building mock API for Discogs record filtering challenge

---

## Project Overview

This is a **data science challenge project** where participants build a pipeline to classify 3M vinyl LP records into confident positives (wants > haves), confident negatives (wants ≤ haves), and uncertain records requiring API verification.

**Challenge constraints:**
- 3M LP catalog (metadata only: artist, title, genre, year, label, etc.)
- API rate limit: 60 calls/minute
- API budget: Maximum 5,000 verification calls
- Training data: 60k enriched records with wants/haves values
- Target: ≥95% precision on confident classifications, ≥80% recall overall

**True base rate**: ~29.1% of records have wants>haves (measured to convergence on 60k sample)
**Estimated total positives**: ~873k in full 3M catalog

---

## Current Status

### ✅ Completed

1. **Mock API scaffold** (`./cataloger/`)
   - FastAPI server that mimics Discogs API
   - Endpoints: `/releases/{id}`, `/stats`, `/health`
   - No rate limiting (for testing)
   - Auto-generated docs at `/docs`

2. **Export script** (`export_enriched_data.py`)
   - Extracts enriched records from Django database in `../ml/`
   - Exports as JSON array for mock API to consume
   - Shows dataset statistics

3. **Training data collection** (ongoing in background)
   - Running `estimate_base_rate.py` to collect 60k random samples
   - Takes ~8 hours at 120 calls/min
   - Creates balanced dataset at natural 28.5% positive rate

4. **Frontend build fixes**
   - Fixed all TypeScript import errors
   - App builds successfully

### 🚧 In Progress / Next Steps

**Immediate**: Get mock API running
1. Export enriched data: `python export_enriched_data.py`
2. Start server: `uvicorn mock_api:app --reload --port 8001`
3. Test endpoints
4. Verify data loads correctly

**Next**: Build participant tooling
1. Example baseline pipeline (load data, extract features, train model, classify catalog)
2. Evaluation script (sample ruled_in/ruled_out, query verify list, calculate metrics)
3. Documentation cleanup
4. Optional: Simple API client wrapper for local testing

---

## File Structure

```
cataloger/
├── mock_api.py              # FastAPI server (mimics Discogs API)
├── export_enriched_data.py  # Export enriched records from Django DB
├── requirements.txt         # FastAPI, uvicorn, pydantic
├── README.md               # Setup and usage instructions
├── .gitignore              # Ignore data/ and cache
└── data/                   # Created by export script
    └── enriched_training.json  # Enriched records for mock API

ml/
├── discogs/
│   └── management/commands/
│       ├── estimate_base_rate.py      # Collect random samples (currently running)
│       └── rough_model_precision_estimate.py  # Baseline model experiment
├── generate_training_csv.py  # Generate balanced training CSV
└── discogs/data/
    ├── lp_catalog_filtered_thrice.json  # 3M LP catalog (metadata only)
    └── training_data.csv                # Current balanced dataset (6.3k records)
```

---

## Key Technical Details

### Mock API Behavior

**Endpoint**: `GET /releases/{release_id}`

**Response format** (mimics Discogs API):
```json
{
  "id": 123456,
  "title": "Kind of Blue",
  "artist": "Miles Davis",
  "year": 1959,
  "community": {
    "want": 15234,
    "have": 12456
  }
}
```

**404 if release not found** in enriched dataset

### Data Format

Enriched records have:
- `discogs_id` - Unique release ID
- `artist`, `title`, `label` - Metadata
- `year` - Release year (optional)
- `genres`, `styles` - Arrays of genre/style strings
- `wants`, `haves` - Community statistics (TARGET DATA)

### Database Schema (Django)

**Model**: `Record` in `bandit/models.py`

**Key fields**:
- `api_enriched=True` - Has wants/haves from API
- `wanted=False` - Unbiased random sample (vs keeper-biased collection)
- `wants`, `haves` - Community stats

**Query for training data**:
```python
Record.objects.filter(api_enriched=True, wanted=False)
```

### Challenge Data Files (for participants)

1. **`lp_catalog.json`** - 3M records, metadata only (no wants/haves)
2. **`enriched_training.json`** - 60k records WITH wants/haves (use however you want)

---

## Commands Reference

### Mock API

```bash
cd cataloger

# Export enriched data from database
python export_enriched_data.py
# Creates: data/enriched_training.json

# Start mock API server
uvicorn mock_api:app --reload --port 8001

# Test endpoints
curl http://localhost:8001/health
curl http://localhost:8001/stats
curl http://localhost:8001/releases/123456  # Use real ID from your data

# Visit interactive docs
open http://localhost:8001/docs
```

### Data Collection (ml/)

```bash
cd ml

# Collect random samples (currently running in background)
python manage.py estimate_base_rate --sample-size 60000 --resume

# Generate balanced training CSV
python generate_training_csv.py
# Creates: discogs/data/training_data.csv

# Run baseline model experiment
python manage.py rough_model_precision_estimate
# Uses records with api_enriched=True, wanted=False
```

### Frontend

```bash
cd frontend

# Build (all import errors fixed)
npm run build

# Dev server
npm run dev
```

---

## Important Context & Decisions

### Why 29.1% positive rate?

- True base rate is ~28.5% (established via random sampling)
- Training data should match this distribution
- Previously had 50k keeper-biased positives (not representative)
- Now using unbiased random samples only

### Why exclude suggested_price?

- Requires authenticated Discogs API access
- Challenge participants have unauthenticated access only
- Must predict wants>haves using metadata only (artist, genre, year, label)

### Why 60k training records?

- Baseline model has ~17 features
- Need ~1000 examples per feature minimum
- 60k gives good statistical power
- Takes ~8 hours to collect at 120 calls/min (removed price fetching for 2x speedup)

### Baseline Performance

Simple gradient boosting with genre/year features:
- **8.3% recall at 80%+ precision**
- This is INTENTIONALLY low - participants use sophisticated techniques to improve:
  - Embeddings (artist/genre/label representations)
  - Bloom filters (sibling propagation via master_id)
  - MinHash + LSH (near-duplicate detection)
  - Uncertainty-based active learning

---

## Known Issues & Gotchas

1. **Path confusion**: Some scripts assume running from `ml/`, others from project root
   - `estimate_base_rate.py`: Fixed to use `discogs/data/...` (relative to ml/)

2. **Virtual environment**: Use ml/.venv with uv
   ```bash
   cd ml
   source .venv/bin/activate
   ```

3. **Django settings**: `export_enriched_data.py` imports Django from `../ml/`
   - Adds ML_DIR to sys.path
   - Sets `DJANGO_SETTINGS_MODULE=config.settings`

4. **Data leakage prevention**: Never use wants/haves as features!
   - Features must be metadata only (what's in lp_catalog.json)
   - Early versions accidentally included wants/haves in features

5. **Rate limiting**:
   - Real Discogs API: 60/min
   - estimate_base_rate.py: Now 120/min (removed price fetching, sleep(0.5))
   - Mock API: NO rate limiting

---

## Next Immediate Tasks

1. **Test mock API thoroughly**
   - Export data
   - Start server
   - Query several releases
   - Verify stats endpoint shows correct counts
   - Check /docs for API documentation

2. **Create simple API client for participants**
   - Thin wrapper around requests
   - Handle rate limiting (60/min)
   - Simple interface: `client.get_release(id) -> {wants, haves}`
   - Both mock and real Discogs API

3. **Build example baseline pipeline**
   - Load lp_catalog.json + enriched_training.json
   - Extract features (genre, year indicators)
   - Train logistic regression or gradient boosting
   - Predict on full catalog
   - Output high-confidence predictions
   - Show bloom filter usage for siblings

4. **Create evaluation script**
   - Accept participant output (list of release_ids)
   - Query mock API for those IDs
   - Calculate precision, recall, F1
   - Validate constraints (≤10% API calls, ≥90% precision)
   - Generate report

5. **Polish documentation**
   - Clean up challenge spec (api_optimization.txt)
   - Write participant README
   - Document data formats
   - Explain evaluation process

---

## Questions to Resolve

1. Should mock API support batch queries? (`POST /releases` with array of IDs)
2. How to handle master_id sibling propagation in evaluation?
3. What's the submission format? (JSON with release_ids + predicted probs + method used?)
4. Should we provide embeddings, or make participants generate them?
5. Test dataset - holdout from training, or separate collection?

---

## Reference Links

- **Project root**: `/Users/benjamincaulfield/Documents/flic-a-disc/`
- **Mock API**: `./cataloger/`
- **ML/Django**: `./ml/`
- **Frontend**: `./frontend/`

---

## Contact Context

Previous conversation covered:
- Fixing frontend import errors (ragStatus, type imports)
- Optimizing data collection (removed suggested_price for 2x speedup)
- Designing balanced training dataset (29.1% positive rate)
- Building mock API infrastructure
- Understanding baseline performance is intentionally low (8.3% recall)

**Key insight**: Challenge participants need sophisticated techniques (embeddings, bloom filters, etc.) to improve beyond the simple baseline. The low baseline demonstrates the problem is hard but solvable.
