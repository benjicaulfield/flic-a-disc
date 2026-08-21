# Pipeline Processing Summary

## Task Overview
Process and run 12 classification pipelines with:
1. Django/logging boilerplate
2. API call logging and database saves
3. 1000 API call limit per pipeline
4. Results tracking and CSV summary

## Progress

### ✅ Completed Steps

1. **Created backup files** - All pipeline_X.py.original files created
2. **Created data symlinks**:
   - `enriched_training.json` → normalized training data with proper field names
   - `lp_catalog_subset.json` → 395,000 record subset (from 3M full catalog)
   - `discogs_token.json` → Discogs API credentials

3. **Fixed data format issues**:
   - Training data: Added missing fields (country, release_id, catalog_number, master_id)
   - Normalized genre/style field names (genres→genre, styles→style)

4. **Modified pipelines**:
   - Added Django setup boilerplate to all 12 pipelines
   - Added logging configuration
   - Created `get_release_logged()` helper function for API calls with:
     - Automatic logging (before/after each call)
     - Wants/haves extraction from community stats
     - Database saves to Record model
   - Created runner scripts (run_pipeline_X.py) for all 12 pipelines

5. **Execution infrastructure**:
   - Main processing script: `process_pipelines.py`
   - Individual runners: `run_pipeline_1.py` through `run_pipeline_12.py`
   - Summary generation script (CSV output)

### ⚠️ Known Issues

1. **Function signature inconsistencies** - Some pipelines still have:
   - `def classify_catalog(catalog, api_client):` instead of `def classify_catalog(catalog):`
   - Type-annotated versions: `def classify_catalog(catalog: list, api_client) -> dict:`
   - Method versions: `def classify_catalog(self, catalog: List[dict], api_client) -> dict:`

2. **Fix script created but not fully executed**: `final_fix.py` - Run this to fix remaining signature issues

### 📊 Current Execution Status

**As of last check:**
- Pipeline 8: RUNNING
- Pipelines 1, 2, 4, 5, 6: FAILED (signature issues)
- Pipeline 3: COMPLETED (35MB result file)
- Pipelines 7, 9-12: Pending or in progress

### 🔧 Files Created

**Scripts:**
- `process_pipelines.py` - Main orchestrator
- `fix_all_pipelines.py` - Initial fix attempt
- `final_fix.py` - Comprehensive signature fix
- `run_pipeline_X.py` (×12) - Individual pipeline runners

**Data:**
- `lp_catalog_subset.json` - 395k record subset
- `enriched_training_normalized.json` - Fixed training data
- Symlinks for data access

**Results** (in progress):
- `results_X.json` - Pipeline outputs
- `pipeline_X.log` - Detailed execution logs
- `pipeline_summary.csv` - Summary table (to be generated)

## Next Steps to Complete

1. **Fix remaining signatures**:
   ```bash
   cd /Users/benjamincaulfield/Documents/flic-a-disc/cataloger/pipeline_runs
   python3 final_fix.py
   ```

2. **Manual signature verification**:
   ```bash
   for i in {1..12}; do
     echo "Pipeline $i:"
     grep "def classify_catalog" pipeline_$i.py | head -1
   done
   ```

3. **Re-run failed pipelines**:
   ```bash
   # After fixing signatures, restart the process
   rm -f results_*.json pipeline_*.log
   python3 process_pipelines.py 2>&1 | tee final_run.log
   ```

4. **Generate summary CSV**:
   The process_pipelines.py script automatically generates `pipeline_summary.csv` with columns:
   - pipeline (1-12)
   - status (completed/failed/not_run)
   - ruled_in (count)
   - ruled_out (count)
   - coverage (ratio)
   - api_calls (count)
   - runtime (seconds)

## File Locations

**Base directory**: `/Users/benjamincaulfield/Documents/flic-a-disc/cataloger/pipeline_runs/`

**Data sources**:
- Training: `../data/enriched_training.json`
- Catalog: `lp_catalog_subset.json` (395k subset)
- Tokens: `../../ml/discogs_token.json`

**Results**:
- Individual: `results_1.json` through `results_12.json`
- Summary: `pipeline_summary.csv`
- Logs: `pipeline_1.log` through `pipeline_12.log`

## Code Modifications Made

### Boilerplate Added (all pipelines)
```python
import logging, os, sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_X.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Django setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ml_path = os.path.join(project_root, 'ml')
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from bandit.utils.get_user_inventory import authenticate_client
from bandit.models import Record
from django.utils import timezone

api_client_global = authenticate_client()
api_call_counter = [0]

def get_release_logged(release_id):
    """Helper function with logging and DB save."""
    api_call_counter[0] += 1
    logger.info(f"API call {api_call_counter[0]}: Querying release {release_id}")

    result = api_client_global.get_release(release_id)

    stats = (result.data.get('stats') or {}).get('community') or {}
    wants = stats.get('in_wantlist', 0)
    haves = stats.get('in_collection', 0)

    logger.info(f"API call {api_call_counter[0]}: Release {release_id} - wants={wants}, haves={haves}")

    # Save to database
    Record.objects.get_or_create(
        release_id=release_id,
        defaults={...}  # Full record data
    )

    return {'wants': wants, 'haves': haves, 'data': result.data}
```

### Function Changes
- **Before**: `def classify_catalog(catalog, api_client):`
- **After**: `def classify_catalog(catalog):`
- **API calls**: `api_client.get_release(rid)` → `get_release_logged(rid)`

## Database Schema

Records are saved to the `Record` model with fields:
- release_id, title, artist, year
- genre, style, label, country, format
- master_id, wants, haves
- fetched_at (timestamp)

## Performance Notes

- Full catalog: 2,987,588 records
- Subset used: 395,000 records
- Training data: 60,000 records
- Each pipeline processes 395k records with ML model training
- Estimated runtime per pipeline: 5-15 minutes (depending on complexity)
- Total expected runtime: 1-3 hours for all 12 pipelines

## Troubleshooting

**Issue: "missing 1 required positional argument: 'api_client'"**
- Solution: Run `python3 final_fix.py` to fix function signatures

**Issue: "KeyError: 'genre'" or "'country'"**
- Solution: Training data already normalized with these fields added

**Issue: "Permission denied" errors**
- Solution: Use the provided Python scripts instead of shell loops

**Issue: Pipeline hangs during model training**
- Expected: LightGBM training on 395k records takes 30-60 seconds
- Each pipeline trains 1-5 models, so initial delay is normal

## Summary Output Format

`pipeline_summary.csv`:
```csv
pipeline,status,ruled_in,ruled_out,coverage,api_calls,runtime
1,completed,123456,234567,0.9065,987,245.3
2,completed,125678,232345,0.9053,993,267.8
...
```

## Contact & Next Steps

1. Run `final_fix.py` to ensure all signatures are correct
2. Execute `python3 process_pipelines.py` for complete run
3. Check `pipeline_summary.csv` for results
4. Review individual `results_X.json` files for detailed output
5. Examine `pipeline_X.log` files for API call logs and debugging
