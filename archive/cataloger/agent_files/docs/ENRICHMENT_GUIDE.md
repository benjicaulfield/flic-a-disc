# Catalog Enrichment Guide

This guide explains how to enrich the lp_catalog.json with wants/haves data by crawling seller inventories.

## The Hack

When you fetch seller inventories via Discogs API, **wants/haves data comes for free**:
- 1 API call to `inventory.page()` returns up to 250 listings
- Each listing includes `wants` and `haves` in `listing.data['release']['stats']['community']`
- **No additional API calls needed** to get community stats

This means:
- **250 records per API call** instead of 1 record per call
- Can enrich entire 3M catalog efficiently by crawling seller inventories

## Quick Start

```bash
cd cataloger

# Install dependencies
pip install python3-discogs-client python-decouple

# Set up Discogs API credentials in .env file
echo "DISCOGS_CONSUMER_KEY=your_key" >> .env
echo "DISCOGS_CONSUMER_SECRET=your_secret" >> .env

# Run enrichment (will prompt for OAuth first time)
python enrich_catalog_from_inventories.py \
    --catalog lp_catalog.json \
    --sellers ../ml/sellers.json \
    --output lp_catalog_enriched.json \
    --max-sellers 50
```

## What It Does

1. **Loads catalog**: Reads lp_catalog.json (3M records)
2. **Loads sellers**: Reads sellers.json (list of Discogs sellers)
3. **Crawls inventories**: For each seller:
   - Fetches up to 100 pages (250 records per page = 25k max per seller)
   - Extracts wants/haves from listing data (free!)
   - Enriches matching catalog records
4. **Saves results**:
   - `lp_catalog_enriched.json` - Records with wants/haves
   - `lp_catalog_enriched_unenriched.json` - Records still missing wants/haves
   - `lp_catalog_enriched_metadata.json` - Stats and coverage info

## Usage Examples

**Test with 10 sellers:**
```bash
python enrich_catalog_from_inventories.py --max-sellers 10
```

**Full enrichment (all sellers):**
```bash
python enrich_catalog_from_inventories.py
```

**Custom paths:**
```bash
python enrich_catalog_from_inventories.py \
    --catalog /path/to/catalog.json \
    --sellers /path/to/sellers.json \
    --output enriched_catalog.json
```

**Limit pages per seller (faster but less coverage):**
```bash
python enrich_catalog_from_inventories.py --max-pages 50
```

## Progress Tracking

The script:
- Saves progress every 10 sellers to `{output}.progress`
- Prints coverage stats after each seller
- Shows records per API call (should be ~200-250)

**Example output:**
```
SELLER: SleepingNekoRecords
============================================================
Total pages: 100 (capped at 100)
  Page 1/100... +152 enriched, 98 skipped, 0 not in catalog
  Page 2/100... +143 enriched, 107 skipped, 0 not in catalog
  ...
✓ Seller complete: 12,458 records enriched

============================================================
PROGRESS: 1/50 sellers crawled
============================================================
  Coverage:      12,458 / 2,987,588 (0.4%)
  API calls:     100
  Records/call:  124.6
  Elapsed:       2.3 minutes
  ✓ Progress saved to lp_catalog_enriched.json.progress
```

## Expected Performance

**With 50 sellers:**
- API calls: ~5,000 (100 pages × 50 sellers)
- Records enriched: ~1,000,000 - 1,250,000
- Coverage: 33-42% of catalog
- Time: ~3-4 hours (with rate limiting)

**With 100 sellers:**
- API calls: ~10,000
- Records enriched: ~2,000,000 - 2,500,000
- Coverage: 67-84% of catalog
- Time: ~6-8 hours

**With 200 sellers:**
- API calls: ~20,000
- Records enriched: ~2,700,000+
- Coverage: 90%+ of catalog
- Time: ~12-16 hours

## After Enrichment

Once you have `lp_catalog_enriched.json`, you can:

1. **Use for evaluation**:
   - Replace lp_catalog.json in eval/data/
   - Now the agentic grader can test on records with ground truth wants/haves
   - No internet needed during evaluation

2. **Analyze coverage**:
   - Check `{output}_metadata.json` for coverage stats
   - See which records are still unenriched
   - Run additional crawls to fill gaps

3. **Test pipeline locally**:
   - Run agent_pipeline.py on enriched catalog
   - Compare predictions to ground truth
   - Measure precision/recall without API calls

## Troubleshooting

**"No module named 'discogs_client'":**
```bash
pip install python3-discogs-client
```

**"Config 'DISCOGS_CONSUMER_KEY' not found":**
- Create a `.env` file in cataloger/ directory
- Add your Discogs API credentials

**"Rate limit exceeded":**
- Discogs allows 60 requests per minute
- Script may pause to respect limits
- Consider running overnight for large crawls

**Low records per API call (<100):**
- This means many records in inventory are not in lp_catalog
- Normal if sellers have rare/niche inventory
- Overall average should still be 150-250 records/call

## Next Steps

After enrichment, update the evaluation to use the enriched catalog:

1. Copy enriched catalog to eval/data:
   ```bash
   cp lp_catalog_enriched.json eval/data/lp_catalog.json
   ```

2. Update evaluate.py to use enriched catalog for both tests

3. Run evaluation with full ground truth!
