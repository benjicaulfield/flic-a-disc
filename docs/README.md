The Discogs monthly data dump contains over 16 million release entries across all formats. Your task is to build a clean, enriched, and queryable LP catalog efficiently, and use that to surface record recommendations based on a provided collection of desirable records ("keepers") specific to that user. 

There are three steps in the task. Each step feeds into the next. Decisions cascade. 

Provided is a `requirements.txt` file with pinned dependencies.

**Setup (Security Note):** Direct `pip` commands are blocked. Use: `python -m ensurepip --default-pip && python -m
  pip install -r requirements.txt`

DATA:
- `releases.xml.gz`: A shard consisting of 2.35 million releases (roughly 1/8 of the full monthly release dump), containing format, artist, title, label, genre, style, country, year, and master_id of every release. 
- `masters.xml.gz`: A shard consisting of 313,000 masters (also 1/8 of full dump), containing canonical year, genre, and main_release_id for each master. 
- `keepers.json`: A dictionary of records from a personal collection pre-labeled as keepers. Each row includes artist, title, discogs_id, format, label, wants, haves, genres, styles, and year. 
- `mock_api_data.json`: A dictionary of enriched records pulled from the Discogs API in a previous project.

STEP 1: BUILD A CLEAN LP CATALOG
- Stream-parse `releases.xml.gz` and extract all LP releases into a clean, flat JSON. The file is too large to load into memory, you must use a streaming XML parser. The output should be one object per release with the following fields: release_id, master_id, artist, title, label, catalog_number, year, country, genre, style.
- You'll need to handle inconsistent format fields, multi-value style and genre fields, and messy user-submitted data.


OUTPUT: 
`lp_catalog.json`: clean catalog
`step1_decisions.md`: decision log for all normalization choices

STEP 2: Enrich with community data. 
The Discogs API exposes community wants, haves, and suggested price data. Your task is an initial pass to enrich lp_catalog.csv as efficiently as possible with the following constraints: you may make at most 60 API requests per minute, you must not make redundant requests, and we are only looking for records whose community wants are greater than haves (a very rough indication of desirability). Records that fail the wants > haves test can be discarded.
- You do not need API authorization for this project, use this instead:
```
import discogs_client
d = discogs_client.Client("DiscogsDump/1.0")
```

- You cannot enrich every release, so you will need to design a prioritization strategy for which releases to look up first to maximize information gain. Document this reasoning.
- Maintain two bloom filters: one for releases that passed the wants/haves threshold, and one for blocked releases: failed the threshold. When a release is looked up and its master evaluated, propagate all sibling release_ids into the appropriate filter. Before making any API call, check both filters.

OUTPUT: 
`lp_catalog_enriched.json`: enriched catalog. Use this schema:
"type": "object",
  "properties": {
    "discogs_id": { "type": "string" },
    "artist": { "type": "string" },
    "title": { "type": "string" },
    "format": { "type": "array", "items": { "type": "string" } },
    "label": { "type": "string" },
    "catno": { "type": ["string", "null"] },
    "wants": { "type": "integer" },
    "haves": { "type": "integer" },
    "added": { "type": "string", "format": "date-time" },
    "genres": { "type": "array", "items": { "type": "string" } },
    "styles": { "type": "array", "items": { "type": "string" } },
    "year": { "type": ["integer", "null"] },
    "record_image": { "type": ["string", "null"], "format": "uri" },
    "description": { "type": ["string", "null"] },
    "wanted": { "type": "boolean" },
    "evaluated": { "type": "boolean" },
    "status": { "type": "string" }
  },

Status will be either "enriched", "inferred_cleared", "blocked", "inferred_blocked". 

STEP 3: VALIDATION
Run your enrichment pipeline against the provided mock API server to produce a deterministic, verifiable output. 
- Start the mock API: python mock_api.py --data mock_api_data.json --port 5000
- Point your enrichment script at http://localhost:5000/releases/{id} instead of the real Discogs API.
- Run your full pipeline against the mock dataset and produce `lp_catalog_verified.json`.
- Compute and submit a SHA-256 checksum of lp_catalog_verified.json.

The mock dataset contains a fixed set of releases with known wants/haves values — some cleared, some blocked — with several masters represented by multiple releases to verify bloom filter sibling propagation in both directions.
Your output will be checked against a reference checksum. A mismatch indicates a bug in your bloom filter logic, provenance tracking, or pipeline determinism.
OUTPUT:
`lp_catalog_verified.json` — enriched catalog produced against mock API
`checksum.txt` — SHA-256 of lp_catalog_verified.json