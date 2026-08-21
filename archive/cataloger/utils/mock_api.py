"""
Mock Discogs API Server

Mimics the Discogs API for testing participant pipelines without rate limits.
Loads enriched training data and serves wants/haves for each release.

Usage:
    uvicorn mock_api:app --reload --port 8001

Endpoints:
    GET /releases/{release_id}  - Get release community data
    GET /health                 - Health check
    GET /stats                  - Dataset statistics
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path
from typing import Dict, Optional

app = FastAPI(
    title="Mock Discogs API",
    description="Mock API for testing record filtering pipelines",
    version="1.0.0"
)

# CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory database of enriched records
RELEASES: Dict[int, dict] = {}


class CommunityStats(BaseModel):
    """Discogs community statistics for a release"""
    want: int
    have: int


class Release(BaseModel):
    """Simplified Discogs release response"""
    id: int
    title: str
    artist: str
    year: Optional[int] = None
    community: CommunityStats


class Stats(BaseModel):
    """Mock API statistics"""
    total_releases: int
    positive_releases: int
    negative_releases: int
    positive_rate: float


def load_enriched_data(filepath: str = "data/enriched_training.json"):
    """
    Load enriched training data into memory.

    Expected format: JSON lines or array of objects with:
    - discogs_id (or release_id)
    - artist
    - title
    - year (optional)
    - wants
    - haves
    """
    global RELEASES

    data_path = Path(filepath)
    if not data_path.exists():
        print(f"⚠️  Warning: {filepath} not found. Mock API will be empty.")
        print("   Generate it with: cd ../ml && python generate_training_csv.py")
        return

    print(f"📂 Loading enriched data from {filepath}...")

    # Try loading as JSON lines first
    records = []
    try:
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except json.JSONDecodeError:
        # Try loading as single JSON array
        with open(data_path) as f:
            records = json.load(f)

    # Index by release_id
    for record in records:
        release_id = record.get('discogs_id') or record.get('release_id')
        if not release_id:
            continue

        RELEASES[int(release_id)] = {
            'id': int(release_id),
            'title': record.get('title', 'Unknown'),
            'artist': record.get('artist', 'Unknown'),
            'year': record.get('year'),
            'wants': record.get('wants', 0),
            'haves': record.get('haves', 0),
        }

    positive = sum(1 for r in RELEASES.values() if r['wants'] > r['haves'])
    print(f"✓ Loaded {len(RELEASES):,} releases")
    print(f"  - Positive (wants>haves): {positive:,} ({positive/len(RELEASES)*100:.1f}%)")
    print(f"  - Negative: {len(RELEASES)-positive:,}")


@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_enriched_data()


@app.get("/")
def root():
    """API root"""
    return {
        "service": "Mock Discogs API",
        "version": "1.0.0",
        "endpoints": {
            "release": "/releases/{release_id}",
            "health": "/health",
            "stats": "/stats"
        }
    }


@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "releases_loaded": len(RELEASES)
    }


@app.get("/stats", response_model=Stats)
def get_stats():
    """Get dataset statistics"""
    positive = sum(1 for r in RELEASES.values() if r['wants'] > r['haves'])

    return Stats(
        total_releases=len(RELEASES),
        positive_releases=positive,
        negative_releases=len(RELEASES) - positive,
        positive_rate=positive / len(RELEASES) if RELEASES else 0.0
    )


@app.get("/releases/{release_id}", response_model=Release)
def get_release(release_id: int):
    """
    Get release data (mimics Discogs API format)

    Returns:
        Release object with community stats (wants/haves)

    Raises:
        404 if release not found in enriched dataset
    """
    if release_id not in RELEASES:
        raise HTTPException(
            status_code=404,
            detail=f"Release {release_id} not found in enriched dataset"
        )

    record = RELEASES[release_id]

    return Release(
        id=record['id'],
        title=record['title'],
        artist=record['artist'],
        year=record['year'],
        community=CommunityStats(
            want=record['wants'],
            have=record['haves']
        )
    )


@app.post("/reload")
def reload_data():
    """Reload enriched data (useful during development)"""
    RELEASES.clear()
    load_enriched_data()
    return {
        "status": "reloaded",
        "releases_loaded": len(RELEASES)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
