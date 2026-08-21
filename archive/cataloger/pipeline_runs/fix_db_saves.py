#!/usr/bin/env python3
"""Fix database save field names in all pipelines"""

import re
import glob

def fix_pipeline(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix: release_id → discogs_id
    content = content.replace(
        'release_id=release_id,',
        'discogs_id=str(release_id),'
    )

    # Fix: fetched_at → added
    content = content.replace(
        "'fetched_at': timezone.now(),",
        "'added': timezone.now(),"
    )

    # Fix: genre/style should be plural (genres/styles)
    content = re.sub(
        r"'genre': result\.data\.get\('genres', \[\]\),",
        "'genres': result.data.get('genres', []),",
        content
    )
    content = re.sub(
        r"'style': result\.data\.get\('styles', \[\]\),",
        "'styles': result.data.get('styles', []),",
        content
    )

    # Also fix field name in catalog access if present
    content = content.replace(
        "catalog[bi].get('catalog_number', '')",
        "catalog[bi].get('catno', '')"
    )

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"✓ Fixed {filepath}")

# Fix all pipelines
for i in range(1, 13):
    filepath = f'/Users/benjamincaulfield/Documents/flic-a-disc/cataloger/pipeline_runs/pipeline_{i}.py'
    fix_pipeline(filepath)

print("\n✓ All pipelines fixed!")
