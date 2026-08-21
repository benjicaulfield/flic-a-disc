#!/usr/bin/env python3
"""Final comprehensive fix for all pipelines."""

import re
from pathlib import Path

def fix_pipeline_signature(pipeline_num):
    """Fix classify_catalog signature for a pipeline."""
    filepath = Path(f"pipeline_{pipeline_num}.py")

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to match various classify_catalog signatures
    # Handles: (catalog, api_client), (self, catalog, api_client), (catalog: type, api_client), etc.
    patterns_to_fix = [
        (r'def classify_catalog\(catalog[^,)]*,\s*api_client\)(\s*->\s*[^:]+)?:', 'def classify_catalog(catalog):'),
        (r'def classify_catalog\(self,\s*catalog[^,)]*,\s*api_client\)(\s*->\s*[^:]+)?:', 'def classify_catalog(self, catalog):'),
    ]

    modified = False
    for pattern, replacement in patterns_to_fix:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True

    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Pipeline {pipeline_num}: FIXED signature")
    else:
        print(f"Pipeline {pipeline_num}: Already correct")

if __name__ == '__main__':
    for i in range(1, 13):
        fix_pipeline_signature(i)
