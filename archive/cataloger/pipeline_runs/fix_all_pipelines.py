#!/usr/bin/env python3
"""Fix all pipeline signatures and API calls using regex."""

import re
from pathlib import Path

PIPELINE_DIR = Path("/Users/benjamincaulfield/Documents/flic-a-disc/cataloger/pipeline_runs")

def fix_pipeline(pipeline_num):
    """Fix a single pipeline."""
    original = PIPELINE_DIR / f"pipeline_{pipeline_num}.py"
    backup = PIPELINE_DIR / f"pipeline_{pipeline_num}.py.original"

    # Read current (may already be partially modified)
    with open(original, 'r') as f:
        content = f.read()

    # Check if already has boilerplate
    has_boilerplate = 'api_client_global' in content

    if not has_boilerplate:
        print(f"Pipeline {pipeline_num}: Adding boilerplate...")
        # Read from backup if exists, else from current
        source_file = backup if backup.exists() else original
        with open(source_file, 'r') as f:
            lines = f.readlines()

        # Find last import
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                last_import_idx = i

        boilerplate = f'''
import logging
import os
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_{pipeline_num}.log'),
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

# Initialize authenticated Discogs client
logger.info("Initializing authenticated Discogs client...")
api_client_global = authenticate_client()
logger.info("Client authenticated successfully")

# API call counter and helper
api_call_counter = [0]  # Use list for mutable global

def get_release_logged(release_id):
    """Helper function to get release with logging and DB save."""
    api_call_counter[0] += 1
    logger.info(f"API call {{api_call_counter[0]}}: Querying release {{release_id}}")

    result = api_client_global.get_release(release_id)

    # Extract wants/haves from community stats
    stats = (result.data.get('stats') or {{}}).get('community') or {{}}
    wants = stats.get('in_wantlist', 0)
    haves = stats.get('in_collection', 0)

    logger.info(f"API call {{api_call_counter[0]}}: Release {{release_id}} - wants={{wants}}, haves={{haves}}")

    # Create a dict-like result for backward compatibility
    result_dict = {{
        'wants': wants,
        'haves': haves,
        'data': result.data,
    }}

    # Save to database
    try:
        Record.objects.get_or_create(
            release_id=release_id,
            defaults={{
                'title': result.data.get('title', ''),
                'artist': ', '.join(a.get('name', '') for a in result.data.get('artists', [])),
                'year': result.data.get('year'),
                'genre': result.data.get('genres', []),
                'style': result.data.get('styles', []),
                'label': result.data.get('labels', [{{}}])[0].get('name', '') if result.data.get('labels') else '',
                'country': result.data.get('country', ''),
                'format': result.data.get('formats', [{{}}])[0].get('name', '') if result.data.get('formats') else '',
                'master_id': result.data.get('master_id'),
                'wants': wants,
                'haves': haves,
                'fetched_at': timezone.now(),
            }}
        )
    except Exception as e:
        logger.warning(f"Failed to save record {{release_id}} to database: {{e}}")

    return result_dict

'''

        new_lines = lines[:last_import_idx+1] + [boilerplate] + lines[last_import_idx+1:]
        content = ''.join(new_lines)

    # Fix function signatures using regex
    # Pattern 1: def classify_catalog(self, catalog: ..., api_client) -> dict:
    # Pattern 2: def classify_catalog(catalog: ..., api_client) -> dict:
    # Pattern 3: def classify_catalog(catalog, api_client):

    # Remove api_client parameter from any classify_catalog signature
    content = re.sub(
        r'def classify_catalog\((self, )?catalog[^,)]*,\s*api_client\)(\s*->\s*dict)?:',
        r'def classify_catalog(\1catalog):',
        content
    )

    # Replace api_client.get_release calls
    content = content.replace('api_client.get_release(', 'get_release_logged(')

    # Write back
    with open(original, 'w') as f:
        f.write(content)

    print(f"Pipeline {pipeline_num}: Fixed!")


if __name__ == '__main__':
    for i in range(1, 13):
        fix_pipeline(i)
    print("\nAll pipelines fixed!")
