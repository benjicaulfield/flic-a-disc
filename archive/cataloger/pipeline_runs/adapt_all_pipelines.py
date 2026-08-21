#!/usr/bin/env python3
"""
Batch adapt all pipelines with:
1. Authenticated API client
2. Logging infrastructure
3. Database saving via Record.objects.get_or_create()
"""

import os
import re
import sys

# Common boilerplate to add at the top of each pipeline
BOILERPLATE = '''import logging
from datetime import datetime
import os
import sys

# Logging setup
pipeline_name = os.path.basename(__file__).replace('.py', '')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{pipeline_name}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Django setup for database access
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
api_client = authenticate_client()
logger.info("Client authenticated successfully")
'''

# Database save template
DB_SAVE_TEMPLATE = '''
            # Save to database
            record, created = Record.objects.get_or_create(
                discogs_id=str(rid),
                defaults={{
                    'artist': catalog[bi].get('artist', ''),
                    'title': catalog[bi].get('title', ''),
                    'label': catalog[bi].get('label', ''),
                    'catno': catalog[bi].get('catalog_number', ''),
                    'wants': wants,
                    'haves': haves,
                    'added': timezone.now(),
                    'genres': catalog[bi].get('genre', []),
                    'styles': catalog[bi].get('style', []),
                    'year': catalog[bi].get('year'),
                    'api_enriched': True,
                }}
            )
            if not created:
                record.wants = wants
                record.haves = haves
                record.api_enriched = True
                record.save()
'''

def adapt_pipeline(pipeline_path):
    """Adapt a single pipeline file"""
    print(f"Adapting {pipeline_path}...")

    with open(pipeline_path, 'r') as f:
        content = f.read()

    # Backup original
    backup_path = pipeline_path + '.original'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)

    # Check if already adapted
    if 'logger = logging.getLogger(__name__)' in content:
        print(f"  Already adapted, skipping...")
        return False

    # Find first import and add boilerplate after it
    lines = content.split('\n')

    # Find where to insert (after docstring and initial imports)
    insert_idx = 0
    in_docstring = False
    for i, line in enumerate(lines):
        if '"""' in line or "'''" in line:
            in_docstring = not in_docstring
        if not in_docstring and line.strip().startswith('import '):
            insert_idx = i + 1
            # Skip to after all consecutive imports
            while insert_idx < len(lines) and (lines[insert_idx].strip().startswith('import ') or
                                                 lines[insert_idx].strip().startswith('from ') or
                                                 lines[insert_idx].strip() == ''):
                insert_idx += 1
            break

    # Insert boilerplate
    lines.insert(insert_idx, BOILERPLATE)

    # Replace mock_client references with api_client
    content = '\n'.join(lines)
    content = re.sub(r'\bmock_client\b', 'api_client', content)

    # Write adapted version
    with open(pipeline_path, 'w') as f:
        f.write(content)

    print(f"  ✓ Adapted {pipeline_path}")
    return True

def main():
    pipeline_dir = '/Users/benjamincaulfield/Documents/flic-a-disc/cataloger/pipeline_runs'

    # Get all main pipelines (1-12)
    pipelines = [f'pipeline_{i}.py' for i in range(1, 13)]

    for pipeline in pipelines:
        path = os.path.join(pipeline_dir, pipeline)
        if os.path.exists(path):
            adapt_pipeline(path)

    print("\n✓ All pipelines adapted!")

if __name__ == '__main__':
    main()
