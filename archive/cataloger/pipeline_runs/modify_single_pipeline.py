#!/usr/bin/env python3
"""
Script to modify a single pipeline file with:
1. Django boilerplate
2. Global api_client_global
3. API call logging and database saves
"""

import re
import sys

def modify_pipeline(pipeline_num, dry_run=False):
    """Modify a single pipeline file."""
    filepath = f"pipeline_{pipeline_num}.py"

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already modified
    if 'api_client_global' in content:
        print(f"Pipeline {pipeline_num} already modified, skipping")
        return

    # =========================================================================
    # STEP 1: Add boilerplate after imports
    # =========================================================================

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

# Global API call counter
api_call_counter = 0
'''

    # Find the last import statement
    import_matches = list(re.finditer(r'^(import|from)\s+.*$', content, re.MULTILINE))
    if not import_matches:
        print(f"ERROR: No imports found in pipeline_{pipeline_num}.py")
        return False

    last_import_pos = import_matches[-1].end()
    content = content[:last_import_pos] + boilerplate + content[last_import_pos:]

    # =========================================================================
    # STEP 2: Change classify_catalog signature
    # =========================================================================

    content = re.sub(
        r'def classify_catalog\(catalog,\s*api_client\)',
        'def classify_catalog(catalog)',
        content
    )

    # =========================================================================
    # STEP 3: Wrap API calls with logging and database saves
    # =========================================================================

    # Find all patterns like: result = api_client.get_release(int(rid))
    # and replace with wrapped version

    def wrap_api_call(match):
        indent = match.group(1)
        var_name = match.group(2)
        rid_var = match.group(3)

        # Build replacement code
        wrapped = f'''{indent}global api_call_counter
{indent}api_call_counter += 1
{indent}logger.info(f"API call {{api_call_counter}}: Querying release {{{rid_var}}}")
{indent}{var_name} = api_client_global.get_release({rid_var})
{indent}stats = ({var_name}.data.get('stats') or {{}}).get('community') or {{}}
{indent}wants = stats.get('in_wantlist', 0)
{indent}haves = stats.get('in_collection', 0)
{indent}{var_name}['wants'] = wants
{indent}{var_name}['haves'] = haves
{indent}logger.info(f"API call {{api_call_counter}}: Release {{{rid_var}}} - wants={{wants}}, haves={{haves}}")
{indent}try:
{indent}    Record.objects.get_or_create(
{indent}        release_id={rid_var},
{indent}        defaults={{
{indent}            'title': {var_name}.data.get('title', ''),
{indent}            'artist': ', '.join(a.get('name', '') for a in {var_name}.data.get('artists', [])),
{indent}            'year': {var_name}.data.get('year'),
{indent}            'genre': {var_name}.data.get('genres', []),
{indent}            'style': {var_name}.data.get('styles', []),
{indent}            'label': {var_name}.data.get('labels', [{{}}])[0].get('name', '') if {var_name}.data.get('labels') else '',
{indent}            'country': {var_name}.data.get('country', ''),
{indent}            'format': {var_name}.data.get('formats', [{{}}])[0].get('name', '') if {var_name}.data.get('formats') else '',
{indent}            'master_id': {var_name}.data.get('master_id'),
{indent}            'wants': wants,
{indent}            'haves': haves,
{indent}            'fetched_at': timezone.now(),
{indent}        }}
{indent}    )
{indent}except Exception as e:
{indent}    logger.warning(f"Failed to save record {{{rid_var}}} to database: {{e}}")'''
        return wrapped

    # Pattern: captures indentation, variable name, and release_id expression
    # Matches: result = api_client.get_release(int(rid))
    pattern = r'(\s+)(\w+)\s*=\s*api_client\.get_release\(([^)]+)\)'
    content = re.sub(pattern, wrap_api_call, content)

    # =========================================================================
    # STEP 4: Write modified content
    # =========================================================================

    if dry_run:
        print(f"DRY RUN - would modify pipeline_{pipeline_num}.py")
        # Show first few lines of modifications
        lines = content.split('\n')
        for i, line in enumerate(lines[:100]):
            if 'api_client_global' in line or 'logger.info' in line:
                print(f"  Line {i}: {line[:80]}")
    else:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Modified pipeline_{pipeline_num}.py")

    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python modify_single_pipeline.py <pipeline_number> [--dry-run]")
        sys.exit(1)

    pipeline_num = int(sys.argv[1])
    dry_run = '--dry-run' in sys.argv

    modify_pipeline(pipeline_num, dry_run=dry_run)
