"""
Master script to process all 12 classification pipelines.

This script:
1. Backs up each pipeline_X.py to pipeline_X.py.original
2. Adds Django setup boilerplate
3. Modifies classify_catalog function
4. Wraps API calls with logging and database saves
5. Creates runner scripts
6. Runs each pipeline with 1000 API call limit
7. Generates summary CSV
"""

import os
import re
import shutil
import subprocess
import json
import time
import csv
from pathlib import Path

PIPELINE_DIR = Path("/Users/benjamincaulfield/Documents/flic-a-disc/cataloger/pipeline_runs")
DATA_DIR = Path("/Users/benjamincaulfield/Documents/flic-a-disc/cataloger/data")
ML_DIR = Path("/Users/benjamincaulfield/Documents/flic-a-disc/ml")

BOILERPLATE = '''import logging
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

# API call counter
api_call_counter = 0

'''


def create_symlinks():
    """Create necessary symlinks in pipeline_runs directory."""
    os.chdir(PIPELINE_DIR)

    symlinks = [
        ('enriched_training.json', DATA_DIR / 'enriched_training.json'),
        ('lp_catalog.json', DATA_DIR / 'lp_catalog.json'),
        ('discogs_token.json', ML_DIR / 'discogs_token.json'),
    ]

    for link_name, target in symlinks:
        link_path = PIPELINE_DIR / link_name
        if link_path.exists() or link_path.is_symlink():
            print(f"Symlink {link_name} already exists, skipping")
            continue
        try:
            os.symlink(target, link_path)
            print(f"Created symlink: {link_name} -> {target}")
        except Exception as e:
            print(f"Error creating symlink {link_name}: {e}")


def backup_pipeline(pipeline_num):
    """Backup original pipeline file."""
    original = PIPELINE_DIR / f"pipeline_{pipeline_num}.py"
    backup = PIPELINE_DIR / f"pipeline_{pipeline_num}.py.original"

    if backup.exists():
        print(f"Backup for pipeline {pipeline_num} already exists")
        return False

    shutil.copy2(original, backup)
    print(f"Backed up pipeline_{pipeline_num}.py")
    return True


def modify_pipeline(pipeline_num):
    """Modify pipeline file with boilerplate and API call wrapping."""
    filepath = PIPELINE_DIR / f"pipeline_{pipeline_num}.py"

    with open(filepath, 'r') as f:
        content = f.read()

    # Find the last import statement
    import_matches = list(re.finditer(r'^(import|from)\s+.*$', content, re.MULTILINE))
    if not import_matches:
        print(f"ERROR: No imports found in pipeline_{pipeline_num}.py")
        return False

    last_import_pos = import_matches[-1].end()

    # Insert boilerplate after imports
    boilerplate = BOILERPLATE.format(pipeline_num=pipeline_num)
    modified_content = content[:last_import_pos] + '\n\n' + boilerplate + content[last_import_pos:]

    # Change classify_catalog signature
    modified_content = re.sub(
        r'def classify_catalog\(catalog,\s*api_client\)',
        'def classify_catalog(catalog)',
        modified_content
    )

    # Wrap API calls with logging and database saves
    # Pattern 1: result = api_client.get_release(...)
    # Pattern 2: result = d.get_release(...)

    def wrap_api_call(match):
        indent = match.group(1)
        var_name = match.group(2)
        api_var = match.group(3)
        rid_expr = match.group(4)

        wrapped = f'''{indent}global api_call_counter
{indent}api_call_counter += 1
{indent}logger.info(f"API call {{api_call_counter}}: Querying release {{{rid_expr}}}")
{indent}{var_name} = api_client_global.get_release({rid_expr})
{indent}stats = ({var_name}.data.get('stats') or {{}}).get('community') or {{}}
{indent}wants = stats.get('in_wantlist', 0)
{indent}haves = stats.get('in_collection', 0)
{indent}logger.info(f"API call {{api_call_counter}}: Release {{{rid_expr}}} - wants={{wants}}, haves={{haves}}")
{indent}try:
{indent}    Record.objects.get_or_create(
{indent}        release_id={rid_expr},
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
{indent}    logger.warning(f"Failed to save record {{{rid_expr}}} to database: {{e}}")'''
        return wrapped

    # Match various patterns of API calls
    patterns = [
        (r'(\s+)(\w+)\s*=\s*api_client\.get_release\(([^)]+)\)', wrap_api_call),
        (r'(\s+)(\w+)\s*=\s*d\.get_release\(([^)]+)\)', wrap_api_call),
    ]

    for pattern, replacer in patterns:
        # Custom replacement to handle the pattern
        def replace_match(m):
            indent = m.group(1)
            var_name = m.group(2)
            rid_expr = m.group(3)
            return wrap_api_call(type('obj', (), {
                'group': lambda i: [None, indent, var_name, 'api_client', rid_expr][i]
            })())

        modified_content = re.sub(pattern, replace_match, modified_content)

    # Write modified content
    with open(filepath, 'w') as f:
        f.write(modified_content)

    print(f"Modified pipeline_{pipeline_num}.py")
    return True


def create_runner_script(pipeline_num):
    """Create run_pipeline_X.py script."""
    runner_path = PIPELINE_DIR / f"run_pipeline_{pipeline_num}.py"

    runner_content = f'''#!/usr/bin/env python3
"""
Runner script for pipeline {pipeline_num}.
Loads catalog and executes classification pipeline.
"""

import json
import time
import sys
from pathlib import Path

# Import the pipeline
import pipeline_{pipeline_num}

def main():
    print(f"Loading LP catalog...")
    with open('lp_catalog.json', 'r') as f:
        catalog = json.load(f)

    print(f"Catalog loaded: {{len(catalog)}} records")
    print(f"Starting pipeline {pipeline_num} classification...")

    start_time = time.time()

    try:
        result = pipeline_{pipeline_num}.classify_catalog(catalog)

        runtime = time.time() - start_time

        print(f"\\nPipeline {pipeline_num} completed!")
        print(f"Runtime: {{runtime:.2f}} seconds")
        print(f"Ruled in: {{len(result['ruled_in'])}}")
        print(f"Ruled out: {{len(result['ruled_out'])}}")
        print(f"Coverage: {{result['metadata']['coverage_ratio']:.4f}}")
        print(f"API calls: {{result['metadata']['api_calls_made']}}")

        # Save results
        result['metadata']['runtime_seconds'] = runtime
        with open('results_{pipeline_num}.json', 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to results_{pipeline_num}.json")
        return 0

    except Exception as e:
        runtime = time.time() - start_time
        print(f"\\nPipeline {pipeline_num} FAILED after {{runtime:.2f}} seconds")
        print(f"Error: {{e}}")
        import traceback
        traceback.print_exc()

        # Save error info
        error_info = {{
            'status': 'failed',
            'error': str(e),
            'runtime_seconds': runtime,
        }}
        with open('results_{pipeline_num}.json', 'w') as f:
            json.dump(error_info, f, indent=2)

        return 1

if __name__ == '__main__':
    sys.exit(main())
'''

    with open(runner_path, 'w') as f:
        f.write(runner_content)

    os.chmod(runner_path, 0o755)
    print(f"Created run_pipeline_{pipeline_num}.py")


def run_pipeline(pipeline_num):
    """Run a single pipeline."""
    print(f"\n{'='*60}")
    print(f"RUNNING PIPELINE {pipeline_num}")
    print(f"{'='*60}\n")

    runner_script = PIPELINE_DIR / f"run_pipeline_{pipeline_num}.py"

    try:
        result = subprocess.run(
            ['python3', str(runner_script)],
            cwd=PIPELINE_DIR,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"Pipeline {pipeline_num} TIMEOUT after 1 hour")
        return False
    except Exception as e:
        print(f"Pipeline {pipeline_num} execution error: {e}")
        return False


def generate_summary():
    """Generate summary CSV from all results."""
    summary_data = []

    for i in range(1, 13):
        result_file = PIPELINE_DIR / f"results_{i}.json"

        if not result_file.exists():
            summary_data.append({
                'pipeline': i,
                'status': 'not_run',
                'ruled_in': 0,
                'ruled_out': 0,
                'coverage': 0.0,
                'api_calls': 0,
                'runtime': 0.0,
            })
            continue

        try:
            with open(result_file, 'r') as f:
                result = json.load(f)

            if 'status' in result and result['status'] == 'failed':
                summary_data.append({
                    'pipeline': i,
                    'status': 'failed',
                    'ruled_in': 0,
                    'ruled_out': 0,
                    'coverage': 0.0,
                    'api_calls': 0,
                    'runtime': result.get('runtime_seconds', 0.0),
                })
            else:
                summary_data.append({
                    'pipeline': i,
                    'status': 'completed',
                    'ruled_in': len(result.get('ruled_in', [])),
                    'ruled_out': len(result.get('ruled_out', [])),
                    'coverage': result.get('metadata', {}).get('coverage_ratio', 0.0),
                    'api_calls': result.get('metadata', {}).get('api_calls_made', 0),
                    'runtime': result.get('metadata', {}).get('runtime_seconds', 0.0),
                })
        except Exception as e:
            print(f"Error reading results for pipeline {i}: {e}")
            summary_data.append({
                'pipeline': i,
                'status': 'error',
                'ruled_in': 0,
                'ruled_out': 0,
                'coverage': 0.0,
                'api_calls': 0,
                'runtime': 0.0,
            })

    # Write summary CSV
    summary_file = PIPELINE_DIR / 'pipeline_summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pipeline', 'status', 'ruled_in', 'ruled_out',
            'coverage', 'api_calls', 'runtime'
        ])
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"\nSummary saved to {summary_file}")

    # Also print summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"{'Pipeline':<10} {'Status':<12} {'In':<8} {'Out':<8} {'Coverage':<10} {'API':<6} {'Time(s)':<8}")
    print("-"*80)
    for row in summary_data:
        print(f"{row['pipeline']:<10} {row['status']:<12} {row['ruled_in']:<8} {row['ruled_out']:<8} "
              f"{row['coverage']:<10.4f} {row['api_calls']:<6} {row['runtime']:<8.1f}")
    print("="*80)


def main():
    """Main processing function."""
    print("Starting pipeline processing...")
    print(f"Pipeline directory: {PIPELINE_DIR}")

    # Step 1: Create symlinks
    print("\n[1] Creating symlinks...")
    create_symlinks()

    # Step 2-4: Process each pipeline
    print("\n[2-4] Processing pipeline files...")
    for i in range(1, 13):
        print(f"\nProcessing pipeline {i}...")
        backup_pipeline(i)
        modify_pipeline(i)
        create_runner_script(i)

    # Step 5-6: Run each pipeline
    print("\n[5-6] Running all pipelines...")
    for i in range(1, 13):
        run_pipeline(i)

    # Step 7: Generate summary
    print("\n[7] Generating summary...")
    generate_summary()

    print("\n\nAll pipelines processed!")


if __name__ == '__main__':
    main()
