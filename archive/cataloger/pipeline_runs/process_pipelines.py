#!/usr/bin/env python3
"""
Process all 12 pipelines: backup, modify, create runners, execute, summarize.

Modification strategy:
1. Add Django/logging boilerplate after imports
2. Change classify_catalog(catalog, api_client) -> classify_catalog(catalog)
3. Replace all "api_client.get_release" with a helper function that logs and saves
"""

import os
import shutil
import json
import time
import csv
import subprocess
from pathlib import Path

PIPELINE_DIR = Path("/Users/benjamincaulfield/Documents/flic-a-disc/cataloger/pipeline_runs")
DATA_DIR = Path("/Users/benjamincaulfield/Documents/flic-a-disc/cataloger/data")
ML_DIR = Path("/Users/benjamincaulfield/Documents/flic-a-disc/ml")

def create_symlinks():
    """Create necessary symlinks."""
    os.chdir(PIPELINE_DIR)
    symlinks = [
        ('enriched_training.json', DATA_DIR / 'enriched_training.json'),
        ('lp_catalog.json', DATA_DIR / 'lp_catalog.json'),
        ('discogs_token.json', ML_DIR / 'discogs_token.json'),
    ]
    for link_name, target in symlinks:
        link_path = PIPELINE_DIR / link_name
        if link_path.exists() or link_path.is_symlink():
            continue
        os.symlink(target, link_path)
        print(f"Created symlink: {link_name}")


def backup_and_modify_pipeline(pipeline_num):
    """Backup and modify a pipeline file."""
    original = PIPELINE_DIR / f"pipeline_{pipeline_num}.py"
    backup = PIPELINE_DIR / f"pipeline_{pipeline_num}.py.original"

    # Backup
    if not backup.exists():
        shutil.copy2(original, backup)
        print(f"Backed up pipeline_{pipeline_num}.py")

    # Read original
    with open(backup, 'r') as f:
        lines = f.readlines()

    # Find last import line
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import_idx = i

    # Insert boilerplate after imports
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

    # Join and process as string for easier replacements
    content = ''.join(new_lines)

    # Change function signature
    content = content.replace(
        'def classify_catalog(catalog, api_client):',
        'def classify_catalog(catalog):'
    )

    # Replace api_client.get_release calls with our helper
    content = content.replace('api_client.get_release(', 'get_release_logged(')

    # Write modified version
    with open(original, 'w') as f:
        f.write(content)

    print(f"Modified pipeline_{pipeline_num}.py")


def create_runner(pipeline_num):
    """Create runner script for a pipeline."""
    runner_path = PIPELINE_DIR / f"run_pipeline_{pipeline_num}.py"

    content = f'''#!/usr/bin/env python3
"""Runner for pipeline {pipeline_num}."""

import json
import time
import sys
import pipeline_{pipeline_num}

def main():
    print("Loading catalog...")
    with open('lp_catalog.json', 'r') as f:
        catalog = json.load(f)

    print(f"Catalog loaded: {{len(catalog)}} records")
    print(f"Starting pipeline {pipeline_num}...")

    start_time = time.time()

    try:
        result = pipeline_{pipeline_num}.classify_catalog(catalog)
        runtime = time.time() - start_time

        print(f"\\nPipeline {pipeline_num} COMPLETED!")
        print(f"Runtime: {{runtime:.2f}}s")
        print(f"Ruled in: {{len(result['ruled_in'])}}")
        print(f"Ruled out: {{len(result['ruled_out'])}}")
        print(f"Coverage: {{result['metadata']['coverage_ratio']:.4f}}")
        print(f"API calls: {{result['metadata']['api_calls_made']}}")

        result['metadata']['runtime_seconds'] = runtime
        with open('results_{pipeline_num}.json', 'w') as f:
            json.dump(result, f, indent=2)

        return 0

    except Exception as e:
        runtime = time.time() - start_time
        print(f"\\nPipeline {pipeline_num} FAILED after {{runtime:.2f}}s")
        print(f"Error: {{e}}")
        import traceback
        traceback.print_exc()

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
        f.write(content)
    os.chmod(runner_path, 0o755)
    print(f"Created run_pipeline_{pipeline_num}.py")


def run_pipeline(pipeline_num):
    """Execute a pipeline."""
    print(f"\n{'='*70}")
    print(f"EXECUTING PIPELINE {pipeline_num}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            ['python3', f'run_pipeline_{pipeline_num}.py'],
            cwd=PIPELINE_DIR,
            capture_output=True,
            text=True,
            timeout=3600
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"Pipeline {pipeline_num} TIMEOUT (1 hour)")
        return False
    except Exception as e:
        print(f"Pipeline {pipeline_num} ERROR: {e}")
        return False


def generate_summary():
    """Generate summary CSV."""
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
            print(f"Error reading results_{i}.json: {e}")
            summary_data.append({
                'pipeline': i,
                'status': 'error',
                'ruled_in': 0,
                'ruled_out': 0,
                'coverage': 0.0,
                'api_calls': 0,
                'runtime': 0.0,
            })

    # Write CSV
    summary_file = PIPELINE_DIR / 'pipeline_summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pipeline', 'status', 'ruled_in', 'ruled_out',
            'coverage', 'api_calls', 'runtime'
        ])
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"\n\nSummary saved to {summary_file}")

    # Print summary
    print("\n" + "="*85)
    print("PIPELINE SUMMARY")
    print("="*85)
    print(f"{'Pipeline':<10} {'Status':<12} {'In':<10} {'Out':<10} {'Coverage':<10} {'API':<8} {'Time(s)':<10}")
    print("-"*85)
    for row in summary_data:
        print(f"{row['pipeline']:<10} {row['status']:<12} {row['ruled_in']:<10} "
              f"{row['ruled_out']:<10} {row['coverage']:<10.4f} "
              f"{row['api_calls']:<8} {row['runtime']:<10.1f}")
    print("="*85)


def main():
    """Main execution."""
    print("PIPELINE PROCESSING SCRIPT")
    print("="*70)

    os.chdir(PIPELINE_DIR)

    # Step 1: Symlinks
    print("\n[Step 1] Creating symlinks...")
    create_symlinks()

    # Step 2: Backup and modify all pipelines
    print("\n[Step 2] Backing up and modifying pipelines...")
    for i in range(1, 13):
        backup_and_modify_pipeline(i)

    # Step 3: Create runners
    print("\n[Step 3] Creating runner scripts...")
    for i in range(1, 13):
        create_runner(i)

    # Step 4: Run all pipelines
    print("\n[Step 4] Running pipelines...")
    for i in range(1, 13):
        run_pipeline(i)

    # Step 5: Generate summary
    print("\n[Step 5] Generating summary...")
    generate_summary()

    print("\n\nALL DONE!")


if __name__ == '__main__':
    main()
