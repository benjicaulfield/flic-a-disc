#!/bin/bash

# End-to-End Backup & Restore Verification Script
# Validates that we can backup the production DB and restore it to a test DB correctly.
# Author: Gemini Agent
# Date: 2026-01-16

set -e

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
REAL_ENV_FILE="$PROJECT_ROOT/flic-a-disc/python_services/.env"
TEST_ENV_FILE="$SCRIPT_DIR/.env.test_restore"
TEST_DB_NAME="flic_restore_verification_db"

BACKUP_SCRIPT="$SCRIPT_DIR/backup.sh"
RESTORE_SCRIPT="$SCRIPT_DIR/restore.sh"

# --- Colors ---
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[TEST] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

cleanup() {
    log "Cleaning up test artifacts..."
    rm -f "$TEST_ENV_FILE"
    
    # Drop test DB
    if [ -n "$DB_HOST" ] && [ -n "$DB_USER" ]; then
        log "Dropping test database $TEST_DB_NAME..."
        export PGPASSWORD="$DB_PASSWORD"
        psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS \"$TEST_DB_NAME\";" > /dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# --- Prerequisites ---
if [ ! -f "$REAL_ENV_FILE" ]; then
    error "Real .env file not found at $REAL_ENV_FILE"
fi

# Load real env to get credentials for psql
export $(grep -v '^#' "$REAL_ENV_FILE" | xargs)

# --- Step 1: Run Backup ---
log "Step 1: Running Production Backup..."
"$BACKUP_SCRIPT"

# --- Step 2: Prepare Test Environment ---
log "Step 2: Preparing Test Environment Config..."
cp "$REAL_ENV_FILE" "$TEST_ENV_FILE"

# Use sed to change DB_NAME in the test env file
# We use a loop to handle potential multiple occurrences or just append if needed, 
# but sed replace is standard. 
# We assume DB_NAME is defined like DB_NAME=value
if grep -q "DB_NAME=" "$TEST_ENV_FILE"; then
    # BSD sed (macOS) requires '' for -i
    sed -i '' "s/DB_NAME=.*/DB_NAME=$TEST_DB_NAME/" "$TEST_ENV_FILE" || sed -i "s/DB_NAME=.*/DB_NAME=$TEST_DB_NAME/" "$TEST_ENV_FILE"
else
    echo "DB_NAME=$TEST_DB_NAME" >> "$TEST_ENV_FILE"
fi

# --- Step 3: Restore to Test DB ---
log "Step 3: Restoring to Test Database ($TEST_DB_NAME)..."
export ENV_FILE="$TEST_ENV_FILE"
"$RESTORE_SCRIPT" --latest -y

# --- Step 4: Verification ---
log "Step 4: Verifying Data Integrity..."

# Get row count from Real DB
REAL_COUNT=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT count(*) FROM bandit_training_example;" | tr -d ' ')

# Get row count from Test DB
TEST_COUNT=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$TEST_DB_NAME" -t -c "SELECT count(*) FROM bandit_training_example;" | tr -d ' ')
