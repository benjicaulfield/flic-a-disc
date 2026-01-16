#!/bin/bash

# Production PostgreSQL Backup Script
# Performs: pg_dump -> gzip -> openssl encryption -> s3cmd upload
# Author: Gemini Agent
# Date: 2026-01-16

set -o pipefail  # Fail if any command in pipe fails

# --- Configuration ---
# Default values, can be overridden by env vars
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR/../../..}" # Adjust relative path to root
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/flic-a-disc/python_services/.env}" # Default location, adjust if needed
LOG_DIR="${LOG_DIR:-/var/log/flic_backup}"
BACKUP_TEMP_DIR="${BACKUP_TEMP_DIR:-/tmp/flic_db_backups}"
LOCK_FILE="/tmp/flic_backup.lock"
RETENTION_DAYS="${RETENTION_DAYS:-30}" # Days to keep logs

# Ensure directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$BACKUP_TEMP_DIR"

# Log file for this run
LOG_FILE="$LOG_DIR/backup_$(date +%Y-%m-%d).log"

# --- Functions ---

log() {
    local level="$1"
    local message="$2"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$LOG_FILE"
}

cleanup() {
    log "INFO" "Cleaning up temporary files..."
    rm -f "$LOCK_FILE"
    if [ -n "$FINAL_FILE_PATH" ] && [ -f "$FINAL_FILE_PATH" ]; then
        rm -f "$FINAL_FILE_PATH"
        log "INFO" "Removed local backup file: $FINAL_FILE_PATH"
    fi
}

error_handler() {
    log "ERROR" "An error occurred on line $1"
    cleanup
    exit 1
}

trap 'error_handler $LINENO' ERR
trap cleanup EXIT

# --- Main Execution ---

# 1. Concurrency Check
if [ -f "$LOCK_FILE" ]; then
    # Check if process is actually running
    PID=$(cat "$LOCK_FILE")
    if ps -p "$PID" > /dev/null; then
        log "ERROR" "Backup already running (PID: $PID). Exiting."
        exit 1
    else
        log "WARN" "Stale lock file found. Removing."
        rm "$LOCK_FILE"
    fi
fi
echo $$ > "$LOCK_FILE"

log "INFO" "Starting backup process..."

# 2. Load Environment
if [ -f "$ENV_FILE" ]; then
    log "INFO" "Loading environment from $ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    log "WARN" ".env file not found at $ENV_FILE. Assuming env vars are set."
fi

# 3. Validate Config
REQUIRED_VARS=("DB_NAME" "DB_USER" "DB_HOST" "BACKUP_ENCRYPTION_KEY" "SPACES_BUCKET")
MISSING_VARS=0
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        log "ERROR" "Missing required environment variable: $var"
        MISSING_VARS=1
    fi
done

if [ "$MISSING_VARS" -eq 1 ]; then
    exit 1
fi

# DB_PASSWORD can be set or provided via .pgpass. Warn if missing but don't fail immediately if .pgpass is used.
if [ -z "$DB_PASSWORD" ]; then
    log "WARN" "DB_PASSWORD is not set. Ensure .pgpass is configured."
fi

# 4. Prepare Filenames
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
FILENAME="${DB_NAME}_${TIMESTAMP}.sql.gz.enc"
FINAL_FILE_PATH="$BACKUP_TEMP_DIR/$FILENAME"

# 5. Execute Backup Pipeline
log "INFO" "Dumping database '$DB_NAME' from host '$DB_HOST'..."
