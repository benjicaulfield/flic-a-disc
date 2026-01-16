#!/bin/bash

# Production PostgreSQL Restore Script
# Performs: s3cmd download -> openssl decrypt -> gunzip -> psql restore
# Author: Gemini Agent
# Date: 2026-01-16

set -o pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR/../../..}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/flic-a-disc/python_services/.env}"
RESTORE_TEMP_DIR="${RESTORE_TEMP_DIR:-/tmp/flic_db_restores}"

mkdir -p "$RESTORE_TEMP_DIR"

# --- Functions ---

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --latest        Restore the most recent backup found in Spaces."
    echo "  --file <name>   Restore a specific backup filename."
    echo "  --list          List available backups."
    echo "  -y              Non-interactive mode (skip confirmation)."
    echo "  -h, --help      Show this help message."
    exit 1
}

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# --- Parse Arguments ---
MODE=""
BACKUP_FILE=""
NON_INTERACTIVE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --latest) MODE="latest" ;;
        --file) MODE="file"; BACKUP_FILE="$2"; shift ;;
        --list) MODE="list" ;;
        -y) NON_INTERACTIVE=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# --- Load Environment ---
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "WARN: .env file not found at $ENV_FILE. Relying on existing env vars."
fi

# Validate minimal vars for listing
if [ -z "$SPACES_BUCKET" ]; then
    echo "ERROR: SPACES_BUCKET env var is not set."
    exit 1
fi

# --- Execution ---

if [ "$MODE" == "list" ]; then
    echo "Available backups in s3://$SPACES_BUCKET/backups/:"
    s3cmd ls "s3://$SPACES_BUCKET/backups/" | awk '{print $4}' | sed "s|s3://$SPACES_BUCKET/backups/||"
    exit 0
fi

if [ -z "$MODE" ]; then
    usage
fi

# Find file if latest
if [ "$MODE" == "latest" ]; then
    log "Finding latest backup..."
    BACKUP_FILE=$(s3cmd ls "s3://$SPACES_BUCKET/backups/" | sort | tail -n 1 | awk '{print $4}' | sed "s|s3://$SPACES_BUCKET/backups/||")
    if [ -z "$BACKUP_FILE" ]; then
        log "ERROR: No backups found."
        exit 1
    fi
    log "Latest backup identified: $BACKUP_FILE"
fi

if [ -z "$BACKUP_FILE" ]; then
    log "ERROR: No backup file specified."
    exit 1
fi

# Confirmation
if [ "$NON_INTERACTIVE" = false ]; then
    echo "⚠️  DANGER: This will OVERWRITE database '$DB_NAME' on '$DB_HOST'."
    echo "Target Backup: $BACKUP_FILE"
    read -p "Are you sure you want to proceed? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Operation cancelled."
        exit 0
    fi
fi

# Check Decryption Key
if [ -z "$BACKUP_ENCRYPTION_KEY" ]; then
    log "ERROR: BACKUP_ENCRYPTION_KEY is missing."
    exit 1
fi

# Download
LOCAL_PATH="$RESTORE_TEMP_DIR/$BACKUP_FILE"
log "Downloading $BACKUP_FILE..."
s3cmd get "s3://$SPACES_BUCKET/backups/$BACKUP_FILE" "$LOCAL_PATH" --force > /dev/null

