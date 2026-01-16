# Automated PostgreSQL Backup & Restore System

This directory contains a production-ready system for backing up PostgreSQL databases to DigitalOcean Spaces, restoring them, and verifying the integrity of the backups.

## Features
- **Backup**: `pg_dump` -> `gzip` (compression) -> `openssl` (AES-256 encryption) -> `s3cmd` (upload to DO Spaces).
- **Restore**: Decrypts, decompresses, and restores to the target database (clean overwrite).
- **Verification**: `test_cycle.sh` runs a full backup of production and restores it to a temporary test database to verify data integrity (row counts).
- **Security**: Uses Environment Variables for all credentials. Local artifacts are cleaned up immediately.

## Prerequisites

1.  **System Tools**:
    - `bash`
    - `postgresql-client` (for `pg_dump`, `psql`)
    - `s3cmd` (installed and configured, or keys provided via env)
    - `openssl`
    - `gzip`

2.  **Environment Variables**:
    The scripts look for a `.env` file at `../../python_services/.env` (relative to this directory) or accept standard environment variables.

    **Required Variables:**
    ```ini
    DB_NAME=my_database
    DB_USER=my_user
    DB_PASSWORD=my_password
    DB_HOST=localhost
    BACKUP_ENCRYPTION_KEY=super_secret_key_at_least_32_chars
    SPACES_BUCKET=my-do-space-name
    SPACES_REGION=nyc3 (or your region)
    # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY if not using ~/.s3cfg
    ```

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    sudo apt-get update
    sudo apt-get install postgresql-client s3cmd openssl gzip
    ```

2.  **Configure s3cmd**:
    Run `s3cmd --configure` and enter your DigitalOcean Spaces keys and endpoint (e.g., `nyc3.digitaloceanspaces.com`).
    *Alternatively, ensure `~/.s3cfg` is present for the user running the scripts.*

3.  **Permissions**:
    Ensure the scripts are executable:
    ```bash
    chmod +x *.sh
    ```

## Usage

### Manual Backup
```bash
./backup.sh
```
