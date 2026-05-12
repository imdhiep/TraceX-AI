#!/usr/bin/env bash
# Fix corrupted PostgreSQL data directory by recreating missing subdirectories.
# Run this when postgres container fails with "could not open directory" errors.
#
# Usage:
#   ./scripts/fix-pgdata.sh [PGDATA_PATH]
#
# Default PGDATA_PATH: ./storage/pgdata (relative to repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PGDATA="${1:-$REPO_ROOT/storage/pgdata}"

if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
  echo "ERROR: $PGDATA does not look like a PostgreSQL data directory (no PG_VERSION)."
  exit 1
fi

PG_VERSION=$(cat "$PGDATA/PG_VERSION" 2>/dev/null || docker run --rm -v "$PGDATA:/data" --user root postgres:16 cat /data/PG_VERSION)
echo "Detected PostgreSQL version: $PG_VERSION"
echo "Fixing PGDATA at: $PGDATA"

REQUIRED_DIRS=(
  pg_notify
  pg_tblspc
  pg_replslot
  pg_stat
  pg_stat_tmp
  pg_snapshots
  pg_commit_ts
  pg_twophase
  pg_logical/snapshots
  pg_logical/mappings
)

docker run --rm \
  -v "$PGDATA:/var/lib/postgresql/data" \
  --user root \
  "postgres:$PG_VERSION" \
  bash -c "
    set -e
    cd /var/lib/postgresql/data
    for d in ${REQUIRED_DIRS[*]}; do
      if [[ ! -d \"\$d\" ]]; then
        mkdir -p \"\$d\"
        echo \"  created: \$d\"
      else
        echo \"  ok:      \$d\"
      fi
    done
    chown -R postgres:postgres /var/lib/postgresql/data
    echo 'Ownership fixed.'
  "

echo ""
echo "Done. You can now start postgres:"
echo "  docker start tracex-postgres"
echo "  # or"
echo "  docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai up -d"
