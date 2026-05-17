"""TRUNCATE all data tables (keep `users`) and migrate `reid_embedding`
from VECTOR(1024) (DINOv2) to VECTOR(384) (PersonViT-S).

Run this ONCE after deploying the PersonViT swap. Idempotent — safe to
re-run if it fails partway.

Usage:
  cd /teamspace/studios/this_studio/TraceX-AI
  python scripts/reset_db_for_personvit.py
  # add --dry-run to print the SQL without executing.

Required env: DATABASE_URL (PostgreSQL DSN).
"""
from __future__ import annotations

import argparse
import os
import sys

import sqlalchemy as sa
from sqlalchemy import text


# Tables to truncate. Order matters: children before parents to avoid FK
# violations even with RESTART IDENTITY CASCADE (CASCADE handles it, but
# explicit order keeps the log readable).
DATA_TABLES = [
    # tracklet-derived
    "tracklets_embeddings",
    "tracklets_actions",
    "tracklet_observations",
    # query layer
    "query_candidate_tracklets",
    "query_candidates",
    "query_jobs",
    "query_history",
    # evidence + grouping
    "evidence_tracklets",
    "evidence_videos",
    "verified_objects_tracklets",
    "verified_objects",
    "spatiotemporal_groups",
    # ingest queue
    "queue_video_assets",
    # core
    "tracklets",
    "videos",
    # cameras (delete data but keep schema; users + camera config are usually
    # paired in production, but the user only asked to keep `users`, so we
    # truncate cameras too — they get re-ingested with metadata anyway).
    "camera_edges",
    "camera_zones",
    "camera_settings",
    "cameras",
]

# Preserve `users` data — listed here as documentation only.
PRESERVE_TABLES = ["users"]


def reset(engine: sa.Engine, dry_run: bool) -> None:
    inspector = sa.inspect(engine)
    existing = set(inspector.get_table_names())

    tables_to_truncate = [t for t in DATA_TABLES if t in existing]
    missing = [t for t in DATA_TABLES if t not in existing]
    if missing:
        print(f"  (skipped — not present in DB: {missing})")

    if not tables_to_truncate:
        print("Nothing to truncate.")
    else:
        truncate_sql = (
            f"TRUNCATE TABLE {', '.join(tables_to_truncate)} "
            f"RESTART IDENTITY CASCADE;"
        )
        print(f"\n── TRUNCATE ──\n{truncate_sql}")
        if not dry_run:
            with engine.begin() as conn:
                conn.execute(text(truncate_sql))
            print("  ✓ truncated")

    # Migrate tracklets_embeddings.reid_embedding dimension
    if "tracklets_embeddings" in existing:
        migrate_sql = (
            "ALTER TABLE tracklets_embeddings "
            "ALTER COLUMN reid_embedding TYPE vector(384) "
            "USING NULL;"
        )
        print(f"\n── ALTER COLUMN dim 1024 → 384 ──\n{migrate_sql}")
        if not dry_run:
            with engine.begin() as conn:
                # Drop any HNSW/IVFFlat index on reid_embedding first
                # (pgvector cannot ALTER dim on an indexed column).
                idx_rows = conn.execute(text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = 'tracklets_embeddings' "
                    "AND indexdef ILIKE '%reid_embedding%';"
                )).fetchall()
                for (idx,) in idx_rows:
                    print(f"  dropping index {idx}")
                    conn.execute(text(f"DROP INDEX IF EXISTS {idx};"))
                conn.execute(text(migrate_sql))
            print("  ✓ migrated (indexes dropped — re-create with new dim if needed)")

    print("\nPreserved tables (data kept):", PRESERVE_TABLES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print SQL only")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL DSN (default: $DATABASE_URL)",
    )
    args = parser.parse_args()

    if not args.database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 1

    print(f"Connecting to {args.database_url.split('@')[-1]}...")
    engine = sa.create_engine(args.database_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  ✓ connected")
    except Exception as exc:
        print(f"ERROR: connection failed: {exc}", file=sys.stderr)
        return 1

    reset(engine, dry_run=args.dry_run)
    if args.dry_run:
        print("\n(dry-run — no changes made)")
    else:
        print("\nDONE. Restart metadata-service to pick up the new schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
