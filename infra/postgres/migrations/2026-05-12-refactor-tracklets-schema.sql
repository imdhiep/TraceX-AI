-- Refactor tracklets + tracklets_embeddings to the minimal schema.
--
-- DROP all duplicate / dead columns. Rename clothing fields to the canonical
-- short prefix (upper_/lower_) so there is exactly ONE attribute per concept.
-- Add per-attribute conf for free-text desc fields (now derived from Qwen
-- token logprobs, not self-reported).
--
-- This is a BREAKING migration. Old rows lose dropped columns. Re-ingest
-- after applying.

BEGIN;

-- ── tracklets ────────────────────────────────────────────────────────────

ALTER TABLE tracklets
  DROP COLUMN IF EXISTS occlusion_score,
  DROP COLUMN IF EXISTS batch_id,
  DROP COLUMN IF EXISTS top_color,
  DROP COLUMN IF EXISTS bottom_color,
  DROP COLUMN IF EXISTS top_color_conf,
  DROP COLUMN IF EXISTS bottom_color_conf,
  DROP COLUMN IF EXISTS hat_color_conf,
  DROP COLUMN IF EXISTS bag_type_conf,
  DROP COLUMN IF EXISTS accessory_conf,
  DROP COLUMN IF EXISTS hair_style_conf,
  DROP COLUMN IF EXISTS hair_color_conf,
  DROP COLUMN IF EXISTS contributing_cameras,
  DROP COLUMN IF EXISTS contributing_video_ids;

-- Rename to canonical short prefix
ALTER TABLE tracklets RENAME COLUMN upper_clothing_color TO upper_color;
ALTER TABLE tracklets RENAME COLUMN upper_clothing_type  TO upper_type;
ALTER TABLE tracklets RENAME COLUMN upper_clothing_desc  TO upper_desc;
ALTER TABLE tracklets RENAME COLUMN upper_clothing_conf  TO upper_conf;
ALTER TABLE tracklets RENAME COLUMN lower_clothing_color TO lower_color;
ALTER TABLE tracklets RENAME COLUMN lower_clothing_type  TO lower_type;
ALTER TABLE tracklets RENAME COLUMN lower_clothing_desc  TO lower_desc;
ALTER TABLE tracklets RENAME COLUMN lower_clothing_conf  TO lower_conf;
ALTER TABLE tracklets RENAME COLUMN is_wearing_mask      TO mask_presence;

-- Per-attribute conf for free-text desc (new — Qwen mean-token logprob)
ALTER TABLE tracklets
  ADD COLUMN IF NOT EXISTS upper_desc_conf      double precision,
  ADD COLUMN IF NOT EXISTS lower_desc_conf      double precision,
  ADD COLUMN IF NOT EXISTS shoes_desc_conf      double precision,
  ADD COLUMN IF NOT EXISTS bag_desc_conf        double precision,
  ADD COLUMN IF NOT EXISTS hat_desc_conf        double precision,
  ADD COLUMN IF NOT EXISTS appearance_summary_conf double precision,
  ADD COLUMN IF NOT EXISTS hair_style_conf      double precision,
  ADD COLUMN IF NOT EXISTS hair_color_conf      double precision;

-- Drop indexes pointing to removed/renamed columns; recreate against new names.
DROP INDEX IF EXISTS ix_tracklets_upper_color;
DROP INDEX IF EXISTS ix_tracklets_upper_type;
DROP INDEX IF EXISTS ix_tracklets_lower_color;
DROP INDEX IF EXISTS ix_tracklets_lower_type;

CREATE INDEX IF NOT EXISTS ix_tracklets_upper_color ON tracklets (upper_color);
CREATE INDEX IF NOT EXISTS ix_tracklets_upper_type  ON tracklets (upper_type);
CREATE INDEX IF NOT EXISTS ix_tracklets_lower_color ON tracklets (lower_color);
CREATE INDEX IF NOT EXISTS ix_tracklets_lower_type  ON tracklets (lower_type);

-- ── tracklets_embeddings ─────────────────────────────────────────────────

ALTER TABLE tracklets_embeddings
  DROP COLUMN IF EXISTS embedding_vector,
  DROP COLUMN IF EXISTS embedding,
  DROP COLUMN IF EXISTS model_version;

COMMIT;
