-- ============================================================
-- Migration: remove legacy columns
-- Generated: 2026-05-11
-- Run manually — no Alembic in this project.
-- BACKUP DATABASE before running.
-- ============================================================

-- 1. tracklets: drop legacy color columns and their conf duplicates
ALTER TABLE tracklets DROP COLUMN IF EXISTS top_color;
ALTER TABLE tracklets DROP COLUMN IF EXISTS bottom_color;
ALTER TABLE tracklets DROP COLUMN IF EXISTS top_color_conf;
ALTER TABLE tracklets DROP COLUMN IF EXISTS bottom_color_conf;

-- 2. tracklets_embeddings: drop DINOv2 JSON fallback and unused pgvector column
ALTER TABLE tracklets_embeddings DROP COLUMN IF EXISTS embedding_vector;
ALTER TABLE tracklets_embeddings DROP COLUMN IF EXISTS embedding;

-- 3. (Optional) update model_version for existing rows written before this migration
UPDATE tracklets_embeddings SET model_version = 'siglip2' WHERE model_version = 'dinov2_vitl14';
UPDATE tracklets_embeddings SET model_version = 'siglip2' WHERE model_version = 'dinov2_vitl14+siglip2';

-- ============================================================
-- Verification queries (run after migration):
-- ============================================================
-- SELECT column_name FROM information_schema.columns
--   WHERE table_name = 'tracklets'
--   AND column_name IN ('top_color','bottom_color','top_color_conf','bottom_color_conf');
-- -- Expected: 0 rows
--
-- SELECT column_name FROM information_schema.columns
--   WHERE table_name = 'tracklets_embeddings'
--   AND column_name IN ('embedding_vector','embedding');
-- -- Expected: 0 rows
