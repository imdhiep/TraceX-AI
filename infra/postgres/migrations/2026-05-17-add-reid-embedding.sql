-- Add a dedicated identity/Re-ID lane beside the existing SigLIP semantic lane.
--
-- `siglip_embedding` remains the text↔image retrieval vector.
-- `reid_embedding` is reserved for same-person association models.
-- Current backend: PersonViT-S MSMT17, 384-dim.
--
-- Earlier local experiments used DINOv2 (1024-dim). If that column already
-- exists, old vectors are intentionally nulled while changing the dimension:
-- they are not comparable to PersonViT vectors and must not survive the swap.

BEGIN;

ALTER TABLE tracklets_embeddings
  ADD COLUMN IF NOT EXISTS reid_model_version character varying(128);

DO $$
DECLARE
  current_reid_type text;
BEGIN
  SELECT format_type(a.atttypid, a.atttypmod)
    INTO current_reid_type
  FROM pg_attribute a
  JOIN pg_class c ON c.oid = a.attrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname = current_schema()
    AND c.relname = 'tracklets_embeddings'
    AND a.attname = 'reid_embedding'
    AND NOT a.attisdropped;

  IF current_reid_type IS NULL THEN
    ALTER TABLE tracklets_embeddings
      ADD COLUMN reid_embedding vector(384);
  ELSIF current_reid_type <> 'vector(384)' THEN
    -- pgvector cannot preserve a 1024-dim DINOv2 vector while converting it to
    -- 384-dim PersonViT. NULL is the only honest value after the model swap.
    ALTER TABLE tracklets_embeddings
      ALTER COLUMN reid_embedding TYPE vector(384)
      USING NULL;
  END IF;
END $$;

COMMIT;
