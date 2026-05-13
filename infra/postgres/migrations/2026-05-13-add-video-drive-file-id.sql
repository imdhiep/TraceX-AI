-- Persist the Google Drive file ID on the videos table so trace-service can
-- download the source clip on-demand when rendering evidence (videos are not
-- kept on local disk after ingest).

BEGIN;

ALTER TABLE videos
    ADD COLUMN IF NOT EXISTS drive_file_id varchar(255);

CREATE INDEX IF NOT EXISTS ix_videos_drive_file_id ON videos(drive_file_id);

COMMIT;
