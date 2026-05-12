-- Add per-frame bbox timeline for each tracklet.
--
-- Trace-service uses this table to render evidence clips where the bounding
-- box follows the subject frame-by-frame, instead of using one static
-- representative_bbox for the whole clip.

BEGIN;

CREATE TABLE IF NOT EXISTS tracklet_observations (
    id serial PRIMARY KEY,
    tracklet_id character varying(255) NOT NULL
        REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    frame_index integer NOT NULL,
    timestamp_second double precision NOT NULL,
    bbox json NOT NULL DEFAULT '[]'::json,
    confidence double precision,
    CONSTRAINT uq_tracklet_observations_tracklet_frame
        UNIQUE (tracklet_id, frame_index)
);

CREATE INDEX IF NOT EXISTS ix_tracklet_obs_tracklet_id
    ON tracklet_observations (tracklet_id);

CREATE INDEX IF NOT EXISTS ix_tracklet_obs_tracklet_frame
    ON tracklet_observations (tracklet_id, frame_index);

COMMIT;
