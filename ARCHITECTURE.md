# TraceX-AI Architecture

Tài liệu này mô tả kiến trúc hiện tại của TraceX-AI: hệ thống tìm kiếm và truy vết người trong nhiều camera bằng mô tả tự nhiên, dữ liệu tracklet, embedding và luồng xác nhận của người vận hành.

Khác với bản thiết kế ban đầu, kiến trúc hiện tại đã tách rõ:

- **Frontend** chạy trên VPS/Coolify.
- **Backend + database + GPU services** chạy trên LightningAI.
- **PostgreSQL + pgvector** là kho dữ liệu chính cho video, tracklet, embedding, query, candidate và evidence.
- Search/trace đều đi qua `metadata-service` để frontend chỉ cần một API base URL.

---

## 1. Bối cảnh và mục tiêu

TraceX-AI phục vụ bài toán vận hành camera giám sát: người dùng cần tìm một người cụ thể trong nhiều video/camera mà không phải xem thủ công từng đoạn.

Mục tiêu hiện tại:

- xử lý video offline thành tracklet có metadata và embedding;
- cho phép tìm kiếm bằng text hoặc ảnh, kèm bộ lọc camera/thời gian;
- trả về candidate có preview, description và toàn bộ tracklet liên quan;
- cho phép người dùng chọn candidate để dựng trace/evidence;
- lưu lại lịch sử query, candidate và evidence video;
- cho phép human-in-the-loop: xóa tracklet sai khỏi candidate, trace nhiều candidate, xem lại history.

Không phải trọng tâm hiện tại:

- realtime streaming;
- nhận dạng danh tính tuyệt đối;
- scale đa GPU tự động;
- training model lớn từ đầu trong app chính.

---

## 2. Kiến trúc triển khai

```text
┌──────────────────────────────────────────────────────────────────┐
│  VPS / Coolify                                                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Frontend (Next.js)                                        │  │
│  │ container: mcpt-frontend                                  │  │
│  │ port 3000 -> Traefik / HTTPS                              │  │
│  │ API rewrite: /api-gw -> LightningAI metadata-service      │  │
│  └──────────────────────────────┬─────────────────────────────┘  │
└─────────────────────────────────┼────────────────────────────────┘
                                  │ HTTPS /api/v1
                                  │ NEXT_PUBLIC_API_BASE_URL
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  LightningAI                                                     │
│                                                                  │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │ metadata-service │   │  query-service   │   │ trace-service│ │
│  │ FastAPI          │   │  FastAPI         │   │ FastAPI      │ │
│  │ public :8002     │   │  internal :8003  │   │ internal:8004│ │
│  │ auth/videos      │   │  search/ranking  │   │ trace/build  │ │
│  │ ingest/history   │   │  translation     │   │ evidence     │ │
│  └────────┬─────────┘   └────────┬─────────┘   └──────┬───────┘ │
│           │                      │                    │         │
│           └──────────────────────┴────────────────────┘         │
│                                  │                              │
│                     ┌────────────▼────────────┐                 │
│                     │ PostgreSQL + pgvector   │                 │
│                     │ internal :5432          │                 │
│                     └─────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────┘
```

### Runtime rule

- Browser gọi non-search API qua `metadata-service` (`/api-gw` / `NEXT_PUBLIC_API_BASE_URL`).
- Search gọi thẳng `query-service` qua `/search-gw` / `NEXT_PUBLIC_SEARCH_API_BASE_URL`; query-service tự auth, tự nhận multipart và tự xử lý ảnh query.
- `metadata-service` vẫn giữ route proxy search để tương thích ngược và gọi nội bộ:
  - `query-service` qua `http://query-service:8003`;
  - `trace-service` qua `http://trace-service:8004`.
- PostgreSQL chỉ nằm trong Docker network của LightningAI, không expose public.

---

## 3. Service responsibilities

### 3.1 Frontend

Vị trí: `frontend/`

Công nghệ:

- Next.js 14;
- React 18;
- TypeScript;
- Tailwind CSS.

Trách nhiệm:

- đăng nhập và giữ session;
- home/search UI;
- candidate result panel;
- candidate detail modal;
- history pages;
- trace page;
- admin/users/settings pages;
- proxy API qua `/api-gw`;
- proxy static files qua `/static`.

Các feature chính:

- `features/home/HomeView.tsx`
- `features/search/SearchResultsPanel.tsx`
- `features/history/HistoryView.tsx`
- `features/history/HistoryCandidatesView.tsx`
- `features/trace/TraceView.tsx`
- `features/candidate/CandidateDetailModal.tsx`

### 3.2 metadata-service

Vị trí: `backend/services/metadata-service/`

Public API chính, port `8002`.

Trách nhiệm:

- auth/JWT và bootstrap admin;
- user management;
- video metadata;
- ingest endpoints;
- video processing endpoints;
- static crop/query-image/trace serving;
- search proxy sang `query-service`;
- trace proxy sang `trace-service`;
- history storage/retrieval.

Routers chính:

- `/api/v1/auth`
- `/api/v1/users`
- `/api/v1/videos`
- `/api/v1/video`
- `/api/v1/ingest`
- `/api/v1/search`
- `/api/v1/candidates`
- `/api/v1/history`
- `/api/v1/trace`

GPU model warmup hiện nằm ở `metadata-service`:

- RT-DETR R50: person detection;
- DINOv2 ViT-L/14: appearance/ReID features;
- SigLIP/SigLIP2 fallback: image-text embedding;
- VideoMAE V2: action recognition;
- Qwen2-VL-7B-Instruct: open-vocabulary person metadata.

### 3.3 query-service

Vị trí: `backend/services/query-service/`

Internal API, port `8003`.

Trách nhiệm:

- nhận query/search request từ `metadata-service`;
- encode text/image query;
- parse metadata từ query;
- tính ranking/fusion score;
- merge candidate theo threshold;
- lưu `query_history`, `query_candidates`, `query_candidate_tracklets`;
- dịch text hiển thị sang tiếng Việt khi cần;
- cung cấp overview metrics.

Model chính:

- SigLIP So400m text/image tower;
- SeamlessM4T v2-large cho translation.

Tham số quan trọng:

- `MIN_FUSION_SCORE`
- `MAX_CANDIDATES`
- `QUERY_MERGE_THRESHOLD`
- `QUERY_SERVICE_LOG_LEVEL`

### 3.4 trace-service

Vị trí: `backend/services/trace-service/`

Internal API, port `8004`.

Trách nhiệm:

- chọn candidate cho trace;
- trả candidate detail gồm toàn bộ tracklet;
- xóa tracklet khỏi candidate;
- build evidence video/timeline;
- trả trace status/timeline;
- nhận feedback;
- continue trace;
- render/cache trace clips;
- đọc video từ local storage hoặc Drive cache khi cần.

Routers chính:

- `/api/v1/trace/select`
- `/api/v1/trace/build`
- `/api/v1/trace/status/{evidence_id}`
- `/api/v1/trace/timeline/{evidence_id}`
- `/api/v1/trace/feedback`
- `/api/v1/trace/candidate-detail`
- `/api/v1/trace/candidate-tracklet/remove`
- `/api/v1/trace/continue`

### 3.5 shared module

Vị trí: `backend/services/shared/`

Trách nhiệm:

- SQLAlchemy models;
- shared database/session config;
- time helpers;
- common schema compatibility.

File quan trọng:

- `shared/models.py`
- `shared/database.py`
- `shared/tracklet_time.py`

---

## 4. Data flow

### 4.1 Ingest / video processing

```text
Video source
  -> metadata-service /api/v1/video/process hoặc /api/v1/ingest/*
  -> decode/sample frames
  -> RT-DETR person detection
  -> tracking + tracklet observations
  -> Qwen/SigLIP/DINOv2/VideoMAE feature extraction
  -> merge/filter tracklets
  -> write DB:
       videos
       tracklets
       tracklets_embeddings
       tracklets_actions
       tracklet_observations
  -> write files:
       crops
       previews
       local cached source/evidence assets
```

Storage entrypoints:

- Local files in `/workspace/storage`;
- optional Google Drive via OAuth secrets;
- helper scripts `move.py` and `ingest_local.py` for local/debug workflows.

### 4.2 Search

```text
User query + optional image + filters
  -> frontend SearchBar
  -> metadata-service POST /api/v1/search
  -> query-service POST /api/v1/search
  -> encode query text/image
  -> DB shortlist from tracklets/embeddings
  -> fusion scoring
  -> candidate merge/grouping
  -> write:
       query_history
       query_candidates
       query_candidate_tracklets
  -> metadata-service returns candidates to frontend
```

Search inputs:

- `query` or `text`;
- `top_k`;
- `offset`;
- `camera_ids`;
- `time_from`;
- `time_to`;
- optional `query_image`.

Search outputs:

- `query_id`;
- ranked candidates;
- candidate preview URL;
- score fields;
- camera/time metadata;
- appearance summary;
- tracklet linkage for later trace/detail.

### 4.3 Candidate review

```text
Candidate selected in frontend
  -> metadata-service POST /api/v1/trace/candidate-detail
  -> trace-service POST /api/v1/trace/candidate-detail
  -> load query_candidate_tracklets + tracklets + observations/actions
  -> localize display fields
  -> frontend displays all tracklets and description
```

The user can:

- inspect all tracklets of a candidate;
- select one or multiple candidates for trace;
- remove wrong tracklets from a candidate;
- go back to history and inspect previous query/candidate state.

### 4.4 Trace / evidence

```text
User selects candidate
  -> POST /api/v1/trace/select
  -> POST /api/v1/trace/build
  -> trace-service builds evidence segments
  -> render/cache clips in /workspace/storage/traces
  -> write:
       evidence_videos
       evidence_tracklets
  -> frontend polls status/timeline
  -> user gives feedback or continues trace
```

Trace output:

- evidence video URL;
- ordered timeline;
- segment count;
- time window;
- confidence;
- linked tracklets.

---

## 5. Database model

Schema chính nằm ở `backend/services/shared/models.py`.

### User/auth

- `users`

### Camera/topology

- `cameras`
- `camera_zones`
- `camera_edges`
- `camera_settings`

Camera config files:

- `backend/config/camera_topology.json`
- `backend/config/camera_calibration.json`
- `backend/config/homography_registry.json`
- `backend/config/world_projection_calibration.json`

### Video/tracklet

- `videos`
- `tracklets`
- `tracklets_embeddings`
- `tracklets_actions`
- `tracklet_observations`

Important tracklet fields:

- `tracklet_id`
- `video_id`
- `camera_id`
- `start_time`, `end_time`
- appearance fields: upper/lower/shoes/bag/hat/mask/hair/gender/age;
- `appearance_summary`;
- `crop_url`;
- `representative_bbox`;
- `quality_score`;
- BEV coordinates.

Embedding:

- table: `tracklets_embeddings`;
- field: `siglip_embedding`;
- dimension: 1152 when pgvector is available.

### Query/candidate

- `query_history`
- `query_candidates`
- `query_candidate_tracklets`
- `query_jobs`
- `spatiotemporal_groups`

Important candidate fields:

- `candidate_id`;
- `query_id`;
- `fusion_score`;
- `vector_score`;
- `text_score`;
- `spatiotemporal_score`;
- `rank_position`;
- `preview_url`;
- `appearance_summary`;
- `primary_camera_id`.

### Evidence/feedback

- `evidence_videos`
- `evidence_tracklets`
- `verified_objects`
- `verified_objects_tracklets`

### Internal staging

- `queue_video_assets`

Used for storage/Drive ingestion staging.

---

## 6. Storage architecture

### Runtime storage

On LightningAI:

```text
storage/
├── pgdata/                 # PostgreSQL bind mount
├── videos/                 # source/local videos
├── traces/                 # evidence clips/videos
├── queue/                  # ingest queue cache
├── candidate-previews/     # candidate preview assets
├── cache/                  # trace/cache workspace
└── model-weights/          # optional local model weights
```

Model cache:

```text
/home/zeus/.cache/huggingface
/home/zeus/.cache/torch
```

Secrets:

```text
secrets/
├── master.env.example
├── shared.env.example
└── oauth/                  # real OAuth files are not committed
```

### Static serving

`metadata-service` mounts:

- `/static/crops`
- `/static/query-images`
- `/static/traces`

Frontend proxies static assets through Next.js rewrite when needed.

---

## 7. API boundary

Frontend only needs one base URL:

```text
NEXT_PUBLIC_API_BASE_URL=https://8002-<workspace>.cloudspaces.litng.ai/api/v1
NEXT_PUBLIC_SEARCH_API_BASE_URL=https://8003-<workspace>.cloudspaces.litng.ai/api/v1
```

`frontend/next.config.js` rewrites:

- `/api-gw/:path*` -> upstream `/api/v1/:path*`;
- `/static/:path*` -> upstream `/static/:path*`.

Primary public APIs:

- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `GET /api/v1/videos`
- `POST /api/v1/video/process`
- `POST /api/v1/video/process/stream`
- `POST /api/v1/video/batch/process`
- `POST /api/v1/search`
- `GET /api/v1/history`
- `GET /api/v1/history/{query_id}/candidates`
- `GET /api/v1/history/{query_id}/evidence`
- `POST /api/v1/trace/candidate-detail`
- `POST /api/v1/trace/select`
- `POST /api/v1/trace/build`
- `GET /api/v1/trace/status/{evidence_id}`
- `GET /api/v1/trace/timeline/{evidence_id}`
- `POST /api/v1/trace/candidate-tracklet/remove`

---

## 8. Deployment architecture

Current deployment path:

| File | Purpose |
| ---- | ------- |
| `docker-compose.yml` | VPS/Coolify frontend only |
| `docker-compose.lightningai.yml` | LightningAI backend + database |
| `.env.vps.example` | frontend env template |
| `.env.lightningai.example` | backend env template |
| `DEPLOY.md` | operational deployment steps |

Service ports:

| Service | Port | Public |
| ------- | ---- | ------ |
| frontend | 3000 | via VPS Traefik HTTPS |
| metadata-service | 8002 | yes, LightningAI public URL |
| query-service | 8003 | public search runtime + internal compatibility path |
| trace-service | 8004 | internal only |
| PostgreSQL | 5432 | internal only |

---

## 9. Scalability considerations

Current design is batch/offline-first.

Reasons:

- video processing is GPU-heavy;
- Qwen/SigLIP/VideoMAE inference is expensive;
- tracklet quality depends on batch tuning;
- trace rendering can be CPU/IO-bound.

Recommended scale path:

1. Keep frontend stateless.
2. Keep PostgreSQL as source of truth.
3. Add explicit job queue for video processing if ingest volume increases.
4. Partition/index DB by camera/time for large deployments.
5. Keep trace/evidence rendering async and pollable.
6. Split model workloads across GPUs only after job boundaries are stable.
7. Add observability for:
   - ingest latency;
   - detection/tracking throughput;
   - candidate count;
   - search latency;
   - trace render time;
   - failed Drive/local video fetches.

---

## 10. Security and privacy

Current requirements:

- no secrets in frontend bundle except public API base URL;
- no real `.env`, OAuth token, HF token, or private Drive credential committed;
- JWT required for user-facing APIs;
- PostgreSQL not publicly exposed;
- logs should avoid dumping full personal descriptions, tokens, or private URLs;
- evidence clips should be scoped to the selected query/candidate;
- admin/user actions should remain auditable through query/history tables.

Operational requirement from `AGENTS.md`:

- run `bash scripts/setup_hooks.sh` before creating PRs;
- do not commit `.ai-log/*.jsonl`.

---

## 11. Known limitations

- No realtime camera stream processing in the current app.
- Single LightningAI backend environment is the main deployment target.
- Search and merge thresholds still require dataset-specific calibration.
- Trace depends on source video availability in storage/Drive.
- Some legacy files remain for older deployment paths, e.g. `docker-compose.backend-included.yml` and `backend/services/lightningai-compose.yml`.
- Automated test coverage for full ingest/search/trace is still limited.

---

## 12. Information still needed for production hardening

The repo is enough to document the current architecture. For production-grade architecture, the following inputs are still needed:

- expected number of cameras active per day;
- average video duration and file size;
- required retention period for source videos, crops, traces and query history;
- expected concurrent users;
- target search latency and trace render latency;
- whether Google Drive remains the storage backend or will be replaced by internal object storage/NVR;
- privacy policy for evidence export and access logs;
- backup/restore requirements for PostgreSQL and `/storage`.
