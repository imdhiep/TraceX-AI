# TraceX-AI

> Hệ thống tìm kiếm và truy vết người trong mạng lưới camera bằng mô tả tự nhiên, dữ liệu tracklet, AI embedding và luồng xác nhận của người vận hành.

TraceX-AI hiện được tổ chức theo hướng **frontend tách riêng khỏi backend GPU**:

- **VPS/Coolify** chạy frontend Next.js.
- **LightningAI** chạy toàn bộ backend: PostgreSQL, metadata-service, query-service và trace-service.
- Video và artefact xử lý được lưu trong `/workspace/storage` hoặc Google Drive, còn database chỉ lưu metadata, tracklet, embedding, lịch sử query, candidate và evidence.

## Liên kết nhanh

| Tài liệu | Mục đích |
| -------- | -------- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Kiến trúc hệ thống, luồng dữ liệu, service responsibilities |
| [DEPLOY.md](DEPLOY.md) | Hướng dẫn deploy frontend trên VPS và backend trên LightningAI |
| [EVALUATION_EVIDENCE.md](EVALUATION_EVIDENCE.md) | Minh chứng đánh giá, benchmark nội bộ, test evidence |
| [WORKLOG.md](WORKLOG.md) | Phân công sprint, task và quyết định kỹ thuật |
| [JOURNAL.md](JOURNAL.md) | Nhật ký phát triển theo tuần |
| [docs/RULES_USER_ACCOUNT.md](docs/RULES_USER_ACCOUNT.md) | Quy tắc tài khoản và phân quyền người dùng |

---

## Mục lục

1. [Tính năng chính](#tính-năng-chính)
2. [Kiến trúc đang chạy](#kiến-trúc-đang-chạy)
3. [Luồng sản phẩm](#luồng-sản-phẩm)
4. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
5. [Tech stack](#tech-stack)
6. [AI pipeline](#ai-pipeline)
7. [Database](#database)
8. [API chính](#api-chính)
9. [Hướng dẫn sử dụng sản phẩm](#hướng-dẫn-sử-dụng-sản-phẩm)
10. [Cài đặt và chạy](#cài-đặt-và-chạy)
11. [Deploy](#deploy)
12. [Biến môi trường quan trọng](#biến-môi-trường-quan-trọng)
13. [Troubleshooting](#troubleshooting)
14. [Ghi chú bảo mật](#ghi-chú-bảo-mật)
15. [Thành viên](#thành-viên)

---

## Tính năng chính

| Nhóm | Mô tả |
| ---- | ---- |
| Đăng nhập và quản trị | JWT auth, bootstrap admin, quản lý user và lịch sử query |
| Video ingestion | Nhận video local/Drive, xử lý offline thành tracklet và metadata |
| AI video processing | Detect người, tracking, merge tracklet, trích xuất thuộc tính, embedding và action |
| Search | Tìm người bằng text/image query, lọc camera/thời gian, trả về candidate đã rank |
| Candidate review | Xem candidate, preview, mô tả, toàn bộ tracklet, chọn nhiều candidate |
| Trace | Chọn candidate để dựng evidence video/timeline theo cửa sổ thời gian |
| Human-in-the-loop | Người dùng xác nhận, loại tracklet khỏi candidate, xem lại history |
| Tiếng Việt hóa UI | Nhãn, lỗi và thông tin hiển thị được dịch/sửa để phù hợp người vận hành |

---

## Kiến trúc đang chạy

```text
┌──────────────────────────────────────────────────────────────────┐
│  VPS / Coolify — Docker Compose                                  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Frontend (Next.js)                                        │  │
│  │ container: mcpt-frontend                                  │  │
│  │ port 3000 → Traefik / HTTPS                               │  │
│  │ API rewrite: /api-gw → LightningAI metadata-service       │  │
│  └──────────────────────────────┬─────────────────────────────┘  │
└─────────────────────────────────┼────────────────────────────────┘
                                  │ HTTPS /api/v1
                                  │ NEXT_PUBLIC_API_BASE_URL
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  LIGHTNINGAI — Backend + Database (GPU)                          │
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
│                     │ users / videos /        │                 │
│                     │ tracklets / queries /   │                 │
│                     │ candidates / evidence   │                 │
│                     └─────────────────────────┘                 │
│                                                                  │
│  metadata-service GPU models:                                    │
│  ├── RT-DETR R50           ← person detection                    │
│  ├── DINOv2 ViT-L/14       ← appearance / ReID features          │
│  ├── SigLIP / SigLIP2      ← image-text embedding                │
│  ├── VideoMAE V2           ← action recognition                  │
│  └── Qwen2-VL-7B           ← open-vocabulary metadata            │
└──────────────────────────────────────────────────────────────────┘
```

Public frontend calls should point to:

```text
https://8002-<LIGHTNINGAI-WORKSPACE>.cloudspaces.litng.ai/api/v1
```

Frontend `next.config.js` rewrites browser calls through `/api-gw` when `NEXT_PUBLIC_API_BASE_URL` is an external LightningAI URL.

---

## Luồng sản phẩm

### 1. Ingest / xử lý video

```text
Video source (local hoặc Google Drive)
  -> metadata-service /api/v1/video/process hoặc /api/v1/ingest/*
  -> detect người bằng RT-DETR
  -> tracking + merge tracklet
  -> trích xuất crop, thuộc tính, embedding, action
  -> lưu videos, tracklets, embeddings, actions, observations vào PostgreSQL
```

### 2. Search

```text
Người dùng nhập mô tả / ảnh / bộ lọc camera-thời gian
  -> frontend
  -> metadata-service /api/v1/search
  -> query-service /api/v1/candidates/search
  -> tính fusion score từ text, vector, metadata, quality/time
  -> lưu query_history, query_candidates, query_candidate_tracklets
  -> frontend hiển thị candidate
```

### 3. Trace

```text
Người dùng chọn candidate
  -> /api/v1/trace/candidate-detail
  -> /api/v1/trace/select
  -> /api/v1/trace/build
  -> trace-service dựng evidence segments và video/timeline
  -> frontend xem trace, history, candidate tracklets
  -> người dùng có thể xóa tracklet sai hoặc trace tiếp
```

---

## Cấu trúc thư mục

```text
TraceX-AI/
├── frontend/                         # Next.js 14 app
│   ├── app/                           # App Router pages: login, home, history, trace, admin
│   ├── components/                    # Layout, search, video, UI components
│   ├── features/                      # Feature views: auth, home, search, history, trace
│   ├── lib/                           # API client, auth session, config, types
│   ├── next.config.js                 # /api-gw and /static rewrites to LightningAI
│   └── Dockerfile
│
├── backend/
│   ├── config/                        # Camera topology, calibration, query vocab
│   └── services/
│       ├── shared/                    # SQLAlchemy models, DB/session config, time helpers
│       ├── metadata-service/          # Public API: auth, users, videos, ingest, search/trace proxy
│       ├── query-service/             # Search ranking, translation, SigLIP online query encoding
│       ├── trace-service/             # Trace build/status/timeline/evidence clips
│       ├── entrypoint.sh
│       └── lightningai-compose.yml    # Legacy/alternate LightningAI compose
│
├── infra/
│   └── postgres/
│       ├── init/                      # DB init scripts
│       └── migrations/                # Schema migration SQL files
│
├── scripts/                           # Deploy, model download, RT-DETR finetune, AI logging hooks
├── secrets/                           # Local secret templates; real values are gitignored
├── docs/                              # Extra rules/docs
├── camera_0002/                       # Benchmark/debug scripts for detector/tracker tuning
├── docker-compose.yml                 # VPS/Coolify frontend-only deployment
├── docker-compose.lightningai.yml     # Backend + DB deployment on LightningAI
├── docker-compose.backend-included.yml# Older all-in-one VPS stack, not the main deploy path
├── DEPLOY.md                          # Detailed deployment guide
├── JOURNAL.md                         # Weekly progress journal
├── WORKLOG.md                         # Sprint/worklog decisions and tasks
├── move.py                            # Google Drive/local file organization helper
└── ingest_local.py                    # Local ingestion/debug helper
```

---

## Tech stack

| Layer | Công nghệ |
| ----- | --------- |
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Public API | FastAPI metadata-service |
| Search | FastAPI query-service, SigLIP, SeamlessM4T |
| Trace | FastAPI trace-service, ffmpeg/evidence rendering |
| AI video processing | PyTorch, Transformers, RT-DETR, DINOv2, SigLIP, VideoMAE, Qwen2-VL |
| Database | PostgreSQL 16, SQLAlchemy 2, pgvector |
| Deploy | Docker Compose, Coolify VPS, LightningAI |
| Storage | `/workspace/storage`, optional Google Drive OAuth |

---

## AI pipeline

### Metadata/video processing

`metadata-service` warmup các model chính khi startup:

| Model | Vai trò |
| ----- | ------- |
| RT-DETR R50 | Person detection |
| DINOv2 ViT-L/14 | Appearance embedding / ReID features |
| SigLIP / SigLIP2 fallback | Image/text embedding cho search |
| VideoMAE V2 | Action recognition theo tracklet |
| Qwen2-VL-7B-Instruct | Open-vocabulary appearance metadata/caption |

Output được lưu thành:

- `videos`: metadata video, camera, thời điểm ghi hình, đường dẫn storage.
- `tracklets`: người được detect/tracking trong video.
- `tracklets_embeddings`: SigLIP embedding 1152-dim, dùng cho text/image search.
- `tracklets_actions`: action classification.
- `tracklet_observations`: bbox theo frame/timestamp để xem chi tiết tracklet.

### Search/ranking

`query-service` nhận shortlist từ DB và tính điểm bằng nhiều tín hiệu:

- SigLIP text/image similarity.
- Metadata/attribute match từ Qwen.
- Chất lượng tracklet/crop.
- Camera/time filters.
- Merge threshold để gom tracklet/candidate cùng người.

Các tham số đang được tune qua env:

- `MIN_FUSION_SCORE`
- `MAX_CANDIDATES`
- `QUERY_MERGE_THRESHOLD`

### Trace/evidence

`trace-service` xử lý:

- `select`: chọn candidate cho query.
- `candidate-detail`: lấy toàn bộ tracklet của candidate.
- `build`: dựng evidence video/timeline.
- `candidate-tracklet/remove`: loại tracklet sai khỏi candidate.
- `continue`: trace tiếp với window mới.
- `feedback`: ghi nhận xác nhận của người dùng.

---

## Database

Schema chính nằm trong [backend/services/shared/models.py](backend/services/shared/models.py). Nhóm bảng hiện tại:

| Nhóm | Bảng |
| ---- | --- |
| User/auth | `users` |
| Camera/topology | `cameras`, `camera_zones`, `camera_edges`, `camera_settings` |
| Video/tracklet | `videos`, `tracklets`, `tracklets_embeddings`, `tracklets_actions`, `tracklet_observations` |
| Query/candidate | `query_history`, `query_candidates`, `query_candidate_tracklets`, `query_jobs`, `spatiotemporal_groups` |
| Evidence/feedback | `evidence_videos`, `evidence_tracklets`, `verified_objects`, `verified_objects_tracklets` |
| Internal queue | `queue_video_assets` |

Migrations hiện có:

```text
infra/postgres/migrations/
├── 2026-05-12-add-tracklet-observations.sql
├── 2026-05-12-refactor-tracklets-schema.sql
└── 2026-05-13-add-video-drive-file-id.sql
```

---

## API chính

Frontend gọi qua metadata-service public prefix `/api/v1`.

| API | Mục đích |
| --- | ------- |
| `POST /api/v1/auth/login` | Đăng nhập |
| `GET /api/v1/auth/me` | Lấy user hiện tại |
| `GET /api/v1/users` | Quản lý user |
| `GET /api/v1/videos` | Danh sách video |
| `POST /api/v1/video/process` | Xử lý một video |
| `POST /api/v1/video/process/stream` | Xử lý video dạng stream |
| `POST /api/v1/video/batch/process` | Xử lý batch video |
| `POST /api/v1/ingest/move-and-process` | Move/ingest rồi process |
| `POST /api/v1/ingest/full-pipeline` | Chạy full ingest pipeline |
| `POST /api/v1/search` | Tìm candidate |
| `GET /api/v1/search/overview` | Overview metrics |
| `GET /api/v1/candidates/{candidate_id}/preview` | Preview image candidate |
| `GET /api/v1/history` | Lịch sử query |
| `GET /api/v1/history/{query_id}/candidates` | Candidate trong một query lịch sử |
| `GET /api/v1/history/{query_id}/evidence` | Evidence video trong history |
| `POST /api/v1/trace/candidate-detail` | Chi tiết candidate và tracklets |
| `POST /api/v1/trace/select` | Chọn candidate |
| `POST /api/v1/trace/build` | Build trace evidence |
| `GET /api/v1/trace/status/{evidence_id}` | Trạng thái evidence |
| `GET /api/v1/trace/timeline/{evidence_id}` | Timeline evidence |
| `POST /api/v1/trace/candidate-tracklet/remove` | Xóa tracklet khỏi candidate |

Health checks:

```bash
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
```

---

## Hướng dẫn sử dụng sản phẩm

Trong sản phẩm đã có video hướng dẫn tại trang **Hướng dẫn**. Sau khi đăng nhập, người dùng có thể mở mục **Hướng dẫn** trên sidebar để xem cách thao tác các luồng chính:

- tìm kiếm bằng mô tả hoặc ảnh;
- lọc theo camera và thời gian;
- xem candidate và toàn bộ tracklet;
- chọn candidate để trace;
- xem evidence video/timeline;
- xem lại lịch sử query và candidate.

---

## Cài đặt và chạy

### Yêu cầu

| Thành phần | Gợi ý |
| ---------- | ----- |
| Node.js | >= 18.17 |
| Python | 3.10+ / 3.11 tùy service |
| Docker | Docker Engine + Compose v2 |
| GPU | A100 80GB trên LightningAI cho full pipeline |
| Storage | `/workspace/storage`, `/workspace/models`, optional Google Drive secrets |

### Cài frontend local

```bash
cd frontend
cp .env.example .env.local
# sửa NEXT_PUBLIC_API_BASE_URL trỏ tới metadata-service:
# NEXT_PUBLIC_API_BASE_URL=http://localhost:8002/api/v1
# hoặc https://8002-<workspace>.cloudspaces.litng.ai/api/v1
npm install
npm run dev
```

Frontend dev server chạy tại:

```text
http://localhost:4000
```

### Chạy backend bằng Docker Compose trên LightningAI

```bash
cp .env.lightningai.example .env.lightningai
# sửa POSTGRES_PASSWORD, DATABASE_URL, JWT_SECRET_KEY, LIGHTNINGAI_PUBLIC_URL

mkdir -p /workspace/storage/videos
mkdir -p /workspace/storage/traces
mkdir -p /workspace/storage/queue
mkdir -p /workspace/storage/candidate-previews
mkdir -p /workspace/storage/cache
mkdir -p /workspace/models/huggingface
mkdir -p /workspace/models/torch

docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai up -d --build
```

Xem log:

```bash
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai logs -f metadata-service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai logs -f query-service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai logs -f trace-service
```

---

## Deploy

Chi tiết đầy đủ nằm ở [DEPLOY.md](DEPLOY.md). Tóm tắt deploy hiện tại:

### 1. LightningAI: backend + database

```bash
cp .env.lightningai.example .env.lightningai
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai build
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai up -d
```

Public API chính là metadata-service:

```text
https://8002-<WORKSPACE-ID>.cloudspaces.litng.ai/api/v1
```

### 2. VPS/Coolify: frontend only

Tạo `.env` trên VPS:

```env
NEXT_PUBLIC_API_BASE_URL=https://8002-<WORKSPACE-ID>.cloudspaces.litng.ai/api/v1
```

Build và chạy:

```bash
docker compose --env-file .env build frontend
docker compose --env-file .env up -d frontend
```

Khi LightningAI URL đổi, phải cập nhật `NEXT_PUBLIC_API_BASE_URL` và rebuild frontend vì URL được dùng lúc build.

---

## Biến môi trường quan trọng

### `.env.lightningai`

| Biến | Mô tả |
| ---- | ---- |
| `LIGHTNINGAI_PUBLIC_URL` | URL public của metadata-service port 8002 |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | Thông tin PostgreSQL |
| `DATABASE_URL` | SQLAlchemy URL, ví dụ `postgresql+psycopg2://...` |
| `JWT_SECRET_KEY` | Key ký JWT, cần đủ mạnh |
| `BOOTSTRAP_ADMIN_EMAIL/PASSWORD/FULL_NAME` | Admin tạo tự động khi startup |
| `STORAGE_BASE_URL` | Base URL cho trace/evidence video |
| `GOOGLE_DRIVE_ENABLED` | Bật/tắt Google Drive ingestion |
| `STORAGE_INGEST_ENABLED` | Bật/tắt auto ingest từ storage |
| `MIN_FUSION_SCORE` | Ngưỡng điểm search |
| `MAX_CANDIDATES` | Số candidate tối đa |
| `QUERY_MERGE_THRESHOLD` | Ngưỡng merge candidate/identity ở query-service |
| `PIPELINE_SAMPLE_FPS` | FPS sampling khi xử lý video |
| `QWEN2VL_MODEL_ID` | Override path/model Qwen2-VL |

### Frontend

| Biến | Mô tả |
| ---- | ---- |
| `NEXT_PUBLIC_API_BASE_URL` | URL metadata-service kèm `/api/v1`; frontend rewrites qua `/api-gw` |

---

## Troubleshooting

### Frontend không gọi được API

Kiểm tra:

```bash
cat .env
docker compose logs -f frontend
curl https://8002-<WORKSPACE-ID>.cloudspaces.litng.ai/health
```

Nếu URL LightningAI mới, cập nhật `.env` rồi rebuild frontend.

### Backend startup lâu

`metadata-service` và `query-service` warmup model GPU khi khởi động. Lần đầu có thể mất vài phút, đặc biệt nếu model cache chưa có.

```bash
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai logs -f metadata-service
nvidia-smi
```

### Search không ra candidate

Kiểm tra DB đã có tracklet chưa:

```sql
SELECT COUNT(*) FROM videos;
SELECT COUNT(*) FROM tracklets;
SELECT COUNT(*) FROM tracklets_embeddings;
```

Sau đó kiểm tra `MIN_FUSION_SCORE`, `QUERY_MERGE_THRESHOLD`, camera filter và time filter.

### Trace không dựng được video

Kiểm tra:

- `trace-service` health.
- File video gốc còn tồn tại trong storage/Drive.
- `time_window_start/time_window_end` và timezone.
- Logs của `trace-service` và static path `/static/traces`.

### PostgreSQL lỗi sau restart LightningAI

Repo có helper:

```bash
bash scripts/fix-pgdata.sh
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai up -d
```

---

## Ghi chú bảo mật

- Không commit `.env`, `.env.lightningai`, token OAuth, HF token hoặc file thật trong `secrets/`.
- `.ai-log/*.jsonl` được gitignore; prompt logging chạy tự động qua hooks.
- Trước khi tạo PR, đảm bảo đã chạy:

```bash
bash scripts/setup_hooks.sh
```

- Không paste token thật vào issue, commit, README hoặc log public.
- PostgreSQL chỉ nên nằm trong Docker network nội bộ, không expose public.

---

## Tài liệu liên quan

- [DEPLOY.md](DEPLOY.md): hướng dẫn deploy chi tiết.
- [WORKLOG.md](WORKLOG.md): phân công sprint và quyết định kỹ thuật.
- [JOURNAL.md](JOURNAL.md): nhật ký phát triển theo tuần.
- [docs/RULES_USER_ACCOUNT.md](docs/RULES_USER_ACCOUNT.md): quy tắc tài khoản người dùng.

---

## Thành viên

| Tên | MSSV | Vai trò |
| --- | --- | --- |
| Dương Văn Hiệp | 2A202600052 | AI Engineer, Backend, Data Pipeline |
| Bùi Văn Đạt | 2A202600355 | Backend, Frontend |
| Cao Diệu Ly | 2A202600356 | Leader, PM, AI Research |
