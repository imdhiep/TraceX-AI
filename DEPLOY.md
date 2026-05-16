# TraceX-AI — Hướng dẫn Deploy

## Kiến trúc tổng quan

```
Browser
  │
  ├──► VPS (tracex-ai.smartnovi.tech)
  │      └─ frontend  [Next.js :3000 → Traefik → HTTPS :443]
  │
  └──► LightningAI (8002-XXXX.cloudspaces.litng.ai)
         ├─ metadata-service  [FastAPI :8002]  ← frontend gọi vào đây
         ├─ query-service     [FastAPI :8003]  ← metadata-service gọi nội bộ
         ├─ trace-service     [FastAPI :8004]  ← metadata-service gọi nội bộ
         └─ postgres          [:5432, KHÔNG expose ra ngoài]
```

**Quy tắc cứng:**
- VPS chỉ chạy `frontend` — KHÔNG database, KHÔNG backend.
- LightningAI chạy tất cả backend và database.
- Video lưu Google Drive (tùy chọn) hoặc `/workspace/storage` — KHÔNG lưu trong database.

---

## Các file deploy

| File | Máy | Mục đích |
|------|-----|----------|
| `docker-compose.yml` | VPS | Chạy `frontend` |
| `docker-compose.lightningai.yml` | LightningAI | Chạy `postgres + metadata-service + query-service + trace-service` |
| `.env` (copy từ `.env.vps.example`) | VPS | Env vars cho frontend |
| `.env.lightningai` (copy từ `.env.lightningai.example`) | LightningAI | Env vars cho backend |

---

## Phần 1 — Deploy LightningAI (Backend + Database)

### 1.1 Chuẩn bị thư mục

```bash
# Trên terminal LightningAI
cd /home/zeus/content/TraceX-AI

# Các path dưới đây là host bind mount được docker-compose.lightningai.yml dùng.
mkdir -p storage/pgdata
mkdir -p storage/videos
mkdir -p storage/traces
mkdir -p storage/queue
mkdir -p storage/candidate-previews
mkdir -p storage/cache
mkdir -p storage/model-weights
mkdir -p secrets

# Cache model dùng trực tiếp từ home của LightningAI.
mkdir -p /home/zeus/.cache/huggingface
mkdir -p /home/zeus/.cache/torch
```

### 1.2 Tạo .env.lightningai

```bash
cd /home/zeus/content/TraceX-AI
cp .env.lightningai.example .env.lightningai
```

Mở `.env.lightningai` và điền các giá trị bắt buộc:

| Biến | Mô tả |
|------|--------|
| `LIGHTNINGAI_PUBLIC_URL` | URL cổng 8002: `https://8002-XXXX.cloudspaces.litng.ai` |
| `POSTGRES_PASSWORD` | Mật khẩu mạnh, không dùng `@ # %` |
| `DATABASE_URL` | `postgresql+psycopg2://mcpt_user:PASSWORD@postgres:5432/mcpt` |
| `JWT_SECRET_KEY` | Tạo bằng `openssl rand -base64 64` |
| `BOOTSTRAP_ADMIN_EMAIL` | Email admin đầu tiên |
| `BOOTSTRAP_ADMIN_PASSWORD` | Mật khẩu admin đầu tiên |

```bash
# Tạo JWT key
openssl rand -base64 64
```

### 1.3 Build Docker images

```bash
# Build tất cả (lần đầu mất 10–30 phút)
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai build

# Hoặc build từng service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai build metadata-service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai build query-service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai build trace-service
```

### 1.4 Khởi chạy

```bash
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai up -d
```

### 1.5 Kiểm tra logs

```bash
# Xem tất cả
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai logs -f

# Xem từng service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai logs -f metadata-service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai logs -f query-service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai logs -f trace-service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai logs -f postgres
```

### 1.6 Test health check

```bash
# Cục bộ trên LightningAI (GPU model warmup ~2 phút)
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health

# Từ bên ngoài
curl https://8002-YOUR-WORKSPACE-ID.cloudspaces.litng.ai/health
```

Kết quả mong đợi: HTTP 200, `{"status": "healthy", ...}`

### 1.7 Test kết nối database

```bash
# Kết nối vào postgres container
docker exec -it tracex-postgres psql -U mcpt_user -d mcpt

# Trong psql
\dt           # Xem danh sách tables (~18 tables)
\q            # Thoát
```

### 1.8 Ghi lại URL public

```bash
# URL này sẽ dùng cho NEXT_PUBLIC_API_BASE_URL trên VPS.
# Lấy URL public của port 8002 trong LightningAI UI, rồi thêm /api/v1.
# Ví dụ:
LIGHTNINGAI_API_URL="https://8002-YOUR-WORKSPACE-ID.cloudspaces.litng.ai/api/v1"
echo "$LIGHTNINGAI_API_URL"
```

---

## Phần 2 — Deploy VPS (Frontend Only)

### 2.1 Tạo .env

```bash
cd /path/to/TraceX-AI
cp .env.vps.example .env
```

Điền URL LightningAI vừa lấy ở bước 1.8:

```env
NEXT_PUBLIC_API_BASE_URL=https://8002-YOUR-WORKSPACE-ID.cloudspaces.litng.ai/api/v1
```

### 2.2 Build và chạy

```bash
# Build (Next.js bake URL vào bundle tại bước này)
docker compose --env-file .env build frontend

# Chạy
docker compose --env-file .env up -d frontend
```

### 2.3 Kiểm tra

```bash
docker compose logs -f frontend
curl http://localhost:3000
```

---

## Phần 3 — Test End-to-End

1. Mở `https://tracex-ai.smartnovi.tech` trong trình duyệt
2. Đăng nhập bằng `BOOTSTRAP_ADMIN_EMAIL` / `BOOTSTRAP_ADMIN_PASSWORD`
3. Dashboard load thành công
4. Kiểm tra Network tab trong DevTools — browser gọi same-origin `/api-gw/...`; Next.js proxy sang `8002-XXXX.cloudspaces.litng.ai`

```bash
# Test login API trực tiếp
curl -X POST https://8002-YOUR-WORKSPACE-ID.cloudspaces.litng.ai/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@tracex.example.com","password":"Admin@123456"}'
```

---

## Phần 4 — Bảo trì

### Restart service

```bash
# LightningAI
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai restart metadata-service

# VPS
docker compose restart frontend
```

### Cập nhật khi code thay đổi

**Backend (LightningAI):**
```bash
git pull
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai build metadata-service
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai up -d metadata-service
```

**Frontend (VPS) — bắt buộc rebuild:**
```bash
git pull
docker compose --env-file .env build frontend
docker compose --env-file .env up -d frontend
```

### Khi LightningAI URL thay đổi

LightningAI URL thay đổi mỗi khi workspace restart:

```bash
# 1. Lấy URL mới
NEW_URL="https://8002-NEW-ID.cloudspaces.litng.ai/api/v1"

# 2. Cập nhật .env trên VPS
sed -i "s|NEXT_PUBLIC_API_BASE_URL=.*|NEXT_PUBLIC_API_BASE_URL=$NEW_URL|" .env

# 3. Rebuild frontend (bắt buộc — URL được baked vào JS bundle)
docker compose --env-file .env build frontend
docker compose --env-file .env up -d frontend
```

### Dừng toàn bộ

```bash
# LightningAI
docker compose -f docker-compose.lightningai.yml --env-file .env.lightningai down

# VPS
docker compose down
```

### Xem resource GPU

```bash
docker stats tracex-metadata-service tracex-query-service tracex-trace-service
nvidia-smi
```

---

## Tham khảo nhanh — Port

| Service | Port | Máy | Expose công khai |
|---------|------|-----|-----------------|
| frontend | 3000 | VPS | Qua Traefik → HTTPS 443 |
| metadata-service | 8002 | LightningAI | ✅ (frontend gọi) |
| query-service | 8003 | LightningAI | Chỉ nội bộ Docker |
| trace-service | 8004 | LightningAI | Chỉ nội bộ Docker |
| postgres | 5432 | LightningAI | ❌ KHÔNG expose |

---

## Bảo mật

- Không commit `.env` và `.env.lightningai` vào git (đã có trong `.gitignore`)
- Dùng mật khẩu mạnh cho `POSTGRES_PASSWORD` và `JWT_SECRET_KEY`
- Port 5432 của postgres chỉ accessible trong Docker network nội bộ
- Port 8003 và 8004 chỉ cần accessible từ nội bộ LightningAI (metadata-service gọi)
