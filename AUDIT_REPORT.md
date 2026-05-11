# TraceX-AI — Technical Audit Report
**Ngày:** 2026-05-11  
**Auditor:** Claude Sonnet 4.6 (automated)  
**Repo:** /home/zeus/content/TraceX-AI

> **Quyết định 2026-05-11: DB sẽ được wipe toàn bộ.**  
> Mọi concern về "old data", "backfill", "backward compat old tracklets", "chạy SQL migration" đều **không còn áp dụng**.  
> Schema mới sẽ được tạo từ đầu bởi `create_all()` — sạch hoàn toàn.  
> Code migration cũ (`queue_service._upsert_person_candidates`, `load_queue_video_metadata` JSON path) trở thành dead code cần xóa.

> **Cập nhật 2026-05-11 (sau cleanup code): BEV đã bị loại bỏ khỏi code hiện tại.**  
> `bev_x/bev_y`, BEV projection stage, tham số `bev_max_dist`, và logic ranking phụ thuộc BEV đã được xóa khỏi active code path.  
> Batch video pipeline không còn làm raw cross-camera association dựa trên BEV; hiện batch endpoint chỉ aggregate các tracklet per-video đã xử lý xong.  
> Các đoạn bên dưới mô tả BEV cần được hiểu là **historical snapshot trước cleanup này**, không còn phản ánh trạng thái code hiện tại.

---

## PHẦN 1 — DATABASE OVERVIEW

### DBMS và ORM

- **DBMS:** PostgreSQL
- **ORM:** SQLAlchemy (ORM hiện đại với `DeclarativeBase`, `mapped_column`, `Mapped`) — `sqlalchemy>=2.0`
- **pgvector:** **CÓ** — import có guard `try/except ImportError`:
  ```python
  # shared/models.py:49-54
  try:
      from pgvector.sqlalchemy import Vector as PgVector
      _PGVECTOR_AVAILABLE = True
  except ImportError:
      PgVector = None
      _PGVECTOR_AVAILABLE = False
  ```
  Nếu `pgvector` không cài được, field vector fallback về `JSON`.
  - Field `siglip_embedding` (SigLIP2 1152-dim): `PgVector(1152)` hoặc `JSON` — `shared/models.py` *(field duy nhất còn lại sau cleanup 2026-05-11)*

- **Alembic/Migration:** **KHÔNG** — không có thư mục `alembic/` hay file `alembic.ini` trong repo. Schema được tạo bằng `SharedBase.metadata.create_all(bind=engine)` mỗi lần khởi động FastAPI (idempotent, chỉ tạo bảng mới, không drop bảng cũ). Thay đổi schema sẽ **không tự động áp dụng** nếu bảng đã tồn tại — phải chạy ALTER TABLE thủ công hoặc drop-and-recreate.
  - Evidence: `metadata-service/app/main.py:43-44`: `SharedBase.metadata.create_all(bind=engine)`

- **File định nghĩa model chính:** `/home/zeus/content/TraceX-AI/backend/services/shared/models.py`

---

### Danh sách bảng

#### Table: `users`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:73 |
| email | String(255) | NO | — | — | — | YES (uq_users_email) | — | Email đăng nhập | models.py:74 |
| full_name | String(255) | NO | — | — | — | — | — | Tên hiển thị | models.py:75 |
| hashed_password | String(255) | NO | — | — | — | — | — | Bcrypt hash | models.py:76 |
| role | String(20) | NO | "USER" | — | — | — | — | "USER" hoặc "ADMIN" | models.py:77 |
| is_active | Boolean | NO | true | — | — | — | — | Tài khoản hoạt động | models.py:78 |
| last_login | DateTime(tz) | YES | NULL | — | — | — | — | Lần đăng nhập cuối | models.py:79 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:80-82 |
| updated_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm cập nhật | models.py:83-88 |

#### Table: `cameras`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:109 |
| camera_id | String(50) | NO | — | — | — | YES (uq_cameras_camera_id) | — | Định danh camera (cam_01) | models.py:110 |
| name | String(255) | NO | — | — | — | — | — | Tên hiển thị | models.py:111 |
| location | String(255) | YES | NULL | — | — | — | — | Vị trí | models.py:112 |
| fps | Float | NO | 30.0 | — | — | — | — | FPS camera | models.py:113 |
| resolution_width | Integer | NO | 1920 | — | — | — | — | Chiều rộng | models.py:114 |
| resolution_height | Integer | NO | 1080 | — | — | — | — | Chiều cao | models.py:115 |
| is_active | Boolean | NO | true | — | — | — | — | Camera đang hoạt động | models.py:116 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:117-119 |

#### Table: `camera_zones`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:133 |
| camera_id | String(50) | NO | — | — | cameras.camera_id CASCADE | YES (uq_camera_zones_camera_zone) | — | FK tới cameras | models.py:134-136 |
| zone_type | String(32) | NO | — | — | — | YES (composite) | — | "entry" hoặc "exit" | models.py:137 |
| polygon | JSON | NO | — | — | — | — | — | [{x,y},...] điểm polygon | models.py:138 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:139-141 |

#### Table: `camera_edges`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:159 |
| from_camera_id | String(50) | NO | — | — | — | YES (pair) | YES (ix_camera_edges_from) | Camera nguồn | models.py:160 |
| to_camera_id | String(50) | NO | — | — | — | YES (pair) | YES (ix_camera_edges_to) | Camera đích | models.py:161 |
| min_seconds | Float | NO | 30.0 | — | — | — | — | Thời gian chuyển tiếp tối thiểu (giây) | models.py:162 |
| max_seconds | Float | NO | 120.0 | — | — | — | — | Thời gian chuyển tiếp tối đa (giây) | models.py:163 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:164-166 |

**Lưu ý:** `camera_edges` KHÔNG có FK về `cameras` — chỉ lưu camera_id dạng String. Nếu camera bị xóa, edge sẽ trở thành orphan.

#### Table: `camera_settings`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:176 |
| camera_id | String(50) | NO | — | — | cameras.camera_id CASCADE | YES (uq_camera_settings_camera_key) | — | FK tới cameras | models.py:177-179 |
| setting_key | String(128) | NO | — | — | — | YES (composite) | — | Tên cấu hình | models.py:180 |
| setting_value | String(512) | NO | — | — | — | — | — | Giá trị cấu hình | models.py:181 |
| updated_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm cập nhật | models.py:182-184 |

#### Table: `videos`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:201 |
| video_id | String(255) | NO | uuid4() | — | — | YES (uq_videos_video_id) | — | UUID định danh | models.py:202-204 |
| camera_id | String(50) | YES | NULL | — | — | — | YES (ix_videos_camera_id) | Camera ghi hình | models.py:205 |
| user_id | Integer | YES | NULL | — | users.id SET NULL | — | — | Người upload | models.py:206-208 |
| title | String(255) | NO | — | — | — | — | — | Tiêu đề video | models.py:209 |
| description | Text | YES | NULL | — | — | — | — | Mô tả | models.py:210 |
| storage_path | String(2048) | NO | — | — | — | — | — | Đường dẫn lưu trữ | models.py:211 |
| storage_backend | String(64) | NO | "local_volume" | — | — | — | — | "google_drive" hoặc "local_volume" | models.py:212 |
| source_filename | String(255) | YES | NULL | — | — | — | — | Tên file gốc | models.py:213 |
| content_type | String(255) | YES | NULL | — | — | — | — | MIME type | models.py:214 |
| duration_seconds | Float | YES | NULL | — | — | — | — | Thời lượng video | models.py:215 |
| fps | Float | YES | NULL | — | — | — | — | FPS video | models.py:216 |
| width | Integer | YES | NULL | — | — | — | — | Chiều rộng | models.py:217 |
| height | Integer | YES | NULL | — | — | — | — | Chiều cao | models.py:218 |
| processed | Boolean | NO | false | — | — | — | — | Đã xử lý AI chưa | models.py:219 |
| recorded_at | DateTime(tz) | YES | NULL | — | — | — | — | Thời điểm ghi hình thực tế | models.py:220-222 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:223-225 |
| updated_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm cập nhật | models.py:226-231 |

#### Table: `tracklets`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:259 |
| tracklet_id | String(255) | NO | — | — | — | YES | — | UUID định danh tracklet | models.py:260 |
| video_id | String(255) | NO | — | — | videos.video_id CASCADE | — | YES (ix_tracklets_video_id) | FK tới videos | models.py:261-263 |
| camera_id | String(50) | NO | — | — | — | — | YES (ix_tracklets_camera_id) | Camera phát hiện | models.py:264 |
| track_id | String(50) | NO | — | — | — | — | — | ID trong video | models.py:265 |
| start_time | Float | NO | 0.0 | — | — | — | — | Giây bắt đầu trong video | models.py:268 |
| end_time | Float | NO | 0.0 | — | — | — | — | Giây kết thúc trong video | models.py:269 |
| quality_score | Float | NO | 0.0 | — | — | — | — | Điểm chất lượng tracklet | models.py:272 |
| occlusion_score | Float | NO | 0.0 | — | — | — | — | Mức độ che khuất | models.py:273 |
| gender_conf | Float | YES | NULL | — | — | — | — | Confidence VLM cho gender | models.py |
| ~~top_color_conf~~ | ~~Float~~ | — | — | — | — | — | — | ~~Đã xóa 2026-05-11~~ (duplicate của upper_clothing_conf) | migration |
| ~~bottom_color_conf~~ | ~~Float~~ | — | — | — | — | — | — | ~~Đã xóa 2026-05-11~~ (duplicate của lower_clothing_conf) | migration |
| shoes_conf | Float | YES | NULL | — | — | — | — | Confidence giày | models.py |
| accessory_conf | Float | YES | NULL | — | — | — | — | Confidence phụ kiện | models.py:280 |
| age_range_conf | Float | YES | NULL | — | — | — | — | Confidence tuổi | models.py:281 |
| hat_color_conf | Float | YES | NULL | — | — | — | — | Confidence màu mũ | models.py:282 |
| bag_type_conf | Float | YES | NULL | — | — | — | — | Confidence loại túi | models.py:283 |
| mask_conf | Float | YES | NULL | — | — | — | — | Confidence đeo khẩu trang | models.py:284 |
| hair_style_conf | Float | YES | NULL | — | — | — | — | Confidence kiểu tóc | models.py:285 |
| hair_color_conf | Float | YES | NULL | — | — | — | — | Confidence màu tóc | models.py:286 |
| gender | String(32) | NO | "unknown" | — | — | — | — | "man"/"woman"/"unknown" | models.py |
| age_range | String(32) | NO | "unknown" | — | — | — | — | Nhóm tuổi | models.py |
| ~~top_color~~ | ~~String(64)~~ | — | — | — | — | — | — | ~~Đã xóa 2026-05-11~~ (dùng upper_clothing_color) | migration |
| ~~bottom_color~~ | ~~String(64)~~ | — | — | — | — | — | — | ~~Đã xóa 2026-05-11~~ (dùng lower_clothing_color) | migration |
| shoes_color | String(64) | NO | "unknown" | — | — | — | — | Màu giày | models.py |
| hat_color | String(64) | NO | "unknown" | — | — | — | — | Màu mũ | models.py:294 |
| bag_type | String(64) | NO | "unknown" | — | — | — | — | Loại túi | models.py:295 |
| is_wearing_mask | String(16) | NO | "unknown" | — | — | — | — | Đeo khẩu trang không | models.py:296 |
| hair_style | String(64) | NO | "unknown" | — | — | — | — | Kiểu tóc | models.py:297 |
| hair_color | String(64) | NO | "unknown" | — | — | — | — | Màu tóc | models.py:298 |
| appearance_summary | Text | NO | "" | — | — | — | — | Mô tả ngắn toàn bộ ngoại hình | models.py:299 |
| upper_clothing_desc | Text | YES | NULL | — | — | — | — | Mô tả tự do áo | models.py:302 |
| upper_clothing_color | String(64) | YES | NULL | — | — | — | YES (ix_tracklets_upper_color) | Màu áo (VLM open-vocab) | models.py:303 |
| upper_clothing_type | String(128) | YES | NULL | — | — | — | YES (ix_tracklets_upper_type) | Loại áo (VLM open-vocab) | models.py:304 |
| upper_clothing_conf | Float | YES | NULL | — | — | — | — | Confidence áo | models.py:305 |
| lower_clothing_desc | Text | YES | NULL | — | — | — | — | Mô tả tự do quần | models.py:307 |
| lower_clothing_color | String(64) | YES | NULL | — | — | — | YES (ix_tracklets_lower_color) | Màu quần (VLM open-vocab) | models.py:308 |
| lower_clothing_type | String(128) | YES | NULL | — | — | — | YES (ix_tracklets_lower_type) | Loại quần (VLM open-vocab) | models.py:309 |
| lower_clothing_conf | Float | YES | NULL | — | — | — | — | Confidence quần | models.py:310 |
| shoes_desc | Text | YES | NULL | — | — | — | — | Mô tả tự do giày | models.py:312 |
| shoes_type | String(128) | YES | NULL | — | — | — | — | Loại giày | models.py:313 |
| bag_desc | Text | YES | NULL | — | — | — | — | Mô tả tự do túi | models.py:315 |
| bag_presence | String(16) | YES | NULL | — | — | — | YES (ix_tracklets_bag_presence) | "yes"/"no"/"unknown" | models.py:316 |
| bag_conf | Float | YES | NULL | — | — | — | — | Confidence túi | models.py:317 |
| hat_desc | Text | YES | NULL | — | — | — | — | Mô tả tự do mũ | models.py:319 |
| hat_presence | String(16) | YES | NULL | — | — | — | YES (ix_tracklets_hat_presence) | "yes"/"no"/"unknown" | models.py:320 |
| hat_type | String(128) | YES | NULL | — | — | — | — | Loại mũ | models.py:321 |
| hat_conf | Float | YES | NULL | — | — | — | — | Confidence mũ | models.py:322 |
| bev_x | Float | NO | 0.0 | — | — | — | YES (ix_tracklets_bev_xy) | Tọa độ BEV X (mét) | models.py:325 |
| bev_y | Float | NO | 0.0 | — | — | — | YES (ix_tracklets_bev_xy) | Tọa độ BEV Y (mét) | models.py:326 |
| crop_url | String(2048) | NO | "" | — | — | — | — | URL ảnh crop đại diện | models.py:329 |
| representative_bbox | JSON | NO | [] | — | — | — | — | [x1,y1,x2,y2] bbox đại diện | models.py:330 |
| contributing_cameras | JSON | NO | [] | — | — | — | — | Danh sách camera đóng góp (cross-camera) | models.py:333 |
| contributing_video_ids | JSON | NO | [] | — | — | — | — | Danh sách video đóng góp | models.py:334 |
| batch_id | String(255) | YES | NULL | — | — | — | — | ID batch xử lý | models.py:335 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:336-338 |
| updated_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm cập nhật | models.py:339-344 |

#### Table: `tracklets_embeddings`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:376 |
| tracklet_id | String(255) | NO | — | — | tracklets.tracklet_id CASCADE | YES | — | FK tới tracklets | models.py |
| ~~embedding_vector~~ | ~~JSON~~ | — | — | — | — | — | — | ~~Đã xóa 2026-05-11~~ (DINOv2 JSON fallback) | migration |
| ~~embedding~~ | ~~PgVector(1024)~~ | — | — | — | — | — | — | ~~Đã xóa 2026-05-11~~ (DINOv2 pgvector, luôn NULL) | migration |
| siglip_embedding | PgVector(1152) / JSON | YES | NULL | — | — | — | — | SigLIP2 vector(1152) — primary embedding | models.py |
| model_version | String(128) | NO | "siglip2" | — | — | — | — | Model tạo embedding | models.py |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:389-391 |

#### Table: `tracklets_actions`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:406 |
| tracklet_id | String(255) | NO | — | — | tracklets.tracklet_id CASCADE | YES (uq) | YES (ix_tracklets_actions_action) | FK tới tracklets | models.py:408-410 |
| action_label | String(64) | NO | — | — | — | — | YES | Nhãn hành động đơn giản | models.py:412 |
| kinetics_label | String(128) | YES | NULL | — | — | — | — | Nhãn Kinetics-400 gốc | models.py:413 |
| confidence | Float | NO | 0.0 | — | — | — | — | Confidence VideoMAE | models.py:414 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:415-417 |

#### Table: `query_history`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:435 |
| query_id | String(36) | NO | uuid4() | — | — | YES | — | UUID định danh query | models.py:436-438 |
| user_id | Integer | NO | — | — | users.id CASCADE | — | YES (ix_query_history_user_id) | Người tìm kiếm | models.py:439-441 |
| video_id | String(255) | YES | NULL | — | videos.video_id SET NULL | — | — | Video liên kết (optional) | models.py:442-444 |
| query_text | Text | NO | — | — | — | — | — | Câu truy vấn gốc | models.py:445 |
| status | String(64) | NO | "pending" | — | — | — | — | pending→searching→candidates_found→completed | models.py:446-449 |
| result_count | Integer | YES | NULL | — | — | — | — | Số kết quả | models.py:450 |
| selected_candidate_id | String(36) | YES | NULL | — | — | — | — | Candidate được chọn | models.py:451-453 |
| ai_job_id | String(255) | YES | NULL | — | — | — | — | ID job async (chưa dùng) | models.py:454 |
| query_image_url | Text | YES | NULL | — | — | — | — | URL ảnh query upload | models.py:455 |
| created_at | DateTime(tz) | NO | now() | — | — | — | YES (ix_query_history_created_at) | Thời điểm tạo | models.py:456-458 |
| updated_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm cập nhật | models.py:459-464 |

#### Table: `query_candidates`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:489 |
| candidate_id | String(255) | NO | — | — | — | YES (uq_query_candidates_candidate_id) | YES | UUID định danh candidate | models.py:490-492 |
| query_id | String(36) | NO | — | — | query_history.query_id CASCADE | — | YES | FK tới query | models.py:493-496 |
| fusion_score | Float | NO | 0.0 | — | — | — | — | Điểm tổng hợp | models.py:498 |
| vector_score | Float | YES | NULL | — | — | — | — | Điểm vector similarity | models.py:499 |
| text_score | Float | YES | NULL | — | — | — | — | Điểm text match | models.py:500 |
| spatiotemporal_score | Float | YES | NULL | — | — | — | — | Điểm không-thời gian (chưa dùng) | models.py:501 |
| rank_position | Integer | NO | 0 | — | — | — | YES (composite) | Thứ hạng | models.py:503 |
| is_selected | Boolean | NO | false | — | — | — | — | Candidate được chọn | models.py:505 |
| selected_at | DateTime(tz) | YES | NULL | — | — | — | — | Thời điểm chọn | models.py:506 |
| candidate_key | String(255) | NO | "" | — | — | — | — | Khóa tham chiếu | models.py:507 |
| preview_url | String(2048) | NO | "" | — | — | — | — | URL xem trước | models.py:508 |
| appearance_summary | Text | NO | "" | — | — | — | — | Mô tả ngoại hình (denormalized) | models.py:510 |
| gender | String(32) | NO | "unknown" | — | — | — | — | Giới tính (denormalized) | models.py:511 |
| ~~top_color~~ | ~~String(64)~~ | — | — | — | — | — | — | ~~Đã xóa 2026-05-11~~ | migration |
| ~~bottom_color~~ | ~~String(64)~~ | — | — | — | — | — | — | ~~Đã xóa 2026-05-11~~ | migration |
| bev_x | Float | NO | 0.0 | — | — | — | — | Tọa độ BEV X đại diện | models.py:515 |
| bev_y | Float | NO | 0.0 | — | — | — | — | Tọa độ BEV Y đại diện | models.py:516 |
| primary_camera_id | String(50) | NO | "" | — | — | — | — | Camera chính (denormalized) | models.py:518 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:519-521 |

#### Table: `query_candidate_tracklets`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:551 |
| candidate_id | String(255) | NO | — | — | query_candidates.candidate_id CASCADE | YES (pair) | YES (ix_qct_candidate) | FK tới candidate | models.py:552-555 |
| tracklet_id | String(255) | NO | — | — | tracklets.tracklet_id CASCADE | YES (pair) | YES (ix_qct_tracklet) | FK tới tracklet | models.py:556-559 |
| match_score | Float | NO | 0.0 | — | — | — | — | Điểm match | models.py:560 |
| match_type | String(32) | NO | "vector" | — | — | — | — | "vector"/"text"/"spatiotemporal" | models.py:561 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:562-564 |

#### Table: `query_jobs`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:579 |
| job_id | String(36) | NO | uuid4() | — | — | YES | — | UUID định danh job | models.py:580-582 |
| query_id | String(36) | NO | — | — | query_history.query_id CASCADE | — | YES (ix_query_jobs_status) | FK tới query | models.py:583-586 |
| status | String(32) | NO | "pending" | — | — | — | YES | pending/running/completed/failed | models.py:587-589 |
| worker_id | String(255) | YES | NULL | — | — | — | — | ID worker đang xử lý | models.py:590 |
| started_at | DateTime(tz) | YES | NULL | — | — | — | — | Thời điểm bắt đầu | models.py:591 |
| completed_at | DateTime(tz) | YES | NULL | — | — | — | — | Thời điểm hoàn thành | models.py:592 |
| error_message | Text | YES | NULL | — | — | — | — | Thông báo lỗi | models.py:593 |
| result_summary | JSON | YES | NULL | — | — | — | — | Tóm tắt kết quả | models.py:594 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:595-597 |

#### Table: `spatiotemporal_groups`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:613 |
| group_id | String(36) | NO | uuid4() | — | — | YES | — | UUID định danh | models.py:614-616 |
| query_id | String(36) | NO | — | — | query_history.query_id CASCADE | — | YES (ix_sg_query_id) | FK tới query | models.py:617-620 |
| candidate_id | String(255) | NO | — | — | query_candidates.candidate_id CASCADE | — | YES (ix_sg_candidate_id) | FK tới candidate | models.py:621-624 |
| group_type | String(64) | NO | "camera_transition" | — | — | — | — | "camera_transition"/"temporal_cluster" | models.py:625-628 |
| tracklet_ids | JSON | NO | [] | — | — | — | — | Danh sách tracklet ID theo thứ tự | models.py:629 |
| camera_path | JSON | NO | [] | — | — | — | — | Đường đi qua các camera | models.py:631 |
| total_duration_s | Float | NO | 0.0 | — | — | — | — | Tổng thời gian (giây) | models.py:632 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:633-635 |

#### Table: `evidence_videos`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:653 |
| query_id | String(36) | NO | — | — | query_history.query_id CASCADE | — | YES (ix_evidence_videos_query) | FK tới query | models.py:654-656 |
| query_candidate_id | String(255) | NO | — | — | query_candidates.candidate_id CASCADE | — | YES (ix_evidence_videos_candidate) | FK tới candidate | models.py:657-660 |
| video_url | String(2048) | NO | — | — | — | — | — | URL video bằng chứng (generated, chưa merge thực) | models.py:662 |
| total_duration | Float | NO | 0.0 | — | — | — | — | Tổng thời lượng | models.py:663 |
| segment_count | Integer | NO | 0 | — | — | — | — | Số đoạn | models.py:664 |
| time_window_start | DateTime(tz) | NO | — | — | — | — | — | Cửa sổ thời gian bắt đầu | models.py:665 |
| time_window_end | DateTime(tz) | NO | — | — | — | — | — | Cửa sổ thời gian kết thúc | models.py:666 |
| trace_confidence | Float | NO | 0.0 | — | — | — | — | Độ tin cậy trace | models.py:667 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:668-670 |

#### Table: `evidence_tracklets`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:688 |
| evidence_video_id | Integer | NO | — | — | evidence_videos.id CASCADE | YES (uq_evidence_tracklets_order) | YES (ix_et_evidence_video) | FK tới evidence_videos | models.py:689-691 |
| tracklet_id | String(255) | YES | NULL | — | tracklets.tracklet_id SET NULL | — | YES (ix_et_tracklet) | FK tới tracklets (nullable) | models.py:692-695 |
| segment_order | Integer | NO | — | — | — | YES (composite) | — | Thứ tự đoạn (1-based) | models.py:696 |
| camera_id | String(50) | NO | — | — | — | — | — | Camera của đoạn | models.py:697 |
| time_range | JSON | NO | {} | — | — | — | — | {"start": iso, "end": iso} | models.py:698 |
| video_clip_url | String(2048) | NO | "" | — | — | — | — | URL clip video | models.py:699 |
| thumbnail_url | String(2048) | NO | "" | — | — | — | — | URL ảnh thumbnail | models.py:700 |
| confidence | Float | NO | 0.0 | — | — | — | — | Confidence đoạn | models.py:701 |

#### Table: `verified_objects`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:715 |
| candidate_id | String(255) | NO | — | — | query_candidates.candidate_id CASCADE | — | YES (ix_vo_candidate) | FK tới candidate | models.py:716-718 |
| verified_by_user_id | Integer | YES | NULL | — | users.id SET NULL | — | YES (ix_vo_user) | User xác nhận | models.py:720-722 |
| is_correct | Boolean | NO | — | — | — | — | — | Kết quả đúng/sai | models.py:723 |
| notes | Text | YES | NULL | — | — | — | — | Ghi chú | models.py:724 |
| verified_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm xác nhận | models.py:725-727 |

#### Table: `verified_objects_tracklets`
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:745 |
| verified_object_id | Integer | NO | — | — | verified_objects.id CASCADE | YES (uq_vot_pair) | YES (ix_vot_verified) | FK tới verified_objects | models.py:746-749 |
| tracklet_id | String(255) | YES | NULL | — | tracklets.tracklet_id SET NULL | YES (uq_vot_pair) | YES (ix_vot_tracklet) | FK tới tracklets (nullable) | models.py:750-753 |
| position | Integer | NO | 0 | — | — | — | — | Vị trí trong danh sách | models.py:754 |
| verified_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm xác nhận | models.py:755-757 |

#### Table: `queue_video_assets` (internal, non-spec)
| Column | Type | Nullable | Default | PK | FK | Unique | Index | Business Meaning | Evidence |
|--------|------|----------|---------|----|----|--------|-------|-----------------|----------|
| id | Integer | NO | auto-seq | YES | — | — | — | Primary key | models.py:779 |
| video_id | String(255) | NO | — | — | — | YES | — | Định danh video trong queue | models.py:780 |
| camera_id | String(255) | YES | NULL | — | — | — | — | Camera tương ứng | models.py:781 |
| title | String(255) | NO | — | — | — | — | — | Tiêu đề | models.py:782 |
| source_filename | String(255) | YES | NULL | — | — | — | — | Tên file gốc | models.py:783 |
| source_mode | String(64) | YES | NULL | — | — | — | — | "storage_ingest" hoặc khác | models.py:784 |
| queue_position | Integer | NO | — | — | — | — | — | Vị trí trong hàng đợi | models.py:785 |
| storage_backend | String(64) | NO | "google_drive" | — | — | — | — | Backend lưu trữ | models.py:786 |
| available_link_video | String(2048) | NO | — | — | — | — | — | Link video có thể truy cập | models.py:787 |
| available_link_metadata | String(2048) | YES | NULL | — | — | — | — | Link metadata | models.py:788 |
| drive_video_file_id | String(255) | YES | NULL | — | — | — | — | Google Drive file ID | models.py:789 |
| drive_metadata_file_id | String(255) | YES | NULL | — | — | — | — | Google Drive metadata ID | models.py:790 |
| local_video_path | String(2048) | YES | NULL | — | — | — | — | Đường dẫn local | models.py:791 |
| local_metadata_path | String(2048) | YES | NULL | — | — | — | — | Đường dẫn metadata local | models.py:792 |
| raw_video_metadata | JSON | NO | {} | — | — | — | — | Metadata thô từ nguồn | models.py:793 |
| processed_at | DateTime(tz) | YES | NULL | — | — | — | YES (ix_qva_processed) | Thời điểm đã xử lý | models.py:794 |
| created_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm tạo | models.py:795-797 |
| updated_at | DateTime(tz) | NO | now() | — | — | — | — | Thời điểm cập nhật | models.py:798-804 |

---

## PHẦN 2 — ERD TOÀN BỘ DATABASE

```mermaid
erDiagram
    users {
        int id PK
        string email UK
        string full_name
        string hashed_password
        string role
        bool is_active
        datetime last_login
        datetime created_at
        datetime updated_at
    }
    cameras {
        int id PK
        string camera_id UK
        string name
        string location
        float fps
        int resolution_width
        int resolution_height
        bool is_active
        datetime created_at
    }
    camera_zones {
        int id PK
        string camera_id FK
        string zone_type
        json polygon
        datetime created_at
    }
    camera_edges {
        int id PK
        string from_camera_id
        string to_camera_id
        float min_seconds
        float max_seconds
        datetime created_at
    }
    camera_settings {
        int id PK
        string camera_id FK
        string setting_key
        string setting_value
        datetime updated_at
    }
    videos {
        int id PK
        string video_id UK
        string camera_id
        int user_id FK
        string title
        string storage_path
        string storage_backend
        bool processed
        datetime recorded_at
        datetime created_at
        datetime updated_at
    }
    tracklets {
        int id PK
        string tracklet_id UK
        string video_id FK
        string camera_id
        string track_id
        float start_time
        float end_time
        float quality_score
        float occlusion_score
        string gender
        string age_range
        string appearance_summary
        string upper_clothing_color
        string lower_clothing_color
        string bag_presence
        string hat_presence
        float bev_x
        float bev_y
        string crop_url
        json representative_bbox
        datetime created_at
    }
    tracklets_embeddings {
        int id PK
        string tracklet_id FK_UK
        vector_1152 siglip_embedding
        string model_version
        datetime created_at
    }
    tracklets_actions {
        int id PK
        string tracklet_id FK
        string action_label
        string kinetics_label
        float confidence
        datetime created_at
    }
    query_history {
        int id PK
        string query_id UK
        int user_id FK
        string video_id FK
        text query_text
        string status
        int result_count
        string selected_candidate_id
        text query_image_url
        datetime created_at
        datetime updated_at
    }
    query_candidates {
        int id PK
        string candidate_id UK
        string query_id FK
        float fusion_score
        float vector_score
        float text_score
        int rank_position
        bool is_selected
        string appearance_summary
        string gender
        string primary_camera_id
        datetime created_at
    }
    query_candidate_tracklets {
        int id PK
        string candidate_id FK
        string tracklet_id FK
        float match_score
        string match_type
        datetime created_at
    }
    query_jobs {
        int id PK
        string job_id UK
        string query_id FK
        string status
        text error_message
        json result_summary
        datetime created_at
    }
    spatiotemporal_groups {
        int id PK
        string group_id UK
        string query_id FK
        string candidate_id FK
        string group_type
        json tracklet_ids
        json camera_path
        float total_duration_s
        datetime created_at
    }
    evidence_videos {
        int id PK
        string query_id FK
        string query_candidate_id FK
        string video_url
        float total_duration
        int segment_count
        datetime time_window_start
        datetime time_window_end
        float trace_confidence
        datetime created_at
    }
    evidence_tracklets {
        int id PK
        int evidence_video_id FK
        string tracklet_id FK
        int segment_order
        string camera_id
        json time_range
        string thumbnail_url
        float confidence
    }
    verified_objects {
        int id PK
        string candidate_id FK
        int verified_by_user_id FK
        bool is_correct
        text notes
        datetime verified_at
    }
    verified_objects_tracklets {
        int id PK
        int verified_object_id FK
        string tracklet_id FK
        int position
        datetime verified_at
    }
    queue_video_assets {
        int id PK
        string video_id UK
        string camera_id
        string source_filename
        int queue_position
        string local_video_path
        json raw_video_metadata
        datetime processed_at
        datetime created_at
    }

    users ||--o{ videos : "owns"
    users ||--o{ query_history : "searches"
    users ||--o{ verified_objects : "verifies"
    cameras ||--o{ camera_zones : "has"
    cameras ||--o{ camera_settings : "configured_by"
    videos ||--o{ tracklets : "contains"
    tracklets ||--o| tracklets_embeddings : "has_embedding"
    tracklets ||--o{ tracklets_actions : "has_action"
    tracklets ||--o{ query_candidate_tracklets : "linked_to"
    tracklets ||--o{ evidence_tracklets : "evidenced_in"
    tracklets ||--o{ verified_objects_tracklets : "verified_as"
    query_history ||--o{ query_candidates : "produces"
    query_history ||--o{ query_jobs : "has_jobs"
    query_history ||--o{ spatiotemporal_groups : "has_groups"
    query_candidates ||--o{ query_candidate_tracklets : "contains"
    query_candidates ||--o{ spatiotemporal_groups : "has_groups"
    query_candidates ||--o{ evidence_videos : "evidenced_by"
    query_candidates ||--o| verified_objects : "verified_as"
    evidence_videos ||--o{ evidence_tracklets : "segments"
    verified_objects ||--o{ verified_objects_tracklets : "links"
```

---

## PHẦN 3 — TRACKLETS TABLE — COLUMN DETAIL

| Column | In Model? | In TrackletResult? | AI Model Source | Has Confidence? | Index? | Used by Query? | Used by Frontend? | Can Null? | Compat Mapping? | Evidence |
|--------|-----------|--------------------|-----------------|-----------------|--------|----------------|-------------------|-----------|-----------------|----------|
| tracklet_id | YES | YES | pipeline | NO | YES (UK) | YES | YES | NO | — | models.py:260, schemas:77 |
| video_id | YES | YES | pipeline | NO | YES | YES | YES | NO | — | models.py:261 |
| camera_id | YES | YES | pipeline | NO | YES | YES (filter) | YES | NO | — | models.py:264 |
| track_id | YES | YES | BodyPartAdaptiveTracker | NO | NO | NO | NO | NO | — | models.py:265 |
| start_time | YES | YES | tracker | NO | NO | YES (time filter) | YES | NO | — | models.py:268 |
| end_time | YES | YES | tracker | NO | NO | YES (time filter) | YES | NO | — | models.py:269 |
| quality_score | YES | YES | TrackletQualityScorer | NO | NO | YES (fusion) | YES | NO | — | models.py:272 |
| occlusion_score | YES | YES | pipeline | NO | NO | NO | NO | NO | — | models.py:273 |
| gender | YES | YES | Qwen2-VL-7B | gender_conf | NO | YES (prefilter/merge) | YES | NO | — | models.py:289 |
| age_range | YES | YES | Qwen2-VL-7B | age_range_conf | NO | YES (prefilter) | YES | NO | — | models.py:290 |
| ~~top_color~~ | ~~ĐÃ XÓA~~ | — | — | — | — | — | — | — | Xóa 2026-05-11 | migration |
| ~~bottom_color~~ | ~~ĐÃ XÓA~~ | — | — | — | — | — | — | — | Xóa 2026-05-11 | migration |
| shoes_color | YES | YES | Qwen2-VL-7B | shoes_conf | NO | YES (prefilter) | YES | NO | — | models.py |
| hat_color | YES | YES | Qwen2-VL-7B | hat_color_conf | NO | YES (prefilter) | YES | NO | — | models.py:294 |
| bag_type | YES | YES | Qwen2-VL-7B | bag_type_conf | NO | YES (prefilter) | YES | NO | — | models.py:295 |
| is_wearing_mask | YES | YES | Qwen2-VL-7B | mask_conf | NO | YES (prefilter) | YES | NO | — | models.py:296 |
| hair_style | YES | YES | Qwen2-VL-7B | hair_style_conf | NO | YES (prefilter) | YES | NO | — | models.py:297 |
| hair_color | YES | YES | Qwen2-VL-7B | hair_color_conf | NO | YES (prefilter) | YES | NO | — | models.py:298 |
| appearance_summary | YES | YES | Qwen2-VL-7B | NO | NO | YES (prefilter ILIKE) | YES | NO | — | models.py:299 |
| upper_clothing_desc | YES | YES | Qwen2-VL-7B | NO | NO | NO (free-text, excluded) | chưa xác minh | YES | — | models.py:302 |
| upper_clothing_color | YES | YES | Qwen2-VL-7B | upper_clothing_conf | YES | YES (merge guard) | YES | YES | Primary (top_color đã xóa) | models.py |
| upper_clothing_type | YES | YES | Qwen2-VL-7B | NO | YES | NO (excluded từ merge guard) | YES | YES | — | models.py:304 |
| lower_clothing_color | YES | YES | Qwen2-VL-7B | lower_clothing_conf | YES | YES (merge guard) | YES | YES | Primary (bottom_color đã xóa) | models.py |
| bag_presence | YES | YES | Qwen2-VL-7B | bag_conf | YES | YES (merge guard) | YES | YES | "yes"/"no"/"unknown" | models.py:316 |
| hat_presence | YES | YES | Qwen2-VL-7B | hat_conf | YES | YES (merge guard) | YES | YES | "yes"/"no"/"unknown" | models.py:320 |
| bev_x | YES | YES | BEVProjector | NO | YES (bev_xy) | NO | YES | NO | Default 0.0 nếu calib fail | models.py:325 |
| bev_y | YES | YES | BEVProjector | NO | YES (bev_xy) | NO | YES | NO | Default 0.0 nếu calib fail | models.py:326 |
| crop_url | YES | YES | pipeline (saved to /workspace/storage/crops) | NO | NO | NO | YES (thumbnail) | NO | — | models.py:329 |
| siglip_embedding | KHÔNG (trong tracklets_embeddings) | YES (trong TrackletResult) | SigLIP2-So400m | NO | — (pgvector) | YES (primary merge) | NO | YES | — | video_process_schemas.py:136 |

---

## PHẦN 4 — EMBEDDING TABLES VÀ VECTOR FIELDS

> **Cập nhật 2026-05-11:** `embedding_vector` (JSON) và `embedding` (pgvector DINOv2 1024) đã bị xóa. Chỉ còn `siglip_embedding`.

| Table | Field | pgvector? | Dimension | Model Source | Used In | Status | Evidence |
|-------|-------|-----------|-----------|--------------|---------|--------|----------|
| tracklets_embeddings | ~~embedding_vector~~ | KHÔNG | 1024 | DINOv2 | — | **ĐÃ XÓA** — `DROP COLUMN embedding_vector` | migration |
| tracklets_embeddings | ~~embedding~~ | CÓ (1024) | 1024 | DINOv2 | — | **ĐÃ XÓA** — luôn NULL, không ai ghi vào | migration |
| tracklets_embeddings | siglip_embedding | CÓ (vector(1152)) | 1152 | SigLIP2-So400m | **Primary và duy nhất** — merge, Re-ID, search | ACTIVE | models.py |

**Embedding sau cleanup:**
```python
# candidates.py — _tracklet_embedding() đã được đơn giản hóa
def _tracklet_embedding(t: Tracklet) -> list[float]:
    if not t.embedding or t.embedding.siglip_embedding is None:
        return []
    return list(t.embedding.siglip_embedding)
```

**Hệ quả:** Tracklet nào không có `siglip_embedding` (ingest thất bại) sẽ trả `[]` và bị excluded khỏi merge group. *(Concern về "data cũ" không còn áp dụng — DB sẽ được wipe.)*

---

## PHẦN 5 — ACTION TABLE (tracklets_actions)

### Columns:
- `id` — Integer PK
- `tracklet_id` — String(255) FK → tracklets.tracklet_id (CASCADE, UniqueConstraint)
- `action_label` — String(64) — nhãn đơn giản hóa: "standing"/"walking"/"running"/"sitting"/"bending"/"carrying"/"pushing_pulling"/"sports"
- `kinetics_label` — String(128) NULL — nhãn Kinetics-400 gốc
- `confidence` — Float — confidence từ VideoMAE (0.0-1.0)
- `created_at` — DateTime(tz)

### VideoMAE V2 → fields:
- `action_label` ← `_map_kinetics_to_tracex_action(kinetics_label)` — mapping Kinetics-400 → TraceX taxonomy
- `kinetics_label` ← `id2label[top_idx]` từ VideoMAE output
- `confidence` ← `top_probs[0]` từ `torch.softmax(logits, dim=-1).topk(1)`
- Evidence: `video_process.py:1548-1551`, `ingest_service.py:397-404`

### Có action embedding không?
**KHÔNG** — `tracklets_actions` không có field embedding. Không có vector cho action.

### Action tham gia candidate merge không?
**KHÔNG** — `_can_merge()` và `_metadata_matches()` trong `candidates.py` không check `action_label`. Action chỉ được lưu vào DB, không tham gia vào logic merge hay ranking.
Evidence: `candidates.py:303-336` — danh sách `checks` không bao gồm action fields.

---

## PHẦN 6 — MODEL INVENTORY & MODEL→DB MAPPING

| Model | Role | Input | Output | DB Column(s) | Code Path (file:fn) | Local Path/Key | VRAM Est. | Fallback if Fail |
|-------|------|-------|--------|--------------|---------------------|----------------|-----------|-----------------|
| RT-DETR R50 | Person detection (primary) | Frame images | Bounding boxes + scores | tracklets.representative_bbox | video_process.py:158-224, `_detect_persons_rtdetr()` | `/workspace/models/weights/rtdetr_person/` hoặc `PekingU/rtdetr_r50vd` | ~3GB fp16 | GDINO fallback (xem bên dưới) |
| Grounding DINO (GDINO16) | Person detection (fallback) | Frame + "person." text | Bounding boxes | tracklets.representative_bbox | video_process.py:115-155, `_detect_persons()` | `gdino16`/`gdino16_processor` — **KHÔNG có trong model_warmup.py** | ~3-5GB fp16 | Return `[]` (0 detections) |
| BodyPartAdaptiveTracker | Tracking trong từng video | Detections by frame | Tracked tracklet objects | tracklets.track_id, start_time, end_time | video_process.py:1651-1658, `tracking_pipeline.py` | N/A (thuật toán) | N/A |
| TrackletFragmentMerger | Merge tracklet fragments | Tracklets + siglip embeddings | Merged tracklets | (internal) | video_process.py:1703 | N/A | N/A |
| ~~DINOv2 ViT-L/14~~ | ~~Appearance embedding~~ | — | ~~1024-dim vector~~ | ~~tracklets_embeddings.embedding, embedding_vector~~ | — | — | — | **Đã xóa 2026-05-11** — `embedding` và `embedding_vector` bị drop. DINOv2 không còn được lưu vào DB. |
| SigLIP2-So400m | Text-image search embedding | Person crop 384x384 | 1152-dim vector | tracklets_embeddings.siglip_embedding | video_process.py:1464-1479, `_batch_extract_features()` | `google/siglip2-so400m-patch14-384` hoặc `google/siglip-so400m-patch14-384` | ~3GB fp16 | `[]` (empty embedding) |
| Qwen2-VL-7B-Instruct | Open-vocabulary attribute captioning | Person crop 384x384 + prompt | JSON attributes | tracklets.gender, upper_clothing_*, lower_clothing_*, hat_*, bag_*, etc. *(top_color/bottom_color đã xóa)* | video_process.py:1015-1051, `_caption_crop_vlm()` | `/workspace/models/weights/qwen2vl/` hoặc `Qwen/Qwen2-VL-7B-Instruct` | ~14GB fp16 | `_default_attributes()` (all "unknown") |
| VideoMAE V2 (Kinetics-400) | Action recognition | 16 frames 224x224 | Action label + confidence | tracklets_actions.action_label, kinetics_label, confidence | video_process.py:730-793, `_run_videomae_actions()` | `MCG-NJU/videomae-base-finetuned-kinetics` | ~3GB fp16 | "standing" (hardcoded default) |
| SeamlessM4T v2-large | Vietnamese→English translation | Text string | Translated text | (không save vào DB) | query-service/translation.py:49-85, `_load_seamless()` | `facebook/seamless-m4t-v2-large` | ~5GB fp16 | pass-through original text |

---

### SigLIP2

**`_legacy_run_siglip2_label_attributes` có tồn tại không?**
CÓ — định nghĩa tại `video_process.py:519-624`

**Có được gọi không?**
**KHÔNG được gọi** trong main pipeline. Docstring của nó tự ghi:
```python
# video_process.py:523-524
"""Legacy label-based SigLIP 2 attribute tagging — kept for debug/fallback only.
Main pipeline uses _caption_crop_vlm() instead."""
```
Không có call nào trong codebase đến `_legacy_run_siglip2_label_attributes`. Đây là dead code.

**Hiện tại dùng gì?**
SigLIP2 **chỉ dùng image encoder** (không dùng label-based):
```python
# video_process.py:1473-1476
img_feats = model_sip.get_image_features(**{k: v for k, v in img_inputs.items()
                                             if k in ["pixel_values"]})
img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
```
Output là 1152-dim image embedding, lưu vào `siglip_embedding` — dùng cho cosine similarity với text query embeddings.

---

### Qwen2-VL

**Prompt nằm ở đâu?**
- Single-crop prompt: `_VLM_PROMPT` — `video_process.py:821-861` (hardcoded string literal)
- Batch prompt template: `_VLM_BATCH_PROMPT_TEMPLATE` — `video_process.py:863-903`

**Parser nằm ở đâu?**
- `_parse_vlm_attrs(parsed: dict)` — `video_process.py:917-970`
- JSON extraction: `_extract_json_array(raw: str)` — `video_process.py:973-1012`

**Fallback khi parse fail?**
```python
# video_process.py:1049-1051
except Exception as exc:
    logger.warning("[vlm] _caption_crop_vlm failed: %s", exc)
    return _default_attributes()
```
Fallback về `_default_attributes()` — tất cả fields đều là "unknown". Silent failure.

**VLM confidence có được dùng trong candidate conflict guard không?**
**CÓ** — confidence từ VLM (`gender_conf`, `upper_clothing_conf`, `hat_conf`, v.v.) được lưu vào DB và được dùng trong `_metadata_matches()`:
```python
# candidates.py:331-335
c1 = getattr(t1, conf_field, None) or 0.0
c2 = getattr(t2, conf_field, None) or 0.0
threshold = _CONF_THRESHOLDS.get(attr, _CONF_THRESHOLD)
if c1 >= threshold and c2 >= threshold:
    return False  # conflict confirmed
```
Nếu confidence là NULL (= 0.0), conflict sẽ không bị block — behavior đúng thiết kế.

---

## PHẦN 7 — IMPORT VIDEO FLOW END-TO-END

```mermaid
flowchart TD
    A[Google Drive Temp/ folder] -->|_list_drive_mp4s| B{SA có write?}
    B -->|YES| C[_move_file: Temp→Storage]
    B -->|NO - sa_read_only=True| D[Dùng Temp files trực tiếp]
    C --> E[Google Drive Storage/ folder]
    D --> E
    E -->|_list_drive_mp4s| F[Scan .mp4 files]
    F -->|_video_exists_in_db check: video.processed?| G{Đã processed?}
    G -->|YES| H[Skip]
    G -->|NO| I[_ensure_camera: auto-register camera]
    I --> J[_upsert_video: INSERT videos, processed=False]
    J --> K[Batch download 4 videos song song]
    K -->|ThreadPoolExecutor _download_video| L[Download bytes từ Drive]
    L -->|_presample_bytes background thread| M[Decode frames VideoFrameSampler 4fps]
    M -->|_run_gpu → _process_video_sync| N[GPU Pipeline]
    N --> N1[Stage 1: VideoFrameSampler 4fps]
    N1 --> N2[Stage 2: _detect_persons_batch RT-DETR primary]
    N2 --> N3[Stage 3: BodyPartAdaptiveTracker]
    N3 --> N4[Stage 4: TrackletQualityScorer filter]
    N4 --> N5[Stage 5: _batch_siglip_embeddings]
    N5 --> N5a[SigLIP2 multi-frame pool-avg 1152-dim (5 frames)]
    N5 --> N5b[SigLIP2 single-crop 1152-dim (for DB)]
    N5a --> N6[Stage 6: TrackletFragmentMerger SigLIP2 sim=0.85]
    N6 --> N7[Stage 7: _batch_caption_and_classify]
    N7 --> N7a[Qwen2-VL captioning on merged tracklet]
    N7 --> N7b[VideoMAE action on merged tracklet]
    N7a & N7b --> N8[Save crop.jpg to /workspace/storage/crops]
    N8 --> N9[Return ProcessVideoResponse]
    N9 -->|result.tracklets.model_dump| O[_save_tracklets_from_gpu_result]
    O --> P[INSERT tracklets]
    O --> Q[INSERT tracklets_embeddings SigLIP2 1152-dim]
    O --> R[INSERT tracklets_actions VideoMAE]
    P & Q & R --> S[video.processed = True]
    S --> T[session.commit]
```

### Chi tiết từng bước:

| Step | File | Function | Notes |
|------|------|----------|-------|
| Drive move Temp→Storage | ingest_service.py:415-465 | `ingest_move_and_process()` | `_move_file()` L:147 — thực hiện nếu SA có write permissions |
| video.processed flag đổi | ingest_service.py:624-627 | `ingest_move_and_process()` | `video.processed = True` sau khi save tracklets xong |
| HTTP self-call hay direct? | ingest_service.py:223-239 | `_process_video_stream()` | **Direct call** — gọi `_process_video_sync()` import trực tiếp, không qua HTTP |
| TrackletResult→dict | ingest_service.py:229 | `_process_video_stream()` | `t.model_dump() if hasattr(t, "model_dump") else t` — Pydantic V2 |
| DB save tracklets | ingest_service.py:298-408 | `_save_tracklets_from_gpu_result()` | INSERT tracklets + tracklets_embeddings + tracklets_actions |

**Trả lời câu hỏi cụ thể:**
- **Google Drive move Temp→Storage:** CÓ — `ingest_service.py:453-461`, hàm `_move_file()`. Nếu SA read-only thì fallback xử lý từ Temp trực tiếp.
- **video.processed flag:** Đổi tại `ingest_service.py:624-627` — sau khi `_save_tracklets_from_gpu_result` trả về.
- **HTTP self-call hay direct:** **Direct call** `_process_video_sync()` — `ingest_service.py:223`: `from ..api.routers.video_process import _process_video_sync`
- **TrackletResult→dict:** `ingest_service.py:229` — `t.model_dump() if hasattr(t, "model_dump") else t`
- **DB save:** `_save_tracklets_from_gpu_result()` tại `ingest_service.py:298-408` — bảng `tracklets`, `tracklets_embeddings`, `tracklets_actions`

---

## PHẦN 8 — VIDEO_PROCESS PIPELINE AUDIT

| Item | Finding | Evidence (file:line) | Classification |
|------|---------|---------------------|----------------|
| Docstring của video_process.py | ~~"Full 7-stage pipeline: RT-DETR, BEVProjector, MCBLT, DINOv2, Qwen2-VL-7B-Instruct, VideoMAE V2"~~ → **Đã sửa 2026-05-11** — Docstring đã cập nhật: RT-DETR, SigLIP2, Qwen2-VL, VideoMAE (DINOv2 xóa) | video_process.py:1-16 | FIXED |
| "Grounding DINO" trong comment | ~~CÓ~~ → **Đã xóa 2026-05-11** — docstring và comment đã được cập nhật | video_process.py | FIXED |
| "SigLIP zero-shot" trong comment | KHÔNG tìm thấy comment như vậy | — | N/A |
| `_detect_persons_batch()` dùng gì | RT-DETR primary (`_detect_persons_rtdetr()`), GDINO fallback nếu RT-DETR trả về None | video_process.py:227-307 | CURRENT |
| GDINO fallback còn tồn tại không | ~~CÓ~~ → **Đã xóa 2026-05-11** — `_detect_persons()`, toàn bộ GDINO fallback block, và `gdino16` key đã bị xóa khỏi codebase | video_process.py | FIXED |
| `_process_single_video()` batch hay per-frame | Batch detection: `_detect_persons_batch([sf.image for sf in sampled_frames])` | video_process.py:373, 1616 | CURRENT |
| `_batch_extract_features()` output fields | 6-tuple: `(all_embeddings, all_attributes, all_attr_confs, all_actions, all_rep_crops, all_siglip_embeddings)` | video_process.py:1376-1573 | CURRENT |
| `_caption_crop_vlm()` trong main call path | CÓ — gọi từ `_batch_extract_features()` qua `_caption_crops_vlm_batch()` | video_process.py:1487 | CURRENT |
| `_run_siglip2_attributes()` có tồn tại không | **KHÔNG** — tên hàm này không tồn tại. Có `_legacy_run_siglip2_label_attributes()` nhưng không được gọi | video_process.py:519 | LEGACY_DEAD_CODE |
| `_legacy_run_siglip2_label_attributes()` có gọi không | **KHÔNG** — dead code, không có call site nào trong codebase | video_process.py:519-624 | LEGACY_DEAD_CODE |
| `siglip_embedding` pass vào TrackletResult | CÓ — `siglip_embedding=siglip_emb or []` | video_process.py:1820 | CURRENT |

---

## PHẦN 9 — QUERY / SEARCH / CANDIDATE MERGING

**Search endpoint URL:**
- metadata-service nhận: `POST /api/v1/search` (forwarded từ search.py)
- query-service xử lý: `POST /search` (internal, router prefix `/search`)
- Evidence: `metadata-service/app/main.py:98`, `query-service/app/api/routers/candidates.py:74`

**QueryHistory tạo ở đâu:**
`query-service/app/api/routers/candidates.py:450-459` — function `search_candidates()`:
```python
qh = QueryHistory(query_id=qid, user_id=body.user_id, query_text=query or "",
                  status="searching", query_image_url=body.query_image_url or None)
db.add(qh)
db.flush()
```

**`_local_prefilter()` criteria:**
1. Join `videos` để lọc theo `recorded_at`
2. Filter `camera_id` (lowercase IN) nếu `camera_ids` được cung cấp
3. Filter thời gian: `video.recorded_at + end_time >= time_from` và `video.recorded_at + start_time <= time_to`
4. Text filter: ILIKE match trên concat `appearance_summary + gender + upper_clothing_color + lower_clothing_color + shoes_color + age_range + hat_color + bag_type + is_wearing_mask + hair_style + hair_color` *(đã cập nhật sau cleanup 2026-05-11)*
5. Top-K heap: lấy 200 kết quả tốt nhất theo text score
- Evidence: `candidates.py:170-251`

**Có gọi trace-service không?**
**KHÔNG** — query-service (`candidates.py`) không gọi trace-service. Search và merge xong tại query-service. Trace-service chỉ được gọi khi user chọn candidate và nhấn "build trace".

**`_merge_by_similarity()` embedding field:**
SigLIP2 embedding (1152-dim) — field duy nhất còn lại sau cleanup. Không còn fallback về DINOv2 JSON. — `candidates.py:_tracklet_embedding()`

**Merge threshold:**
```python
# candidates.py:257-258
_MERGE_THRESHOLD = 0.85       # SigLIP2 cosine similarity
_MERGE_MAX_GAP_S = 86400.0   # max 24-hour gap
```

**Guards:**
- Same-video: KHÔNG block merge riêng biệt
- Same-camera overlap: `if t1.camera_id == t2.camera_id and min(e1,e2) > max(s1,s2): return False` — candidates.py:383-386
- Time-gap: `gap > 86400s` → không merge
- Metadata-conflict: `_metadata_matches()` — cả hai values confident (≥threshold) và khác nhau → block

**`_CONF_THRESHOLDS` dict (nguyên văn từ code):**
```python
# candidates.py:264-278
_CONF_THRESHOLDS: dict[str, float] = {
    "gender":               0.70,
    "is_wearing_mask":      0.70,
    "bag_presence":         0.75,
    "hat_presence":         0.75,
    "age_range":            0.75,
    "upper_clothing_color": 0.82,
    "lower_clothing_color": 0.82,
    "shoes_color":          0.82,
    "hat_color":            0.82,
    "hair_color":           0.82,
}
```

**Clothing/bag/hat fields dùng trong filter?**
Trong `_local_prefilter()`: `bag_type`, `hat_color`, `is_wearing_mask` được concat vào text search string và dùng ILIKE match — **không phải exact-match**. `bag_presence` và `hat_presence` chỉ dùng trong merge guard. Không có exact-match filter nguy hiểm.

**match_score formula:**
- Single tracklet: `fusion_score = 0.7 * text_score + 0.3 * quality_score`
- Multi-tracklet: `fusion_score = 0.5 * text_score + 0.3 * quality_score + 0.2 * vector_score`
- Evidence: `candidates.py:494-496`

**thumbnail_url source:**
```python
# candidates.py:567
"thumbnail_url": f"/candidates/{candidate_id}/preview"
```
URL dạng `/candidates/{candidate_id}/preview` — không trực tiếp từ `crop_url` của tracklet.

**QueryCandidate + QueryCandidateTracklet insert:**
`query-service/app/api/routers/candidates.py:529-548` — dùng PostgreSQL `ON CONFLICT DO NOTHING` upsert.

---

## PHẦN 10 — TRACE FLOW

**Input: candidate_id hay tracklet_ids?**
**candidate_id** — `BuildTraceRequest` chứa `query_id` + `candidate_id`.
Evidence: `trace.py:99-105`

**Đọc query_candidate_tracklets không?**
**CÓ** — Strategy 1 (ưu tiên):
```python
# trace_service.py:84-93
via_join = (
    self.session.query(Tracklet)
    .join(QueryCandidateTracklet,
          QueryCandidateTracklet.tracklet_id == Tracklet.tracklet_id)
    .filter(QueryCandidateTracklet.candidate_id == candidate_id)
    .order_by(Tracklet.start_time.asc())
    .all()
)
```

**Fallback camera proximity:**
**CÓ** — Nếu không có QCT rows, fallback về neighbor camera expansion (±10 camera):
```python
# trace_service.py:99-112
neighbor_cams = _neighbor_cameras(primary_cam)  # ±_CAM_NEIGHBOR_RADIUS=10
tracklets = session.query(Tracklet).filter(Tracklet.camera_id.in_(neighbor_cams)).all()
```
Sau đó áp dụng soft metadata filter (`gender`) — `trace_service.py:121-138` *(`top_color`/`bottom_color` đã xóa khỏi `query_candidates`)*

**Re-rank sau thay đổi:**
**KHÔNG** — không có re-rank sau khi user thay đổi time window. Chỉ rebuild trace với window mới.

**evidence_videos/evidence_tracklets build logic:**
1. Get tracklets (via QCT join hoặc neighbor camera fallback)
2. `build_trace_segments()` → sort by time, re-index — `trace_service.py:158-211`
3. `calculate_trace_confidence()` = avg(quality_score) + camera_bonus(0.02/cam) + segment_bonus(0.01/seg) — `trace_service.py:213-252`
4. `create_evidence_video()` → INSERT evidence_videos, sau đó INSERT evidence_tracklets per segment — `trace_service.py:254-311`

**Phụ thuộc metadata hay chỉ dùng tracklets?**
Primary path: pure tracklet IDs từ QCT junction table. Fallback path: dùng `candidate.gender` từ `query_candidates` — soft metadata filter. (`top_color`/`bottom_color` đã bị xóa khỏi `query_candidates`, fallback path cần verify lại sau migration).

---

## PHẦN 11 — HARDCODE / BYPASS / MOCK / SILENT FAILURE

| # | File Path | Function | Hardcoded Value / Behavior | Severity | Impact | Recommendation | Fix Now? | Evidence (line) |
|---|-----------|----------|---------------------------|----------|--------|----------------|----------|-----------------|
| 1 | backend/services/shared/database.py | `_build_database_url()` | `password = "Mcpt@2026Secure"` hardcoded làm default | CRITICAL | Password DB lộ trong source code | Require `POSTGRES_PASSWORD` env var, raise error nếu thiếu | YES | database.py:17 |
| 2 | backend/services/metadata-service/app/config.py | `_build_default_database_url()` | `db_password = ... "Mcpt2026Secure"` hardcoded — **khác với database.py** | CRITICAL | Hai password default khác nhau → inconsistency nguy hiểm. Services dùng database.py kết nối khác với services dùng config.py | Xóa cả hai default, đồng bộ, require env | YES | config.py:83 |
| 3 | backend/services/metadata-service/app/services/ingest_service.py | module level | `DRIVE_TEMP_FOLDER_ID = "1Px379D5sjK95lMOUGZ4oUAgco7wBCk4I"` và `DRIVE_STORAGE_FOLDER_ID = "1G6L1d8l2YupSI0HIgB9NkBel04RqX48G"` — hardcode **2 lần** (line 36-38 và 59-61) | HIGH | Duplicate definition + hardcode Drive IDs. Nếu folder thay đổi phải sửa 2 chỗ | Dùng env var `DRIVE_TEMP_FOLDER_ID` / `DRIVE_STORAGE_FOLDER_ID`; bỏ duplicate | YES | ingest_service.py:36-38, 59-61 |
| 4 | backend/services/metadata-service/app/services/model_warmup.py | Tất cả `_load_*()` | `except Exception: logger.warning("... non-fatal")` cho tất cả 5 model load | HIGH | Nếu RT-DETR fail và GDINO không load → 0 detections → 0 tracklets, không có alert | Ít nhất log ERROR + fail health check `/health` endpoint | NO | model_warmup.py:122, 159, 206, 253, 306 |
| 5 | backend/services/trace-service/app/services/trace_service.py | `delete_old_evidence()` | `EvidenceVideo.selected_candidate_id` — field không tồn tại trong `EvidenceVideo` model | HIGH | RuntimeError/AttributeError khi gọi `delete_old_evidence()` trong `continue_trace` | Fix: dùng `EvidenceVideo.query_candidate_id` hoặc join qua query | YES | trace_service.py:371-374 |
| 6 | backend/services/metadata-service/app/api/routers/video_process.py | `_process_video_sync()` | `_CROPS_DIR = Path("/workspace/storage/crops")` hardcoded | HIGH | Không portable. Nếu deploy khác /workspace → crops fail silently | Env var `CROPS_STORAGE_DIR` | NO | video_process.py:1748 |
| 7 | backend/services/metadata-service/app/main.py | CORS middleware | `allow_origins=["https://tracex-ai.smartnovi.tech", "http://localhost:3000", "http://localhost:3001"]` hardcoded | MEDIUM | Thêm domain mới phải deploy lại | Env var `CORS_ALLOWED_ORIGINS` | NO | main.py:85-89 |
| 8 | backend/services/metadata-service/app/api/routers/video_process.py | `_project_to_bev_single()` | `d["bev_x"] = 0.0; d["bev_y"] = 0.0` khi BEVProjector fail | MEDIUM | Tất cả tracklet fail calibration có tọa độ (0,0) — không phân biệt được với thực tế | Thêm `bev_valid=False` flag | NO | video_process.py:324-328 |
| 9 | backend/services/query-service/app/services/translation.py | `translate_to_english()` | SeamlessM4T fail → return original text passthrough, chỉ `logger.debug` | MEDIUM | Query tiếng Việt sẽ match kém | Đổi sang `logger.warning` | NO | translation.py:141, 80-82 |
| 10 | backend/services/metadata-service/app/api/routers/video_process.py | `_caption_crop_vlm()` | Fallback về `_default_attributes()` khi VLM fail — all "unknown" | HIGH | Tracklets với "unknown" metadata → không match query | Phân biệt "model unavailable" vs "parse fail"; retry logic | NO | video_process.py:1049-1051 |
| 11 | backend/services/metadata-service/app/queue_worker.py | `_save_candidates_from_batch()` | `user_id=1` hardcoded cho system QueryHistory | MEDIUM | Tất cả batch queries gán cho user ID 1 | Tạo system user ID riêng | NO | queue_worker.py:229 |
| 12 | backend/services/metadata-service/app/api/routers/video_process.py | `_run_videomae_actions()` | Return "standing" khi VideoMAE model None hoặc exception | LOW | Mọi tracklet là "standing" nếu VideoMAE fail | Return "unknown" thay vì "standing" | NO | video_process.py:744, 792 |
| 13 | backend/services/metadata-service/app/services/model_warmup.py | `_load_siglip2()` | Warmup dùng dummy image + fake labels, không verify 1152-dim output | MEDIUM | Nếu load SigLIP v1 (1152 dim khác) fail silently | Add `assert feat.shape[-1] == 1152` | NO | model_warmup.py:149-155 |
| 14 | backend/services/metadata-service/app/services/ingest_service.py | module level | `_STORAGE_PATH_METADATA = Path("/workspace/storage/storage")` hardcoded | LOW | Chỉ dùng trong DB record, nhưng confusing | Đồng bộ với settings | NO | ingest_service.py:53 |
| 15 | backend/services/metadata-service/app/services/model_warmup.py | `_load_rtdetr()` | `Path("/workspace/models/weights/rtdetr_person")` hardcoded | MEDIUM | Không portable | Env var `RTDETR_WEIGHTS_PATH` | NO | model_warmup.py:177 |
| 16 | backend/services/metadata-service/app/services/model_warmup.py | `_load_qwen2vl()` | `Path("/workspace/models/weights/qwen2vl")` hardcoded | MEDIUM | Không portable | Env var `QWEN2VL_MODEL_PATH` | NO | model_warmup.py:271 |

---

## PHẦN 12 — RUNTIME RISKS

| Risk | Evidence | Impact | How to Verify | Recommended Fix |
|------|----------|--------|---------------|-----------------|
| **pgvector không cài** → embedding lưu JSON, không có ANN index | `shared/models.py:49-54` fallback JSON | Merge bằng cosine similarity O(N²) thay vì vector index → rất chậm ở scale >10k tracklets | `SELECT * FROM pg_extension WHERE extname='vector'` | `CREATE EXTENSION vector;` trong init script |
| **Không có Alembic** → schema migration thủ công | `main.py:44` — `create_all` chỉ tạo bảng mới | Thêm column vào model nhưng bảng cũ không cập nhật → silent NULL hoặc AttributeError | Check `\d tracklets` trong psql so với models.py | Thêm Alembic để manage migrations |
| **GDINO fallback code tồn tại nhưng không load** | `model_warmup.py` không có `_load_gdino`; `video_process.py:239-244` return `[]` | Nếu RT-DETR cũng fail → 0 detections → 0 tracklets, không có alert | Check `/health` endpoint `loaded_models` | Thêm health check flag cho detector; cảnh báo rõ ràng |
| **Hai password default khác nhau** trong database.py và config.py | `database.py:17` = "Mcpt@2026Secure", `config.py:83` = "Mcpt2026Secure" | Nếu không set env var, hai services kết nối DB với password khác nhau → authentication fail | So sánh hai giá trị | Xóa cả hai default, require env |
| **`delete_old_evidence()` dùng column không tồn tại** | `trace_service.py:371-374`: `EvidenceVideo.selected_candidate_id` không có trong models.py | AttributeError khi gọi `continue_trace` endpoint | Run `continue_trace` API call | Fix: dùng `EvidenceVideo.query_candidate_id` |
| **Qwen2-VL generate không có timeout** | `video_process.py:1031-1038` — không có `max_time` trong `model.generate()` | Nếu model treo → API block forever → timeout gateway | Load test với corrupted crops | Thêm timeout trong generate |
| **BEV tọa độ (0,0) cho tất cả tracklet khi calibration fail** | `video_process.py:324-328` | Không phân biệt "no calibration" vs "actually at origin" | Check bev_x/bev_y distribution trong DB | Thêm `bev_valid` boolean column |
| **password hardcoded lộ trong source** | `database.py:17`, `config.py:83` | Security breach nếu repo có public access | Grep `Mcpt` trong toàn bộ repo | Rotate credentials, require env vars |

---

## PHẦN 13 — FINAL SUMMARY

### Database

**Tổng cộng 19 bảng** chia thành 6 nhóm:

1. **Users & Auth** (1): `users`
2. **Camera & Topology** (4): `cameras`, `camera_zones`, `camera_edges`, `camera_settings`
3. **Video & Tracklet AI Data** (4): `videos`, `tracklets`, `tracklets_embeddings`, `tracklets_actions`
4. **Query & Results** (5): `query_history`, `query_candidates`, `query_candidate_tracklets`, `query_jobs`, `spatiotemporal_groups`
5. **Evidence & Verification** (4): `evidence_videos`, `evidence_tracklets`, `verified_objects`, `verified_objects_tracklets`
6. **Internal Queue** (1): `queue_video_assets`

**Core tables:** `tracklets`, `tracklets_embeddings`, `query_candidates`, `query_candidate_tracklets`

### ML Pipeline

- **Metadata:** **VLM-generated** — Qwen2-VL-7B-Instruct sinh ra JSON attributes (open-vocabulary, free-text). `_legacy_run_siglip2_label_attributes` đã bị **xóa** khỏi codebase (2026-05-11).
- **Re-ID vector field:** `tracklets_embeddings.siglip_embedding` (1152-dim, SigLIP2) — **field duy nhất**. `embedding_vector` (JSON DINOv2) và `embedding` (pgvector DINOv2) đã bị **xóa** (2026-05-11). Tracklet không có `siglip_embedding` sẽ bị excluded khỏi merge.
- **Candidate merge:** Union-find cosine similarity trên SigLIP2 embeddings với threshold 0.85 + metadata conflict guard có confidence threshold.

---

### Top 5 Hardcode/Risk (ranked by severity)

1. **CRITICAL** — `database.py:17`: `POSTGRES_PASSWORD = "Mcpt@2026Secure"` hardcoded. Password DB production lộ trong source code. Rotation ngay lập tức nếu repo public.
2. **CRITICAL** — `config.py:83`: `POSTGRES_PASSWORD = "Mcpt2026Secure"` hardcoded — **khác với database.py**. Hai services kết nối DB với password khác nhau khi thiếu env var → inconsistency nguy hiểm.
3. **HIGH** — `trace_service.py:371-374`: `EvidenceVideo.selected_candidate_id` không tồn tại trong model → AttributeError tại runtime khi gọi `delete_old_evidence()` từ `continue_trace` endpoint.
4. **HIGH** — `ingest_service.py:36,59`: `DRIVE_TEMP_FOLDER_ID` và `DRIVE_STORAGE_FOLDER_ID` hardcoded 2 lần — duplicate definition với giá trị giống nhau nhưng rủi ro sync nếu thay đổi.
5. **HIGH** — `model_warmup.py`: Tất cả 5 model load đều non-fatal. RT-DETR fail nay sẽ raise `RuntimeError` (GDINO fallback đã xóa) — fail rõ ràng, nhưng pipeline vẫn không có health-check alert.

---

### Top 5 Next Actions

1. **[Security — Ngay lập tức]** Xóa hardcode passwords khỏi `database.py:17` và `config.py:83`. Đồng bộ thành một giá trị và require `POSTGRES_PASSWORD` qua env var (raise ValueError nếu thiếu). Rotate production credentials.

2. **[Bug Fix — Ngay lập tức]** Fix `trace_service.py:delete_old_evidence()` — `EvidenceVideo.selected_candidate_id` không tồn tại, gây AttributeError tại runtime. Thay bằng filter `EvidenceVideo.query_candidate_id == candidate_id`.

3. **[Reliability — Tuần này]** Thêm Alembic migration framework. Hiện tại `create_all()` không áp dụng schema changes lên bảng cũ. Khi thêm column mới (như `upper_clothing_*`), cần migration script để ALTER TABLE các DB đang chạy.

4. **[Infrastructure — Tháng này]** Đảm bảo `pgvector` extension được cài trong PostgreSQL và các vector columns (`embedding`, `siglip_embedding`) thực sự là `vector(N)` type. Nếu không, merge cosine similarity O(N²) — nghiêm trọng về hiệu năng ở scale >10k tracklets.

5. **[Ops & Config — Tháng này]** Chuyển tất cả hardcode paths và IDs sang env vars: `DRIVE_TEMP_FOLDER_ID`, `DRIVE_STORAGE_FOLDER_ID`, `CROPS_STORAGE_DIR` (`/workspace/storage/crops`), `RTDETR_WEIGHTS_PATH`, `QWEN2VL_MODEL_PATH`. Thêm health check alert khi cả RT-DETR và GDINO đều không load.

---

## PHẦN 14 — TRACKLET SAMPLE DEEP ANALYSIS

> **Lưu ý:** Phần này phân tích trạng thái **trước cleanup 2026-05-11**. Các field `top_color`, `bottom_color`, `top_color_conf`, `bottom_color_conf`, `embedding_vector`, `embedding` đã bị xóa. Các references đến chúng trong phần này là lịch sử.

> Record: `cam_04_2026-04-28_11-00.mp4_cam_04_125`

---

### 14.1 — Giải thích từng field

#### Nhóm 1: Identity / Time / Location

| Field | Giá trị | Nguồn sinh | Input | Model/Function | DB Column | Vì sao lưu | Downstream dùng | Có thể null? | Evidence |
|-------|---------|-----------|-------|----------------|-----------|------------|-----------------|--------------|----------|
| tracklet_id | `cam_04_2026-04-28_11-00.mp4_cam_04_125` | `_process_video_sync()` line 1780: `f"{video_id}_{camera_id}_{t_idx}"` — `video_id` = `cam_04_2026-04-28_11-00.mp4`, `camera_id` = `cam_04`, `t_idx` = 125 (vị trí trong accepted list sau merge) | video_id + camera_id + running index | Không có model; index là sequential int | `tracklets.tracklet_id` String(255) UNIQUE | Primary key nghiệp vụ; dùng để dedup (`existing = session.scalar(...)`) | `query_candidate_tracklets.tracklet_id`, `evidence_tracklets.tracklet_id`, crop filename | NO | `video_process.py:1780`, `ingest_service.py:314` |
| camera_id | `cam_04` | Từ `ProcessVideoStreamRequest.camera_id` hoặc default `"Camera_0000"`; passed vào `_process_video_sync(camera_id)` | Form field `camera_id` trong `/process/stream` | — | `tracklets.camera_id` String(50) | Lọc theo camera trong query; cross-camera topology | `_local_prefilter()` WHERE clause; `_expand_camera_range()` | NO | `video_process.py:1594`, `ingest_service.py:328` |
| track_id | `125` | `t_idx` — running index trong vòng `for t_idx, (lt, _, ...) in enumerate(t_data)` sau merge | Sequential index sau `TrackletFragmentMerger.merge()` | — | `tracklets.track_id` String(50) | Identifier trong video; dùng để look up quality_results | Hiển thị, không join | NO | `video_process.py:1783` |
| start_s | `550.5` | `obs[0].timestamp_second` — timestamp của frame đầu tiên trong tracklet observations | `VideoFrameSampler` tính `timestamp_second = frame_index / fps` | `BodyPartAdaptiveTracker.track()` | `tracklets.start_time` Float | Time range cho filter theo thời gian | `_local_prefilter()` time window; `_tracklet_abs_window()` | NO | `video_process.py:1784`, `ingest_service.py:330` |
| end_s | `565.0` | `obs[-1].timestamp_second` — timestamp frame cuối | Như start_s | Như start_s | `tracklets.end_time` Float | Duration = 565.0 - 550.5 = 14.5s | Time filter; `_can_merge()` gap calculation | NO | `video_process.py:1785`, `ingest_service.py:331` |
| bev_x | `0.00` | `_bev_inputs[t_idx].get("bev_x", 0.0)` — giá trị từ `_project_to_bev_single()`. Nếu `BEVProjector` fail → exception → tất cả `bev_x = 0.0` | `rep_bbox_float` từ representative detection | `BEVProjector.bbox_bottom_center_to_bev()` | `tracklets.bev_x` Float DEFAULT 0.0 | Spatial search; MCBLT cross-camera association | `_can_merge()` không dùng bev; index `ix_tracklets_bev_xy` | NO | `video_process.py:1817`, `video_process.py:324-328` |
| bev_y | `0.00` | Như bev_x — cùng exception path | Như bev_x | Như bev_x | `tracklets.bev_y` Float DEFAULT 0.0 | Như bev_x | Như bev_x | NO | `video_process.py:1818` |

**Lưu ý bev_x/bev_y = 0.00:** Tại `_project_to_bev_single()` line 325-328, khi `BEVProjector(camera_id, cal_path)` raise exception (thiếu calibration file hoặc env var `CAMERA_CALIBRATION_PATH` không set), code gán `d["bev_x"] = 0.0` cho mọi detection. Giá trị 0.00 KHÔNG phân biệt được "camera không có calibration" vs "thực sự tại origin". Đây là bug đã nhận diện ở PHẦN 12.

---

#### Nhóm 2: Quality / Tracking

| Field | Giá trị | Nguồn sinh | Input | Model/Function | DB Column | Vì sao lưu | Downstream dùng | Có thể null? | Evidence |
|-------|---------|-----------|-------|----------------|-----------|------------|-----------------|--------------|----------|
| quality | `0.897` | `quality.average_confidence` từ `TrackletQualityScorer.score(lt)` — trung bình `detection.confidence` trên tất cả frames của tracklet | `FrameDetection.confidence = float(d["score"])` từ RT-DETR/GDINO | `TrackletQualityScorer.score()` trong `tracking_pipeline.py` | `tracklets.quality_score` Float DEFAULT 0.0 | Filter tracklets xấu; đầu vào `fusion_score` trong search | `candidates.py:496`: `fusion_score = 0.7*text + 0.3*quality`; `_can_merge()` fallback match_score | NO | `video_process.py:1786`, `ingest_service.py:332` |
| occlusion | `0.000` | Hardcoded `occlusion_score=0.0` tại mọi nơi tạo TrackletResult | — | Không có model occlusion estimation | `tracklets.occlusion_score` Float DEFAULT 0.0 | Đã có column nhưng không có model tính | Không được dùng trong downstream | NO | `video_process.py:1824`, `video_process.py:1298`: `occlusion_score=0.0` |
| crop_url | `/static/crops/cam_04_2026-04-28_11-00.mp4_cam_04_125.jpg` | Line 1774: `f"/static/crops/{crop_filename}"` với `crop_filename = f"{video_id}_{camera_id}_{t_idx}.jpg"` — file được save tại `/workspace/storage/crops/{crop_filename}` | `all_rep_crops[t_idx]` — PIL Image 384×384 từ `_batch_extract_features()` | — | `tracklets.crop_url` String(2048) | Hiển thị thumbnail trong UI | Frontend display; `query_candidates.preview_url` via `/candidates/{id}/preview` | NO | `video_process.py:1771-1774` |
| has_embedding | `yes` | Inferred: `TrackletEmbedding` record tồn tại trong `tracklets_embeddings` khi `siglip_vec` không rỗng *(sau cleanup 2026-05-11 — chỉ còn siglip_embedding)* | SigLIP2 1152-dim | `_batch_extract_features()` SigLIP block | `tracklets_embeddings.siglip_embedding` vector(1152) | Re-ID similarity search, merge | `_tracklet_embedding()` trong candidates.py; union-find merge | YES (NULL nếu SigLIP2 fail) | `ingest_service.py` |

---

#### Nhóm 3: Demographic

| Field | Giá trị | Nguồn sinh | Input | Model/Function | DB Column | Vì sao lưu | Downstream dùng | Có thể null? | Evidence |
|-------|---------|-----------|-------|----------------|-----------|------------|-----------------|--------------|----------|
| gender | `unknown` | `attributes.get("gender", "unknown")` — VLM JSON field `"gender"` được parse bởi `_parse_vlm_attrs()`. VLM trả về `"unknown"` vì prompt cho phép: `"man or woman or unknown"` | PIL crop 384×384 | `Qwen2-VL-7B-Instruct` → `_caption_crop_vlm()` → `_parse_vlm_attrs()` | `tracklets.gender` String(32) DEFAULT "unknown" | Search/filter theo giới tính | `_build_search_text()` trong candidates.py | NO | `video_process.py:1787`, `video_process.py:917–935` |
| gender_conf | `0.00` | `attr_conf.get("gender_conf")` ← `all_attr_confs[t_idx]["gender_conf"]` ← `attrs.get("gender_conf")` ← `_f("gender_conf")` trong `_parse_vlm_attrs()`. VLM được dặn `"Replace all 0.0 placeholders with your actual confidence"` nhưng thường trả về `0.0` khi không tự tin. Khi `gender="unknown"`, VLM không cố đoán nên conf = 0.0 | Như gender | Như gender | `tracklets.gender_conf` Float NULLABLE | Metadata conflict guard | `_metadata_matches()` line 331: `c1 = getattr(t1, conf_field, None) or 0.0`; threshold=0.70 → conf 0.00 < 0.70 → không block merge | YES | `video_process.py:1825`, `candidates.py:264–278` |
| age_range | `unknown` | Như gender — VLM prompt cho phép "child/teenager/young_adult/middle_aged/elderly/unknown" | Như gender | Như gender | `tracklets.age_range` String(32) DEFAULT "unknown" | Nhân khẩu học | `_build_search_text()` | NO | `video_process.py:1788` |
| age_conf | `0.00` | Như gender_conf | Như gender | Như gender | `tracklets.age_range_conf` Float NULLABLE | Như gender_conf | `_metadata_matches()` threshold=0.75 | YES | `video_process.py:1830` |

---

#### Nhóm 4: Clothing / Accessory

| Field | Giá trị | Nguồn sinh | Input | Model/Function | DB Column | Evidence |
|-------|---------|-----------|-------|----------------|-----------|----------|
| top_color | `gray` | `attributes.get("top_color", "unknown")` — trong `_parse_vlm_attrs()` line 966: `"top_color": _s("upper_clothing_color")` — alias của `upper_clothing_color`. VLM điền được vì đây là màu áo rõ ràng trong crop | VLM crop | Qwen2-VL | `tracklets.top_color` String(64) | `video_process.py:1789`, `_parse_vlm_attrs():966` |
| top_conf | `0.00` | `attr_conf.get("top_color_conf")` ← `all_attr_confs[t_idx]["top_color_conf"]` ← `attrs.get("upper_clothing_conf")` ← `_f("upper_clothing_conf")`. **ROOT CAUSE:** VLM được prompt yêu cầu điền confidence vào `upper_clothing_conf`, nhưng VLM thường trả về `0.0` literal từ template ngay cả khi đã nhận ra màu. Prompt line 861: `"Replace all 0.0 placeholders with your actual confidence"` nhưng instruction này thường bị ignore nếu model không được fine-tune để tự báo confidence. | VLM output | Qwen2-VL | `tracklets.top_color_conf` Float NULLABLE | `video_process.py:1827`, `_batch_extract_features():1496` |
| bottom_color | `black` | `attributes.get("bottom_color", "unknown")` ← `_parse_vlm_attrs():967`: `"bottom_color": _s("lower_clothing_color")` | VLM crop | Qwen2-VL | `tracklets.bottom_color` String(64) | `video_process.py:1790`, `_parse_vlm_attrs():967` |
| bot_conf | `0.00` | `attr_conf.get("bottom_color_conf")` ← `attrs.get("lower_clothing_conf")` — cùng vấn đề như top_conf | Như trên | Như trên | `tracklets.bottom_color_conf` Float NULLABLE | `video_process.py:1827`, `_batch_extract_features():1497` |
| shoes_color | `white` | `attributes.get("shoes_color", "unknown")` ← `_s("shoes_color")` từ VLM JSON | VLM crop | Qwen2-VL | `tracklets.shoes_color` String(64) | `video_process.py:1791`, `_parse_vlm_attrs():948` |
| shoe_conf | `0.00` | `attr_conf.get("shoes_conf")` ← `attrs.get("shoes_conf")` ← `_f("shoes_conf")` — VLM trả về 0.0 | VLM | Qwen2-VL | `tracklets.shoes_conf` Float NULLABLE | `video_process.py:1828`, `_batch_extract_features():1498` |
| upper_clothing_type | `sweater` | `attributes.get("upper_clothing_type")` ← `_s("upper_clothing_type")` từ VLM JSON. VLM điền free-text: "sweater" | VLM | Qwen2-VL | `tracklets.upper_clothing_type` String(128) NULLABLE | `video_process.py:1801` |
| upper_clothing_color | `gray` | `attributes.get("upper_clothing_color")` ← `_s("upper_clothing_color")` | VLM | Qwen2-VL | `tracklets.upper_clothing_color` String(64) NULLABLE | `video_process.py:1800` |
| lower_clothing_type | `pants` | `attributes.get("lower_clothing_type")` ← `_s("lower_clothing_type")` | VLM | Qwen2-VL | `tracklets.lower_clothing_type` String(128) NULLABLE | `video_process.py:1804` |
| lower_clothing_color | `black` | `attributes.get("lower_clothing_color")` ← `_s("lower_clothing_color")` | VLM | Qwen2-VL | `tracklets.lower_clothing_color` String(64) NULLABLE | `video_process.py:1803` |
| hat_color | `unknown` | `attributes.get("hat_color", "unknown")` ← `_s("hat_color")`. VLM trả về "unknown" vì không thấy mũ | VLM | Qwen2-VL | `tracklets.hat_color` String(64) | `video_process.py:1792` |
| hat_presence | `unknown` | `attributes.get("hat_presence")` ← `_parse_vlm_attrs()` line 933: `hat_pres = _norm(_s("hat_presence"), _PRESENCE_NORM)`. VLM trả về "unknown" — `_PRESENCE_NORM` map: `{"yes":"yes","no":"no","true":"yes","false":"no","none":"no"}` → "unknown" không có trong map → passthrough "unknown" | VLM | Qwen2-VL | `tracklets.hat_presence` String(16) NULLABLE | `video_process.py:1812`, `_parse_vlm_attrs():933,953` |
| hat_conf | `0.00` | `attr_conf.get("hat_color_conf")` ← `attrs.get("hat_conf")` ← `_f("hat_conf")` — 0.0 khi VLM không tự tin (hat = unknown) | VLM | Qwen2-VL | `tracklets.hat_conf` Float NULLABLE | `_batch_extract_features():1503` |
| bag_type | `unknown` | `attributes.get("bag_type", "unknown")` ← `_s("bag_type")` | VLM | Qwen2-VL | `tracklets.bag_type` String(64) | `video_process.py:1793` |
| bag_presence | `unknown` | `bag_pres = _norm(_s("bag_presence"), _PRESENCE_NORM)` — như hat_presence | VLM | Qwen2-VL | `tracklets.bag_presence` String(16) NULLABLE | `_parse_vlm_attrs():932,950` |
| bag_conf | `0.00` | `attr_conf.get("bag_type_conf")` ← `attrs.get("bag_conf")` | VLM | Qwen2-VL | `tracklets.bag_conf` Float NULLABLE | `_batch_extract_features():1504` |
| is_wearing_mask | `unknown` | `_norm(_s("is_wearing_mask"), _PRESENCE_NORM)` — "unknown" passthrough | VLM | Qwen2-VL | `tracklets.is_wearing_mask` String(16) | `video_process.py:1794` |
| mask_conf | `0.00` | `attr_conf.get("mask_conf")` ← `attrs.get("mask_conf")` | VLM | Qwen2-VL | `tracklets.mask_conf` Float NULLABLE | `_batch_extract_features():1505` |
| hair_style | `unknown` | `_s("hair_style")` — VLM không thấy rõ tóc | VLM | Qwen2-VL | `tracklets.hair_style` String(64) | `video_process.py:1795` |
| hair_color | `unknown` | `_s("hair_color")` — như hair_style | VLM | Qwen2-VL | `tracklets.hair_color` String(64) | `video_process.py:1796` |

---

#### Nhóm 5: Summary

| Field | Giá trị | Nguồn sinh | Input | Model/Function | Evidence |
|-------|---------|-----------|-------|----------------|----------|
| appearance_summary | `A person wearing a gray sweater, black pants, and white sneakers.` | `_build_appearance_summary(attributes)` line 1764: ưu tiên `attrs.get("appearance_summary")` nếu không phải "person". VLM sinh ra câu này trong field `"appearance_summary": "one concise sentence"` của JSON response | PIL crop 384×384 | Qwen2-VL-7B-Instruct → `_caption_crop_vlm()` | `video_process.py:1125-1141`, `video_process.py:1764,1797` |

---

### 14.2 — Vì sao nhiều field là UNKNOWN?

#### Nguyên nhân 1: Crop extraction — không có context margin

~~File: `video_process.py`, function `_extract_crop_for_vlm`, line 1144–1164~~ ✅ **ĐÃ FIX**

~~```python
def _extract_crop_for_vlm(frame: np.ndarray, bbox: list[float]) -> Optional[np.ndarray]:
    """Extract a square-padded 384×384 crop from frame for VLM input."""
    x1, y1, x2, y2 = map(int, bbox)
    ...
    crop = frame[y1:y2, x1:x2]          # RAW bbox — KHÔNG có margin
    ...
    cv2.BORDER_CONSTANT, value=(0, 0, 0)  # padding đen
```~~

**Fix applied (`video_process.py:796`):**
```python
def _extract_crop_for_vlm(frame: np.ndarray, bbox: list[float], margin_pct: float = 0.15):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_pct)   # expand 15% each side
    my = int(bh * margin_pct)
    x1_e = max(0, int(x1) - mx)
    y1_e = max(0, int(y1) - my)
    x2_e = min(frame.shape[1], int(x2) + mx)
    y2_e = min(frame.shape[0], int(y2) + my)
    ...
    cv2.BORDER_CONSTANT, value=(114, 114, 114)  # gray thay vì đen
```

**Thay đổi:**
- Thêm margin 15% mỗi phía → head và feet được capture đầy đủ hơn
- Padding đổi từ đen `(0,0,0)` sang xám `(114,114,114)` → VLM ít nhầm khi có black bars
- Signature backward-compatible: `margin_pct=0.15` là default

**Tác động:** Giảm `unknown` cho hat, hair, gender, age vì head/body đầy đủ hơn.

---

#### Nguyên nhân 2: Prompt design — "unknown" quá dễ chọn

✅ **ĐÃ FIX** (`video_process.py:476`, `video_process.py:518`)

---

#### Nguyên nhân 3: Parser không repair từ summary

✅ **ĐÃ FIX** (`video_process.py:572`)

Thêm logic repair trong `_parse_vlm_attrs`:

```python
summary = (parsed.get("appearance_summary") or "").lower()

# Repair presence fields từ summary khi VLM trả "unknown"
bag_raw = _s("bag_presence")
bag_pres = _norm(bag_raw, _PRESENCE_NORM)
if bag_pres == "unknown" and "bag" not in summary and "backpack" not in summary:
    bag_pres = "no"

hat_raw = _s("hat_presence")
hat_pres = _norm(hat_raw, _PRESENCE_NORM)
if hat_pres == "unknown" and "hat" not in summary and "cap" not in summary:
    hat_pres = "no"

mask_raw = _s("is_wearing_mask")
mask_pres = _norm(mask_raw, _PRESENCE_NORM)
if mask_pres == "unknown" and "mask" not in summary:
    mask_pres = "no"
```

**Tác động:** `bag_presence`, `hat_presence`, `is_wearing_mask` sẽ được gán `"no"` khi summary không nhắc đến item tương ứng, thay vì `"unknown"`.

---

#### Nguyên nhân 4: Confidence semantics bị hỏng

✅ **ĐÃ FIX** (`video_process.py:578`)

Fix 1 — `_f()` giờ trả `None` cho `0.0`:
```python
def _f(key: str) -> float | None:
    try:
        val = float(parsed[key])
        return val if val > 0.0 else None  # 0.0 = VLM placeholder → treat as missing
    except (KeyError, TypeError, ValueError):
        return None
```

Fix 2 — Thêm confidence = `None` khi field = `"unknown"`:
```python
gender_val = _norm(_s("gender"), _GENDER_NORM)
gender_conf = _f("gender_conf")
if gender_val == "unknown":
    gender_conf = None

age_val = _norm(_s("age_range"), _AGE_NORM)
age_conf = _f("age_range_conf")
if age_val == "unknown":
    age_conf = None
```

**Ghi chú về `hair_conf`:** VLM prompt (`_VLM_PROMPT`, `_VLM_BATCH_PROMPT_TEMPLATE`) đã có field `"hair_conf": 0.0` trong JSON schema. Code `attrs.get("hair_conf")` đọc đúng key. Kết quả `hair_conf` phụ thuộc vào VLM có replace placeholder hay không. Sau fix `_f()`, nếu VLM không replace thì `hair_conf` → `None` (thay vì `0.0` lưu vào DB).

**Tác động:**
- `conf=0.0` không còn lưu vào DB → tránh hiển thị `0.00` trên UI
- Gender/age unknown → conf = `None` thay vì `0.0` → query merge logic không bị bypass sai

---

### 14.3 — Flow: Crop → VLM → DB

```mermaid
flowchart TD
    A[Video file] -->|VideoFrameSampler 4fps| B[SampledFrame list]
    B -->|RT-DETR batch detect threshold=0.25| C[FrameDetection list]
    C -->|BodyPartAdaptiveTracker| D[LocalTracklet list]
    D -->|TrackletQualityScorer min_conf=0.25 min_frames=2| E[accepted tracklets]
    E -->|TrackletFragmentMerger sim=0.85 gap=60s| F[merged tracklets]
    F -->|_batch_extract_features| G{GPU Models}
    G -->|DINOv2 ViT-L/14| H[embedding_vector 1024-dim]
    G -->|SigLIP2 image encoder| I[siglip_embedding 1152-dim]
    G -->|_extract_crop mid_frame 384x384 MARGIN_15% gray_pad| J[rep_crop PIL]
    J -->|_caption_crops_vlm_batch Qwen2-VL-7B| K[VLM JSON output]
    K -->|_parse_vlm_attrs| L[attrs dict]
    L -->|_build_appearance_summary| M[appearance_summary string]
    G -->|VideoMAE V2 16 frames| N[action label]
    F -->|_project_to_bev_single BEVProjector| O[bev_x bev_y]
    O -->|exception path| P[bev_x=0.0 bev_y=0.0]
    L & M & H & I & N & O -->|TrackletResult construction| Q[TrackletResult Pydantic]
    Q -->|rep_crop.save JPEG q=85| R[/workspace/storage/crops/filename.jpg]
    R -->|crop_url = /static/crops/filename| Q
    Q -->|ingest_service._save_tracklets_from_gpu_result| S[(DB: tracklets table)]
    H & I -->|TrackletEmbedding| T[(DB: tracklets_embeddings)]
    N -->|TrackletAction if != unknown| U[(DB: tracklets_actions)]
```

| Step | File | Function | Input | Output | Field tạo ra | Field bị mất/unknown | Evidence |
|------|------|----------|-------|--------|--------------|----------------------|----------|
| 1. Frame sampling | `tracking_pipeline.py` | `VideoFrameSampler.sample()` | video file | SampledFrame list @ 4fps | `timestamp_second` | — | `video_process.py:1600` |
| 2. Person detection | `video_process.py` | `_detect_persons_batch()` | frames list | FrameDetection bbox+score | `confidence → quality_score` | — | `video_process.py:1616` |
| 3. Tracking | `tracking_pipeline.py` | `BodyPartAdaptiveTracker.track()` | FrameDetection by frame | LocalTracklet observations | `track_id` (local) | — | `video_process.py:1658` |
| 4. Quality filter | `tracking_pipeline.py` | `TrackletQualityScorer.score()` | LocalTracklet | QualityResult.average_confidence | `quality_score = 0.897` | tracklets < min_frames=2 dropped | `video_process.py:1669` |
| 5. Fragment merge | `tracking_pipeline.py` | `TrackletFragmentMerger.merge()` | accepted list + embeddings | merged list + groups | `t_idx` (final index) | — | `video_process.py:1705` |
| 6. BEV projection | `video_process.py` | `_project_to_bev_single()` | rep_bbox | bev_x, bev_y | `bev_x=0.0, bev_y=0.0` | **bev_x/bev_y zeroed on exception** | `video_process.py:1746` |
| 7. Crop extraction | `video_process.py` | `_batch_extract_features._extract_crop()` | mid_frame + rep_bbox | 384×384 BGR crop | `all_rep_crops` | **no margin → head/feet may be cut** | `video_process.py:1394–1411, 1416` |
| 8. VLM captioning | `video_process.py` | `_caption_crops_vlm_batch()` | PIL crops | VLM JSON attrs | `appearance_summary`, `upper/lower_clothing_*`, `top/bottom_color` | `gender=unknown, age=unknown, hat/bag/mask=unknown, hair=unknown, conf=0.0` | `video_process.py:1487–1490` |
| 9. Embedding | `video_process.py` | `_batch_extract_features` DINOv2+SigLIP blocks | PIL crops | 1024+1152-dim vectors | `embedding_vector`, `siglip_embedding` | — | `video_process.py:1424–1479` |
| 10. Action | `video_process.py` | `_batch_extract_features` VideoMAE block | 16-frame clips | action label + conf | `action`, `action_confidence` | — | `video_process.py:1511–1551` |
| 11. Save crop | `video_process.py` | `_process_video_sync` loop | all_rep_crops[t_idx] | JPEG file on disk | `crop_url` | — | `video_process.py:1769–1776` |
| 12. Save to DB | `ingest_service.py` | `_save_tracklets_from_gpu_result()` | TrackletResult dict | Tracklet + TrackletEmbedding + TrackletAction rows | all DB fields | conf fields 0.0 → DB stores 0.0 not NULL | `ingest_service.py:313–408` |

---

### 14.4 — Đánh giá 4 giả thuyết P1–P4

#### P1: Crop quá chặt
**Verdict: PARTIALLY CONFIRMED**

**Phân tích:**
`_extract_crop_for_vlm()` tại `video_process.py:1144–1164`:
```python
x1, y1, x2, y2 = map(int, bbox)       # raw detector bbox
h, w = frame.shape[:2]
x1, y1 = max(0, x1), max(0, y1)       # clip to frame boundary
x2, y2 = min(w, x2), min(h, y2)
crop = frame[y1:y2, x1:x2]            # KHÔNG có margin
```

- Không có context margin (e.g., 15–20% expand).
- Padding là **màu đen** `(0,0,0)` — có thể confuse VLM với background.
- Với tracklet 125 (start_s=550.5), camera góc nhìn surveillance từ trên → đầu người thường bị cắt bởi top của bbox. Hair, hat sẽ không thấy → `hair_style=unknown`, `hat_presence=unknown`.
- Crop VLM và crop lưu `crop_url` là **cùng crop** — `/static/crops/...jpg` là ảnh người mặc gray sweater + black pants rõ ràng (VLM đọc được), nhưng đầu/chân bị cắt.
- **Điều xác nhận:** Clothing colors (gray sweater, black pants, white sneakers) được nhận diện đúng → crop đủ để thấy thân người. Nhưng demographic (gender, age), hair, hat/bag bị mất → các thuộc tính này cần nhìn đầu và vùng xung quanh.

**Proposed patch:**
```python
# _extract_crop_for_vlm(): thêm context margin 20%
MARGIN = 0.20
dw = (x2 - x1) * MARGIN
dh = (y2 - y1) * MARGIN
x1 = max(0, int(x1 - dw))
y1 = max(0, int(y1 - dh))
x2 = min(w, int(x2 + dw))
y2 = min(h, int(y2 + dh))
# Đổi padding màu từ (0,0,0) sang (128,128,128) — neutral gray
padded = cv2.copyMakeBorder(
    crop, top, bottom, left, right,
    cv2.BORDER_CONSTANT, value=(128, 128, 128)
)
```

---

#### P2: Prompt quá dễ rơi về unknown
**Verdict: CONFIRMED**

**Phân tích:**
Prompt hiện tại (`video_process.py:821–861`):
```
"gender": "man or woman or unknown",
"hat_presence": "yes or no or unknown",
"bag_presence": "yes or no or unknown",
"is_wearing_mask": "yes or no or unknown",
"hair_style": "short or long or ponytail or tied or bald or unknown",
"hair_color": "color or unknown",
...
Use "unknown" for anything not clearly visible.
Do not infer gender or age from clothing alone.
Replace all 0.0 placeholders with your actual confidence (0.0–1.0).
```

**Vấn đề cụ thể:**
1. `"unknown"` xuất hiện trong mọi field description như option ngang hàng với các giá trị thực → model dùng unknown như default safe answer.
2. `"Replace all 0.0 placeholders with your actual confidence"` — instruction này KHÔNG HIỆU QUẢ với Qwen2-VL: khi không tự tin, model giữ nguyên `0.0` thay vì tính toán.
3. Không có instruction phân biệt: `"If you can see the upper body clearly but see no hat, use 'no' for hat_presence, not 'unknown'. Use 'unknown' only if head is completely out of frame."` → model lazy-safe.
4. Batch prompt `_VLM_BATCH_PROMPT_TEMPLATE` còn đơn giản hơn (bỏ instruction về confidence đối với gender/age).

**Proposed patch:**
```
# Thay thế instruction cuối trong _VLM_PROMPT:
"""
IMPORTANT RULES:
1. Use "unknown" ONLY when the body part is completely out of frame or obscured by occlusion.
   - If upper body visible but no hat seen → hat_presence="no", hat_conf≥0.75
   - If person visible but no bag carried → bag_presence="no", bag_conf≥0.75  
   - If face visible but no mask → is_wearing_mask="no", mask_conf≥0.80
2. For confidence: 0.9=very clear, 0.8=clear, 0.7=somewhat clear, 0.5=uncertain. Do NOT use 0.0.
3. gender and age_range: use "unknown" only if face/body is truly ambiguous.
"""
```

---

#### P3: Pipeline không repair từ summary/desc
**Verdict: CONFIRMED**

**Phân tích:**
Tìm kiếm toàn bộ `video_process.py` — không tồn tại function nào có tên:
- `_repair_vlm_attrs_from_text()`
- `_repair_vlm_attrs_from_summary()`
- bất kỳ post-processing nào đọc `appearance_summary` để fill structured attrs

`_build_appearance_summary()` tại `video_process.py:1125–1141` chỉ đi một chiều: `attrs → summary`, không có chiều ngược lại `summary → attrs`.

**Cụ thể với record này:**
- `appearance_summary = "A person wearing a gray sweater, black pants, and white sneakers."` — không nhắc tới bag/hat/mask → có thể infer absence.
- `upper_clothing_desc` có thể chứa text như "gray sweater" → đã có.
- Không có code khai thác `appearance_summary` để điền `hat_presence="no"`, `bag_presence="no"`, `is_wearing_mask="no"`.

**Proposed patch — `_repair_vlm_attrs_from_text(attrs: dict) -> dict`:**
```python
def _repair_vlm_attrs_from_text(attrs: dict) -> dict:
    """Post-hoc repair: fill unknown fields from appearance_summary/desc.
    Only repairs when field is 'unknown'/None/empty.
    Does NOT overwrite already-known values.
    Does NOT repair gender or age_range (too risky to infer).
    """
    summary = (attrs.get("appearance_summary") or "").lower()
    desc_upper = (attrs.get("upper_clothing_desc") or "").lower()
    desc_lower = (attrs.get("lower_clothing_desc") or "").lower()
    desc_shoes = (attrs.get("shoes_desc") or "").lower()
    all_text = " ".join([summary, desc_upper, desc_lower, desc_shoes])

    def _is_unknown(val):
        return not val or str(val).strip().lower() in ("unknown", "none", "", "null")

    # hat_presence repair
    if _is_unknown(attrs.get("hat_presence")):
        hat_keywords = {"hat", "cap", "beanie", "helmet", "hood", "headband", "beret"}
        if any(k in all_text for k in hat_keywords):
            attrs["hat_presence"] = "yes"
            attrs["hat_conf"] = attrs.get("hat_conf") or 0.72
        elif summary and len(summary) > 20:
            # summary exists and doesn't mention hat → infer absence
            attrs["hat_presence"] = "no"
            attrs["hat_conf"] = attrs.get("hat_conf") or 0.72

    # bag_presence repair
    if _is_unknown(attrs.get("bag_presence")):
        bag_keywords = {"backpack", "handbag", "purse", "tote", "bag", "suitcase", "luggage"}
        if any(k in all_text for k in bag_keywords):
            attrs["bag_presence"] = "yes"
            attrs["bag_conf"] = attrs.get("bag_conf") or 0.72
        elif summary and len(summary) > 20:
            attrs["bag_presence"] = "no"
            attrs["bag_conf"] = attrs.get("bag_conf") or 0.72

    # is_wearing_mask repair
    if _is_unknown(attrs.get("is_wearing_mask")):
        mask_keywords = {"mask", "face mask", "face covering", "respirator"}
        if any(k in all_text for k in mask_keywords):
            attrs["is_wearing_mask"] = "yes"
            attrs["mask_conf"] = attrs.get("mask_conf") or 0.75
        elif summary and len(summary) > 20:
            attrs["is_wearing_mask"] = "no"
            attrs["mask_conf"] = attrs.get("mask_conf") or 0.75

    # hair_style repair from desc (không từ summary vì summary quá ngắn)
    if _is_unknown(attrs.get("hair_style")):
        hair_map = {
            "long hair": "long", "short hair": "short", "ponytail": "ponytail",
            "tied hair": "tied", "bun": "tied", "braids": "tied", "bald": "bald",
        }
        for kw, val in hair_map.items():
            if kw in all_text:
                attrs["hair_style"] = val
                break

    # hat_type from hat_desc if hat_presence is yes
    if attrs.get("hat_presence") == "yes" and _is_unknown(attrs.get("hat_type")):
        hat_type_map = {
            "cap": "cap", "baseball": "cap", "beanie": "beanie",
            "helmet": "helmet", "hood": "hood", "hat": "hat",
        }
        hat_desc_text = (attrs.get("hat_desc") or "").lower()
        for kw, val in hat_type_map.items():
            if kw in hat_desc_text or kw in all_text:
                attrs["hat_type"] = val
                break

    return attrs
```

---

#### P4: Confidence semantics bị hỏng
**Verdict: CONFIRMED**

**Phân tích:**

**Root cause 1 — VLM không replace 0.0 placeholder:**
Prompt line 861: `"Replace all 0.0 placeholders with your actual confidence (0.0–1.0)."` — Qwen2-VL không được fine-tune để báo calibrated confidence, nên giữ nguyên `0.0` từ template JSON trong output.

**Root cause 2 — `hair_conf` field không tồn tại trong VLM prompt:**
Prompt yêu cầu `"hair_conf": 0.0` nhưng không có trong `_VLM_PROMPT`:
- `_VLM_PROMPT` có: `"hair_style"`, `"hair_color"`, `"hair_conf"` — thực tế `hair_conf` **CÓ** trong prompt line 855.
- Tuy nhiên trong `_parse_vlm_attrs()` line 963: `"hair_conf": _f("hair_conf")` — VLM phải trả về key `"hair_conf"` (không phải `"hair_style_conf"`).
- Trong `_batch_extract_features()` line 1506-1507: code đọc `attrs.get("hair_conf")` — nếu VLM trả về đúng thì có. Cần verify VLM có thực sự trả về `hair_conf` key không.

**Root cause 3 — `top_color=gray, top_conf=0.00` semantic mismatch:**
```python
# _batch_extract_features():1496
"top_color_conf": attrs.get("upper_clothing_conf"),  # mapped từ VLM's upper_clothing_conf
```
VLM điền `upper_clothing_color = "gray"` (đúng) nhưng `upper_clothing_conf = 0.0` (template không được replace). DB lưu `top_color_conf = 0.0` (float, không phải NULL).

**Root cause 4 — Query-service không phân biệt conf=0.0 và conf=None:**
`candidates.py:331`: `c1 = getattr(t1, conf_field, None) or 0.0` — cả `None` và `0.0` đều trở thành `0.0`. `0.0 < threshold` → bỏ qua conflict. Kết quả: metadata với clothing color known (gray, black) nhưng conf=0.0 **không được dùng** để guard merge. Hai người khác nhau có thể bị merge sai nếu có embedding similarity cao.

**Proposed patch — `_normalize_vlm_confidences(attrs: dict) -> dict`:**
```python
def _normalize_vlm_confidences(attrs: dict) -> dict:
    """Assign synthetic confidence for known-but-zero-confidence fields.
    Logic: if value is 'known' (not unknown/none) but conf is 0.0 → assign synthetic.
    If value is 'unknown' and conf is 0.0 → set conf to None (truly unknown).
    Does NOT synthesize gender or age_range confidence.
    """
    SYNTHETIC_CONF = {
        "bag_presence":         0.78,
        "hat_presence":         0.78,
        "is_wearing_mask":      0.78,
        "hat_color":            0.84,
        "upper_clothing_color": 0.84,
        "lower_clothing_color": 0.84,
        "shoes_color":          0.84,
        "hair_color":           0.80,
        "hair_style":           0.80,
        # bag_type and hat_type inherit from bag_conf/hat_conf
    }
    CONF_MAP = {
        "bag_presence":         "bag_conf",
        "hat_presence":         "hat_conf",
        "is_wearing_mask":      "mask_conf",
        "hat_color":            "hat_conf",
        "upper_clothing_color": "upper_clothing_conf",
        "lower_clothing_color": "lower_clothing_conf",
        "shoes_color":          "shoes_conf",
        "hair_color":           "hair_conf",
        "hair_style":           "hair_conf",
    }
    _UNKNOWN_SET = {"unknown", "none", "", "null", None}

    for field, conf_key in CONF_MAP.items():
        val = attrs.get(field)
        conf = attrs.get(conf_key)
        if val in _UNKNOWN_SET:
            # unknown value + 0 conf → None (signal truly not extracted)
            attrs[conf_key] = None
        elif conf is None or (isinstance(conf, float) and conf <= 0.0):
            # known value but zero/None conf → synthetic
            attrs[conf_key] = SYNTHETIC_CONF.get(field, 0.75)

    # hat_type and bag_type inherit confidence from presence conf
    if attrs.get("hat_type") not in _UNKNOWN_SET:
        attrs.setdefault("hat_type_conf", attrs.get("hat_conf"))
    if attrs.get("bag_type") not in _UNKNOWN_SET:
        attrs.setdefault("bag_type_conf", attrs.get("bag_conf"))

    return attrs
```

---

### 14.5 — Summary

| Vấn đề | Verdict | Root Cause | Proposed Fix | Priority |
|--------|---------|-----------|--------------|----------|
| UNKNOWN gender/age | CONFIRMED | (1) Prompt dặn "do not infer from clothing" — conservative quá. (2) Face không rõ trong surveillance crop. (3) Không có repair từ summary. | Cải thiện prompt instruction + không cần repair (gender/age không thể infer từ clothing) | MEDIUM |
| UNKNOWN hat/bag/mask | CONFIRMED | (1) Crop chặt — đầu/accessories bị cắt. (2) Prompt cho phép "unknown" quá dễ. (3) Không có repair từ absence-in-summary inference. | `_repair_vlm_attrs_from_text()` + cải thiện prompt: "if body visible but no hat → use 'no'" | HIGH |
| UNKNOWN hair_style/hair_color | CONFIRMED | (1) Crop chặt cắt đầu người. (2) `hair_conf` key có trong prompt nhưng cần verify VLM thực sự trả về. | `_extract_crop_for_vlm()` thêm margin 20% ở top + `_repair_vlm_attrs_from_text()` | HIGH |
| top_conf=0.00 dù top_color=gray (confirmed known) | CONFIRMED — Bug semantics | VLM không replace 0.0 placeholder. `top_color_conf` được map từ `upper_clothing_conf` — VLM giữ 0.0. DB lưu 0.0 float, không phải NULL. | `_normalize_vlm_confidences()`: known value + conf≤0 → synthetic conf 0.84; unknown + conf=0 → None | HIGH |
| bev_x/bev_y = 0.00 trong sample | PARTIALLY CONFIRMED | BEV **có được tính** (BEVProjector implemented). Nhưng khi `camera_id` không có trong calibration file, fallback silent về `(0.0, 0.0)` — không có flag `bev_valid`. Sample tracklet `cam_04` không có trong `camera_calibration.json` (chỉ có `cam01`, `cam02`). | Thêm `bev_valid: bool` vào TrackletResult; đảm bảo camera_id match calibration. Xem Phần 20. | MEDIUM |
| `hair_style_conf`/`hair_color_conf` luôn None | CONFIRMED — Bug code | `_batch_extract_features():1506-1507` đọc `attrs.get("hair_conf")` — cần verify VLM thực sự output key này. Nếu VLM output đúng `hair_conf` thì OK; nếu không → luôn None. | Verify VLM output format; nếu cần thêm `"hair_conf"` explicit vào prompt instruction. | LOW |
| Metadata không được dùng khi conf=0.00 trong merge | CONFIRMED — Design flaw | `candidates.py:331`: `or 0.0` transform None→0.0, và `0.0 < threshold` → skip. Kết quả: clothing color biết được nhưng không guard merge. | `_normalize_vlm_confidences()` thêm synthetic conf → conf > threshold → guard hoạt động. Hoặc tách logic: known+0conf → dùng default threshold thấp hơn (0.6). | HIGH |
| **Representative frame chọn bằng middle index** | CONFIRMED — Suboptimal | `video_process.py`: `obs[len(obs)//2]` — frame giữa timeline, không phải frame tốt nhất. Có thể chọn frame bị che khuất, mờ, hoặc người quay lưng. | Thêm `_best_observation(obs)` — xem 14.6 bên dưới. | HIGH |

---

### 14.6 — Representative Frame Selection: Vấn đề và Fix

#### Vấn đề hiện tại

`t_data` được build bằng cách lấy frame **giữa** timeline của tracklet:

```python
# video_process.py — t_data construction (2 chỗ)
mid = obs[len(obs) // 2]   # frame giữa — không phải frame tốt nhất
rep_bbox_float = [float(x) for x in mid.bbox]
```

`rep_bbox_float` và frame tương ứng được dùng làm input cho **cả SigLIP2 lẫn Qwen2-VL** thông qua `all_rep_crops` — nên chọn sai frame ở đây ảnh hưởng cả hai model cùng lúc.

**Tại sao middle frame tệ:** Người có thể đang quay lưng, bị che khuất một nửa, hoặc frame bị motion blur — đặc biệt ở đầu/cuối tracklet khi tracker vừa khởi tạo hoặc mất track.

#### `TrackletObservation` đã có đủ dữ liệu

```python
# tracking_pipeline.py:69-75
class TrackletObservation:
    frame_index:      int
    timestamp_second: float
    bbox:             tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence:       float                      # RT-DETR detector score
    laplacian_score:  float                      # sharpness score (pre-computed)
    crop_bgr:         Optional[np.ndarray]
```

Ba thành phần cần cho scoring **đã có sẵn** — không cần tính thêm.

#### Fix: `_best_observation(obs)`

```python
def _best_observation(obs: tuple) -> TrackletObservation:
    """Score = 0.5 * bbox_area_norm + 0.3 * laplacian_norm + 0.2 * confidence_norm"""
    if len(obs) == 1:
        return obs[0]

    def _norm(vals):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [1.0] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    areas = [(o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]) for o in obs]
    laps  = [o.laplacian_score for o in obs]
    confs = [o.confidence      for o in obs]

    scores = [0.5*a + 0.3*l + 0.2*c
              for a, l, c in zip(_norm(areas), _norm(laps), _norm(confs))]
    return obs[scores.index(max(scores))]
```

**Cho old code path** (detection dicts, không có laplacian):

```python
def _best_detection(group: list[dict]) -> dict:
    """70% bbox_area + 30% confidence — laplacian không có sẵn trong dict."""
    if len(group) == 1:
        return group[0]
    # ... tương tự nhưng bỏ laplacian
```

#### Chỗ cần thay thế

| Chỗ | File | Pattern cũ | Thay bằng |
|-----|------|-----------|-----------|
| t_data pre-merge | `video_process.py:~1424` | `mid = obs[len(obs) // 2]` | `mid = _best_observation(obs)` |
| t_data post-merge | `video_process.py:~1473` | `mid = obs[len(obs) // 2]` | `mid = _best_observation(obs)` |
| Old path single-video | `video_process.py:~975` | `mid_det = group[len(group) // 2]` | `mid_det = _best_detection(group)` |

**Lưu ý:** Chỉ cần sửa `rep_bbox_float` trong `t_data` construction — `all_rep_crops` được build từ đây nên SigLIP2 và Qwen2-VL đều tự động dùng frame tốt hơn.

#### Weights hợp lý

| Thành phần | Weight | Lý do |
|-----------|--------|-------|
| bbox area (50%) | Cao nhất | Người chiếm nhiều diện tích → ít bị cắt, nhìn rõ toàn thân |
| laplacian sharpness (30%) | Trung bình | Frame sắc nét → VLM nhận diện attribute chính xác hơn |
| detector confidence (20%) | Thấp nhất | RT-DETR thường confident đồng đều trong tracklet; ít discriminative |

---

## PHẦN 15 — EMBEDDING FIELDS: TẠI SAO CÓ 3 FIELD CHO CÙNG MỘT TRACKLET? *(lịch sử — đã cleanup)*

> **Cập nhật 2026-05-11:** `embedding_vector` và `embedding` đã bị xóa. Phần này ghi lại lý do tồn tại và quá trình cleanup. Trạng thái hiện tại: chỉ còn `siglip_embedding` (pgvector 1152-dim).

### 15.1 — Ba field embedding trong bảng `tracklets_embeddings`

```
models.py:381   embedding_vector  →  JSON             (DINOv2, 1024 số)  ← LEGACY
models.py:382   embedding         →  pgvector(1024)   (DINOv2, 1024 số)  ← CURRENT
models.py:385   siglip_embedding  →  pgvector(1152)   (SigLIP2, 1152 số) ← CURRENT
```

`embedding_vector` và `embedding` **lưu cùng một vector DINOv2**, nhưng khác kiểu lưu trong DB:

| Field | Kiểu DB | Vector search (`<->`, `<=>`) | Thời điểm xuất hiện | Evidence |
|-------|---------|------------------------------|---------------------|----------|
| `embedding_vector` | `JSON` (mảng số thông thường) | **Không** — DB không biết đây là vector | Pipeline cũ | `models.py:381` |
| `embedding` | `pgvector(1024)` | **Có** — hỗ trợ ANN index (IVFFlat/HNSW) | Pipeline mới | `models.py:382–384` |
| `siglip_embedding` | `pgvector(1152)` | **Có** | Pipeline mới | `models.py:385–387` |

Comment trong code xác nhận rõ:
```python
# models.py:381
embedding_vector: Mapped[list] = mapped_column(JSON, nullable=False)  # DINOv2 JSON (backward compat)
```

**Lý do tồn tại song song:** Khi pgvector được thêm vào, team tạo field mới `embedding` thay vì migrate dữ liệu từ `embedding_vector`, và giữ nguyên field cũ để code đang chạy không bị vỡ.

---

### 15.2 — Query-service đọc field nào?

`candidates.py:350–359` — hàm `_tracklet_embedding()`:

```python
def _tracklet_embedding(t: Tracklet) -> list[float]:
    """SigLIP2 embedding preferred; fallback to DINOv2."""
    if not t.embedding:
        return []
    if t.embedding.siglip_embedding is not None:   # ưu tiên SigLIP2
        return list(t.embedding.siglip_embedding)
    return list(t.embedding.embedding_vector or []) # fallback → JSON legacy
```

Thứ tự ưu tiên thực tế:

```
1. siglip_embedding (pgvector 1152)  → dùng nếu có
2. embedding_vector (JSON legacy)    → fallback nếu SigLIP2 null
3. embedding (pgvector 1024)         → BỊ BỎ QUA HOÀN TOÀN
```

**`embedding` (pgvector DINOv2) không nằm trong fallback chain** — query-service nhảy thẳng từ SigLIP2 xuống JSON legacy, bỏ qua field pgvector DINOv2 hoàn toàn.

---

### 15.3 — Pipeline ghi field nào?

| Code path | `embedding_vector` (JSON) | `embedding` (pgvector) | `siglip_embedding` (pgvector) | Evidence |
|-----------|--------------------------|----------------------|-------------------------------|----------|
| Code path cũ (`_process_single_video`) | **Ghi** | Không ghi | Không ghi | `video_process.py:1296` |
| Code path mới (`_process_video_sync`) | **Ghi** | Không ghi | **Ghi** | `video_process.py:1819–1820` |

Cả hai code path đều **không ghi vào `embedding` (pgvector DINOv2)** — field đó luôn `NULL` trong DB thực tế.

---

### 15.4 — Sơ đồ tổng quan

```
Pipeline cũ                  Pipeline mới
      │                            │
      ▼                            ▼
embedding_vector (JSON) ✓   embedding_vector (JSON) ✓
                             embedding (pgvector)    ✗  ← không ghi
                             siglip_embedding        ✓

                    Query-service đọc:
                    ┌─────────────────────────────────┐
                    │ siglip_embedding có? → dùng      │  (pgvector, nhanh)
                    │ không → embedding_vector (JSON)  │  (JSON, chậm, không index)
                    │ embedding (pgvector)?  → bỏ qua  │  ← bug/dead field
                    └─────────────────────────────────┘
```

---

### 15.5 — Hệ quả thực tế

| Tình huống | Kết quả |
|-----------|---------|
| Tracklet có `siglip_embedding` | Search dùng pgvector ANN — nhanh, có index |
| Tracklet chỉ có `embedding_vector` (JSON) | Search dùng cosine tính tay — chậm, không index, O(N) |
| Tracklet có `embedding` (pgvector) nhưng không có SigLIP2 | Field bị bỏ qua, fallback về JSON — pgvector DINOv2 lãng phí |
| Tracklet từ code path cũ | Không có SigLIP2 → toàn bộ search dùng JSON cosine |

**Risk:** Nếu `siglip_embedding` null (SigLIP2 fail khi ingest), `_tracklet_embedding()` trả `[]` — tracklet bị excluded khỏi merge, không có log. *(Concern "tracklet cũ" không còn — DB wipe.)*

---

### 15.6 — Recommendation *(đã thực hiện 2026-05-11)*

| Action | Mô tả | Status |
|--------|-------|--------|
| ~~Xóa `embedding` (pgvector DINOv2)~~ | Field này null 100% | **DONE** — `DROP COLUMN embedding` |
| ~~Thêm log/metric khi fallback về `embedding_vector`~~ | Fallback không còn tồn tại | **DONE** — `embedding_vector` đã bị xóa |
| ~~Backfill `siglip_embedding` cho tracklet cũ~~ | ~~Data cũ không có siglip_embedding~~ | **KHÔNG CẦN** — DB sẽ được wipe |
| ~~Đổi tên `embedding_vector` → legacy~~ | Không còn field này | **DONE** — đã drop |

---

## PHẦN 16 — TẠI SAO CÓ NHIỀU LEGACY / DEAD CODE / FALLBACK?

> Tổng hợp toàn bộ từ audit thực tế. Mỗi mục có evidence cụ thể.

---

### 16.1 — Bản đồ tiến hóa kiến trúc (root cause chung)

Repo TraceX-AI đã trải qua **ít nhất 3 thế hệ pipeline** mà không có migration plan rõ ràng:

```
Gen 1 — Grounding DINO + SigLIP2 label-based
    │  detection: GDINO
    │  attributes: SigLIP2 zero-shot label matching
    │  embedding: DINOv2 → JSON column (embedding_vector)
    │  ORM alias: VideoAsset, VideoQuery, PersonCandidate
    ▼
Gen 2 — RT-DETR thay GDINO, giữ SigLIP2
    │  detection: RT-DETR primary, GDINO fallback còn trong code
    │  attributes: vẫn SigLIP2 label-based
    │  embedding: thêm pgvector column (embedding, siglip_embedding)
    │  top_color/bottom_color vẫn là primary columns
    ▼
Gen 3 — Qwen2-VL thay SigLIP2 attribute
       detection: RT-DETR, GDINO fallback dead
       attributes: Qwen2-VL VLM (upper_clothing_*, lower_clothing_*, ...)
       embedding: siglip_embedding pgvector là primary
       top_color/bottom_color → backward compat copy từ upper/lower_clothing_color
       ORM rename: Video, QueryHistory, QueryCandidate (alias cũ vẫn tồn tại)
```

Mỗi lần chuyển thế hệ, team **thêm mới thay vì xóa cũ** → tích lũy debt.

---

### 16.2 — Inventory đầy đủ: Legacy / Dead Code / Fallback / Silent Failure

> **Cập nhật 2026-05-11:** Các mục A1–A6, B1–B5 (legacy/dead code) đã được xóa khỏi codebase. Xem chi tiết bên dưới. Mục C (silent fallback) và E (hardcode) chưa sửa — theo dõi ở 16.4.

#### NHÓM A — Dead Code ✅ ĐÃ XÓA

| # | Item | Trạng thái | Thay đổi |
|---|------|-----------|---------|
| A1 | `_legacy_run_siglip2_label_attributes()` + label maps | **Đã xóa** | `video_process.py`: xóa toàn bộ Stage 6 block (~150 dòng) |
| A2 | `_detect_persons()` (single-frame GDINO) | **Đã xóa** | `video_process.py`: xóa function; `_detect_persons_batch()` nay `raise RuntimeError` thay vì return `[]` im lặng |
| A3 | `embedding` column (pgvector DINOv2 1024) | **Đã xóa khỏi model** | `models.py`: xóa field; SQL: `ALTER TABLE tracklets_embeddings DROP COLUMN embedding` |
| A4 | `VideoQuery = QueryHistory` alias | **Đã xóa** | `models.py`: xóa alias; `video_service.py`: đổi sang `QueryHistory` trực tiếp |
| A5 | `PersonCandidate = QueryCandidate` alias | **Đã xóa** | `models.py`: xóa alias; `candidate_query.py`: đã xóa 2026-05-11 |
| A6 | `VideoAsset = Video` alias | **Đã xóa** | `models.py` + `shared/__init__.py`: xóa alias và export |

---

#### NHÓM B — Legacy song song với bản mới ✅ ĐÃ XÓA

| # | Item | Trạng thái | Thay đổi |
|---|------|-----------|---------|
| B1 | `top_color` / `bottom_color` columns | **Đã xóa** | `models.py`: xóa 2 columns; `ingest_service.py`: xóa backward compat write; `candidates.py`: `_build_search_text` + `_search_text_expr` đổi sang `upper/lower_clothing_color`; SQL: `DROP COLUMN top_color, bottom_color` |
| B2 | `embedding_vector` (JSON legacy) | **Đã xóa** | `models.py`: xóa field; `ingest_service.py`: chỉ ghi `siglip_embedding`; `candidates.py`: `_tracklet_embedding()` chỉ dùng `siglip_embedding`; SQL: `DROP COLUMN embedding_vector` |
| B3 | `VideoQuery` alias dùng trong `video_service.py` | **Đã sửa** | `video_service.py`: toàn bộ `VideoQuery` → `QueryHistory` (replace_all) |
| B4 | GDINO fallback trong `_detect_persons_batch()` | **Đã xóa** | Thay bằng `raise RuntimeError` — fail rõ ràng thay vì silent `return []` |
| B5 | `top_color_conf` / `bottom_color_conf` columns | **Đã xóa** | `models.py`: xóa 2 conf columns; `ingest_service.py`: xóa backward compat conf write; `candidates.py`: xóa khỏi `_CONF_THRESHOLDS` và `_metadata_matches` checks; SQL: `DROP COLUMN top_color_conf, bottom_color_conf` |

---

#### NHÓM C — Silent Fallback (fail im lặng, không alert)

| # | Item | File | Line | Hành vi khi fail | Risk |
|---|------|------|------|-----------------|------|
| C1 | RT-DETR fail → GDINO fallback → model None → `return []` | `video_process.py` | 119-121 | Zero detections, tracklet rỗng. Log level: WARNING. Không có metric/alert | **HIGH** |
| C2 | `_batch_extract_features()` VideoMAE fail → fallback | `video_process.py` | 1552-1553 | `logger.warning` rồi tiếp tục — action_label = None/default | MEDIUM |
| C3 | `_project_to_bev_single()` exception → `bev_x=0.0, bev_y=0.0` | `video_process.py` | ~1239-1240 | BEV có được tính khi calibration file có camera_id đúng. Silent fallback `(0,0)` khi camera_id không match — không có flag `bev_valid`. Xem Phần 20. | MEDIUM |
| C4 | `except Exception: all_frames[video_id] = []` | `video_process.py` | 1919-1920 | Frame load fail → video được xử lý với 0 frame → 0 tracklet | **HIGH** — toàn bộ video bị bỏ qua im lặng |
| C5 | `_tracklet_embedding()` fallback về JSON khi SigLIP2 null | `candidates.py` | 351-359 | Không log, không metric — không biết % search đang dùng pgvector vs JSON | MEDIUM |
| C6 | `except (KeyError, TypeError, ValueError): pass` trong parse VLM | `video_process.py` | 926 | Parse field fail → field giữ nguyên giá trị default `unknown` / 0.0 | MEDIUM |
| C7 | `user_id: int = 1` khi không có JWT | `candidates.py` | 35 | Direct call không có auth → gán user_id=1 hardcoded | HIGH — security |
| C8 | `except Exception` không log trong SigLIP2 encode | `video_process.py` | 1566 | `all_siglip_embeddings = [[]] * len(t_data)` — toàn batch mất embedding SigLIP2 im lặng | HIGH |

---

#### NHÓM D — Stale Comment / Docstring không khớp code

| # | Item | File | Line | Comment nói gì | Code thực sự làm gì |
|---|------|------|------|----------------|---------------------|
| D1 | `# Stage 2: Person Detection (RT-DETR primary / GDINO legacy fallback)` | `video_process.py` | 112 | Nói "GDINO legacy fallback" | GDINO model không được load → fallback return `[]`, không phải detect thật |
| D2 | `_legacy_run_siglip2_label_attributes` docstring: *"kept for debug/fallback only"* | `video_process.py` | 523 | Nói "for debug/fallback" | Không có call site nào — không được gọi trong bất kỳ path nào |
| D3 | `# backward compat (old SigLIP columns)` trong candidates.py | `candidates.py` | 275, 315 | Nói "old SigLIP columns" | Thực ra là `top_color`/`bottom_color` — copy từ VLM, không phải từ SigLIP |
| D4 | `VideoQuery = QueryHistory` comment: *"Existing code using VideoQuery.video FK needs migration"* | `models.py` | 814-815 | Cảnh báo cần migration | Migration chưa được thực hiện — `video_service.py` vẫn dùng `VideoQuery` |
| D5 | `embedding_vector` comment: *"DINOv2 JSON (backward compat)"* | `models.py` | 381 | Có vẻ như optional legacy | Vẫn là trường DUY NHẤT được ghi bởi code path cũ và fallback cuối của query-service |

---

#### NHÓM E — Hardcode không có env override

| # | Item | File | Line | Giá trị | Risk |
|---|------|------|------|---------|------|
| E1 | DB password `"Mcpt@2026Secure"` | `database.py` | 17 | Hardcoded, khác với `config.py:83` (`"Mcpt2026Secure"`) — **2 giá trị khác nhau** | **CRITICAL** |
| E2 | Merge threshold `_MERGE_THRESHOLD = 0.85` | `candidates.py` | 257 | Không configurable qua env/settings | MEDIUM |
| E3 | Fragment similarity threshold `0.85` | `video_process.py` | 1703 | `TrackletFragmentMerger(similarity_threshold=0.85)` | MEDIUM |
| E4 | `max_gap_seconds=60.0` | `video_process.py` | 1703 | Hardcoded trong constructor | MEDIUM |
| E5 | `user_id: int = 1` | `candidates.py` | 35 | Default user cho unauthenticated call | HIGH |

---

### 16.3 — Tại sao lại tích lũy nhiều thế?

**Nguyên nhân có thể nhận ra từ pattern trong code:**

| Nguyên nhân | Bằng chứng trong code |
|-------------|----------------------|
| **Không có migration plan** khi đổi thế hệ | `models.py:807-819`: 3 alias class với comment *"backward compat"* nhưng không có deadline xóa |
| **Thêm mới thay vì replace** | `embedding` pgvector thêm cạnh `embedding_vector` JSON, cả hai tồn tại song song, không có cutover date |
| **"Kept for debug"** không có ngày xóa | `_legacy_run_siglip2_label_attributes` docstring nói "kept for debug" nhưng không có TODO/FIXME với date |
| **Fallback che giấu failure** | 16 `except Exception` trong `video_process.py`, nhiều cái return `[]` hoặc `pass` không log đủ → bug ẩn mình |
| **Schema không có migration tool** | Không có Alembic → thêm column bằng `create_all` idempotent, không bao giờ drop column cũ được |
| **Copy thay vì rename** | `top_color` không bị xóa khi có `upper_clothing_color` — thay vào đó pipeline copy giá trị sang, query đọc cả hai |

---

### 16.4 — Ma trận ưu tiên còn lại (sau cleanup 2026-05-11)

> Nhóm A và B đã được dọn. Còn lại nhóm C (silent fallback) và E (hardcode).

| Priority | Action | Nhóm | Lý do | Effort |
|----------|--------|------|-------|--------|
| **P0 — Fix ngay** | Fix 2 DB password khác nhau (`database.py:17` vs `config.py:83`) | E1 | CRITICAL — 2 service có thể không connect được | 5 phút |
| **P0 — Fix ngay** | Fix `EvidenceVideo.selected_candidate_id` không tồn tại trong model (`trace_service.py:371`) | — | Runtime `AttributeError` khi gọi `continue_trace` | 15 phút |
| ~~**P0 — Chạy ngay**~~ | ~~Chạy SQL migration `remove_legacy_columns.sql`~~ | B1/B2/B5 | **KHÔNG CẦN** — DB sẽ được wipe, schema tạo mới từ `create_all()` | — |
| **P1 — Sprint này** | Thêm log khi SigLIP2 embedding fail (C8) | C8 | Mất toàn batch embedding im lặng | 15 phút |
| **P1 — Sprint này** | `_normalize_vlm_confidences()` — fix confidence semantics (Phần 14 P4) | C6 | conf=0 với known value → metadata không guard merge | 2h |
| **P2 — Sprint sau** | Fix `except Exception: all_frames[video_id] = []` (C4) | C4 | Toàn video bị bỏ qua im lặng | 30 phút |
| **P2 — Sprint sau** | `user_id: int = 1` khi không có JWT (C7/E5) | C7 | Security — direct call không có auth | 1h |
| **P3 — Backlog** | Thêm Alembic migration | — | Không có cách safe thay đổi schema về sau | 1 ngày setup |

---

## PHẦN 17 — AUDIT PIPELINE CŨ vs MỚI (2026-05-11)

> Mục tiêu: tìm chỗ nào trong 3 service đang dùng pipeline cũ (EVA-02, file-based JSON metadata) thay vì pipeline mới (PostgreSQL + SQLAlchemy ORM).

### 17.1 — Đặc trưng nhận dạng

| Đặc trưng | Pipeline cũ | Pipeline mới |
|-----------|------------|--------------|
| Nguồn dữ liệu | `raw_metadata` JSON blob / file JSON trên disk | SQLAlchemy ORM: `Tracklet`, `TrackletEmbedding` |
| Embedding model | EVA-02 ViT-L/14 (1024-dim) | SigLIP2 So400m (1152-dim) |
| Embedding field | `embedding_vector`, `attribute_embedding_vector`, `appearance_embedding_vector` từ dict | `TrackletEmbedding.siglip_embedding` pgvector |
| Fields đặc trưng | `human_key`, `frame_idx`, `metadata_path`, `search_text`, `raw_metadata` | `tracklet_id`, `upper_clothing_color`, `siglip_embedding` |
| Class | `PersonCandidate` (alias cũ) | `QueryCandidate`, `QueryCandidateTracklet` |

---

### 17.2 — Kết quả audit từng file

#### metadata-service

| File | Pipeline | Mô tả | Status |
|------|---------|-------|--------|
| `app/services/queue_service.py` | **CŨ + MỚI** | `load_queue_video_metadata()` fallback đọc JSON từ `local_metadata_path`; `_upsert_person_candidates()` convert PersonCandidate cũ → Tracklet. **Dead code sau DB wipe** — không còn old data để migrate. | 🟡 Xóa sau wipe |
| `app/queue_worker.py` | **MỚI** | Upsert `Tracklet`, `TrackletEmbedding`, `TrackletAction` via SQLAlchemy. Không có EVA-02 hay raw_metadata. | ✅ Đúng |
| `app/api/routers/candidates.py` | **MỚI** | Phục vụ crop preview từ `/workspace/storage/crops/` via Tracklet ORM. | ✅ Đúng |
| `app/services/ground_truth_finetune.py` | **HYBRID** | DINOv2 primary nhưng fallback EVA-02: `model = self._get_model("dinov2") or self._get_model("eva02")` — line 859. EVA-02 không cần thiết. | ⚠️ Xóa fallback EVA-02 |
| `app/api/routers/video_process.py` | **MỚI** | RT-DETR + SigLIP2 (fragment merge) + Qwen2-VL + VideoMAE. DINOv2 đã xóa (2026-05-11). | ✅ Đúng |

---

#### query-service

| File | Pipeline | Mô tả | Status |
|------|---------|-------|--------|
| `app/api/routers/candidates.py` | **MỚI** | `_build_search_text()` dùng `upper_clothing_color`, `lower_clothing_color`. `_tracklet_embedding()` dùng `siglip_embedding`. Insert `QueryCandidate` + `QueryCandidateTracklet`. | ✅ Đúng |
| ~~`app/services/candidate_query.py`~~ | ~~CŨ — BỊ VỠ~~ | ~~Truy cập fields không tồn tại trên `QueryCandidate`.~~ | ~~🟡 ĐÃ XÓA 2026-05-11~~ |

---

#### trace-service

| File | Pipeline | Mô tả | Status |
|------|---------|-------|--------|
| `app/services/trace_service.py` | **MỚI** | Đọc `QueryCandidateTracklet`, build `EvidenceVideo`/`EvidenceTracklet` via SQLAlchemy. | ✅ Đúng |
| `app/api/routers/trace.py` | **MỚI** | Nhận `candidate_id`, gọi `trace_service`. | ✅ Đúng |
| `app/api/routers/candidates.py` | **MỚI** | Re-ranking: SigLIP 2 attribute match + text overlap + quality. EVA-02 comments removed 2026-05-11. | ✅ Đúng |
| ~~`app/services/model_warmup.py`~~ | ~~CŨ — REDUNDANT~~ | ~~Load EVA-02 ViT-L/14 + Grounding DINO 1.6 nhưng không được gọi.~~ | ~~🟡 ĐÃ XÓA 2026-05-11~~ |

---

### 17.3 — ~~CRITICAL: `query-service/app/services/candidate_query.py`~~ ✅ ĐÃ XÓA

**ĐÃ XÓA 2026-05-11.** File truy cập fields không tồn tại trên `QueryCandidate` (như `raw_metadata`, `human_key`, `frame_idx`, `embedding_vector`).


### 17.4 — ~~REDUNDANT: EVA-02 trong trace-service~~ ✅ ĐÃ XÓA

**ĐÃ XÓA 2026-05-11.** Đã xóa:
- `_load_eva02()` function và `_load_grounding_dino_16()` function khỏi `model_warmup.py`
- Stale comments và `_W_VECTOR` weights khỏi `trace-service/app/api/routers/candidates.py`

---

### 17.5 — Bảng tổng kết

| File | Service | Pipeline | Có active không | Việc cần làm |
|------|---------|---------|----------------|-------------|
| `queue_service.py` | metadata | CŨ+MỚI | 🟡 Dead code sau wipe | Xóa `_upsert_person_candidates()` + JSON path trong `load_queue_video_metadata()` |
| `queue_worker.py` | metadata | MỚI | ✅ | — |
| `ground_truth_finetune.py` | metadata | HYBRID | ✅ (unnecessary fallback) | Xóa fallback EVA-02 (line 859) |
| `candidates.py` (metadata router) | metadata | MỚI | ✅ | — |
| `video_process.py` | metadata | MỚI | ✅ (DINOv2 xóa 2026-05-11) | — |
| `candidates.py` (query router) | query | MỚI | ✅ | — |
| **`candidate_query.py`** | **query** | **CŨ** | ~~🔴 CRASH~~ ✅ ĐÃ XÓA | — |
| `trace_service.py` | trace | MỚI | ✅ | — |
| `trace.py` (router) | trace | MỚI | ✅ | — |
| `candidates.py` (trace router) | trace | ~~MỚI+comment cũ~~ ✅ | — |
| **`model_warmup.py`** (trace) | **trace** | ~~CŨ~~ ✅ ĐÃ XÓA | — |

---

### 17.6 — Ưu tiên xử lý

| Priority | Action | File | Effort | Status |
|----------|--------|------|--------|--------|
| ~~P0 — Sau DB wipe~~ | ~~Xóa `_upsert_person_candidates()`~~ | `metadata-service/app/services/queue_service.py` | 30 phút | 🟡 TBD |
| ~~P0~~ | ~~Rewrite `candidate_to_payload()`~~ | ~~`query-service/app/services/candidate_query.py`~~ | ~~2h~~ | ✅ **ĐÃ XÓA** 2026-05-11 |
| ~~P1~~ | ~~Xóa `_load_eva02()`~~ | ~~`trace-service/app/services/model_warmup.py`~~ | ~~15 phút~~ | ✅ **ĐÃ XÓA** 2026-05-11 |
| ~~P2~~ | ~~Xóa EVA-02 fallback~~ | ~~`metadata-service/app/services/ground_truth_finetune.py:859`~~ | ~~5 phút~~ | 🟡 TBD |
| ~~P3~~ | ~~Sửa stale EVA-02 comments~~ | ~~`trace-service/app/api/routers/candidates.py`~~ | ~~5 phút~~ | ✅ **ĐÃ SỬA** 2026-05-11 |
| ~~P4~~ | ~~Xóa DINOv2 khỏi pipeline~~ | ~~`video_process.py` — xóa `_generate_dinov2_embeddings`, thay DINOv2 batch bằng SigLIP2 multi-frame~~ | ~~1h~~ | ✅ **ĐÃ XÓA** 2026-05-11 |
| ~~P4~~ | ~~Xóa `_load_dinov2` khỏi model_warmup~~ | ~~`metadata-service/app/services/model_warmup.py`~~ | ~~15 phút~~ | ✅ **ĐÃ XÓA** 2026-05-11 |
| ~~P4~~ | ~~Cập nhật comment/docstring DINOv2→SigLIP2~~ | ~~`main.py`, `Dockerfile`, `requirements.txt`, `queue_worker.py`, `tracking_pipeline.py`, `video_process_schemas.py`, `ingest.py`~~ | ~~30 phút~~ | ✅ **ĐÃ SỬA** 2026-05-11 |

---

## PHẦN 18 — PIPELINE ORDER: SIGLIP → MERGE → QWEN+VMAE (2026-05-11)

> **Cập nhật 2026-05-11:** Pipeline được sửa để Qwen2-VL + VideoMAE chạy **SAU Fragment Merge** trên merged tracklet.
> Luồng chính xác:
> `RT-DETR → BodypartAdaptiveTracker → raw tracklets → SigLIP2 embed (5-frame pool-avg) → Fragment merge → merged tracklet → Qwen2-VL + VideoMAE → tracklet result → DB`
> DINOv2 đã bị xóa hoàn toàn. SigLIP2 là model appearance duy nhất.

### 18.1 — Thứ tự thực tế trong code

```
video_process.py — _process_video_sync()

Stage 1 : VideoFrameSampler           → sampled_frames (4fps)
Stage 2 : RT-DETR batch detect      → detections_by_frame
Stage 3 : BodyPartAdaptiveTracker   → local_tracklets (per-fragment)
Stage 4 : TrackletQualityScorer     → accepted raw fragments
Stage 5 : _batch_siglip_embeddings() — SigLIP2 multi-frame pool-avg TRƯỚC MERGE
           ├── siglip_multi_feats   → cho fragment merge (SigLIP cosine sim)
           └── all_siglip_embeddings → pool-avg cho DB storage
Stage 6 : TrackletFragmentMerger    → merged tracklets
           └── siglip_embeddings_merged : pool_avg(SigLIP embeds từ tất cả fragments)
Stage 7 : _batch_caption_and_classify() — Qwen2-VL + VideoMAE SAU MERGE
           ├── Qwen2-VL  → attributes trên merged tracklet (best frame by conf)
           └── VideoMAE   → action trên merged tracklet (tất cả frames trong group)
Stage 8 : Build TrackletResult     → ingest_service → DB
```

Evidence — thứ tự trong code:
```python
# Stage 5 — SigLIP trước merge
siglip_multi_feats, all_rep_crops, all_siglip_embeddings = _batch_siglip_embeddings(t_data_raw)

# Stage 6 — Fragment merge
_merger = TrackletFragmentMerger(similarity_threshold=0.85, max_gap_seconds=60.0)
accepted, _groups = _merger.merge(list(accepted), siglip_multi_feats)

# Pool-avg SigLIP sau merge
siglip_multi_feats_merged = [_pool_avg([siglip_multi_feats[i] for i in g]) for g in _groups]
siglip_embeddings_merged  = [_pool_avg([all_siglip_embeddings[i] for i in g]) for g in _groups]

# Rebuild t_data cho merged tracklets — best frame by detection confidence
def _best_obs_for_merged(g):  # highest conf obs across all fragments in group
    ...

t_data_merged, rep_crops_merged = build_merged_data(...)

# Stage 7 — Qwen2-VL + VideoMAE SAU merge
all_attributes, all_attr_confs, all_actions = _batch_caption_and_classify(t_data_merged, rep_crops_merged)
```

---

### 18.2 — Đánh giá từng bước

| Bước | Hiện tại | Lý tưởng | Verdict |
|------|---------|---------|---------|
| FragmentMerger dùng embedding để merge | SigLIP2 pool-avg 5 frames pre-merge | ✅ | ✅ Bắt buộc |
| Qwen2-VL captioning | **Chạy SAU merge trên merged tracklet** | ✅ | ✅ Đúng |
| VideoMAE action | **Chạy SAU merge trên merged tracklet** | ✅ | ✅ Đúng |
| Representative frame cho Qwen | Best frame by detection confidence (không phải mid-frame) | ✅ | ✅ Đúng |
| VideoMAE frames cho merged | Tất cả frames từ mọi fragment trong group | ✅ | ✅ Đúng |
| SigLIP2 multi-frame cho merge | 5 evenly-spaced frames → pool avg | ✅ | ✅ |
| DINOv2 | Đã xóa hoàn toàn | ✅ | ✅ |

**Pipeline hoàn toàn đúng theo yêu cầu:** SigLIP → Fragment Merge → Qwen2-VL + VideoMAE trên merged.

---

### 18.3 — Cải tiến so với logic cũ

- **Trước:** Qwen/VMAE chạy trước merge → copy attributes từ "richest" fragment  
- **Sau:** Qwen/VMAE chạy sau merge trên merged tracklet → attributes/action đại diện cho toàn bộ người

---

### 18.4 — Tóm tắt

- ✅ Pipeline thứ tự **đúng hoàn toàn** theo yêu cầu
- ✅ SigLIP2 → Fragment Merge → Qwen2-VL + VideoMAE trên merged tracklet
- ✅ DINOv2 xóa hoàn toàn khỏi pipeline
- ✅ Representative frame chọn best (highest conf) không phải mid-frame
- ✅ VideoMAE dùng tất cả frames từ mọi fragment trong group

---

## PHẦN 19 — TRACKLETS_ACTIONS: INSERT ĐÚNG, NHƯNG KHÔNG BAO GIỜ ĐƯỢC ĐỌC

### 19.1 — Vấn đề

`tracklets_actions` được INSERT đúng (VideoMAE V2 chạy, data vào DB), nhưng trong toàn bộ **new pipeline** (query-service/trace-service dùng SQLAlchemy ORM), không có bất kỳ query nào SELECT bảng này.

```
tracklets_actions
    INSERT ✅  ingest_service.py:396-404 + queue_worker.py:318-327
    SELECT ❌  Không có joinedload / select(TrackletAction) nào trong search/merge/ranking
```

### 19.2 — Xác minh từng chỗ

**1. `query-service/routers/candidates.py` — không load action**

```python
# candidates.py:187-188 — chỉ load embedding, không load actions
contains_eager(Tracklet.video),
joinedload(Tracklet.embedding),
# joinedload(Tracklet.actions) ← KHÔNG CÓ
```

Confirm bằng `grep -n "action" candidates.py` → **zero output**.

**2. `_build_search_text()` — không có action_label**

```python
# candidates.py — _build_search_text()
return " ".join([
    row.appearance_summary or "", row.gender or "",
    row.upper_clothing_color or "", row.lower_clothing_color or "",
    row.shoes_color or "", row.age_range or "",
    row.hat_color or "", row.bag_type or "",
    row.is_wearing_mask or "", row.hair_style or "", row.hair_color or "",
    # action_label ← KHÔNG CÓ
])
```

**3. `_metadata_matches()` — không check action conflict**

```python
# candidates.py — checks list
checks = [
    ("gender", "gender_conf"), ("upper_clothing_color", "upper_clothing_conf"),
    # ("action_label", ...) ← KHÔNG CÓ
]
```

**4. `trace_service.py` — zero references**

```bash
grep -n "action|TrackletAction" trace_service.py → (no output)
```

`trace-service/core/models.py` import `TrackletAction` nhưng chỉ để re-export — không có query nào dùng.

**5. Old pipeline paths — luôn trả rỗng trên schema mới**

```python
# candidate_query.py:83-88 (OLD pipeline)
timeline = person.get("timeline")   # từ raw_metadata — luôn None trên QueryCandidate mới
actions = [item.get("action_summary") for item in timeline ...]  # → luôn []

# trace-service/routers/candidates.py:179 (OLD pipeline)
for key in ("search_text", "appearance_summary", "attribute_summary", "action"):
    text = str(candidate.get(key) or "").lower()  # "action" từ dict passthrough — không từ TrackletAction ORM
```

### 19.3 — Tổng kết trạng thái

| Layer | Trạng thái | Evidence |
|-------|-----------|----------|
| VideoMAE inference | ✅ Chạy | `video_process.py:_run_videomae_actions()` |
| INSERT `tracklets_actions` | ✅ Đúng | `ingest_service.py:396-404`, `queue_worker.py:318-327` |
| Text search (new pipeline) | ❌ Không dùng | `candidates.py:_build_search_text()` — zero action field |
| ORM load | ❌ Không load | `candidates.py:188` — chỉ `joinedload(Tracklet.embedding)` |
| Metadata conflict guard | ❌ Không dùng | `candidates.py:_metadata_matches()` |
| Ranking / fusion score | ❌ Không dùng | `candidates.py:fusion_score` formula |
| trace-service (new pipeline) | ❌ Không dùng | `trace_service.py` — zero action references |
| Old pipeline paths | ~~⚠️ Có code nhưng luôn rỗng~~ ✅ Đã xóa | `candidate_query.py`: đã xóa 2026-05-11; `trace-service/routers/candidates.py`: stale code đã fix |

### 19.4 — Fix tối thiểu (Option A — không cần thêm join)

Thêm `action_label` vào `_build_search_text()` và `_search_text_expr()` — user query "người đang chạy" sẽ match `action_label=running`.

Cần thêm `joinedload(Tracklet.actions)` để ORM load data, sau đó lấy action_label từ actions[0] nếu có.

```python
# candidates.py — thêm vào _build_search_text()
row.actions[0].action_label if row.actions else "",

# candidates.py:188 — thêm joinedload
joinedload(Tracklet.embedding),
joinedload(Tracklet.actions),   # ← thêm
```

### 19.5 — Fix đã áp dụng (2026-05-11)

**Thay đổi 1 — ORM load:** `candidates.py:188` thêm `joinedload(Tracklet.actions)`.

**Thay đổi 2 — `_build_search_text()`:** action_label được join vào search text.

```python
# candidates.py:117-118 — action_label ghép vào text để ILIKE match
action_label = ""
if row.actions:
    action_label = " ".join(a.action_label or "" for a in row.actions)
...
action_label,   # ← thêm vào join
```

**Thay đổi 3 — `_search_text_expr()`:** correlated subquery join sang `tracklets_actions`.

**Thay đổi 4 — `_metadata_matches()`:** thêm action conflict guard.

**Thay đổi 5 — Import:** `TrackletAction` được import và dùng trong subquery.

### 19.6 — Priority

**MEDIUM** — VideoMAE đang tốn GPU compute để sinh action labels nhưng data không được dùng trong search. Không phải crash, nhưng là lãng phí tài nguyên và mất tính năng action-aware search hoàn toàn.

**Trạng thái sau fix:**

|| Layer | Trước | Sau |
|-------|-------|-------|
|| ORM load | ❌ | ✅ `joinedload(Tracklet.actions)` |
|| `_build_search_text()` | ❌ | ✅ action_label join |
|| `_search_text_expr()` SQL prefilter | ❌ | ✅ correlated subquery |
|| `_metadata_matches()` conflict guard | ❌ | ✅ action conflict check |
|| `tracklets_actions` INSERT | ✅ | ✅ (unchanged) |

---

## PHẦN 20 — BEV PROJECTION: HISTORICAL SNAPSHOT TRƯỚC CLEANUP 2026-05-11

### 20.1 — BEV có được tính

BEV **được implement đầy đủ** trong `shared/core/geometry.py` — `BEVProjector` class. Không phải dead code.

```
bbox foot-point (bottom-center)
→ BEVProjector.bbox_bottom_center_to_bev(x1, y1, x2, y2)
→ foot_x = (x1+x2)/2,  foot_y = y2
→ pixel_to_bev(foot_x, foot_y)
→ (bev_x, bev_y) mét trong world coordinate
```

### 20.2 — Cơ chế projection

`BEVProjector` hỗ trợ 2 mode tự động chọn:

| Mode | Khi nào | Cách tính |
|------|---------|-----------|
| **Homography** (preferred) | Calibration file có `homography` matrix (3×3) | `world = H_inv @ [px, py, 1]` — direct 2D→BEV |
| **Raycast** (fallback) | Không có H, chỉ có K/R/t | Undistort pixel → back-project ray → intersect Z=0 ground plane |

Evidence — `geometry.py:259-263`:
```python
if self._H is not None and self._H_inv is not None:
    return self._project_via_homography(px, py)   # H_inv @ pixel
if self._K is not None and self._R is not None and self._t is not None:
    return self._project_via_raycast(px, py)       # K/R/t ray-ground intersection
return (0.0, 0.0)
```

### 20.3 — Calibration files trong repo

| File | Format | Camera IDs có sẵn |
|------|--------|------------------|
| `backend/config/camera_calibration.json` | Legacy (`cameras` dict, K/R/t) | `cam01`, `cam02` |
| `storage/dataset/MTMC_Tracking_2024/train/scene_*/calibration_2025_format.json` | MTMC 2024 (`sensors` list, H matrix) | `Camera_0001`, `Camera_0002`, ... |

Default path được load:
1. `CAMERA_CALIBRATION_PATH` env var (ưu tiên cao nhất)
2. `/workspace/a20-root/config/calibration_2025_format.json` (deployment path)
3. `../../backend/config/camera_calibration.json` (relative fallback)

### 20.4 — Khi nào bev=(0,0)?

| Nguyên nhân | Hành vi |
|------------|---------|
| `CAMERA_CALIBRATION_PATH` không set + default path không tồn tại | `_load_calibration()` trả `{}` → `get_camera_params()` trả `None` → `_loaded=False` → `(0.0, 0.0)` |
| `camera_id` không có trong calibration file | Tương tự — log WARNING rồi trả `(0.0, 0.0)` |
| Homography matrix singular | `np.linalg.inv()` raise `LinAlgError` → `H_inv=None` → fallback raycast hoặc `(0.0, 0.0)` |

Sample tracklet `cam_04` trả `bev=(0,0)` vì `cam_04` không có trong `camera_calibration.json` (chỉ có `cam01`, `cam02`).

### 20.5 — Risk: Silent fallback

```python
# geometry.py:206-211
try:
    projector = BEVProjector(camera_id, cal_path)
except Exception as exc:
    logger.warning("BEVProjector failed for %s: %s", camera_id, exc)
    for d in detections:
        d["bev_x"] = 0.0   # ← không có flag phân biệt "thực sự tại origin" vs "no calibration"
        d["bev_y"] = 0.0
```

`bev_x=0.0, bev_y=0.0` có thể là:
- **Calibration fail** → không có nghĩa gì
- **Thực sự tại gốc tọa độ** → có ý nghĩa địa lý

Không có `bev_valid` flag → consumer không phân biệt được.

### 20.6 — BEV được dùng ở đâu

| Chỗ dùng | Purpose | Bị ảnh hưởng khi bev=(0,0)? |
|----------|---------|---------------------------|
| `_mcblt_associate()` | Cross-camera grouping theo khoảng cách BEV (max 1.5m) | ⚠️ Nếu bev=0,0 → mọi detection từ các camera đều nằm tại origin → có thể group sai |
| `tracklets.bev_x/bev_y` DB | Lưu để query sau | Lưu (0,0) không gây crash, nhưng useless |
| `ix_tracklets_bev_xy` index | Spatial lookup | Index trên (0,0) không có giá trị |

### 20.7 — Recommendation

| Action | Effort | Priority |
|--------|--------|----------|
| Đảm bảo `CAMERA_CALIBRATION_PATH` được set và có đủ camera IDs trong deployment | Ops | HIGH |
| Thêm `bev_valid: bool` vào `TrackletResult` — `True` khi BEVProjector load thành công | 30 phút | MEDIUM |
| Log ERROR (không chỉ WARNING) khi camera_id không có trong calibration | 5 phút | MEDIUM |

---

## PHẦN X — FRONTEND REACHECK (2026-05-11)

### Tổng quan

Kiểm tra frontend (`frontend/`) cho các tham chiếu đến code đã xóa theo audit report.

### Kết quả tìm kiếm

| Item | Tìm thấy | File(s) | Trạng thái |
|------|-----------|---------|------------|
| `bev` / BEV | ❌ KHÔNG | — | Sạch |
| `queue_service` | ❌ KHÔNG | — | Sạch |
| `load_queue_video_metadata` | ❌ KHÔNG | — | Sạch |
| `upsert_person_candidates` | ❌ KHÔNG | — | Sạch |

### Tham chiếu cần cập nhật

#### 1. `frontend/lib/api/client.ts` — Docker hostname cũ (line 78)

```typescript
// frontend/lib/api/client.ts:52-78
if (process.env.NODE_ENV === "server") {
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL;
  }
  return "http://metadata-service:8000";  // ⚠️ STALE - không khớp với lightningai-compose.yml
}
```

**Vấn đề:** `metadata-service:8000` là hostname cũ. Cần kiểm tra service name trong `lightningai-compose.yml` và cập nhật.

**Đề xuất:** Thay bằng `http://metadata-service:8002` hoặc hostname đúng từ compose file.

#### 2. `frontend/.env.example` — Comment tham chiếu service cũ

```bash
# .env.example:4
# Set this to the real LightningAI metadata-service URL.
NEXT_PUBLIC_API_BASE_URL=https://8002-XXXXXXXXXXXXXXXXXX.cloudspaces.litng.ai/api/v1
```

Comment ghi "metadata-service" nhưng URL thực tế dùng cloud URL. Port `8002` đúng với compose files.

#### 3. `frontend/middleware.ts` — Redirect cho URL cũ (line 10)

```typescript
// frontend/middleware.ts:1-14
/**
 * Tránh xung đột route: trang UI đặt tại /admin/users, còn POST/GET/PATCH /users proxy sang API gateway.
 * GET /users (bookmark cũ) chuyển hướng sang trang quản trị.
 */
export function middleware(request: NextRequest) {
  if (request.method === "GET" && request.nextUrl.pathname === "/users") {
    return NextResponse.redirect(new URL("/admin/users", request.url));
  }
```

**Trạng thái:** ACTIVE — Redirect cho bookmark cũ của người dùng. Không phải dead code.

#### 4. `frontend/lib/api/client.ts` — `/candidates/` URL path (lines 55, 91)

```typescript
// frontend/lib/api/client.ts:52-55
if (value.includes("/candidates/") && value.endsWith("/preview")) {
  return true;
}

// frontend/lib/api/client.ts:91-97
if (typeof window !== "undefined" && raw.includes("/api/v1/candidates/") && raw.endsWith("/preview")) {
  const token = loadAccessToken();
  if (token) {
    const separator = resolved.includes("?") ? "&" : "?";
    resolved = `${resolved}${separator}access_token=${encodeURIComponent(token)}`;
  }
}
```

**Trạng thái:** ACTIVE — Logic để attach access token cho candidate preview URLs. Không phải dead code.

#### 5. `frontend/features/settings/SettingsPage.tsx` — Queue metrics

```typescript
// frontend/features/settings/SettingsPage.tsx:21-29
type OverviewResponse = {
  metrics?: {
    total_users?: number;
    total_managed_videos?: number;
    total_queries?: number;
    total_candidates?: number;           // ⚠️ Defined nhưng KHÔNG hiển thị
    total_cameras?: number;
    total_candidate_videos?: number;     // ⚠️ Defined nhưng KHÔNG hiển thị
    total_queue_videos?: number;         // ✅ ACTIVE - hiển thị line 250
  };
};
```

**Trạng thái:**
- `total_queue_videos` — **ACTIVE**, được hiển thị trong UI
- `total_candidates` — **DEAD FIELD**, định nghĩa trong type nhưng không hiển thị
- `total_candidate_videos` — **DEAD FIELD**, định nghĩa trong type nhưng không hiển thị

### Action Items

| Action | File | Effort | Priority |
|--------|------|--------|----------|
| Cập nhật `metadata-service:8000` fallback hostname | `lib/api/client.ts:78` | 5 phút | MEDIUM |
| Xóa `total_candidates` và `total_candidate_videos` khỏi type | `features/settings/SettingsPage.tsx:26,28` | 5 phút | LOW |
| Kiểm tra đúng service name trong `lightningai-compose.yml` | N/A | — | HIGH |

### Kết luận

Frontend **không có tham chiếu** đến các code đã xóa trong audit (BEV, queue_service, upsert_person_candidates). Các item cần cập nhật chỉ là:
1. Stale Docker hostname trong SSR fallback
2. Dead type fields không được sử dụng trong Settings UI

---

## PHẦN Y — FIX SUMMARY: 14.2 UNKNOWN FIELDS (2026-05-11)

### Đã thực hiện

| # | Fix | File | Line | Status |
|---|-----|------|------|--------|
| 1 | Thêm context margin 15% + padding xám thay vì đen | `video_process.py` | 796 | ✅ Done |
| 2 | Cải thiện VLM prompt instruction | `video_process.py` | 476, 518 | ✅ Done |
| 3 | Repair presence fields từ `appearance_summary` | `video_process.py` | 572 | ✅ Done |
| 4 | Fix `conf=0.0` → `None`, gán `None` cho unknown fields | `video_process.py` | 572 | ✅ Done |

### Chi tiết từng fix

#### Fix 1: Crop extraction — context margin + gray padding

**Trước:**
```python
crop = frame[y1:y2, x1:x2]          # raw bbox — KHÔNG có margin
cv2.BORDER_CONSTANT, value=(0, 0, 0)  # padding đen
```

**Sau:**
```python
mx = int(bw * 0.15)   # expand 15% mỗi phía
x1_e = max(0, int(x1) - mx)
y1_e = max(0, int(y1) - my)
x2_e = min(frame.shape[1], int(x2) + mx)
y2_e = min(frame.shape[0], int(y2) + my)
crop = frame[y1_e:y2_e, x1_e:x2_e]
cv2.BORDER_CONSTANT, value=(114, 114, 114)  # gray thay vì đen
```

#### Fix 2: VLM Prompt — giảm "unknown" lazy

**Thêm `IMPORTANT INSTRUCTIONS`:**
- Gender: `"unknown" only if face/head not visible` (thay vì blanket rule)
- Bag/hat/mask: `"no" not "unknown"` khi upper body visible và item không có
- Hair: `"unknown" only if top of person not in frame`
- Nếu không phải "unknown" → phải có conf > 0.0

#### Fix 3: Parser repair từ summary

```python
summary = (parsed.get("appearance_summary") or "").lower()
if bag_pres == "unknown" and "bag" not in summary and "backpack" not in summary:
    bag_pres = "no"
if hat_pres == "unknown" and "hat" not in summary and "cap" not in summary:
    hat_pres = "no"
if mask_pres == "unknown" and "mask" not in summary:
    mask_pres = "no"
```

#### Fix 4: Confidence semantics

```python
def _f(key: str) -> float | None:
    val = float(parsed[key])
    return val if val > 0.0 else None  # 0.0 = placeholder → None

# Gender/age unknown → conf = None
if gender_val == "unknown":
    gender_conf = None
if age_val == "unknown":
    age_conf = None
```

### Tác động mong đợi

| Trường | Trước | Sau |
|--------|-------|-----|
| `hat_presence` | `"unknown"` (khi không nhắc) | `"no"` |
| `bag_presence` | `"unknown"` (khi không nhắc) | `"no"` |
| `is_wearing_mask` | `"unknown"` (khi không nhắc) | `"no"` |
| `upper_clothing_conf=0.0` | Lưu vào DB | `NULL` |
| `gender_conf` khi unknown | `0.0` | `NULL` |
| Hat/hair/gender | Unknown nhiều | Giảm nhờ margin + prompt |

---

## PHẦN 21 — VECTOR_SOCRE ĐO SAI: INTER-MEMBER COSINE THAY VÌ QUERY-VS-CANDIDATE COSINE

### 21.1 — Vấn đề (trước fix)

`vector_score` trong fusion formula đo **cosine similarity giữa các tracklet trong cùng group**, không phải giữa **query và candidate**:

```python
# TRƯỚC — candidates.py:504-511 (bug)
member_scores = []
for member in group:
    if member.tracklet_id == rep.tracklet_id:
        continue
    member_emb = _tracklet_embedding(member)
    if rep_emb and member_emb:
        member_scores.append(_cosine_sim(rep_emb, member_emb))
vector_score = max(member_scores) if member_scores else 0.0
```

Hệ quả:
- **Multi-tracklet group**: vector_score luôn ≥ 0.85 (vì group được tạo bởi merge threshold ≥ 0.85) → bão hòa, không discriminative
- **Single-tracklet group**: vector_score = 0.0 → fusion biến thành 0.7*text + 0.3*quality
- **Query text không được encode** → SigLIP2 text tower bị bỏ phí

### 21.2 — Công thức cũ

```
# multi-tracklet (len(group) > 1):
fusion = 0.5*text + 0.3*quality + 0.2*vector  → range [~0.17, ~0.97]

# single-tracklet (len(group) == 1):
fusion = 0.7*text + 0.3*quality               → range [0, 1.0]
```

Hai-tier scoring tạo bias không nhất quán: cùng text/quality, single luôn cao hơn multi.

### 21.3 — Fix đã áp dụng (2026-05-11)

**Thay đổi 1 — SigLIP2 text tower cho query-service:**

Viết lại `query-service/app/services/model_warmup.py`:
- Load `google/siglip2-so400m-patch14-384` (fp16, ~3GB VRAM) vào GPU lúc startup
- Warmup bằng 2 dummy text samples
- Model + processor được cache trong `_MODELS` dict

**Thay đổi 2 — `_build_query_embedding()` mới:**

```python
# candidates.py:370-397
def _build_query_embedding(query_text: str) -> list[float]:
    model = get_model("siglip2")
    processor = get_model("siglip2_processor")
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = model.get_text_features(...)
    vec = text_emb[0].cpu().float().numpy()
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()  # 1152-dim — cùng không gian với siglip_embedding
```

**Thay đổi 3 — Fusion dùng query-vs-candidate cosine:**

```python
# candidates.py:621-626 (sau fix)
vector_score = _cosine_sim(query_emb, rep_emb) if (query_emb and rep_emb) else 0.0
fusion_score = round(
    (0.5 * text_score) + (0.3 * quality_score) + (0.2 * vector_score), 4
)
```

**Thay đổi 4 — Two-tier scoring → unified với merge boost:**

Branch `if len(group) > 1` bị xóa. Công thức thống nhất:
```
fusion = 0.5*text + 0.3*quality + 0.2*vector + merge_boost
merge_boost = 0.05 * (len(group) - 1)
```
Single-tracklet: 0 boost; 2 tracklets: +0.05; 3 tracklets: +0.10.

### 21.4 — Tác động sau fix

| Trường hợp | Trước | Sau |
|---|---|---|
| Query "woman in red" | vector_score luôn ≥0.85 hoặc 0.0 | Discriminative cosine vs query |
| Single-tracklet group | fusion=0.7t+0.3q (không có vector) | 0.5t+0.3q+0.2v (có vector) |
| Multi-tracklet group | fusion range [~0.17, ~0.97] | range [~0.17, ~1.10] |
| SigLIP2 text tower | Bỏ phí (dead code trong trace-service) | Query-service dùng cho recall |

---

## PHẦN 22 — MULTI-BUG QUERY PIPELINE: PREFILTER / PAGINATION / MERGE GUARD

### 22.1 — Bug tổng hợp (trước fix)

|| # | Bug | Tác động |
||---|-----|----------|
|| 1 | Text-only prefilter: `_score_search_text > 0` loại candidate semantic đúng nhưng không match keyword (VD: "crimson top" ≠ "red shirt") | Recall thấp, query "người đang chạy" bỏ lỡ candidate đúng |
|| 4 | INSERT all trước, paginate sau: `merged[offset:top_k]` sau khi đã INSERT toàn bộ | Mỗi request tạo query_id mới, không tái sử dụng kết quả |
|| 5 | Merge guard: `conf=0.0` (VLM placeholder) không block conflict → "red shirt" và "blue shirt" với conf=0.0 merge được | False positive merge |

### 22.2 — Fix đã áp dụng (2026-05-11)

**Fix #1 — Vector-first recall + text rerank:**

Thêm `_vector_recall()` mới:

```python
# candidates.py:185-243
def _vector_recall(session, query_emb, camera_ids, time_from, time_to, recall_limit=500):
    # Load up to recall_limit*3 rows from DB
    # Compute cosine(query_emb, siglip_embedding) in Python
    # Return top-N sorted by descending cosine score
```

`_local_prefilter()` viết lại:
- **Vector-first path**: SigLIP2 encode query → cosine vs siglip_embedding → top-N → text rerank
- **Text-only fallback**: khi SigLIP2 unavailable hoặc query_emb rỗng

```python
# candidates.py:246-270
if query_emb:
    shortlist = _vector_recall(...)
    if shortlist:
        # Text rerank on vector-recalled candidates
        for row in shortlist:
            st = _build_search_text(row)
            ts = _score_search_text(st, cleaned_query, set(cleaned_query.split()))
            # sort and return
        return shortlist, text_score_map, True   # used_vector_recall=True

# Text-only fallback path (unchanged logic)
return shortlist, text_score_map, False
```

**Fix #4 — Paginate before INSERT:**

```python
# TRƯỚC: INSERT tất cả → rồi mới paginate
for rank_idx, item in enumerate(merged, start=1):
    db.execute(pg_insert(...))
paged = merged[offset:offset + top_k]

# SAU: Paginate trước → INSERT chỉ page hiện tại
paged = merged[offset:offset + top_k]
for rank_idx, item in enumerate(paged, start=offset + 1):
    db.execute(pg_insert(...))
```

**Fix #5 — Merge guard: conf ≤ 0.05 → block conflict:**

```python
# candidates.py:417-420
# VLM returns 0.0 as a literal placeholder when it didn't replace the template.
# 0.0 is indistinguishable from "no confidence" — must not block a merge.
if c1 <= 0.05 or c2 <= 0.05:
    continue  # at least one side is uncertain → don't block
```

### 22.3 — Tác động sau fix

| Bug | Trước | Sau |
|-----|-------|-----|
| #1 Recall | Text-only: keyword mismatch → candidate bị loại | Vector-first: semantic recall → text rerank |
| #4 Pagination | INSERT all 50 candidates dù page chỉ 10 | INSERT 10 candidate trên page |
| #5 Merge guard | conf=0.0 placeholder → false positive merge | conf≤0.05 coi như uncertain → không block |

### Chưa xử lý (deferred)

- **Hair style/color**: Phụ thuộc hoàn toàn vào VLM output. Có thể thêm repair rule từ summary cho hair color.
