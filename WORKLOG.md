# Worklog

Ghi lại các quyết định kỹ thuật, phân công, và brainstorming của nhóm.

> Cập nhật **bất cứ khi nào** nhóm ra quyết định kỹ thuật quan trọng hoặc thay đổi hướng đi.

---

## Template

### Quyết định kỹ thuật

```markdown
### [ADR-N] Tiêu đề quyết định — DD/MM/YYYY

**Bối cảnh:** Vấn đề cần giải quyết là gì?

**Các lựa chọn đã xem xét:**
- Option A: ...
- Option B: ...

**Quyết định:** Chọn option nào và tại sao.

**Hệ quả:** Những gì bị ảnh hưởng / trade-off.
```

### Phân công

```markdown
### Sprint N — DD/MM → DD/MM/YYYY

| Task | Người làm | Deadline | Trạng thái |
| ---- | --------- | -------- | ---------- |
|      |           |          |            |
```

### Brainstorming

```markdown
### Brainstorm: [Chủ đề] — DD/MM/YYYY

**Câu hỏi:** ...

**Các ý tưởng:**
- Ý tưởng 1: ...
- Ý tưởng 2: ...

**Kết luận:** ...
```

---

## Phân công nhóm — đề tài Semantic Video Search / Multi-Camera Person Tracking (AI20K-243)

**Thành viên:** Bùi Văn Đạt, Dương Văn Hiệp, Cao Diệu Ly.

**Ghi chú cập nhật:** Bảng dưới được tổng hợp lại từ `JOURNAL.md`, README hiện tại và lịch sử commit đến **16/05/2026**. Các đầu việc được chia theo sprint để phản ánh tiến độ thực tế: từ prototype Semantic Video Search ban đầu, chuyển sang pipeline Multi-Camera Person Tracking, rồi hoàn thiện search, trace, history, deploy và tuning model.

### Sprint 1 — 02/04 → 07/04/2026

| Task | Người làm | Deadline | Trạng thái |
| ---- | --------- | -------- | ---------- |
| Chốt đề tài, pain point và phạm vi MVP: tìm kiếm người trong video/camera bằng mô tả tự nhiên | Cả nhóm | 06/04 | ✅ Xong |
| Research bài toán semantic video search, dữ liệu CCTV, công cụ AI hỗ trợ và các hướng VLM/embedding | Cả nhóm | 07/04 | ✅ Xong |
| Khởi tạo project, starter code app và cấu hình AI prompt logging hooks | Bùi Văn Đạt | 07/04 | ✅ Xong |
| Ghi nhận rủi ro ban đầu: model demo đẹp chưa chắc phù hợp video dài và truy vết đa camera | Cao Diệu Ly | 07/04 | ✅ Xong |

### Sprint 2 — 08/04 → 12/04/2026

| Task | Người làm | Deadline | Trạng thái |
| ---- | --------- | -------- | ---------- |
| Làm MVP demo end-to-end mức đơn giản với backend API và giao diện thử nghiệm | Bùi Văn Đạt | 12/04 | ✅ Xong |
| Thử hướng LaVA / VLM cho truy vấn hình ảnh, đánh giá điểm lệch so với bài toán video search | Cả nhóm | 10/04 | ✅ Xong |
| Thử Qwen + ViT, ghi nhận vấn đề VRAM/GPU và tốc độ vòng lặp infer | Cao Diệu Ly | 12/04 | ✅ Xong |
| Chuẩn hóa README, ARCHITECTURE, JOURNAL và mô tả project identity / MVP / system architecture | Cả nhóm | 12/04 | ✅ Xong |
| Thêm unit tests ban đầu cho config, dataset parsing, enrichment inference, runtime và search stack | Dương Văn Hiệp | 12/04 | ✅ Xong |
| Cập nhật requirements cho surveillance search stack và dọn các file demo tạm | Dương Văn Hiệp | 12/04 | ✅ Xong |

### Sprint 3 — 13/04 → 19/04/2026

| Task | Người làm | Deadline | Trạng thái |
| ---- | --------- | -------- | ---------- |
| Chuyển trọng tâm từ prototype đơn lẻ sang kiến trúc data / DB → detection → tracking → embedding → API → frontend | Cả nhóm | 19/04 | ✅ Xong |
| Thêm YOLOv3/YOLOv4 và thử nghiệm các detector / embedding khác nhau cho bài toán người trong camera | Dương Văn Hiệp | 16/04 | ✅ Xong |
| Xây pipeline hospital / multi-camera person tracking và re-identification thử nghiệm | Dương Văn Hiệp | 16/04 | ✅ Xong |
| Triển khai tracking service với FastAPI và PostgreSQL, bắt đầu tách metadata service / tracking service | Dương Văn Hiệp | 17/04 | ✅ Xong |
| Phát triển frontend React, authentication flow và màn hình kết nối backend/model | Bùi Văn Đạt | 17/04 | ✅ Xong |
| Tích hợp Google Drive loading, PostgreSQL setup và upload/lưu trữ dữ liệu | Cao Diệu Ly | 19/04 | ✅ Xong |
| Refactor secret management, project initialization và OAuth2 cho Google Drive | Cao Diệu Ly | 19/04 | ✅ Xong |

### Sprint 4 — 20/04 → 26/04/2026

| Task | Người làm | Deadline | Trạng thái |
| ---- | --------- | -------- | ---------- |
| Refactor runtime configuration, deployment workflow và cấu trúc service để dễ chạy production hơn | Dương Văn Hiệp | 22/04 | ✅ Xong |
| Bổ sung báo cáo kỹ thuật cho kiến trúc Multi-Camera Person Tracking và ingestion flow | Cao Diệu Ly | 22/04 | ✅ Xong |
| Thêm Tailwind CSS / PostCSS setup và cập nhật cấu trúc frontend | Bùi Văn Đạt | 22/04 | ✅ Xong |
| Cập nhật Google Drive folder IDs / metadata, đồng bộ metadata-service và tracking-service | Cao Diệu Ly | 23/04 | ✅ Xong |
| Cải thiện ranked limit, cấu trúc dự án, README và nội dung hướng dẫn home | Bùi Văn Đạt | 25/04 | ✅ Xong |
| Thêm metadata JSON cho camera 16–31, triển khai storage ingestion và video moving | Dương Văn Hiệp | 26/04 | ✅ Xong |
| Dọn duplicated infra / secret directories để giảm nhầm lẫn khi deploy | Cả nhóm | 26/04 | ✅ Xong |

### Sprint 5 — 27/04 → 03/05/2026

| Task | Người làm | Deadline | Trạng thái |
| ---- | --------- | -------- | ---------- |
| Thêm user permissions, giao diện login mới, dashboard guide và luồng quản trị người dùng | Bùi Văn Đạt | 27/04 | ✅ Xong |
| Hoàn thiện Google Drive file management, public download URL, ingestion queue sync và error logging | Cao Diệu Ly | 28/04 | ✅ Xong |
| Xây candidate search request và pipeline xử lý candidate từ dữ liệu tracklet | Dương Văn Hiệp | 29/04 | ✅ Xong |
| Triển khai trace pipeline 6 stage cho person trajectory reconstruction | Dương Văn Hiệp | 29/04 | ✅ Xong |
| Tối ưu secret import/export, HTTP client management, weight path resolution và logging setup | Cao Diệu Ly | 30/04 | ✅ Xong |
| Cập nhật sidebar, TopK select, note time/location và ổn định preview image/auth forwarding | Bùi Văn Đạt | 01/05 | ✅ Xong |
| Triển khai async ingestion job queue, polling trạng thái, dedup source key và batch processing Google Drive | Dương Văn Hiệp | 01/05 | ✅ Xong |
| Refactor tracker theo hướng HeadBoxTracker / ByteTrack-style, tuning sample FPS, ReID thresholds và chunked DB commits | Dương Văn Hiệp | 03/05 | ✅ Xong |

### Sprint 6 — 04/05 → 10/05/2026

| Task | Người làm | Deadline | Trạng thái |
| ---- | --------- | -------- | ---------- |
| Thêm proxy candidate preview qua LightningAI và `/internal/candidates/preview` endpoint | Dương Văn Hiệp | 04/05 | ✅ Xong |
| Restructure services architecture, thêm AI/tracking services và tài liệu Quickstart/README cho A100 GPU | Cả nhóm | 05/05 | ✅ Xong |
| Sửa Docker/Coolify deploy: build context, frontend Dockerfile, Traefik routing, health check, DATABASE_URL và bcrypt | Cao Diệu Ly | 05/05 | ✅ Xong |
| Deploy được phiên bản đầu tiên trên hạ tầng hiện tại, ghi nhận mốc “đã deploy được” | Cả nhóm | 05/05 | ✅ Xong |
| Tích hợp video moving script với metadata service và API list videos | Bùi Văn Đạt | 06/05 | ✅ Xong |
| Sửa API router/session/DB critical crashes, bootstrap admin user, debug routes/config và Next.js standalone output | Dương Văn Hiệp | 07/05 | ✅ Xong |
| Hoàn thiện routing `/v1`, rewrites, CORS, `/auth/me`, loại hardcoded HF token và thống nhất API base URL | Cao Diệu Ly | 08/05 | ✅ Xong |
| Thêm RT-DETR person detection, candidates router, candidate preview, confidence scoring, user management và search UI | Cả nhóm | 09/05 | ✅ Xong |
| Bổ sung thuộc tính người: age range, hat color, bag type, mask wearing, hair style, hair color và schema xử lý video | Dương Văn Hiệp | 10/05 | ✅ Xong |

### Sprint 7 — 11/05 → 16/05/2026

| Task | Người làm | Deadline | Trạng thái |
| ---- | --------- | -------- | ---------- |
| Hoàn thiện database và query, kiểm tra lại luồng truy vấn dữ liệu mới | Dương Văn Hiệp | 11/05 | ✅ Xong |
| Update thêm data cho DB query để có dữ liệu kiểm thử search/trace | Bùi Văn Đạt | 11/05 | ✅ Xong |
| Kiểm tra DB mới và test luồng trace end-to-end | Cao Diệu Ly | 11/05 | ✅ Xong |
| Hoàn thiện DB schema để build và phục vụ các service phía sau | Dương Văn Hiệp | 12/05 | ✅ Xong |
| Thêm trang hiển thị toàn bộ tracklet của candidate | Bùi Văn Đạt | 12/05 | ✅ Xong |
| Tiếp tục xây DB và sửa lại trace theo schema mới | Cao Diệu Ly | 12/05 | ✅ Xong |
| Chạy luồng trace để kiểm tra dữ liệu và kết quả truy vết | Dương Văn Hiệp | 13/05 | ✅ Xong |
| Tối ưu hiển thị toàn bộ tracklet của candidate | Bùi Văn Đạt | 13/05 | ✅ Xong |
| Tối ưu pipeline, sửa logic query, thêm trang description và chạy full luồng trace | Cao Diệu Ly | 13/05 | ✅ Xong |
| Tiếp tục hoàn thiện trace và xử lý các lỗi trong luồng truy vết | Dương Văn Hiệp | 14/05 | ✅ Xong |
| Hoàn thiện chức năng cơ bản, sửa luồng lịch sử không lưu nhiều video | Bùi Văn Đạt | 14/05 | ✅ Xong |
| Sửa merge tracklet bị ghép nhầm, sinh lại DB để test, thêm trace nhiều candidate, xóa tracklet khỏi candidate và dịch tiếng Việt cho hiển thị | Cao Diệu Ly | 14/05 | ✅ Xong |
| Hoàn chỉnh luồng để submit | Dương Văn Hiệp | 15/05 | ✅ Xong |
| Test hoàn chỉnh các luồng search, candidate, trace và history | Bùi Văn Đạt | 15/05 | ✅ Xong |
| Sửa full luồng detect, track, merge tracklet, similarity và sinh lại DB | Cao Diệu Ly | 15/05 | ✅ Xong |
| Hoàn thiện DB cuối cho bản submit | Dương Văn Hiệp | 16/05 | ✅ Xong |
| Fix time UTC ở bộ lọc thời gian | Bùi Văn Đạt | 16/05 | ✅ Xong |
| Sửa logic query và fix trọng số tìm kiếm | Cao Diệu Ly | 16/05 | ✅ Xong |

---

## Quyết định kỹ thuật đã ghi nhận

### [ADR-1] Chuyển từ LaVA demo sang pipeline search/index/trace — 12/04/2026

**Bối cảnh:** LaVA phù hợp VQA hoặc hỏi đáp trên ảnh đơn, nhưng sản phẩm cần tìm kiếm và truy vết trong video dài / nhiều camera.

**Quyết định:** Tách hệ thống thành offline indexing và online search/trace. Pipeline ưu tiên detection, tracking, embedding, metadata, sau đó mới tới UI và workflow xác nhận của người vận hành.

**Hệ quả:** Phải refactor lại prototype ban đầu, nhưng hệ thống bám sát use case CCTV hơn và dễ tối ưu từng tầng.

### [ADR-2] Tách service và lưu metadata có cấu trúc — 19/04/2026

**Bối cảnh:** Pipeline model nặng, ingestion, search và frontend có tốc độ phát triển khác nhau.

**Quyết định:** Tách metadata-service, query-service, trace-service, dùng PostgreSQL làm kho metadata/tracklet/candidate, Google Drive làm nguồn video.

**Hệ quả:** Deploy phức tạp hơn, nhưng dễ mở rộng, debug và thay thế model.

### [ADR-3] Dùng LightningAI/A100 cho AI inference nặng — 05/05/2026

**Bối cảnh:** Qwen, SigLIP, detector và tracking pipeline gây áp lực lớn lên GPU/VRAM local.

**Quyết định:** Đẩy trace/AI inference sang LightningAI A100, VPS chạy frontend/backend/database/orchestration.

**Hệ quả:** Cần quản lý URL/token/secrets chặt hơn và xử lý cold start, nhưng throughput inference tốt hơn cho demo.

### [ADR-4] Ưu tiên human-in-the-loop trong luồng trace — 13/05/2026

**Bối cảnh:** ReID/search tự động có thể merge nhầm hoặc bỏ sót người khi camera đông, góc nhìn thay đổi hoặc tracklet vụn.

**Quyết định:** Cho người vận hành chọn candidate, xem evidence clip/history, xác nhận hoặc loại tracklet/candidate để trace lại.

**Hệ quả:** UX phức tạp hơn, nhưng giảm rủi ro kết quả sai và giúp hệ thống phù hợp kịch bản vận hành thật.
