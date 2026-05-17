# Weekly Journal

Ghi lại hành trình xây dựng sản phẩm mỗi tuần — những gì đã làm, học được gì, AI giúp như thế nào.

> **Cập nhật mỗi cuối tuần** (trước khi tạo PR). Không cần dài, chỉ cần thật.

---


## Template

```markdown
## Tuần N — DD/MM/YYYY

### Đã làm
-

### Khó nhất tuần này
-

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| Claude Code | | |

### Học được
-

### Nếu làm lại, sẽ làm khác
-

### Kế hoạch tuần tới
-
```

---

## Ví dụ

### Tuần 1 — 02/04/2026

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Quá trình chọn đề tài
- **Khởi đầu:** Nhóm brainstorm theo hướng “AI + thị giác máy tính”, liệt kê vài hướng (giám sát an ninh, tìm kiếm trong video, hỗ trợ người khiếm thị, v.v.).
- **Tiêu chí gạn lọc:** (1) có pain point rõ trong đời sống / doanh nghiệp, (2) có thể làm MVP với dữ liệu tìm được hoặc thu thập được, (3) không vượt quá năng lực GPU và thời gian môn học.
- **Chốt hướng:** Tập trung **Search Engine / truy vấn video bằng ngôn ngữ tự nhiên** (gần với camera giám sát, người dùng mô tả đối tượng cần tìm(người) thay vì xem lại cả giờ footage). Đề tài gắn mã AI20K-243, định hướng VLM + embedding (ví dụ CLIP) trong tài liệu đề xuất ban đầu.


### Đã làm
- Chốt thành viên nhóm, chốt đề tài và phạm vi bài toán (tìm kiếm ngữ nghĩa trong video).
- Học cách xác định **bài toán AI** (problem framing), phân biệt demo “đẹp” và bài toán deploy được.

### Khó nhất tuần này
- Đề tài lúc đầu **chưa khóa chặt pain point**.
- Chưa có data thật đủ tốt để validate ngay từ đầu.
### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Tìm hiểu bài toán, nhiều góc nhìn, scope và pain point | Hiểu rộng hơn về không gian bài toán và rủi ro chọn sai model |
| Perplexity | Research bài toán, phương pháp và thị trường | Tìm được hướng bài toán có dữ liệu khả thi hơn |

### Học được
- Học cách xác định bài toán cho AI một cách có kiểm chứng (pain point, metric, dữ liệu).

### Nếu làm lại, sẽ làm khác
- Xác định pain point từ người dùng / kịch bản thực tế, không chỉ từ paper.

### Kế hoạch tuần tới (lúc đó)
- Tìm hiểu sâu bài toán thực tế, thu thập / mock data, thử pipeline gần đúng với CCTV hơn.

---

## Tuần 02 - 12/04/2026 

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Đã làm
- Thu nhỏ scope sản phẩm, tập trung **một use case chính** (truy vấn bằng text, trả về đoạn video / kết quả liên quan) thay vì làm rộng.
- Code MVP demo với các tính năng cơ bản để validate ý tưởng; có thể chạy **end-to-end** ở mức đơn giản.
- **Thử nghiệm model:** hướng hiện tại dùng **Qwen** (xử lý ngôn ngữ / đa phương thức tùy cấu hình) kết hợp **ViT** (Vision Transformer) cho nhánh thị giác. Nhận thấy bộ đôi này **rất nặng**, chiếm nhiều **VRAM/GPU**, khó chạy ổn định trên máy cấu hình vừa phải → ảnh hưởng tốc độ thử nghiệm và khả năng demo liên tục.
- **Định hướng stack sản phẩm (web):** nhóm **chọn frontend React** để xây giao diện; **backend** xây dựng dạng **API** (phục vụ infer, dữ liệu, session người dùng) — tách bạch khỏi phần model để sau này thay đổi backbone hoặc tối ưu GPU dễ hơn. (Chi tiết triển khai có thể điều chỉnh theo repo và deadline.)

### Khó nhất tuần này
- **Dữ liệu:** tìm dataset / footage phù hợp bài toán CCTV-semantic search; data hiện có đôi khi **không đủ sát** hoặc **không đủ nhãn** → chất lượng output dao động, khó đánh giá đúng sai của model.
- **Di sản hướng LaVA / pipeline cũ:** vẫn phải dọn chỗ không khớp use case, đồng thời giữ tiến độ MVP.
- **Tài nguyên GPU:** Qwen + ViT khiến vòng lặp thử (train / infer) chậm và dễ OOM nếu không giảm batch hoặc không quantize / dùng model nhỏ hơn.

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Hỗ trợ code MVP, tài liệu, thu nhỏ scope | Có bản demo tối thiểu để test ý tưởng |

### Sai lầm kỹ thuật ban đầu: chọn LaVA (Large Language and Vision Assistant)
- **Lý do hồi đó:** LaVA nổi, dễ tìm tutorial “hỏi đáp trên ảnh”, demo trông ấn tượng nên nhóm tạm chọn làm “model chính” mà chưa khóa kỹ **bài toán đích**.
- **Vấn đề:** LaVA mạnh ở kiểu **VQA / mô tả ảnh / hội thoại đa phương thức**, không phải pipeline tối ưu cho **truy vấn ngữ nghĩa trên video dài, cắt clip, index theo thời gian** như CCTV. Nhóm lỡ **lệch đối tượng xử lý**: code và thử nghiệm xoay quanh format “một ảnh + một câu hỏi”, trong khi sản phẩm cần “video + truy vấn + trả về đoạn video / khung thời gian / embedding”.
- **Hệ quả tới hiện tại:** Phải **refactor lại kiến trúc** (tách bước embed, index, search), một phần thời gian đã đổ vào hướng không khớp use case; vẫn còn **nợ kỹ thuật** (chuẩn hoá dữ liệu, format output, đồng bộ giữa prototype cũ và MVP mới). Đây là một trong những “lỗi gốc” khiến team vừa làm MVP vừa phải sửa lại tư duy model.


### Học được
- Xác định đúng **pain point** quan trọng hơn nhồi nhiều tính năng; MVP là làm **đúng lõi** trước.
- **Data** quyết định lớn chất lượng, nhất là bài toán video + VLM.
- Cần cân nhắc **ngân sách GPU** ngay từ khi chọn backbone, không chỉ nhìn benchmark.

### Nếu làm lại, sẽ làm khác
- Validate và **mock pipeline + data** trước khi khóa bộ model nặng.
- Lên plan rõ **bài toán → metric → kích thước model tối đa** trên GPU đang có.

### Kế hoạch tuần tới
- Hoàn thiện MVP (ổn định hơn, xử lý edge case cơ bản).
- Cải thiện hoặc thu thập bộ data sát bài toán hơn.
- Cân nhắc **nhẹ hóa model** (distill, model nhỏ hơn, hoặc tách bước embed/search) để giảm áp lực GPU; tiếp tục chỉnh **frontend React + backend API** cho luồng người dùng thật.


---

## Tuần 03 - 19/04/2026

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Đã làm
- **Dương Văn Hiệp:** thử nghiệm thêm các đề xuất CPU-based theo góp ý office hour; tiếp tục tối ưu lưu trữ; tìm kiếm và thử các mô hình detection / embedding khác nhau để trích xuất metadata tốt hơn; xây dựng metric đánh giá, tối ưu độ chính xác; unit test và nối các luồng xử lý để chạy được pipeline.
- **Bùi Văn Đạt:** xây dựng frontend bằng React; nghiên cứu hướng giải quyết mới; tìm nguồn dataset; nghiên cứu cách lấy, lưu dataset và mô phỏng streaming data; thiết kế luồng kết nối backend, frontend và model.
- **Cao Diệu Ly:** phát triển lại architecture theo user và workflow mới; research và điều chỉnh architecture theo scope mới; triển khai thử nghiệm; research VLM hiệu quả hơn; đánh giá chất lượng tracklet sau tracking; tìm kiếm, setup data và dựng DB; lựa chọn model tracking tối ưu theo IDF1.
- Nhóm chuyển trọng tâm từ prototype rời rạc sang pipeline rõ hơn: data / DB → model detection, tracking, embedding → backend API → frontend React.
- Bắt đầu đánh giá hệ thống bằng metric cụ thể hơn thay vì chỉ nhìn output demo; đặc biệt quan tâm metadata, tracklet, IDF1 và khả năng chạy trên tài nguyên hạn chế.

### Chi tiết daily standup
- **13/04/2026:** Hiệp thử đề xuất CPU-based theo góp ý office hour; Đạt xây dựng FE bằng React; Ly phát triển architecture theo user và workflow mới.
- **14/04/2026:** Hiệp điều chỉnh theo hướng dẫn mentor về tối ưu lưu trữ; Đạt tìm hướng giải quyết bài toán mới; Ly research và thay đổi architecture theo scope mới.
- **15/04/2026:** Hiệp tìm mô hình embedding để trích xuất metadata tốt hơn; Đạt nghiên cứu nguồn dataset; Ly triển khai thử nghiệm.
- **16/04/2026:** Hiệp xây metric đánh giá và tối ưu độ chính xác; Ly research VLM hiệu quả hơn và đánh giá tracklet sau tracking; Đạt nghiên cứu cách lấy / lưu dataset, mô phỏng streaming data.
- **17/04/2026:** Hiệp thử nghiệm các model detection, embedding khác nhau; Ly đưa ra phương án cải thiện architecture mới; Đạt tiếp tục mô phỏng theo hướng streaming data.
- **18/04/2026:** Hiệp tiếp tục thử model embedding tốt hơn; Đạt thiết kế luồng BE, FE với model; Ly tìm kiếm, setup data và viết lại architecture.
- **19/04/2026:** Hiệp unit test và nối các luồng để chạy; Đạt tiếp tục thiết kế luồng FE / BE; Ly tìm và dựng DB, lựa chọn model track tối ưu IDF1.

### Khó nhất tuần này
- Vừa phải đổi kiến trúc theo scope mới, vừa phải giữ tiến độ triển khai thử nghiệm.
- Dataset và cách lưu / mô phỏng streaming data vẫn là phần tốn thời gian vì ảnh hưởng trực tiếp tới pipeline backend, frontend và model.
- Cần cân bằng giữa độ chính xác của detection / embedding / tracking và khả năng chạy trên CPU hoặc tài nguyên GPU hạn chế.

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Hỗ trợ brainstorm architecture, gợi ý hướng model / metric, rà soát pipeline và viết tài liệu | Có thêm hướng điều chỉnh architecture, metric đánh giá và cách diễn đạt journal rõ hơn |

### Học được
- Pipeline video search cần được thiết kế theo luồng dữ liệu trước: ingest / lưu trữ / tracking / embedding / search / hiển thị, không chỉ chọn model trước.
- Metric như IDF1, chất lượng tracklet và độ chính xác metadata giúp nhóm đánh giá thực tế hơn so với chỉ xem kết quả demo.
- Frontend, backend và model cần thống nhất contract sớm để tránh mỗi phần phát triển theo một giả định khác nhau.

### Nếu làm lại, sẽ làm khác
- Chốt format dataset, metadata và API contract sớm hơn trước khi thử nhiều model.
- Tách rõ thử nghiệm model, mô phỏng streaming data và phần giao diện để dễ đo tiến độ từng mảng.

### Kế hoạch tuần tới
- Hoàn thiện pipeline chạy được ổn định hơn từ data / DB đến model, backend và frontend.
- Tiếp tục so sánh model detection, embedding và tracking theo metric đã chọn.
- Chuẩn hoá dataset, metadata và API contract để phục vụ demo end-to-end.

---

## Tuần 04 - 26/04/2026

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Đã làm
- **Dương Văn Hiệp:** refactor runtime configuration và deployment workflow; tiếp tục làm gọn cấu trúc code; bổ sung metadata JSON cho camera 16–31; triển khai storage ingestion và video moving để chuẩn bị pipeline xử lý video theo lô.
- **Bùi Văn Đạt:** thêm Tailwind CSS / PostCSS setup; cập nhật cấu trúc frontend; sửa ranked limit; cập nhật README và nội dung home guide để giao diện bám sát sản phẩm hơn.
- **Cao Diệu Ly:** viết engineering code report cho kiến trúc Multi-Camera Person Tracking; cập nhật tài liệu AI architecture, core profiles, ingestion flow; cập nhật Google Drive folder IDs / names trong metadata-service và tracking-service.
- Nhóm bắt đầu chuyển rõ từ “demo tìm kiếm video” sang hệ thống **Multi-Camera Person Tracking & Re-Identification** có ingest, lưu metadata, tracking và search.
- Dọn các thư mục infra / secrets bị trùng để giảm nhầm lẫn khi cấu hình môi trường.

### Chi tiết daily standup
- **21/04/2026:** Hiệp refactor runtime configuration và deployment workflow; Đạt rà lại cấu trúc frontend; Ly kiểm tra lại kiến trúc tổng thể sau tuần đổi scope.
- **22/04/2026:** Hiệp tiếp tục refactor code; Đạt thêm Tailwind CSS và PostCSS; Ly bổ sung engineering code report và tài liệu AI architecture.
- **23/04/2026:** Ly cập nhật Google Drive folder IDs / names cho metadata-service và tracking-service; cả nhóm đồng bộ lại naming giữa Drive, DB và service.
- **25/04/2026:** Đạt sửa ranked limit, cập nhật cấu trúc dự án, dọn phần `/model` cũ và cập nhật README / home guide.
- **26/04/2026:** Hiệp thêm metadata JSON cho camera 16–31 và triển khai storage ingestion / video moving; nhóm dọn duplicated infra và secret directories.

### Khó nhất tuần này
- Naming giữa camera, Google Drive, metadata và backend service dễ lệch nhau, nếu không thống nhất sớm sẽ gây lỗi dây chuyền.
- Việc tách repo thành nhiều service làm hệ thống đúng hướng hơn nhưng khiến cấu hình môi trường và secrets phức tạp hơn.
- Cần cân bằng giữa viết tài liệu kiến trúc và tiếp tục code để pipeline chạy được.

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Hỗ trợ rà soát kiến trúc, viết lại tài liệu và mô tả luồng ingestion | Tài liệu rõ hơn, dễ giải thích pipeline hơn |
| Codex | Hỗ trợ refactor code, đọc repo và cập nhật tài liệu theo cấu trúc mới | Giảm thời gian dọn code / docs lặp lại |

### Học được
- Với hệ thống nhiều camera, naming convention và metadata contract quan trọng không kém model.
- Tài liệu kiến trúc cần bám sát code thật, nếu không sẽ nhanh chóng lệch khỏi sản phẩm.
- Nên chuẩn hóa luồng storage / ingestion trước khi tối ưu model sâu hơn.

### Nếu làm lại, sẽ làm khác
- Khóa sớm format camera ID, folder Drive và metadata schema để tránh phải sửa nhiều nơi.
- Tách rõ tài liệu “ý tưởng kiến trúc” và tài liệu “cách chạy production” ngay từ đầu.

### Kế hoạch tuần tới
- Hoàn thiện Google Drive ingestion và queue xử lý video.
- Xây candidate search request và trace pipeline rõ hơn.
- Chuẩn hóa secret management và luồng deploy lên VPS / GPU cloud.

---

## Tuần 05 - 03/05/2026

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Đã làm
- **Dương Văn Hiệp:** tối ưu queue runtime, remote endpoint, local ingestion batch processing và action clip builder; triển khai candidate search request; xây trace pipeline 6 stage; refactor tracker từ BoT-SORT / TransReID sang HeadBoxTracker, DINOv2 và within-camera identity resolution.
- **Bùi Văn Đạt:** triển khai user permissions, giao diện login, dashboard guide; cập nhật sidebar, TopK select, note time/location; sửa preview auth forwarding và ổn định hiển thị candidate image.
- **Cao Diệu Ly:** refactor secret export/import, sync secrets, HTTP client management và weight path resolution; rà soát deployment / environment setup; hỗ trợ cập nhật cấu hình VPS và Google Drive ingestion.
- Nhóm thêm async video ingestion job queue, status polling, dedup source key, progress logging và endpoint `/api/v1/ingestion/jobs`.
- Pipeline bắt đầu có luồng gần production hơn: Drive storage → queue → GPU processing → DB → search/candidate → trace.

### Chi tiết daily standup
- **27/04/2026:** Đạt thêm user permissions và cập nhật giao diện login / dashboard guide; nhóm kiểm tra lại quyền người dùng trong MVP.
- **28/04/2026:** Ly và Hiệp làm Google Drive file management, public download URL, queue sync, logging và error handling cho ingestion.
- **29/04/2026:** Hiệp nâng cấp queue runtime, batch processing, candidate search request và triển khai trace pipeline 6 stage.
- **30/04/2026:** Ly refactor secret management; Hiệp cập nhật model adapters, weight path và logging; Đạt cập nhật sidebar, TopK select và note time/location.
- **01/05/2026:** Nhóm sửa preview auth forwarding, permissions, sampling, async ingestion job queue, polling, dedup và nhiều lỗi runtime khi submit/poll video.
- **02/05/2026:** Hiệp tối ưu embedding similarity, sample FPS, VideoMAE batch processing, chunked feature extraction và tracker.
- **03/05/2026:** Hiệp cải thiện HeadBoxTracker, DINOv2 Re-ID, confidence thresholds và candidate search với tham số Drive URL.

### Khó nhất tuần này
- Ingestion qua Google Drive có nhiều edge case: URL lỗi, 503, 404 sau restart, file đã xử lý rồi nhưng queue vẫn chạy lại.
- Tracker ban đầu còn dễ vụn tracklet hoặc merge chưa ổn, phải thử nhiều hướng như BoT-SORT, TransReID, HeadBoxTracker và DINOv2.
- Quyền người dùng và preview image cần đi qua nhiều lớp auth/proxy nên dễ lỗi khi deploy.

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Gợi ý xử lý edge case cho queue, ingestion và tracking | Có thêm hướng retry, polling và phân rã lỗi |
| Codex | Hỗ trợ refactor nhiều file, đọc commit/logic cũ và sửa lỗi runtime | Tăng tốc việc nối pipeline và dọn service |

### Học được
- Với video ingestion, “đã submit được” chưa đủ; cần trạng thái job, retry, fatal error và skip duplicate rõ ràng.
- Tracking chất lượng phụ thuộc rất lớn vào threshold, sample FPS, feature extraction và logic merge tracklet.
- UI candidate phải ổn định ảnh/thumbnail trước khi đánh giá được chất lượng search.

### Nếu làm lại, sẽ làm khác
- Thiết kế ingestion job state machine sớm hơn thay vì bổ sung dần theo lỗi.
- Log chuẩn các bước queue / GPU / DB ngay từ đầu để debug nhanh hơn.

### Kế hoạch tuần tới
- Đưa hệ thống lên hạ tầng deploy ổn định hơn.
- Sửa các lỗi Docker, API routing, DB schema và auth.
- Hoàn thiện candidate preview, user/search UI và video processing schema.

---

## Tuần 06 - 10/05/2026

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Đã làm
- **Dương Văn Hiệp:** thêm proxy candidate preview qua LightningAI; nâng cấp video processing / tracking capabilities; triển khai RT-DETR person detection, candidates router và các thuộc tính người như age range, hat color, bag type, mask wearing, hair style, hair color.
- **Bùi Văn Đạt:** thêm API list videos; sửa frontend API base URL, route path, SearchBar, Sidebar, MainShell, user management và search UI; cải thiện dark mode nhưng cũng revert các thay đổi UI chưa ổn để giữ sản phẩm ổn định.
- **Cao Diệu Ly:** restructure services architecture, tài liệu Quickstart/README cho A100 GPU; sửa Docker/Coolify deploy, build context, Traefik, health check, DATABASE_URL, bcrypt/passlib, CORS, `/auth/me` và loại hardcoded HF token.
- Nhóm deploy được phiên bản đầu tiên trên hạ tầng hiện tại vào **05/05/2026**.
- Nhóm sửa nhiều lỗi production: DB table bootstrap, admin user, FastAPI router imports, session leaks, JSON/JSONB mismatch, API prefix `/v1`, Next.js rewrites và frontend standalone build.

### Chi tiết daily standup
- **04/05/2026:** Hiệp thêm proxy candidate preview và tracking service URL; nhóm chuẩn bị các nhánh deploy.
- **05/05/2026:** Ly tập trung sửa Docker/Coolify, Traefik, DB URL, bcrypt, frontend build; nhóm đạt mốc deploy được.
- **06/05/2026:** Đạt tích hợp video moving script với metadata service và thêm API list videos.
- **07/05/2026:** Ly và Hiệp sửa bootstrap DB, admin user, router imports, session, route prefix, debug endpoints, Next.js standalone và rewrites.
- **08/05/2026:** Ly hoàn thiện routing/config, bỏ hardcoded HF token, sửa `/auth/me` và double-prefix.
- **09/05/2026:** Hiệp thêm RT-DETR / candidates router; Đạt cải thiện candidate preview, user management, layout và search UI; nhóm revert phần UI không ổn.
- **10/05/2026:** Hiệp bổ sung schema thuộc tính người và refactor video processing models.

### Khó nhất tuần này
- Deploy thực tế phát sinh lỗi khác hoàn toàn local: build image, env build-time/runtime, Traefik host, API prefix và DB driver.
- FastAPI router/session/DB schema có nhiều lỗi dây chuyền khi đổi kiến trúc service.
- Frontend cần vừa đẹp hơn vừa không phá luồng search/candidate đang dùng để demo.

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Hỗ trợ phân tích lỗi deploy, Docker, CORS, routing và DB | Có checklist debug theo từng lớp hạ tầng |
| Codex | Sửa nhiều lỗi code/config lặp lại, rà route/API prefix và cập nhật docs | Đẩy nhanh quá trình ổn định production |

### Học được
- Deploy là một phần của sản phẩm, không phải bước cuối “cho có”.
- API prefix và env build-time của frontend phải được thống nhất cực kỳ chặt.
- Khi UI thay đổi nhiều nhưng chưa chắc chắn, revert có chọn lọc giúp giữ demo ổn định.

### Nếu làm lại, sẽ làm khác
- Dựng staging deploy sớm hơn để phát hiện lỗi Docker/API trước khi pipeline đã quá lớn.
- Viết checklist health check cho từng service: frontend, metadata, query, trace, postgres.

### Kế hoạch tuần tới
- Tăng chất lượng search/candidate ranking.
- Hoàn thiện trace evidence, history và human-in-the-loop.
- Tuning Qwen/SigLIP/tracker để giảm tracklet vụn và merge nhầm.

---

## Tuần 07 - 16/05/2026

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Đã làm
- **Dương Văn Hiệp - 2A202600052:** hoàn thiện DB và query; hoàn thiện DB schema để build; chạy và tiếp tục hoàn thiện luồng trace; hoàn chỉnh luồng để submit; chốt DB cuối cho bản nộp.
- **Bùi Văn Đạt - 2A202600355:** update thêm data cho DB query; thêm và tối ưu trang hiển thị toàn bộ tracklet của candidate; hoàn thiện các chức năng cơ bản; sửa luồng lịch sử không lưu nhiều video; test hoàn chỉnh các luồng; fix time UTC ở bộ lọc thời gian.
- **Cao Diệu Ly - 2A202600356:** kiểm tra DB mới và test trace; tiếp tục xây DB và sửa trace; tối ưu pipeline/query; thêm trang description; chạy full luồng trace; sửa merge tracklet bị ghép nhầm; sinh lại DB; thêm trace nhiều candidate, xóa tracklet khỏi candidate và dịch tiếng Việt; sửa full luồng detect, track, merge tracklet, similarity; fix trọng số tìm kiếm.
- Nhóm tập trung tuần này vào việc đưa search → candidate → trace → history về trạng thái đủ ổn định để submit, ưu tiên DB đúng schema, trace chạy được, candidate hiển thị rõ và query/search có trọng số hợp lý hơn.
- Nguồn tổng hợp chính: daily submissions trong tuần 11/05 → 16/05/2026, đối chiếu thêm với commit về DB/query, trace, history, timezone, merge tracklet và search threshold.

### Chi tiết daily standup
- **11/05/2026:** Hiệp hoàn thiện DB và query; Đạt update thêm data cho DB query; Ly kiểm tra DB mới và test luồng trace.
- **12/05/2026:** Hiệp hoàn thiện DB schema để build; Đạt thêm trang hiển thị toàn bộ tracklet của candidate; Ly tiếp tục xây DB và sửa lại trace.
- **13/05/2026:** Ly tiếp tục tối ưu pipeline, sửa logic query cho chính xác hơn, thêm trang description và chạy full luồng trace; Hiệp chạy luồng trace; Đạt tối ưu hiển thị toàn bộ tracklet.
- **14/05/2026:** Hiệp tiếp tục hoàn thiện trace; Đạt hoàn thiện các chức năng cơ bản và sửa luồng lịch sử không lưu nhiều video; Ly sửa logic merge tracklet do bị ghép nhầm quá nhiều, sinh lại DB để test hiệu quả, thêm lựa chọn trace nhiều candidate, thêm lựa chọn xóa tracklet khỏi candidate và dịch tiếng Việt cho hiển thị.
- **15/05/2026:** Ly sửa lại toàn bộ full luồng logic detect, track, merge tracklet và tính similarity, sau đó sinh lại DB; Hiệp hoàn chỉnh luồng để submit; Đạt test hoàn chỉnh các luồng.
- **16/05/2026:** Hiệp hoàn thiện DB cuối; Ly sửa logic query và fix trọng số tìm kiếm; Đạt fix time UTC ở bộ lọc time.

### Khó nhất tuần này
- Search và trace phụ thuộc nhiều lớp: ranking, threshold, timezone, candidate merge, evidence render và history; sửa một lớp có thể làm lệch lớp khác.
- Tracklet vụn và merge nhầm là hai lỗi đối nghịch: giảm vụn quá mạnh có thể merge nhầm, siết quá mạnh lại mất hành trình.
- UX history/candidate cần đủ thông tin cho người vận hành nhưng không làm màn hình quá rối.

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Brainstorm logic search/trace, diễn giải lỗi timezone và cách trình bày UX | Có thêm hướng kiểm tra edge case và viết nhãn tiếng Việt rõ hơn |
| Codex | Hỗ trợ sửa logic, refactor query/tracker/frontend và cập nhật tài liệu | Tăng tốc vòng lặp debug trong tuần nước rút |

### Học được
- Human-in-the-loop không chỉ là tính năng phụ; nó là cách giảm rủi ro khi ReID/search chưa thể hoàn hảo.
- Threshold phải được tune theo dữ liệu thật và theo mục tiêu demo: ưu tiên không merge nhầm hay ưu tiên không bỏ sót.
- Timezone, timestamp và hiển thị thời gian là phần nhỏ nhưng ảnh hưởng trực tiếp tới niềm tin của người dùng.

### Nếu làm lại, sẽ làm khác
- Tạo bộ test nhỏ cho timezone, ranking và trace window để không phải kiểm tra thủ công quá nhiều.
- Lưu lại bảng tuning threshold theo ngày để biết thay đổi nào làm search tốt hơn hoặc tệ hơn.

### Kế hoạch tiếp theo
- Chuẩn bị demo cuối: chọn video/camera/query ổn định, kiểm tra search → candidate → trace → history.
- Bổ sung test tối thiểu cho query ranking, trace time window và ingestion job state.
- Viết ngắn gọn phần giới hạn hiện tại: single A100, batch ingestion, threshold cần calibration và chưa có realtime streaming.
