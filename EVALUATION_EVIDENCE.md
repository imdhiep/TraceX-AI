# Evaluation Evidence

Tài liệu này tổng hợp minh chứng đánh giá cho TraceX-AI: thay đổi kỹ thuật quan trọng, kết quả test/benchmark nội bộ, bộ câu hỏi kiểm thử và các bằng chứng cần chụp lại khi demo.

> Lưu ý: các số liệu bên dưới là ghi nhận nội bộ trong quá trình phát triển. Khi nộp/chấm, nên bổ sung ảnh chụp màn hình, log chạy thử hoặc video demo tương ứng để biến các ghi nhận này thành bằng chứng đầy đủ.

---

## 1. Phạm vi đánh giá

Tài liệu này chỉ giữ các minh chứng nội bộ cần thiết cho quá trình đánh giá TraceX-AI:

- các câu hỏi đánh giá chính;
- kết quả benchmark và tối ưu pipeline;
- thay đổi kỹ thuật có thể dùng làm evidence;
- luồng test chức năng đã kiểm tra;
- danh sách ảnh chụp, log và dữ liệu mẫu cần chuẩn bị khi nộp/demo.

Các dẫn chứng từ bài báo, tài liệu tham khảo bên ngoài và ví dụ vụ việc thực tế đã được loại bỏ để file tập trung vào evaluation evidence của chính dự án.

---

## 2. Câu hỏi đánh giá chính

| Câu hỏi | Trạng thái | Minh chứng hiện có |
| ------- | ---------- | ------------------ |
| Sản phẩm có đúng mục tiêu không? | Đạt ở mức MVP | Hệ thống hỗ trợ search người bằng mô tả/ảnh, xem candidate, chọn trace và xem history |
| Agent/AI pipeline có xử lý chính xác không? | Đạt có điều kiện | Pipeline đã chuyển sang tracklet + embedding + metadata; vẫn cần tuning threshold theo dữ liệu |
| Hệ thống có ổn định không? | Đạt ở luồng demo | Đã deploy frontend/backend tách riêng, có health check và luồng history/trace |
| Đã kiểm thử nhiều tình huống chưa? | Có kiểm thử thủ công và benchmark cục bộ | Có test luồng search/candidate/trace/history, benchmark tracker trong `camera_0002/` |

---

## 3. Evaluation Summary

| Hạng mục | Trước tối ưu | Sau tối ưu | Tác động |
| -------- | ------------ | ---------- | -------- |
| Detector / video processing flow | YOLO / pipeline cũ, thời gian xử lý khoảng 180s trong một luồng test nội bộ | RT-DETR thay YOLO, thiết kế lại chia luồng hoạt động, còn khoảng 60s trong cùng nhóm test | Giảm thời gian xử lý xấp xỉ 3 lần, luồng detect ổn định hơn cho pipeline hiện tại |
| Embedding stack | Kết hợp SigLIP2 + DINOv2, pipeline nặng và khó đồng bộ embedding space | Chuyển trọng tâm sang SigLIP/SigLIP-compatible embedding cho search/query | Giảm độ phức tạp khi ranking, tránh mismatch embedding và dễ tune threshold hơn |
| Metadata extraction | Qwen tạo một description tổng hợp duy nhất, chậm và khó parse | Qwen2-VL 7B sinh JSON metadata theo từng thuộc tính | Metadata có cấu trúc hơn, query/filter dễ hơn |
| Qwen metadata runtime | Khoảng 20 phút cho 1 video 10 phút trong ghi nhận nội bộ | Khoảng 3 phút cho 1 video 10 phút sau khi chia JSON/batch | Giảm mạnh thời gian metadata extraction, phù hợp hơn với demo batch processing |
| Trace / history UX | Trace và history còn mất candidate/tracklet, trạng thái render chưa rõ | Lưu query/candidate/video trace, hiển thị all tracklet, cho phép xóa tracklet khỏi candidate | Người dùng kiểm tra được evidence và sửa kết quả sai trong luồng human-in-the-loop |
| Time filter | Có lỗi lệch timezone UTC ở bộ lọc | Fix UTC/timezone filter | Search theo thời gian đáng tin cậy hơn |

---

## 4. Technical Changes Used As Evidence

### 4.1 RT-DETR thay YOLO cho detection

**Mục tiêu:** cải thiện luồng phát hiện người và giảm thời gian xử lý video.

**Thay đổi chính:**

- thay hướng detector cũ bằng RT-DETR;
- tách lại luồng detect / track / merge để giảm nghẽn;
- thêm tuning threshold, Kalman filter và matching gate trong tracker;
- benchmark/debug nằm trong thư mục `camera_0002/`.

**Kết quả nội bộ:**

- thời gian xử lý trong một test giảm từ khoảng `180s` xuống `60s`;
- luồng detect -> raw tracklet rõ hơn;
- giảm over-fragment tracklet sau các vòng tuning.

**Cần bổ sung bằng chứng khi nộp:**

- ảnh/log trước và sau khi chạy cùng một video;
- tên video test, độ dài video, GPU/máy chạy;
- commit hoặc log liên quan tới RT-DETR và tracker tuning.

### 4.2 SigLIP/SigLIP2 + DINOv2 -> SigLIP-compatible search

**Mục tiêu:** đơn giản hóa không gian embedding cho query/search.

**Thay đổi chính:**

- ban đầu dùng kết hợp SigLIP2 + DINOv2 cho các tín hiệu appearance/search;
- chuyển search/ranking về hướng SigLIP-compatible embedding để query-service và metadata-service đồng bộ hơn;
- query-service warmup SigLIP text/image tower để encode query online;
- tracklet embedding lưu ở `tracklets_embeddings.siglip_embedding`.

**Kết quả nội bộ:**

- ranking dễ tune hơn bằng `MIN_FUSION_SCORE` và `QUERY_MERGE_THRESHOLD`;
- giảm rủi ro mismatch giữa vector của query và vector lưu DB;
- search theo text/image phù hợp hơn với dữ liệu candidate hiện tại.

**Cần bổ sung bằng chứng khi nộp:**

- screenshot kết quả search trước/sau với cùng query;
- log top candidates và fusion score;
- query mẫu có ảnh/text.

### 4.3 Qwen2-VL 7B sinh metadata JSON thay vì một description duy nhất

**Mục tiêu:** biến metadata từ mô tả tự do thành dữ liệu có cấu trúc để search/filter tốt hơn.

**Thay đổi chính:**

- trước đó Qwen sinh một description tổng hợp duy nhất;
- sau đó dùng Qwen2-VL 7B để sinh JSON theo từng nhóm thuộc tính;
- metadata được tách thành các trường như giới tính, tuổi, áo, quần, giày, túi, mũ, khẩu trang, tóc và appearance summary;
- dữ liệu được lưu vào các cột tracklet thay vì chỉ nằm trong text tự do.

**Kết quả nội bộ:**

- runtime metadata extraction giảm từ khoảng `20 phút / video 10 phút` xuống khoảng `3 phút / video 10 phút`;
- kết quả dễ parse, dễ hiển thị và dễ match với query hơn;
- frontend có thể dịch/hiển thị description rõ hơn.

**Cần bổ sung bằng chứng khi nộp:**

- một JSON output mẫu;
- ảnh UI candidate detail hiển thị metadata;
- log runtime của cùng một video trước/sau.

---

## 5. Functional Test Evidence

| Luồng test | Kết quả mong đợi | Trạng thái |
| ---------- | ---------------- | ---------- |
| Đăng nhập bằng bootstrap admin | Vào được dashboard/home | Đã kiểm thử thủ công |
| Search bằng text query | Trả candidate có preview, score, camera/time | Đã kiểm thử thủ công |
| Search bằng ảnh/query chỉ có ảnh | Có candidate phù hợp, không lỗi form | Đã sửa và kiểm thử thủ công |
| Lọc theo thời gian | Không lệch UTC/timezone | Đã fix ngày 16/05 |
| Mở candidate detail | Hiển thị all tracklet và description | Đã làm trong tuần 7 |
| Xóa tracklet khỏi candidate | Candidate cập nhật lại, tracklet sai bị loại | Đã làm trong tuần 7 |
| Build trace | Tạo evidence row và render clip/timeline | Đã kiểm thử luồng submit |
| History | Lưu query, candidate, evidence video đúng | Đã sửa luồng không lưu nhiều video |

---

## 6. Benchmark / Debug Assets In Repo

Các file sau có thể dùng làm evidence kỹ thuật hoặc phụ lục:

| File | Mục đích |
| ---- | -------- |
| `camera_0002/bench_fps_comparison.py` | So sánh ảnh hưởng FPS sampling tới tracker |
| `camera_0002/bench_g2.py` | Benchmark gate/matching logic |
| `camera_0002/bench_raw_tracker.py` | Benchmark raw tracker độc lập với detection |
| `camera_0002/test_detection_changes.py` | Test crop quality, bbox, low-quality detection cases |
| `backend/config/camera_topology.json` | Cấu hình topology camera phục vụ trace |
| `backend/config/query_metadata_vocab.json` | Từ vựng metadata/query phục vụ search |

---

## 7. Evidence cần chụp để nộp

Nên chuẩn bị các ảnh/video sau:

1. Screenshot dashboard/home sau khi login.
2. Screenshot search text query có candidate.
3. Screenshot search bằng ảnh hoặc query ảnh.
4. Screenshot candidate detail hiển thị all tracklet.
5. Screenshot thao tác xóa tracklet khỏi candidate.
6. Screenshot trace status/timeline/evidence video.
7. Screenshot history query và candidate history.
8. Log hoặc terminal output cho benchmark `180s -> 60s`.
9. Log hoặc terminal output cho Qwen metadata `20p/video 10p -> 3p/video 10p`.
10. Một JSON metadata mẫu sinh từ Qwen2-VL.

---

## 8. Những điểm cần ghi rõ khi báo cáo

- Các số liệu runtime phụ thuộc vào video, GPU, batch size và cache model.
- Search/trace hiện là MVP batch/offline-first, chưa phải realtime streaming.
- Threshold merge/search vẫn cần calibration theo dataset.
- Evidence mạnh nhất là so sánh cùng một video, cùng máy, cùng cấu hình trước/sau.
