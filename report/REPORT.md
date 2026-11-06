# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (CUDA)

**Python / PyTorch / Ultralytics:**
Python: 3.13.16

PyTorch: 2.11.0+cu130

Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

---

## Cell 2 – Tải ảnh COCO và kiểm tra dữ liệu đầu vào

### Mục tiêu

Tải ba ảnh mẫu từ bộ dữ liệu **COCO 2017 Validation** để sử dụng trong các bước thực nghiệm YOLO11.

Mỗi ảnh được xác định bằng COCO Image ID và kiểm tra tính toàn vẹn thông qua mã SHA256 nhằm đảm bảo file tải về khớp với dữ liệu được khai báo trong notebook.

### Danh sách ảnh đã tải

| Sample ID | COCO Image ID | Tên ảnh gốc | SHA256 (12 ký tự đầu) | Kết quả |
|---|---:|---|---|---|
| `traffic` | `210273` | Wuhan | `3ec23de63592` | PASS |
| `kitchen` | `397133` | Kitchen | `09e1d25c75f7` | PASS |
| `dining` | `166918` | Big Wine Thing | `a7f8457580a2` | PASS |

**Kết quả:** Cả ba ảnh đều được tải thành công và mã SHA256 khớp với giá trị tham chiếu trong notebook.

### Nguồn gốc và giấy phép sử dụng ảnh

Các ảnh được lấy từ COCO 2017 Validation. Thông tin tác giả, nguồn ảnh và giấy phép sử dụng được ghi lại trong file `day1_lab_outputs/IMAGE_ATTRIBUTION.md`.

| Sample ID | Tác giả | Giấy phép |
|---|---|---|
| `traffic` | Tauno Tõhk (toehk) | CC BY 2.0 |
| `kitchen` | Maggie Stephens (Pot Noodle) | CC BY 2.0 |
| `dining` | WordRidden | CC BY 2.0 |

Các ảnh PNG minh chứng được tạo ở những bước tiếp theo có thể chứa phần hiển thị prediction hoặc biểu đồ của model. Đây là những phiên bản ảnh đã được bổ sung nội dung từ kết quả thực nghiệm.

### Hiển thị và kiểm tra ảnh

Notebook sử dụng:

- **PIL (Python Imaging Library):** Mở ảnh và chuyển đổi sang định dạng RGB.
- **Matplotlib:** Hiển thị ba ảnh mẫu cùng thông tin kích thước tương ứng.

Các ảnh được lưu trong thư mục:

```text
day1_lab_images/
├── traffic.jpg
├── kitchen.jpg
└── dining.jpg
```

### Kết luận Cell 2

Cell 2 hoàn thành thành công các công việc:

- Tải ba ảnh mẫu từ COCO 2017 Validation.
- Xác minh tính toàn vẹn của ảnh bằng SHA256.
- Tạo file `IMAGE_ATTRIBUTION.md` để ghi nhận nguồn ảnh, tác giả và giấy phép sử dụng.
- Mở và hiển thị ảnh thành công bằng PIL và Matplotlib.

**Kết luận:** Dữ liệu ảnh đầu vào đã được chuẩn bị và xác minh thành công, sẵn sàng cho các bước thực hiện prediction bằng YOLO11.

---

## 1. Phân loại ảnh (Image Classification)

### 1.1. Mục tiêu

Sử dụng mô hình YOLO11 Classification (`yolo11n-cls.pt`) để thực hiện phân loại ba ảnh đầu vào đã tải từ bộ dữ liệu COCO 2017 Validation.

Mô hình trả về Top-5 dự đoán có điểm tin cậy (score) cao nhất cho mỗi ảnh. Kết quả được lưu trong file `day1_lab_outputs/classification_predictions.json`.

Trong phần phân tích này, chỉ sử dụng ảnh `sample_id = traffic` và record có `rank = 1` theo yêu cầu của bài thực hành.

### 1.2. Kết quả dự đoán

Thông tin record có thứ hạng cao nhất của ảnh `traffic`:

| Thuộc tính | Giá trị |
|---|---|
| `sample_id` | `traffic` |
| `coco_image_id` | `210273` |
| `image_width` | `640` pixels |
| `image_height` | `428` pixels |
| `class_id` | `468` |
| `class_name` | `cab` |
| `rank` | `1` |
| `score` | `0.510915` |
| `taxonomy_name` | `ImageNet-1K` |
| `model_file` | `yolo11n-cls.pt` |
| `ultralytics_version` | `8.4.145` |

**Record hạng 1:**

- `class_id` = `468`
- `class_name` = `cab`
- `rank` = `1`
- `score` = `0.510915`
- `taxonomy_name` = `ImageNet-1K`

### 1.3. Phân tích kết quả

**Câu 1. Record này nói về cả bức ảnh hay một vật thể riêng trong ảnh? Vì sao?**

Record này mô tả nội dung của **toàn bộ bức ảnh**, không phải kết quả xác định vị trí của một vật thể riêng lẻ.

Lý do: Image Classification nhận ảnh đầu vào và dự đoán lớp đại diện cho nội dung ảnh. Kết quả không chứa tọa độ bounding box hoặc vùng mask để xác định vị trí cụ thể của từng vật thể.

**Câu 2. Danh sách tên lớp (class list) do đâu mà có?**

Danh sách lớp không phải do model tự nghĩ ra trong quá trình dự đoán. Danh sách này được xác định từ taxonomy dùng khi huấn luyện mô hình và được lưu kèm checkpoint.

Trong trường hợp này, mô hình sử dụng taxonomy `ImageNet-1K`, bao gồm 1.000 lớp đối tượng được định nghĩa sẵn.

**Câu 3. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**

Cần lưu cả ba thông tin vì:

- `class_id`: Mã định danh số của lớp, thuận tiện cho xử lý dữ liệu và đối chiếu.
- `class_name`: Tên lớp giúp con người đọc và hiểu kết quả dự đoán.
- `taxonomy_name`: Xác định hệ thống phân loại đang được sử dụng, tránh nhầm lẫn khi nhiều taxonomy có thể sử dụng tên lớp hoặc cách định nghĩa khác nhau.

Việc lưu đầy đủ ba trường giúp dữ liệu prediction có tính nhất quán, dễ kiểm tra và thuận tiện khi tích hợp vào pipeline dữ liệu AI.

**Câu 4. Nếu trong ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**

Guideline cần quy định rõ tiêu chí lựa chọn lớp đại diện cho toàn bộ ảnh, chẳng hạn:

- Ưu tiên chủ thể chính của ảnh.
- Xác định đối tượng chiếm vị trí hoặc vai trò nổi bật nhất.
- Nếu có nhiều đối tượng cùng xuất hiện, cần thống nhất quy tắc ưu tiên.
- Nếu ảnh không thể xác định được một chủ thể chính rõ ràng, cần có quy định về nhãn phù hợp hoặc chuyển reviewer/Lab Coach xử lý.

Điều này giúp các annotator đưa ra kết quả nhất quán khi gán nhãn cùng một loại dữ liệu.

**Câu 5. Vì sao không được lấy `score` của model làm ground truth?**

`score` là mức độ tin cậy trong dự đoán của mô hình, không phải nhãn đúng đã được xác minh bởi con người.

Model có thể dự đoán sai nhưng vẫn đưa ra score cao. Ngược lại, model có thể dự đoán đúng nhưng score không cao.

Ground truth phải được xác định thông qua quá trình gán nhãn của con người dựa trên guideline và được reviewer kiểm tra chất lượng.

Vì vậy, `score = 0.510915` chỉ thể hiện mức độ tin cậy của model đối với lớp `cab`, không chứng minh rằng `cab` là ground truth của ảnh.

### 1.4. Kết luận

Mô hình YOLO11 Classification đã thực hiện dự đoán thành công trên ảnh `traffic` (COCO Image ID `210273`).

Kết quả Top-1 là lớp `cab`, có `class_id = 468` và `score = 0.510915`, thuộc taxonomy `ImageNet-1K`.

Qua thực nghiệm, có thể phân biệt được prediction và ground truth: prediction là kết quả do mô hình tạo ra, trong khi ground truth phải được xác định theo guideline và kiểm tra bởi con người.

### Trực quan hóa kết quả phân loại

**Mục tiêu:** Trực quan hóa kết quả Top-5 của mô hình YOLO11 Classification trên ảnh `traffic`, giúp quan sát ảnh đầu vào và so sánh điểm tin cậy giữa các nhãn dự đoán.

**Các bước thực hiện:**

1. Lọc các bản ghi trong `classification_records` với `sample_id = "traffic"`.
2. Hiển thị ảnh đầu vào ở biểu đồ bên trái.
3. Vẽ biểu đồ thanh ngang ở bên phải, thể hiện điểm tin cậy (`score`) của 5 nhãn dự đoán.
4. Giới hạn trục điểm tin cậy từ 0 đến 1 để thuận tiện so sánh.
5. Lưu hình ảnh kết quả tại `day1_lab_outputs/visuals/classification_top5.png`.
6. Giải phóng mô hình khỏi bộ nhớ và làm trống CUDA cache nếu GPU khả dụng.

**Kết quả đầu ra:**

- File trực quan hóa: `classification_top5.png`
- Vị trí lưu: `day1_lab_outputs/visuals/classification_top5.png`
- Ảnh được trực quan hóa: `traffic` (COCO Image ID: 210273).
- Biểu đồ thể hiện 5 nhãn có điểm tin cậy cao nhất:

| Thứ hạng | Nhãn dự đoán | Score |
|---:|---|---:|
| 1 | cab | 0.510915 |
| 2 | minibus | 0.164284 |
| 3 | police_van | 0.085848 |
| 4 | recreational_vehicle | 0.054110 |
| 5 | streetcar | 0.048194 |

**Phân tích:**

- Ảnh bên trái cung cấp ngữ cảnh trực quan của dữ liệu đầu vào; biểu đồ bên phải thể hiện kết quả dự đoán của mô hình.
- `cab` có điểm tin cậy cao nhất, đạt `0.510915`, cao hơn đáng kể so với các nhãn còn lại.
- Các nhãn `minibus`, `police_van`, `recreational_vehicle` và `streetcar` là những phương án dự đoán tiếp theo của mô hình.
- Biểu đồ giúp người đánh giá nhận biết trường hợp mô hình có nhiều nhãn dự đoán cạnh tranh, từ đó xác định những mẫu cần xem xét kỹ hơn.
- Điểm `score` phản ánh mức độ tin cậy của mô hình đối với dự đoán, không chứng minh nhãn đó là Ground Truth chính xác.

**Lưu ý về xử lý tài nguyên:** Sau khi hoàn tất trực quan hóa, biến `classification_model` được giải phóng bằng `del`. Nếu hệ thống sử dụng GPU CUDA, lệnh `torch.cuda.empty_cache()` được gọi để giải phóng bộ nhớ GPU chưa được sử dụng.

**Kết luận:** Cell này hoàn thành bước trực quan hóa kết quả Image Classification và tạo ảnh minh chứng `classification_top5.png`. Kết quả cho thấy mô hình ưu tiên nhãn `cab` trên ảnh `traffic`, nhưng vẫn cần đối chiếu với dữ liệu thực tế và tiêu chí gán nhãn trước khi xem đây là nhãn Ground Truth.


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

### 2.1. Mục tiêu

Sử dụng mô hình YOLO11 Object Detection (`yolo11n.pt`) để phát hiện các vật thể trong ba ảnh mẫu từ COCO 2017 Validation. Mô hình xác định lớp vật thể, điểm tin cậy và tọa độ bounding box tương ứng.

Kết quả được lưu trong `day1_lab_outputs/detection_predictions.json`. Ngưỡng confidence mặc định được sử dụng trong lần chạy này là `0.35`.

### 2.2. Cấu hình thực nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Model | `yolo11n.pt` |
| Task | Object Detection |
| Taxonomy | COCO-80 |
| Ultralytics | 8.4.145 |
| Confidence threshold | 0.35 |
| Coordinate unit | Pixel |
| Bounding box format | XYXY |
| Tổng số prediction trên ba ảnh | 53 |

### 2.3. Kết quả thực nghiệm

Mô hình thực hiện inference trên ba ảnh `traffic`, `kitchen` và `dining`. Với ngưỡng confidence `0.35`, tổng cộng ghi nhận 53 prediction.

Mỗi prediction bao gồm các thông tin:
- `class_id`: mã định danh lớp vật thể.
- `class_name`: tên lớp vật thể theo taxonomy COCO-80.
- `score`: điểm tin cậy của mô hình.
- `bbox_xyxy`: tọa độ bounding box.
- `bbox_width`, `bbox_height`: chiều rộng và chiều cao bounding box tính theo pixel.

**Ví dụ record minh họa từ output ảnh traffic:**

- `class_name`: bus
- `score`: 0.912558
- `bbox_xyxy`: [93.17, 187.95, 223.01, 320.91]
- `bbox_width`: 129.84
- `bbox_height`: 132.96

Record trên chỉ là ví dụ minh họa cách đọc dữ liệu prediction. Phần báo cáo chính thức yêu cầu chọn một record thuộc `sample_id = kitchen`; cần bổ sung record tương ứng từ file JSON thực tế.

### 2.4. Giải thích tọa độ Bounding Box

Bounding box được biểu diễn theo định dạng XYXY:

`[x_min, y_min, x_max, y_max]`

Trong đó:
- `(x_min, y_min)` là tọa độ góc trên-trái của hộp.
- `(x_max, y_max)` là tọa độ góc dưới-phải của hộp.
- `bbox_width = x_max - x_min`.
- `bbox_height = y_max - y_min`.

Gốc tọa độ `(0, 0)` nằm tại góc trên-trái của ảnh. Trục X tăng theo chiều ngang từ trái sang phải, trục Y tăng theo chiều dọc từ trên xuống dưới.

Các tọa độ được tính theo pixel của ảnh đầu vào, không phải tọa độ chuẩn hóa theo tỷ lệ phần trăm.

### 2.5. So sánh ngưỡng Confidence

Ngưỡng confidence quyết định prediction nào được giữ lại trong kết quả inference. Ngưỡng thấp cho phép giữ lại nhiều prediction hơn, trong khi ngưỡng cao chỉ giữ những prediction có score lớn hơn hoặc bằng ngưỡng được đặt.

Trong bài lab, ba ngưỡng cần so sánh là `0.20`, `0.35` và `0.60`.

| Confidence threshold | Số prediction |
|---:|---:|
| 0.20 | Chưa có output |
| 0.35 | 53 trên cả ba ảnh |
| 0.60 | Chưa có output |

Lưu ý: Số lượng 53 là tổng prediction của cả ba ảnh, không phải riêng ảnh `kitchen`. Cần sử dụng output của cell so sánh threshold để điền số lượng chính xác cho từng ngưỡng và đúng phạm vi ảnh mà notebook yêu cầu.

### 2.6. Vai trò của Reviewer khi hạ ngưỡng

Khi giảm confidence threshold, mô hình có thể giữ lại nhiều bounding box hơn, bao gồm cả những prediction có độ tin cậy thấp hoặc dự đoán sai.

Reviewer cần:
- Kiểm tra các bounding box có thực sự bao quanh vật thể trong ảnh hay không.
- Phát hiện prediction sai lớp hoặc không có vật thể thực tế.
- Phát hiện các bounding box trùng lặp trên cùng một vật thể.
- Kiểm tra các vật thể nhỏ, bị che khuất hoặc nằm sát mép ảnh.
- Đánh giá tính nhất quán giữa các annotator theo guideline.
- Ghi nhận những trường hợp không rõ ràng để escalation lên Lab Coach.

Mục tiêu là đảm bảo chất lượng Ground Truth, không đơn thuần giữ lại càng nhiều prediction càng tốt.

### 2.7. Quy tắc Bounding Box đề xuất

**Quy tắc:** Người gán nhãn phải vẽ bounding box bao quanh toàn bộ phần vật thể có thể quan sát được trong ảnh, sử dụng hộp chữ nhật nhỏ nhất có thể nhưng không cắt mất phần vật thể nhìn thấy. Không đưa vùng nền không thuộc vật thể vào bounding box. Đối với vật thể bị che khuất, chỉ bao quanh phần có thể xác định chắc chắn theo guideline đã thống nhất.

### 2.8. Vật thể bị che khuất hoặc cắt mép và Escalation

Cần kiểm tra trực tiếp ảnh `kitchen` để xác định một vật thể cụ thể bị che khuất hoặc bị cắt mép.

Nếu trường hợp đó đã được guideline quy định rõ, annotator cần áp dụng thống nhất theo hướng dẫn. Nếu chưa có quy tắc rõ ràng, hoặc không thể xác định ranh giới vật thể một cách đáng tin cậy, cần escalation lên Lab Coach để thống nhất cách xử lý trước khi tiếp tục gán nhãn.

Không tự suy đoán phần bị che khuất hoặc tự đặt ra quy tắc mới khi chưa được phê duyệt.

### 2.9. Kết luận

Thực nghiệm đã hoàn thành bước Object Detection bằng YOLO11 với checkpoint `yolo11n.pt`, taxonomy COCO-80 và confidence threshold `0.35`. Mô hình tạo ra 53 prediction trên ba ảnh mẫu và lưu kết quả dưới dạng JSON.

Kết quả cho thấy mỗi prediction bao gồm thông tin lớp, điểm tin cậy và tọa độ bounding box theo pixel. Tuy nhiên, prediction của mô hình chỉ đóng vai trò gợi ý ban đầu. Ground Truth cần được annotator kiểm tra và reviewer xác nhận theo guideline để đảm bảo độ chính xác và tính nhất quán của dữ liệu.

### Trực quan hóa kết quả và so sánh Confidence Threshold

**Mục tiêu:**

Trực quan hóa kết quả Object Detection trên ảnh `kitchen`, đồng thời đánh giá ảnh hưởng của confidence threshold đến số lượng vật thể được mô hình YOLO11 phát hiện.

**Các bước thực hiện:**

1. Chọn ảnh `kitchen` từ bộ dữ liệu đầu vào.
2. Sử dụng mô hình `yolo11n.pt` để thực hiện Object Detection với ngưỡng mặc định `0.35`.
3. Sử dụng `result.plot()` để tạo ảnh kết quả có hiển thị bounding box, tên lớp và điểm tin cậy.
4. Lưu ảnh trực quan hóa tại `day1_lab_outputs/visuals/detection_predictions.png`.
5. Thực hiện inference lần lượt với ba ngưỡng confidence `0.20`, `0.35` và `0.60`.
6. Ghi nhận số lượng prediction và danh sách lớp vật thể tương ứng ở mỗi ngưỡng.
7. Giải phóng mô hình và bộ nhớ GPU sau khi hoàn thành.

**Kết quả trực quan hóa:**

- Ảnh đầu vào: `kitchen` (COCO Image ID: 397133).
- Mô hình: `yolo11n.pt`.
- Confidence threshold mặc định: `0.35`.
- File kết quả: `detection_predictions.png`.
- Đường dẫn: `day1_lab_outputs/visuals/detection_predictions.png`.

**Kết quả so sánh ngưỡng Confidence:**

| Confidence threshold | Số prediction | Các lớp vật thể được phát hiện |
|---:|---:|---|
| 0.20 | 17 | person, bowl, oven, cup, spoon, potted plant, dining table, bottle |
| 0.35 | 11 | person, bowl, oven, cup |
| 0.60 | 6 | person, bowl, oven |

**Phân tích kết quả:**

- Ở ngưỡng `0.20`, mô hình phát hiện 17 prediction. Đây là ngưỡng thấp nên giữ lại nhiều bounding box hơn, bao gồm cả những vật thể có điểm tin cậy thấp như `spoon`, `potted plant`, `dining table` và `bottle`.
- Ở ngưỡng `0.35`, số prediction giảm còn 11. Một số lớp có score thấp hơn ngưỡng bị loại khỏi kết quả.
- Ở ngưỡng `0.60`, mô hình chỉ còn 6 prediction, tập trung vào ba lớp `person`, `bowl` và `oven`.

Khi tăng confidence threshold từ `0.20` lên `0.60`, số lượng prediction giảm từ 17 xuống 6, tương đương giảm 11 prediction (khoảng 64,7%).

Kết quả cho thấy confidence threshold ảnh hưởng trực tiếp đến số lượng bounding box được giữ lại. Ngưỡng thấp giúp hạn chế bỏ sót các ứng viên vật thể nhưng có thể làm tăng số prediction sai hoặc không chắc chắn. Ngược lại, ngưỡng cao giúp lọc bớt các prediction có score thấp nhưng có nguy cơ bỏ qua những vật thể nhỏ hoặc khó nhận diện.

**Ý nghĩa đối với công việc Annotation và QC:**

Khi sử dụng ngưỡng thấp, Reviewer cần dành thêm thời gian kiểm tra:
- Bounding box có thực sự bao quanh một vật thể tồn tại trong ảnh hay không.
- Nhãn lớp có chính xác với vật thể được quan sát hay không.
- Có nhiều bounding box trùng lặp trên cùng một vật thể hay không.
- Các vật thể nhỏ, bị che khuất hoặc có score thấp có bị bỏ sót hay không.

Confidence threshold chỉ là cơ chế lọc kết quả dự đoán của mô hình, không phải tiêu chí quyết định vật thể nào được đưa vào Ground Truth. Annotator vẫn phải tuân thủ guideline khi gán nhãn.

**Quản lý tài nguyên:**

Sau khi hoàn tất trực quan hóa và so sánh ngưỡng, mô hình `detection_model` được giải phóng bằng `del`. Nếu GPU CUDA khả dụng, chương trình gọi `torch.cuda.empty_cache()` để giải phóng bộ nhớ GPU chưa sử dụng.

**Kết luận:**

Cell này hoàn thành việc trực quan hóa kết quả phát hiện vật thể trên ảnh `kitchen` và đánh giá ảnh hưởng của confidence threshold. Số lượng prediction lần lượt là 17, 11 và 6 tương ứng với các ngưỡng `0.20`, `0.35` và `0.60`. Qua đó, có thể thấy việc lựa chọn ngưỡng ảnh hưởng đến khối lượng công việc kiểm tra của Reviewer, nhưng không thay thế quy tắc xây dựng Ground Truth.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Phân đoạn theo từng đối tượng (Instance Segmentation)

### 3.1. Mục tiêu

Sử dụng mô hình YOLO11 Segmentation (`yolo11n-seg.pt`) để xác định và phân đoạn từng đối tượng riêng biệt trong ảnh. Khác với Object Detection chỉ xác định vị trí vật thể bằng bounding box, Instance Segmentation tạo ra polygon mô tả đường biên của từng đối tượng.

Kết quả được lưu trong file `day1_lab_outputs/segmentation_predictions.json`, sử dụng taxonomy COCO-80 và confidence threshold mặc định là `0.35`.

### 3.2. Cấu hình thực nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Model | `yolo11n-seg.pt` |
| Task | Instance Segmentation |
| Taxonomy | COCO-80 |
| Ultralytics | 8.4.145 |
| Confidence threshold | 0.35 |
| Coordinate unit | Pixel |
| Polygon format | Danh sách tọa độ `[x, y]` |
| Tổng số instance trên ba ảnh | 49 |

### 3.3. Kết quả thực nghiệm

Mô hình thực hiện segmentation trên ba ảnh mẫu: `traffic`, `kitchen` và `dining`.

Với confidence threshold `0.35`, chương trình ghi nhận tổng cộng 49 instance. Mỗi instance được lưu thành một record riêng trong JSON.

Các trường dữ liệu quan trọng:

- `instance_id`: mã định danh riêng của từng instance trong kết quả lab.
- `class_id`: mã số lớp vật thể.
- `class_name`: tên lớp vật thể theo taxonomy COCO-80.
- `score`: điểm tin cậy của mô hình.
- `bbox_xyxy`: bounding box bao quanh đối tượng.
- `polygon_point_count`: số lượng điểm tạo thành polygon.
- `polygon_xy`: danh sách tọa độ pixel mô tả đường biên đối tượng.

**Ví dụ minh họa từ output ảnh traffic:**

- `instance_id`: traffic-001
- `class_name`: bus
- `score`: 0.925745
- `polygon_point_count`: 120
- 5 điểm polygon đầu tiên:

```json
[
  [148.0, 189.0],
  [147.0, 190.0],
  [145.0, 190.0],
  [143.0, 192.0],
  [142.0, 192.0]
]
```

Đây là record minh họa cách đọc cấu trúc dữ liệu. Phần báo cáo chính thức cần chọn một record thuộc `sample_id = kitchen` theo yêu cầu bài lab.

### 3.4. So sánh Bounding Box và Polygon

Bounding Box và Polygon cung cấp hai mức độ thông tin hình học khác nhau:

- **Bounding Box:** Là hình chữ nhật bao quanh đối tượng, xác định vị trí và kích thước tổng thể. Cách biểu diễn đơn giản nhưng có thể chứa cả những vùng nền không thuộc đối tượng.
- **Polygon:** Là tập hợp các điểm tọa độ bám theo đường biên đối tượng, mô tả hình dạng và phạm vi của vật thể chi tiết hơn.

Ví dụ, với một vật thể có hình dạng không đều, bounding box chỉ biểu diễn được vùng chữ nhật bao quanh, trong khi polygon có thể mô tả các phần lồi, lõm và đường viền thực tế của vật thể.

Do đó, polygon phù hợp hơn khi cần xác định chính xác vùng pixel thuộc về từng instance, đặc biệt trong các bài toán yêu cầu phân tích hình dạng hoặc diện tích đối tượng.

### 3.5. Ý nghĩa của instance_id

`instance_id` là mã định danh riêng cho từng đối tượng trong phạm vi kết quả của bài lab.

Ví dụ:
- `kitchen-001`
- `kitchen-002`

Hai vật thể có thể cùng thuộc một lớp `bowl`, nhưng vẫn có `instance_id` khác nhau vì chúng là hai instance riêng biệt.

Cần phân biệt ba khái niệm:

- `instance_id`: Xác định từng đối tượng riêng biệt trong kết quả segmentation.
- `class_id`: Xác định loại đối tượng theo taxonomy, ví dụ `bowl`, `person` hoặc `cup`.
- `tracking_id`: Dùng để theo dõi cùng một đối tượng qua nhiều frame trong video.

Như vậy, `instance_id` không đại diện cho loại vật thể và cũng không thể sử dụng trực tiếp để xác định một đối tượng xuyên suốt nhiều frame. Trong notebook này, ID được tạo theo thứ tự xuất hiện trong kết quả của từng ảnh.

### 3.6. Quy tắc vẽ biên Mask đề xuất

**Quy tắc:** Annotator phải vẽ polygon bám sát đường biên của phần đối tượng có thể quan sát được, sử dụng số điểm đủ để mô tả chính xác hình dạng nhưng tránh tạo các điểm thừa hoặc đường biên bị gấp khúc không cần thiết. Không đưa vùng nền vào mask và không gộp hai đối tượng riêng biệt thành một polygon, kể cả khi chúng cùng lớp hoặc nằm sát nhau.

Đối với phần bị che khuất, chỉ gán nhãn vùng có thể xác định được theo guideline đã thống nhất; không tự suy diễn phần hình dạng nằm phía sau vật thể khác.

### 3.7. Xử lý biên mờ, vật thể tiếp xúc và vùng bị che khuất

Trong Instance Segmentation, các trường hợp biên mờ, hai vật thể tiếp xúc hoặc một phần đối tượng bị che khuất có thể gây ra sự không thống nhất giữa các annotator.

Nguyên tắc xử lý:

- Nếu guideline đã quy định rõ cách xác định biên, annotator áp dụng thống nhất theo hướng dẫn.
- Nếu đường biên không rõ ràng nhưng vẫn có thể xác định theo tiêu chí đã được phê duyệt, thực hiện annotation theo tiêu chí đó.
- Nếu hai vật thể cùng chạm nhau và không thể xác định ranh giới đáng tin cậy, hoặc guideline chưa đề cập đến trường hợp này, cần escalation lên Lab Coach.
- Không tự ý gộp hai instance hoặc suy đoán vùng mask bị che khuất.

Việc escalation giúp đảm bảo các trường hợp khó được xử lý nhất quán và có thể cập nhật thành guideline chung cho toàn bộ nhóm.

### 3.8. Kết luận

Thực nghiệm đã hoàn thành bước Instance Segmentation bằng YOLO11 với checkpoint `yolo11n-seg.pt`, taxonomy COCO-80 và confidence threshold `0.35`. Mô hình tạo ra tổng cộng 49 instance trên ba ảnh mẫu. Đồng thời, ảnh segmentation_prediction.png được tạo để trực quan hóa kết quả phân đoạn trên ảnh kitchen.

Kết quả được biểu diễn bằng bounding box và polygon tọa độ pixel. Trong đó, polygon cung cấp thông tin chi tiết hơn về hình dạng và đường biên của từng đối tượng. Mỗi instance có một `instance_id` riêng, giúp phân biệt các đối tượng cùng lớp.

kết quả dự đoán của mô hình chỉ là dữ liệu tham khảo ban đầu. Annotator vẫn cần kiểm tra và hiệu chỉnh mask theo guideline; những trường hợp biên không rõ ràng hoặc không có quy tắc xử lý cần được escalation để đảm bảo chất lượng Ground Truth.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground Truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| **Phân loại ảnh** | Một hoặc nhiều nhãn lớp gắn với toàn bộ ảnh, tùy guideline. Đơn vị nhãn là ảnh. | JSON `classification_predictions.json` cho ảnh `traffic` có Top-1 là `cab`, score `0.510915`; các lớp `minibus`, `police_van` cũng xuất hiện trong Top-5. Điều này cho thấy có nhiều lớp cạnh tranh, cần xác định tiêu chí chọn nhãn chính. | Quan sát toàn bộ ảnh, xác định lớp phù hợp theo guideline; không lấy trực tiếp Top-1 prediction làm Ground Truth. Nếu ảnh có nhiều đối tượng, áp dụng quy tắc chọn nhãn chính hoặc multi-label đã được quy định. | Kiểm tra nhãn có phản ánh đúng nội dung ảnh không; cách chọn nhãn chính có nhất quán giữa các annotator không; kiểm tra các trường hợp nhiều lớp dễ nhầm lẫn. |
| **Phát hiện vật thể** | Mỗi đối tượng có một nhãn lớp và một bounding box `[x_min, y_min, x_max, y_max]`. Đơn vị nhãn là từng instance. | Với ảnh `kitchen`, số prediction thay đổi theo threshold: 17 ở mức `0.20`, 11 ở mức `0.35` và 6 ở mức `0.60`. Điều này cho thấy việc chọn threshold ảnh hưởng đến số dự đoán được giữ lại; các đối tượng có score thấp có thể cần được kiểm tra thêm. | Xác định từng đối tượng thực tế, gán đúng class và vẽ bounding box bao sát phần đối tượng nhìn thấy theo guideline. Không tự động xem mọi prediction là đối tượng hợp lệ. | Kiểm tra thiếu hoặc thừa box, sai class, box quá rộng hoặc quá hẹp, box trùng lặp và cách xử lý đối tượng bị che khuất hoặc nằm sát mép ảnh. |
| **Instance segmentation** | Mỗi đối tượng có một nhãn lớp và một polygon/mask mô tả vùng pixel thuộc đối tượng. Đơn vị nhãn là từng instance. | JSON `segmentation_predictions.json` ghi nhận 49 instance trên ba ảnh. Ví dụ hai record đầu của ảnh `traffic` là `bus` và `car`, có lần lượt 120 và 92 điểm polygon. Số điểm polygon khác nhau cho thấy mức độ phức tạp của đường biên; cần kiểm tra polygon có bám sát vật thể hay không. | Vẽ polygon theo đường biên của từng đối tượng, phân biệt các instance cùng lớp và chỉ gán nhãn vùng có thể xác định theo guideline. Không suy diễn vùng bị che khuất. | Kiểm tra mask có bám sát đường biên không, có chứa nền hoặc bỏ sót vùng đối tượng không, các instance có bị gộp nhầm không và cách xử lý vùng tiếp xúc hoặc bị che khuất. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

### 5.1. Quy tắc bảo vệ dữ liệu

Chỉ sử dụng dữ liệu được cung cấp hoặc cho phép trong phạm vi bài lab, không tự ý chia sẻ, tải lên dịch vụ bên ngoài, công khai hoặc tiết lộ thông tin cá nhân và dữ liệu nhạy cảm; đồng thời bảo vệ các file đầu vào và kết quả thực nghiệm theo đúng quy định của chương trình.

### 5.2. Xử lý dữ liệu không đúng phạm vi

Nếu phát hiện ảnh hoặc dữ liệu không thuộc phạm vi được phép xử lý, có chứa thông tin nhạy cảm hoặc có dấu hiệu vi phạm quy định bảo mật, tôi sẽ dừng xử lý, không sao chép hay chia sẻ dữ liệu và báo ngay cho **Lab Coach** để được hướng dẫn xử lý theo quy trình.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
