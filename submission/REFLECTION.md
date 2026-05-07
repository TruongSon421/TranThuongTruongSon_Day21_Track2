# REFLECTION.md

## Tổng Quan Project

Đây là project MLOps end-to-end sử dụng **Wine Quality dataset** (phân loại rượu thành 3 nhãn: thấp / trung bình / cao). Project được xây dựng qua 3 bước với mục tiêu tự động hóa toàn bộ vòng đời ML: huấn luyện, đánh giá, và triển khai.

---

## Kiến Trúc Hệ Thống

```
Local                        GitHub Actions                      Cloud (AWS)
-----                        --------------                     ----------
code + data ----git push---> GitHub Actions CI/CD
                                  |
                            [Unit Test] ----+
                                  |
                            [Train] --------+--> S3 bucket
                                  |                  |
                                  |                  +-- models/latest/model.pkl
                                  |                  +-- models/latest/metrics.json
                                  |
                            [Eval] (accuracy >= 0.65 ?)
                                  |
                            [Deploy] ----SSH----> EC2 VM
                                                  |
                                            FastAPI (port 8000)
                                            + systemd service
                                            + download model from S3 on startup
```

### Cloud Stack (thay vì GCP mặc định của task)

| Thành phần | GCP (template) | Thực tế (bạn chọn) |
|---|---|---|
| Object Storage | GCS + `gsutil` | **S3** + `boto3` |
| VM | Compute Engine | **EC2** |
| Credentials | Service Account JSON | **AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY** |
| DVC remote | `gs://bucket/dvc` | **s3://mlops-track2-dev-model-store-52de15/dvc** |
| Deploy action | generic SSH | **appleboy/ssh-action** với EC2_HOST / EC2_USER / EC2_SSH_KEY |

---

## Những Gì Đã Làm Từng Bước

### Bước 1 — Thực nghiệm cục bộ

Bạn đã cấu hình MLflow để ghi lại các thí nghiệm huấn luyện trên máy local, thử nghiệm nhiều bộ siêu tham số, và chọn ra cấu hình tốt nhất để đưa vào pipeline tự động.

Cấu hình cuối cùng trong `params.yaml`:

```yaml
model_type: extra_trees
n_estimators: 500
max_features: 1.0
criterion: gini
class_weight: balanced
random_state: 42
```

Model `ExtraTreesClassifier` được chọn vì hoạt động tốt trên dữ liệu dạng bảng với phân phối lớp không cân bằng (label 2 chỉ chiếm ~20%).

### Bước 2 — Pipeline CI/CD tự động

**Những gì đã triển khai:**

1. **GitHub Actions workflow** (`.github/workflows/mlops.yml`) với 4 jobs nối tiếp:
   - `test`: Chạy 9 unit tests bằng `pytest tests/ -v` trên dữ liệu tạo ngẫu nhiên (không cần cloud credentials).
   - `train`: DVC pull dữ liệu từ S3 → chạy `python src/train.py` → đọc accuracy từ `outputs/metrics.json` → upload `model.pkl` + `metrics.json` lên S3 `models/latest/`.
   - `eval`: Hai gate kiểm tra:
     - Accuracy phải >= 0.65 (dưới ngưỡng 0.70 trong spec gốc, đây là điều chỉnh thực tế của bạn).
     - Accuracy mới phải >= accuracy của lần deploy trước (so sánh với `models/latest/metrics.json` trên S3). Nếu model mới kém hơn, pipeline dừng — đây là một bổ sung tốt giúp đảm bảo chất lượng model không bị regression.
   - `deploy`: SSH vào EC2 bằng `appleboy/ssh-action`, restart `mlops-serve` service, chờ 5 giây, gọi `/health` endpoint để xác nhận.

2. **FastAPI inference server** (`src/serve.py`):
   - Dùng `boto3` (thay vì `google-cloud-storage`) để tải model từ S3 về khi khởi động.
   - Biến môi trường: `AWS_REGION`, `S3_BUCKET`, `S3_MODEL_KEY`.
   - Endpoint `/health`: trả `{"status": "ok"}`.
   - Endpoint `/predict`: nhận 12 features, trả `{"prediction": int, "label": str}`.

3. **Unit tests** (`tests/test_train.py`):
   - 9 test cases bao gồm: kiểm tra giá trị trả về của `train()`, tạo file `metrics.json` / `model.pkl` / `report.txt`, hỗ trợ nhiều model type (`random_forest`, `gradient_boosting`, `logistic_regression`, `extra_trees`), xử lý `None` params cho `extra_trees`, và kiểm tra báo lỗi khi dùng model type không hỗ trợ.
   - `conftest.py` redirect MLflow tracking vào thư mục tạm mỗi test để tránh conflict giữa các lần chạy.

4. **DVC data versioning**:
   - Remote: `s3://mlops-track2-dev-model-store-52de15/dvc`
   - Track 3 files: `train_phase1.csv`, `train_phase2.csv`, `eval.csv`

5. **Systemd service** trên EC2 để service tự khởi động khi VM reboot.

**Kết quả Bước 2:**

```json
{"accuracy": 0.678, "f1_score": 0.676}
```

### Bước 3 — Huấn luyện liên tục khi có dữ liệu mới

Quy trình thêm dữ liệu mới:

1. Chạy `python add_new_data.py` — ghép `train_phase2.csv` (2998 mẫu) vào `train_phase1.csv` (tổng: 5996 mẫu).
2. `dvc add data/train_phase1.csv` — thông báo cho DVC biết dữ liệu đã thay đổi.
3. `git add data/train_phase1.csv.dvc` — commit con trỏ DVC (không commit CSV — CSV đã nằm trong `.gitignore`).
4. `dvc push` — đẩy dữ liệu mới lên S3 **trước** git push (nếu ngược lại, CI sẽ pull thất bại).
5. `git push origin master` — kích hoạt GitHub Actions.

**Điều quan trọng cần nhớ:** Phải `dvc push` **trước** `git push`. Nếu không, CI runner bắt đầu job trước khi dữ liệu có mặt trên S3 → `dvc pull` thất bại.

Pipeline chạy lại hoàn toàn tự động: test → train (với 5996 mẫu) → eval (accuracy 0.756 ≥ 0.65 ✅, 0.756 ≥ 0.678 ✅ so với Bước 2) → deploy.

**Kết quả Bước 3:**

```json
{"accuracy": 0.756, "f1_score": 0.755}
```

---

## So Sánh Kết Quả

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) | Thay đổi |
|---|---|---|---|
| accuracy | 0.678 | 0.756 | **+0.078** (+11.5%) |
| f1_score | 0.676 | 0.755 | **+0.079** (+11.7%) |

Việc nhân đôi dữ liệu huấn luyện cải thiện đáng kể cả accuracy lẫn f1_score. Điều này cho thấy model `extra_trees` vẫn chưa bão hòa và tiếp tục học được pattern mới từ dữ liệu bổ sung.

---

## Thành Tựu Nổi Bật

1. **Tùy biến cloud provider linh hoạt**: Thay GCP mặc định bằng AWS S3 + EC2 mà vẫn đảm bảo đầy đủ chức năng — DVC remote, boto3 SDK, SSH deploy đều hoạt động đúng.

2. **Eval gate hai lớp**: Ngoài kiểm tra ngưỡng tuyệt đối (>= 0.65), còn so sánh với accuracy của lần deploy trước để tránh regression. Đây là cải tiến thông minh so với spec gốc.

3. **Số lượng test nhiều hơn spec**: Spec chỉ yêu cầu 3 tests, nhưng bạn đã viết 9 tests bao gồm nhiều edge cases (multi-model support, None params, report content, value ranges).

4. **Hệ thống tự phục hồi**: systemd service trên EC2 đảm bảo API tự khởi động sau khi VM reboot mà không cần can thiệp thủ công.

5. **Pipeline hoàn toàn không cần thao tác thủ công sau git push**: Từ dữ liệu mới → CI/CD → model trên production chỉ qua `dvc push` + `git push`.

---

## Hạn Chế & Điều Cần Cải Thiện

1. **Ngưỡng eval thấp hơn spec**: Spec yêu cầu accuracy >= 0.70, nhưng pipeline của bạn dùng ngưỡng 0.65. Dù Bước 3 đạt 0.756 (vượt cả hai), nhưng nếu accuracy rơi vào khoảng 0.66–0.69, model vẫn được deploy trong khi spec gốc sẽ chặn.

2. **Không thấy Bước 1 chi tiết**: Task buoc-1.md có trong repo nhưng chưa xem chi tiết các thí nghiệm MLflow. Không rõ đã thử bao nhiêu model type và siêu tham số trước khi chọn `extra_trees`.

3. **Không có Rollback tự động**: Nếu model mới deploy thành công nhưng inference sai nhiều hơn (data drift), hệ thống chưa có cơ chế tự động quay về model cũ.

4. **Không có Monitoring**: Không có alert khi accuracy trên production giảm, không có logging cho các request dự đoán, không có Prometheus/Grafana dashboard.

5. **DVC credentials không lưu trong pipeline**: Trong GitHub Actions, credentials AWS được set qua secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) nhưng DVC remote config (url `s3://...`) chỉ lưu trong `.dvc/config` (đã commit git). Điều này OK cho repo công khai nếu bucket không cần credentials để đọc, nhưng cần kiểm tra kỹ IAM policy.

---

## Kiến Thức MLOps Áp Dụng Được

Dự án này thể hiện nhiều nguyên tắc MLOps thực tế:

- **CI/CD cho ML**: Code và data versioning tách biệt (git vs DVC), mỗi push có thể tạo model mới.
- **Eval gate**: Quality gate trước deploy — chỉ model đạt ngưỡng mới được triển khai.
- **Data versioning**: DVC theo dõi thay đổi dữ liệu, đảm bảo reproducibility.
- **Continuous Training (CT)**: Dữ liệu mới tự động trigger pipeline huấn luyện lại.
- **Model registry pattern**: Mỗi lần deploy lưu vào `models/latest/`, đồng thời giữ được artifact cũ để so sánh và rollback nếu cần.
