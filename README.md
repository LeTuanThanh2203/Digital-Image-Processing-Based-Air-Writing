# Air Writing: Trajectory Extraction and Handwriting Reconstruction

> **Đồ án môn học Xử lý Ảnh Kỹ thuật số (Digital Image Processing - DIP)**  
> **Trường Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE)**

---

## 👥 Thành viên nhóm thực hiện

| STT | MSSV | Họ và tên | Vai trò |
| :---: | :---: | :--- | :--- |
| 1 | **24110075** | **Phạm Phước Anh** | Thành viên |
| 2 | **24110127** | **Ca Ngọc Tân** | Thành viên |
| 3 | **24110128** | **Lê Tuấn Thành** | Thành viên |

---

## 📖 Giới thiệu Dự án

Hệ thống viết chữ trên không thời gian thực (Real-time Air Writing) kết hợp các kỹ thuật **Xử lý Ảnh Kỹ thuật số (DIP)** và **Trí tuệ Nhân tạo (CNN)**:
- **Theo dõi bàn tay (Tracking)**: Sử dụng MediaPipe Hand Landmarker trích xuất 21 điểm khớp tay 3D theo thời gian thực qua Webcam.
- **Xử lý quỹ đạo (Trajectory DIP)**: Lọc nhảy tọa độ (Jump filter) và làm mượt nét vẽ bằng thuật toán Moving Average.
- **Tái tạo chữ viết (Reconstruction)**: Render nét chữ mượt mà trên Canvas 640x480.
- **Nhận dạng ký tự (Recognition)**: Bộ phân loại Hybrid kết hợp **NumPy CNN Inference Engine** (chạy trực tiếp model `.keras` thuần túy không cần cài đặt TensorFlow) và so khớp hình thái học **SSIM** trên ảnh nhị phân 64x64.
- **Gợi ý từ thông minh (Trie Search)**: Tự động tìm kiếm và gợi ý từ vựng tiếng Anh theo tiền tố với từ điển 500.000 từ.

---

## 🛠️ Hướng dẫn Cài đặt & Khởi chạy

### 1. Yêu cầu môi trường
- **Python**: Khuyến nghị phiên bản `3.10` - `3.13` (đã kiểm thử ổn định trên Python 3.13).
- **Webcam**: Camera tích hợp hoặc cắm ngoài qua cổng USB.

### 2. Cài đặt thư viện

Mở Terminal / PowerShell tại thư mục dự án và chọn **1 trong 2 cách** sau:

#### Cách 1: Sử dụng Python chuẩn (Khuyên dùng)
```bash
# Bước 1: Tạo môi trường ảo
# Trên Windows (dùng 'py' launcher để tránh xung đột nếu máy có cài MSYS2/MinGW):
py -m venv .venv
# Hoặc: python -m venv .venv

# Bước 2: Kích hoạt môi trường ảo
# Trên Windows (PowerShell / CMD):
.venv\Scripts\activate
# Trên macOS / Linux:
# source .venv/bin/activate

# Bước 3: Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

#### Cách 2: Sử dụng trình quản lý gói  `uv` (Nhanh nhất)
```bash
# Tự động tạo venv và cài đặt toàn bộ thư viện chỉ trong 2 giây:
uv sync
```

### 3. Khởi chạy chương trình

```bash
# Chạy trực tiếp sau khi kích hoạt venv:
python main.py

# Hoặc chạy không cần kích hoạt venv (nếu dùng uv):
uv run python main.py
```

> **Lưu ý:** Hệ thống đã tích hợp sẵn **NumPy CNN Engine**, tự động đọc weights từ file `.keras` và suy luận bằng `numpy` + `h5py`. Bạn **hoàn toàn không cần cài đặt TensorFlow hay Keras**.

---

## 📚 Danh mục Thư viện (`requirements.txt`)

| Thư viện | Phiên bản | Vai trò trong đồ án |
| :--- | :--- | :--- |
| **`opencv-python`** | `>=4.8.0` | Thu thập video Webcam, xử lý hình thái học ảnh (DIP), render UI Dashboard & Canvas |
| **`mediapipe`** | `>=0.10.0` | Nhận diện cử chỉ và định vị 21 khớp tay theo model offline `hand_landmarker.task` |
| **`numpy`** | `>=1.24.0` | Tính toán ma trận ảnh, làm mượt tọa độ nét vẽ và chạy mạng CNN thuần |
| **`h5py`** | `>=3.8.0` | Đọc trọng số weights nơ-ron từ file mô hình `.keras` |
| **`Pillow`** | `>=10.0.0` | Kết xuất văn bản tiếng Việt Unicode sắc nét trên giao diện điều khiển |

---

## 🖐️ Cử chỉ Điều khiển (Gesture Controls)

| Cử chỉ (Gesture) | Tư thế ngón tay | Hành động kích hoạt |
| :--- | :--- | :--- |
| **WRITING** | Ngón trỏ duỗi thẳng (các ngón khác co) | Viết chữ / số trên không theo tọa độ đầu ngón trỏ |
| **MODE** | Ngón trỏ + ngón giữa duỗi (V-sign) | Chuyển đổi qua lại giữa chế độ `LETTER (A-Z)` và `NUMBER (0-9)` |
| **SPACE** | Ngón trỏ + ngón út duỗi (Rock sign) | Thêm dấu cách trắng giữa các từ (Giữ ~0.5s) |
| **DELETE** | Nắm chặt toàn bộ bàn tay (Nắm đấm) | Xóa ký tự vừa viết gần nhất (Giữ ~0.5s) |
| **CYCLE** | Ngón cái + ngón trỏ tạo góc vuông (Chữ L) | Duyệt qua các từ gợi ý tiếng Anh hiển thị trên màn hình |
| **CONFIRM** | 3 ngón (Trỏ + Giữa + Áp út) duỗi thẳng | Xác nhận chọn từ gợi ý đang highlight |
| **IDLE** | Xòe 5 ngón tay hoặc hạ tay xuống | Nghỉ; sau 1 giây ngưng vẽ hệ thống sẽ tự động nhận diện nét chữ |

---

## ⌨️ Phím tắt Bàn phím

- **`c`**: Xóa toàn bộ nội dung văn bản và canvas nét vẽ.
- **`s`**: Lưu ảnh canvas nét vẽ hiện tại vào thư mục `output/`.
- **`d`**: Bật / Tắt cửa sổ Debug xem ảnh 64x64 đưa vào bộ phân loại.
- **`q`**: Thoát an toàn khỏi chương trình.

---

## 🏗️ Cấu trúc Thư mục Dự án

```text
Digital-Image-Processing-Based-Air-Writing/
├── main.py                               # Vòng lặp chính, điều khiển State Machine và luồng xử lý
├── config.py                             # File cấu hình tham số, ngưỡng tracking và màu sắc
├── requirements.txt                      # Danh sách thư viện Python cần cài đặt
├── hand_landmarker.task                  # Model MediaPipe Hand Landmarker chạy offline
├── frequency_dictionary_en_500_000.txt   # Từ điển tiếng Anh 500k từ phục vụ Trie Search
├── assets/
│   └── logo_ute.jpg                      # Logo trường HCMUTE trên Dashboard
├── models/
│   ├── airwriting_letters_cnn.keras      # Model CNN nhận diện 26 chữ cái tiếng Anh (A-Z)
│   └── airwriting_digits_cnn.keras       # Model CNN nhận diện 10 chữ số (0-9)
├── dataset/
│   └── custom/                           # Dữ liệu ảnh mẫu phục vụ bộ phân loại Hybrid SSIM
├── tracking/
│   ├── webcam.py                         # Module điều khiển và thu thập khung hình Webcam
│   └── hand_tracker.py                   # Module phát hiện và nhận diện cử chỉ bàn tay
├── trajectory/
│   ├── smoothing.py                      # Bộ lọc nhảy (Jump filter) và làm mượt Moving Average
│   └── trajectory_manager.py             # Quản lý tập hợp các điểm nét vẽ (Strokes)
├── processing/
│   └── pipeline_processor.py             # Pipeline DIP tiền xử lý, căn giữa và chuẩn hóa ảnh 64x64
├── reconstruction/
│   ├── handwriting.py                    # Vẽ nét chữ và hiển thị Canvas trắng 640x480
│   └── ui_dashboard.py                   # Giao diện Dashboard hiển thị thông tin và Unicode font
└── recognition/
    ├── classifier.py                     # Trình suy luận CNN thuần NumPy Engine & Keras loader
    ├── hybrid_classifier.py              # Bộ phân loại kết hợp SSIM hình thái học + Xác suất CNN
    └── trie_search.py                    # Cấu trúc dữ liệu Trie gợi ý từ thông minh
```

---

## 📦 Lưu ý Dọn dẹp khi Nộp bài (Cleaning & Submission)

Để file zip nộp bài nhẹ, đúng quy cách và không phát sinh lỗi:

### 1. Thư mục BẮT BUỘC XÓA trước khi nén zip:
- **`.venv/`** (~310 MB): Môi trường ảo của máy cá nhân. Người chấm sẽ tự cài qua `requirements.txt`.
- **`__pycache__/`**: Bộ nhớ đệm bytecode của Python.
- **`scratch/`**: Thư mục chứa script test nháp trong quá trình phát triển.
- Các file ảnh chụp màn hình thử nghiệm trong thư mục **`output/`**.

### 2. Thư mục BẮT BUỘC GIỮ LẠI:
- `models/` (chứa 2 file model `.keras`)
- `hand_landmarker.task` (~7.8 MB)
- `frequency_dictionary_en_500_000.txt` (~7.4 MB)
- `dataset/custom/` (~1.7 MB)
- `assets/`
- Toàn bộ mã nguồn các thư mục `.py` và file `requirements.txt`.

**Lệnh PowerShell dọn dẹp nhanh 1-click trước khi zip:**
```powershell
Remove-Item -Recurse -Force .venv, scratch -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force
```


