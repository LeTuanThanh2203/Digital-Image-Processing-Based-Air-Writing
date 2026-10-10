# Air Writing - Trajectory Extraction & Handwriting Reconstruction

Hệ thống viết trên không (Air Writing) dựa trên Xử lý ảnh Kỹ thuật số (Digital Image Processing) và Trí tuệ Nhân tạo (CNN). Hệ thống cho phép theo dõi cử chỉ bàn tay qua webcam, trích xuất quỹ đạo viết, làm mượt nét vẽ, tái tạo chữ viết trên canvas và nhận dạng ký tự tiếng Anh/chữ số bằng mô hình CNN.

---

## 🛠️ Thư viện và Cài đặt (Requirements & Installation)

### 1. Cài đặt nhanh qua `requirements.txt` (Khuyên dùng)

Chạy lệnh sau trong Terminal / Command Prompt:

```bash
pip install -r requirements.txt
```

### 2. Cài đặt thủ công từng thư viện

Nếu bạn muốn cài đặt lẻ từng thư viện bằng `pip`:

```bash
# Thư viện xử lý ảnh OpenCV
pip install opencv-python

# Thư viện theo dõi bàn tay MediaPipe
pip install mediapipe

# Thư viện tính toán mảng NumPy
pip install numpy

# Thư viện đọc weights mô hình HDF5 (.keras) cho NumPy CNN Engine
pip install h5py
```

### 📋 Đơn giản hóa cài đặt trong một dòng lệnh:

```bash
pip install opencv-python mediapipe numpy h5py
```

> **Lưu ý về TensorFlow / Keras:**
> Hệ thống đã được tích hợp sẵn **NumPy CNN Inference Engine** thuần `NumPy` + `h5py`, cho phép chạy mô hình `.keras` trực tiếp mà **KHÔNG CẦN** cài đặt TensorFlow hoặc Keras cồng kềnh.
> (Nếu môi trường của bạn đã cài sẵn `tensorflow` hoặc `keras`, hệ thống sẽ tự động ưu tiên sử dụng).

---

## 🚀 Huớng dẫn Khởi chạy (How to Run)

Chạy file `main.py` bằng Python:

```bash
python main.py
# hoặc trên Windows nếu có nhiều phiên bản Python:
py main.py
```

---

## 🖐️ Cử chỉ Điều khiển (Gesture Controls)

| Cử chỉ (Gesture) | Trạng thái tay | Hành động |
| :--- | :--- | :--- |
| **WRITING** | Ngón trỏ giơ lên (các ngón khác gập lại) | Vẽ/viết chữ theo đầu ngón trỏ |
| **SPACE** | Ngón trỏ + Ngón út giơ lên | Thêm dấu cách (Giữ ~0.5s) |
| **DELETE** | Bóp tay thành nắm đấm | Xóa ký tự vừa viết (Giữ ~0.5s) |
| **MODE** | Ngón trỏ + Ngón giữa giơ lên (V-sign) | Chuyển đổi giữa chế độ `LETTER` (A-Z) và `NUMBER` (0-9) |

---

## ⌨️ Phím tắt Bàn phím (Keyboard Shortcuts)

- `d`: Bật/Tắt cửa sổ Debug xem ảnh nét chữ 64x64 đưa vào CNN.
- `c`: Xóa toàn bộ nội dung văn bản và canvas hiện tại.
- `s`: Lưu ảnh Canvas chữ viết ra file.
- `q`: Thoát chương trình.

---

## 🏗️ Cấu trúc Dự án (Project Structure)

```text
Digital-Image-Processing-Based-Air-Writing/
├── main.py                     # Chương trình chính điều khiển State Machine
├── config.py                   # Cấu hình ngưỡng tracking, canvas, CNN, v.v.
├── requirements.txt            # Danh sách thư viện cần cài đặt
├── hand_landmarker.task        # Model tracking MediaPipe Hand Landmarker
├── models/
│   └── airwriting_cnn.keras    # Model CNN nhận dạng chữ viết
├── tracking/
│   └── hand_tracker.py         # Theo dõi khớp bàn tay & nhận diện gesture
├── trajectory/
│   ├── smoothing.py            # Thuật toán làm mượt Moving Average & lọc jump
│   └── trajectory_manager.py   # Quản lý stroke và chuỗi điểm nét vẽ
├── processing/
│   └── pipeline_processor.py   # DIP Pipeline: Squarify, Padding, Bounding Box, Render 64x64
├── reconstruction/
│   └── handwriting.py          # Render canvas nét chữ 640x480
└── recognition/
    └── classifier.py           # Module nhận dạng CNN (hỗ trợ NumPy Engine & Keras)
```
