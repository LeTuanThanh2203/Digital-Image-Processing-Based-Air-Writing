# config.py
# Chứa các hằng số và cấu hình cho toàn bộ dự án

# --- MediaPipe Tracking Configuration ---
# Khoảng cách tối đa (pixels) giữa ngón cái (Thumb) và ngón trỏ (Index) để được tính là "Pinch" (Đang viết)
PINCH_THRESHOLD = 40

# Độ tin cậy cho việc phát hiện và tracking tay
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# --- Trajectory & Smoothing Configuration ---
# Số điểm dùng để tính Moving Average (càng lớn càng mượt nhưng trễ)
SMOOTHING_WINDOW_SIZE = 5
MAX_JUMP_DISTANCE = 50.0           # Khoảng cách điểm tối đa (pixel) để lọc cú nhảy nhiễu tracking
MIN_STROKE_POINTS = 3              # Số điểm tối thiểu để giữ lại một stroke hợp lệ
CONFIDENCE_THRESHOLD = 40.0        # Ngưỡng tin cậy phần trăm tối thiểu để nhận ký tự

# --- Reconstruction Configuration ---
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
CANVAS_BG_COLOR = (255, 255, 255) # Nền trắng
DRAWING_COLOR = (0, 0, 0)         # Nét vẽ đen
DRAWING_THICKNESS = 8             # Độ dày nét chữ (tăng từ 5 -> 8px cho chữ to, rõ nét)

# --- Gesture & Writing State Configuration ---
GESTURE_STABILIZATION_FRAMES = 5   # Majority-vote window to stabilize gestures
PENDING_DURATION = 1.0             # Seconds to wait after stopping writing before CNN fires
SPACE_HOLD_FRAMES = 15             # Consecutive frames SPACE gesture must be held (~0.5s at 30fps)
DELETE_HOLD_FRAMES = 15            # Consecutive frames DELETE gesture (fist) must be held (~0.5s at 30fps)
MIN_MOVEMENT_THRESHOLD = 3.0       # Minimum pixel movement to consider fingertip moving
MIN_TRAJECTORY_POINTS = 8          # Minimum trajectory points before CNN is called

# --- CNN Recognition Configuration ---
from pathlib import Path
MODEL_PATH = Path(__file__).resolve().parent / "models" / "airwriting_cnn.keras"
CNN_INPUT_SIZE = (64, 64)
CNN_NUM_CLASSES = 36
SHOW_RECOGNITION_DEBUG = False     # Bật/tắt hiển thị cửa sổ debug 64x64 CNN input

# Danh sách nhãn ký tự (CLASS_LABELS):
# 1. Nếu đặt CLASS_LABELS = None: Hệ thống sẽ TỰ ĐỘNG đọc số lớp từ file model .keras.
#    - Model 26 lớp -> Tự động ánh xạ: ['A'..'Z']
#    - Model 10 lớp -> Tự động ánh xạ: ['0'..'9']
#    - Model 36 lớp -> Mặc định tự động ánh xạ: ['0'..'9'] + ['A'..'Z']
#
# 2. Nếu muốn tùy chỉnh thủ công, bạn có thể bỏ comment một trong các cấu hình dưới đây:

# --- Cấu hình A: Model 26 lớp (Chỉ chữ cái A-Z) ---
# CLASS_LABELS = [chr(ord('A') + i) for i in range(26)]

# --- Cấu hình B: Model 36 lớp (Chữ cái A-Z từ 0..25, Số 0-9 từ 26..35) ---
# CLASS_LABELS = [chr(ord('A') + i) for i in range(26)] + [str(i) for i in range(10)]

# --- Cấu hình C: Model 36 lớp (Số 0-9 từ 0..9, Chữ cái A-Z từ 10..35) ---
# CLASS_LABELS = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]

CLASS_LABELS = None



