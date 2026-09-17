# config.py
# Chứa các hằng số và cấu hình cho toàn bộ dự án

# --- OpenCV DIP Configuration ---
# Kích thước kernel cho Gaussian Blur
BLUR_KERNEL_SIZE = (5, 5)

# HSV Color ranges (Mặc định cho da người hoặc màu một đối tượng cụ thể)
# Ở đây dùng dải màu cho màu da (skin color) trong điều kiện ánh sáng bình thường
HSV_LOWER = (0, 20, 70)
HSV_UPPER = (20, 255, 255)

# Kích thước kernel cho Morphological operations
MORPH_KERNEL_SIZE = (5, 5)

# --- MediaPipe Tracking & ROI Configuration ---
# Bật/tắt chế độ dùng ROI (Cắt ảnh theo mask DIP trước khi đưa vào MediaPipe)
USE_ROI_MODE = True

# Padding cho Bounding Box (pixel) để đảm bảo không bị cắt mất ngón tay
ROI_PADDING = 30

# Khoảng cách tối đa (pixels) giữa ngón cái (Thumb) và ngón trỏ (Index) để được tính là "Pinch" (Đang viết)
PINCH_THRESHOLD = 40

# Độ tin cậy cho việc phát hiện và tracking tay
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# --- Trajectory & Smoothing Configuration ---
# Số điểm dùng để tính Moving Average (càng lớn càng mượt nhưng trễ)
SMOOTHING_WINDOW_SIZE = 5

# --- Reconstruction Configuration ---
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
CANVAS_BG_COLOR = (255, 255, 255) # Nền trắng
DRAWING_COLOR = (0, 0, 0)         # Nét vẽ đen
DRAWING_THICKNESS = 5             # Độ dày nét chữ
