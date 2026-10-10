# processing/pipeline_processor.py
import cv2
import numpy as np

def compute_robust_bounding_box(all_points, quantile_low=0.02, quantile_high=0.98):
    """
    Tính Bounding Box kháng nhiễu điểm xa (Stray Outliers).
    Dùng Percentile & 2.5*IQR chỉ để loại bỏ ranh giới của các điểm nhiễu cực xa khỏi Bounding Box.
    Không xóa điểm khỏi stroke, bảo tồn hoàn toàn nét chữ xấu/nghiêng/méo.
    """
    xs = np.array([p[0] for p in all_points], dtype=np.float32)
    ys = np.array([p[1] for p in all_points], dtype=np.float32)

    if len(all_points) < 10:
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

    q_x_low, q_x_high = np.quantile(xs, quantile_low), np.quantile(xs, quantile_high)
    q_y_low, q_y_high = np.quantile(ys, quantile_low), np.quantile(ys, quantile_high)

    iqr_x = max(1.0, q_x_high - q_x_low)
    iqr_y = max(1.0, q_y_high - q_y_low)

    # Ngưỡng an toàn cực rộng 2.5 * IQR để không lỡ loại các nét của chữ xấu
    valid_mask = (
        (xs >= q_x_low - 2.5 * iqr_x) & (xs <= q_x_high + 2.5 * iqr_x) &
        (ys >= q_y_low - 2.5 * iqr_y) & (ys <= q_y_high + 2.5 * iqr_y)
    )

    valid_xs = xs[valid_mask]
    valid_ys = ys[valid_mask]

    if len(valid_xs) > 0 and len(valid_ys) > 0:
        return float(valid_xs.min()), float(valid_xs.max()), float(valid_ys.min()), float(valid_ys.max())
    else:
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def keep_main_group(image_mask: np.ndarray, kernel_size=(5, 5)) -> np.ndarray:
    """
    keep_main_group():
    1. Phép nở Morphological Dilation với kernel nhẹ (5x5) chỉ làm mask tạm thời tìm cụm nét chính.
    2. Phân tích thành phần liên thông (Connected Components) giữ lại cụm chữ chính lớn nhất.
    3. Trả về nét chữ gốc (chưa bị làm bít lỗ) sau khi lọc bỏ các điểm rác tách biệt ở xa.
    """
    if image_mask is None or cv2.countNonZero(image_mask) == 0:
        return image_mask

    # 1. Dilation với kernel hình ellipse nhẹ (5x5) làm mask liên thông tạm thời
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated = cv2.dilate(image_mask, kernel, iterations=1)

    # 2. Phân tích thành phần liên thông Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    if num_labels <= 1:
        return image_mask

    # Tìm nhãn có diện tích lớn nhất (bỏ qua background nhãn 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Tạo mask vùng chính từ nhãn lớn nhất
    main_group_mask = np.uint8(labels == largest_label) * 255

    # Lấy giao của nét chữ GỐC với mask nhóm chính (bảo tồn nguyên vẹn lỗ bụng chữ B, D, O, P)
    cleaned_mask = cv2.bitwise_and(image_mask, image_mask, mask=main_group_mask)
    return cleaned_mask


def render_trajectory_to_64x64(strokes, target_size=(64, 64), padding=6, line_thickness=2) -> np.ndarray:
    """
    render_trajectory():
    1. Lấy tất cả các điểm tọa độ từ danh sách strokes.
    2. Chuẩn hóa: Dịch chuyển về tâm/bbox (kháng nhiễu điểm xa), scale giữ đúng tỉ lệ aspect ratio.
    3. Vẽ trực tiếp các nét mượt lên mảng ảnh 64x64 bằng cv2.LINE_AA (khử răng cưa).
    4. Áp dụng keep_main_group() lọc các nét lạc (kernel 5x5 nhẹ).
    5. Áp dụng Dilation 2x2 làm đậm nhẹ nét chữ, giữ nguyên độ rộng lỗ bụng chữ B, D, O, P.
    """
    all_points = [pt for stroke in strokes for pt in stroke]
    if not all_points:
        return np.zeros(target_size, dtype=np.uint8)

    # 1. Lấy Robust Bounding Box (kháng stray points ở xa)
    min_x, max_x, min_y, max_y = compute_robust_bounding_box(all_points)

    w = max(1, max_x - min_x)
    h = max(1, max_y - min_y)

    target_w, target_h = target_size
    drawable_w = target_w - 2 * padding
    drawable_h = target_h - 2 * padding

    # 2. Tính scale giữ tỷ lệ aspect ratio
    scale = min(drawable_w / w, drawable_h / h)

    # Căn giữa trong vùng vẽ
    scaled_w = w * scale
    scaled_h = h * scale
    offset_x = padding + (drawable_w - scaled_w) / 2.0
    offset_y = padding + (drawable_h - scaled_h) / 2.0

    # Chuyển đổi các điểm sang tọa độ không gian 64x64
    normalized_strokes = []
    for stroke in strokes:
        norm_stroke = []
        for (x, y) in stroke:
            nx = int(round((x - min_x) * scale + offset_x))
            ny = int(round((y - min_y) * scale + offset_y))
            if not norm_stroke or norm_stroke[-1] != (nx, ny):
                norm_stroke.append((nx, ny))
        if norm_stroke:
            normalized_strokes.append(norm_stroke)

    # 3. Vẽ lên canvas 64x64 với cv2.LINE_AA (khử răng cưa, nét dày vừa đủ 2px)
    img_64 = np.zeros(target_size, dtype=np.uint8)
    for stroke in normalized_strokes:
        if len(stroke) == 1:
            pt = (int(round(float(stroke[0][0]))), int(round(float(stroke[0][1]))))
            cv2.circle(img_64, pt, line_thickness, 255, -1)
        else:
            for i in range(1, len(stroke)):
                pt1 = (int(round(float(stroke[i - 1][0]))), int(round(float(stroke[i - 1][1]))))
                pt2 = (int(round(float(stroke[i][0]))), int(round(float(stroke[i][1]))))
                cv2.line(img_64, pt1, pt2, 255, line_thickness, cv2.LINE_AA)

    # 4. keep_main_group(): Loại bỏ các điểm rác / nét vẽ lạc ở xa (kernel 5x5 nhẹ)
    img_64 = keep_main_group(img_64, kernel_size=(5, 5))

    # 5. Dilation 2x2 nhẹ để nét chữ vừa vặn CNN mà không bít lỗ ruột chữ
    kernel_2x2 = np.ones((2, 2), np.uint8)
    img_64 = cv2.dilate(img_64, kernel_2x2, iterations=1)

    return img_64
