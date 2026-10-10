# trajectory/smoothing.py
import math
import numpy as np

def filter_coordinate_jumps(points, max_jump_distance=50.0):
    """
    Lọc bỏ các điểm bị nhảy vị trí bất thường do nhiễu tracking camera.
    """
    if not points:
        return []

    filtered = [points[0]]
    for i in range(1, len(points)):
        pt1 = filtered[-1]
        pt2 = points[i]
        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist <= max_jump_distance:
            filtered.append(pt2)
    return filtered


def apply_simple_moving_average(points, window_size=5):
    """
    Áp dụng Simple Moving Average (SMA) làm mượt tọa độ 2D (giữ độ chính xác sub-pixel float).
    """
    if len(points) < window_size:
        return points

    smoothed_points = []
    for i in range(len(points)):
        if i < window_size - 1:
            smoothed_points.append(points[i])
        else:
            window = points[i - window_size + 1 : i + 1]
            avg_x = float(np.mean([p[0] for p in window]))
            avg_y = float(np.mean([p[1] for p in window]))
            smoothed_points.append((avg_x, avg_y))

    return smoothed_points


def to_int_points(points):
    """Chuyển đổi các điểm float sub-pixel sang tuple (int, int) khi vẽ bằng OpenCV."""
    return [(int(round(pt[0])), int(round(pt[1]))) for pt in points if pt is not None]

