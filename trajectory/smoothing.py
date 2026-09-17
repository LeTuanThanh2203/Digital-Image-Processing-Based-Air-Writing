# trajectory/smoothing.py
import numpy as np

def apply_simple_moving_average(points, window_size):
    """
    Áp dụng Simple Moving Average (SMA) để làm mượt danh sách các tọa độ 2D.
    """
    if len(points) < window_size:
        return points
        
    smoothed_points = []
    # Khởi tạo cửa sổ
    for i in range(len(points)):
        if i < window_size - 1:
            smoothed_points.append(points[i])
        else:
            window = points[i - window_size + 1 : i + 1]
            avg_x = int(np.mean([p[0] for p in window]))
            avg_y = int(np.mean([p[1] for p in window]))
            smoothed_points.append((avg_x, avg_y))
            
    return smoothed_points
