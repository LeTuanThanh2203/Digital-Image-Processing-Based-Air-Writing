# trajectory/trajectory_manager.py
import math
import numpy as np
from .smoothing import apply_simple_moving_average, filter_coordinate_jumps

class TrajectoryManager:
    def __init__(self, smoothing_window=5, max_jump_distance=50.0, min_stroke_points=3, max_consecutive_rejects=3):
        self.raw_strokes = []
        self.current_stroke = []
        self._cached_smoothed_raw_strokes = []  # Cache cho các nét đã hoàn thành
        self.smoothing_window = smoothing_window
        self.max_jump_distance = max_jump_distance
        self.min_stroke_points = min_stroke_points
        self.max_consecutive_rejects = max_consecutive_rejects
        
        # Thống kê Debug theo dõi hiệu năng và mất nét
        self.points_received = 0
        self.points_rejected = 0
        self.large_jumps = 0
        self.reanchor_count = 0
        self.consecutive_rejects = 0

    def add_point(self, point):
        """Thêm một điểm vào stroke hiện tại (kèm lọc nhảy vị trí & cơ chế RE-ANCHOR khi viết nhanh)."""
        if point is not None:
            self.points_received += 1
            if self.current_stroke:
                last_pt = self.current_stroke[-1]
                dist = math.hypot(point[0] - last_pt[0], point[1] - last_pt[1])
                if dist > self.max_jump_distance:
                    self.large_jumps += 1
                    self.consecutive_rejects += 1
                    if self.consecutive_rejects < self.max_consecutive_rejects:
                        # Nhiễu nhảy ngẫu nhiên (Impulse Noise Spike): Bỏ qua
                        self.points_rejected += 1
                        return
                    else:
                        # Người dùng viết nhanh liên tục: RE-ANCHOR tiếp tục tracking, không bị freeze!
                        self.reanchor_count += 1
                        self.consecutive_rejects = 0
                else:
                    self.consecutive_rejects = 0
            
            self.current_stroke.append(point)

    def finish_stroke(self):
        """Kết thúc một nét chữ (khi dừng viết) và lưu lại nếu không quá ngắn."""
        if len(self.current_stroke) >= self.min_stroke_points:
            self.raw_strokes.append(self.current_stroke)
            # Tối ưu: Mượt nét vừa xong 1 lần và lưu vào cache (không lọc lại lần 2 để tránh mất nét)
            smoothed = apply_simple_moving_average(self.current_stroke, self.smoothing_window)
            self._cached_smoothed_raw_strokes.append(smoothed)
        self.current_stroke = []
        self.consecutive_rejects = 0

    def get_smoothed_strokes(self):
        """Trả về toàn bộ các stroke (sử dụng cache nét cũ + mượt nét hiện tại)."""
        smoothed_strokes = list(self._cached_smoothed_raw_strokes)
        if len(self.current_stroke) >= self.min_stroke_points:
            smoothed = apply_simple_moving_average(self.current_stroke, self.smoothing_window)
            smoothed_strokes.append(smoothed)

        return smoothed_strokes

    def get_trajectory_stats(self):
        """
        Tính toán các chỉ số thống kê đặc trưng của toàn bộ Trajectory.
        Phục vụ cho quy trình Holistic Multi-Signal Validation & Console Debug Logging.
        """
        smoothed_strokes = self.get_smoothed_strokes()
        all_points = [pt for stroke in smoothed_strokes for pt in stroke]
        total_pts = len(all_points)

        if total_pts < 2:
            return {
                'total_pts': total_pts,
                'median_step_dist': 0.0,
                'max_step_dist': 0.0,
                'ratio_max_to_median': 1.0,
                'bbox_w': 0.0,
                'bbox_h': 0.0,
                'aspect_ratio': 1.0,
                'num_strokes': len(smoothed_strokes)
            }

        step_dists = []
        for stroke in smoothed_strokes:
            for i in range(1, len(stroke)):
                d = math.hypot(stroke[i][0] - stroke[i - 1][0], stroke[i][1] - stroke[i - 1][1])
                step_dists.append(d)

        median_step = float(np.median(step_dists)) if step_dists else 0.0
        max_step = float(np.max(step_dists)) if step_dists else 0.0
        ratio_max_to_median = max_step / (median_step + 1e-5)

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        bbox_w = float(max(xs) - min(xs))
        bbox_h = float(max(ys) - min(ys))
        min_dim = max(1.0, min(bbox_w, bbox_h))
        max_dim = max(bbox_w, bbox_h)
        aspect_ratio = max_dim / min_dim

        return {
            'total_pts': total_pts,
            'median_step_dist': round(median_step, 2),
            'max_step_dist': round(max_step, 2),
            'ratio_max_to_median': round(ratio_max_to_median, 2),
            'bbox_w': round(bbox_w, 1),
            'bbox_h': round(bbox_h, 1),
            'aspect_ratio': round(aspect_ratio, 2),
            'num_strokes': len(smoothed_strokes)
        }

    def clear(self):
        """Xóa toàn bộ các nét đã viết và reset thống kê."""
        self.raw_strokes = []
        self.current_stroke = []
        self._cached_smoothed_raw_strokes = []
        self.consecutive_rejects = 0


