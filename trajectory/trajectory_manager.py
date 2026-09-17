# trajectory/trajectory_manager.py
from .smoothing import apply_simple_moving_average

class TrajectoryManager:
    def __init__(self, smoothing_window=5):
        # raw_strokes là list của các strokes. Mỗi stroke là một list tọa độ.
        self.raw_strokes = []
        self.current_stroke = []
        self.smoothing_window = smoothing_window
        
    def add_point(self, point):
        """Thêm một điểm vào stroke hiện tại."""
        if point is not None:
            self.current_stroke.append(point)
            
    def finish_stroke(self):
        """Kết thúc một nét chữ (khi dừng viết) và lưu lại."""
        if len(self.current_stroke) > 0:
            self.raw_strokes.append(self.current_stroke)
            self.current_stroke = []
            
    def get_smoothed_strokes(self):
        """Trả về toàn bộ các stroke đã được làm mượt."""
        smoothed_strokes = []
        # Làm mượt các strokes đã xong
        for stroke in self.raw_strokes:
            smoothed = apply_simple_moving_average(stroke, self.smoothing_window)
            smoothed_strokes.append(smoothed)
            
        # Làm mượt stroke hiện tại đang vẽ (nếu có)
        if len(self.current_stroke) > 0:
            smoothed = apply_simple_moving_average(self.current_stroke, self.smoothing_window)
            smoothed_strokes.append(smoothed)
            
        return smoothed_strokes

    def clear(self):
        """Xóa toàn bộ các nét đã viết."""
        self.raw_strokes = []
        self.current_stroke = []
