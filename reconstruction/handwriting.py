# reconstruction/handwriting.py
import cv2
import numpy as np
import os
import time

class HandwritingCanvas:
    def __init__(self, width, height, bg_color, draw_color, thickness):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.draw_color = draw_color
        self.thickness = thickness
        self.canvas = self._create_blank_canvas()
        
        # Đảm bảo thư mục output tồn tại
        if not os.path.exists("output"):
            os.makedirs("output")
            
    def _create_blank_canvas(self):
        """Tạo một canvas trắng mới."""
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color
        return canvas
        
    def update_canvas(self, smoothed_strokes):
        """Vẽ toàn bộ nét chữ lên canvas với đường nét mượt chống răng cưa."""
        self.canvas = self._create_blank_canvas()
        if not smoothed_strokes:
            return
        for stroke in smoothed_strokes:
            for i in range(1, len(stroke)):
                pt1 = (int(round(float(stroke[i - 1][0]))), int(round(float(stroke[i - 1][1]))))
                pt2 = (int(round(float(stroke[i][0]))), int(round(float(stroke[i][1]))))
                cv2.line(self.canvas, pt1, pt2, self.draw_color, self.thickness, cv2.LINE_AA)


                
    def get_canvas(self):
        return self.canvas
        
    def save_image(self):
        """Lưu canvas thành file hình ảnh."""
        timestamp = int(time.time())
        filename = f"output/handwriting_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        print(f"Đã lưu kết quả tại: {filename}")
