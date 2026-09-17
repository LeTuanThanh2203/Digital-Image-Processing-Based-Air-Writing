# processing/morphology.py
import cv2
import numpy as np

def apply_morphology(mask, kernel_size):
    """
    Áp dụng các phép toán hình thái học (Morphology) để làm sạch mask.
    Sử dụng Erosion, Dilation, Opening, Closing.
    """
    kernel = np.ones(kernel_size, np.uint8)
    
    # Opening (Erosion sau đó Dilation) để xóa các điểm nhiễu nhỏ ngoài nền
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Closing (Dilation sau đó Erosion) để lấp đầy các lỗ hổng nhỏ trong đối tượng
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask

def find_and_draw_contours(mask, frame_to_draw):
    """
    Tìm và vẽ các đường viền (contours) lên một frame để hiển thị.
    Đồng thời trả về bounding box (x, y, w, h) của contour lớn nhất (được xem là vùng tay).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Chỉ giữ lại các contour lớn để tránh nhiễu
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    
    best_bbox = None
    if filtered_contours:
        # Tìm contour có diện tích lớn nhất (vùng bàn tay)
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        best_bbox = cv2.boundingRect(largest_contour)
        
        # Vẽ các contour màu xanh lá
        cv2.drawContours(frame_to_draw, filtered_contours, -1, (0, 255, 0), 2)
        
        # Vẽ Bounding Box lớn nhất để theo dõi (màu vàng)
        x, y, w, h = best_bbox
        cv2.rectangle(frame_to_draw, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
    return frame_to_draw, best_bbox
