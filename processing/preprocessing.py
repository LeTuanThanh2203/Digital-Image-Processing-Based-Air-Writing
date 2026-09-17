# processing/preprocessing.py
import cv2

def preprocess_frame(frame, blur_kernel_size):
    """
    Tiền xử lý frame: Lật ảnh và làm mờ giảm nhiễu.
    """
    # Lật ảnh theo trục ngang để như gương
    frame = cv2.flip(frame, 1)
    
    # Làm mờ bằng Gaussian Blur để giảm nhiễu tần số cao
    blurred_frame = cv2.GaussianBlur(frame, blur_kernel_size, 0)
    
    return frame, blurred_frame
