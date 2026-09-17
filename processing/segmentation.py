# processing/segmentation.py
import cv2
import numpy as np

def segment_hsv(frame, lower_bound, upper_bound):
    """
    Chuyển đổi BGR sang HSV và tạo binary mask dựa trên ngưỡng.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_bound, dtype="uint8")
    upper = np.array(upper_bound, dtype="uint8")
    mask = cv2.inRange(hsv_frame, lower, upper)
    return mask
