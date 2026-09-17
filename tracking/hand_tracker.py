# tracking/hand_tracker.py
import cv2
import math
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Định nghĩa các khớp nối để vẽ skeleton bàn tay
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class HandTracker:
    def __init__(self, min_detection_con=0.7, min_tracking_con=0.5, pinch_threshold=40):
        # Tự động tải file model nếu chưa có
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Đang tải model hand_landmarker.task...")
            import urllib.request
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                model_path
            )
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_con,
            min_hand_presence_confidence=min_tracking_con
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.pinch_threshold = pinch_threshold

    def process_frame(self, frame):
        """
        Xử lý frame bằng MediaPipe Tasks API và trả về trạng thái vẽ + tọa độ ngón trỏ.
        Trạng thái: "Writing", "Tracking", "No Hand"
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = self.detector.detect(mp_image)
        
        state = "No Hand"
        fingertip_pos = None
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = frame.shape
                
                # Vẽ các đường nối skeleton
                for connection in HAND_CONNECTIONS:
                    pt1 = hand_landmarks[connection[0]]
                    pt2 = hand_landmarks[connection[1]]
                    x1, y1 = int(pt1.x * w), int(pt1.y * h)
                    x2, y2 = int(pt2.x * w), int(pt2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
                # Vẽ các điểm landmark
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                # Điểm 8 là đầu ngón trỏ
                lm8 = hand_landmarks[8]
                px8, py8 = int(lm8.x * w), int(lm8.y * h)
                fingertip_pos = (px8, py8)
                
                # Điểm 4 là đầu ngón cái
                lm4 = hand_landmarks[4]
                px4, py4 = int(lm4.x * w), int(lm4.y * h)
                
                # Tính khoảng cách Euclidean
                distance = math.hypot(px8 - px4, py8 - py4)
                
                if distance < self.pinch_threshold:
                    state = "Writing"
                else:
                    state = "Tracking"
                    
        return frame, state, fingertip_pos
