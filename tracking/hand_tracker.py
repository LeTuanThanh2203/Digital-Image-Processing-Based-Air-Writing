# tracking/hand_tracker.py
import cv2
import math
import os
from collections import Counter, deque
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


class GestureStabilizer:
    """
    Lớp ổn định cử chỉ bàn tay bằng majority voting trong cửa sổ N frames liên tiếp.
    Giúp chống nhiễu / jitter do MediaPipe nhận sai 1-2 frame lẻ.
    Khi không có đa số rõ ràng, giữ nguyên gesture ổn định trước đó (sticky)
    thay vì chuyển theo gesture thô mới nhất (tránh giật khi rút tay).
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.last_stable = "IDLE"   # Giữ gesture ổn định cuối cùng

    def update(self, raw_gesture: str) -> str:
        self.history.append(raw_gesture)
        if not self.history:
            return "IDLE"
        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]
        # Cần đa số trong cửa sổ để chốt gesture
        if count >= (len(self.history) // 2 + 1):
            self.last_stable = most_common
            return most_common
        # Không có đa số: giữ gesture ổn định trước đó thay vì jitter theo raw mới nhất
        return self.last_stable

    def reset(self):
        self.history.clear()
        self.last_stable = "IDLE"


def classify_hand_gesture(hand_landmarks) -> str:
    """
    Nhận diện các cử chỉ điều khiển Air-Writing từ 21 landmarks:
    - WRITING: Chỉ 1 ngón trỏ duỗi (ngón cái và 3 ngón còn lại co).
    - MODE: 2 ngón (V-sign: Ngón trỏ & giữa duỗi, áp út & út co).
    - SPACE: Ngón trỏ & ngón út duỗi (Rock/Spider-man).
    - DELETE: Nắm tay hoàn toàn (Tất cả ngón co).
    
    CỬ CHỈ ĐIỀU HƯỚNG & CHỌN TỪ GỢI Ý:
    - CYCLE: Giơ ngón cái + ngón trỏ duỗi vuông góc (Chữ L 👆👈), 3 ngón còn lại co -> Chuyển đổi qua lại giữa các lựa chọn 1 - 5 (~10 frames).
    - CONFIRM: Giơ 3 ngón (Trỏ + Giữa + Áp út duỗi thẳng 🤟, ngón út co) -> Chốt chọn từ đang được highlight!
    - IDLE: Xòe cả bàn tay 5 ngón hoặc các tư thế khác -> Trạng thái nghỉ tay an toàn.
    """
    # Trạng thái mở của 4 ngón dài (y của đầu ngón < y của đốt giữa)
    index_open = hand_landmarks[8].y < hand_landmarks[6].y
    middle_open = hand_landmarks[12].y < hand_landmarks[10].y
    ring_open = hand_landmarks[16].y < hand_landmarks[14].y
    pinky_open = hand_landmarks[20].y < hand_landmarks[18].y

    # Tính kích thước lòng bàn tay (Wrist 0 -> Middle MCP 9) làm thước đo tỉ lệ
    palm_size = math.hypot(hand_landmarks[0].x - hand_landmarks[9].x, hand_landmarks[0].y - hand_landmarks[9].y)
    # Khoảng cách từ đầu ngón cái (4) đến gốc ngón trỏ (5)
    dist_thumb_index = math.hypot(hand_landmarks[4].x - hand_landmarks[5].x, hand_landmarks[4].y - hand_landmarks[5].y)
    norm_thumb_dist = dist_thumb_index / (palm_size + 1e-6)

    # Chữ L (CYCLE): Ngón cái mở rộng ra xa lòng bàn tay (norm_thumb_dist > 0.60)
    # Viết chữ (WRITING): Ngón cái khép/áp vào cạnh bàn tay (norm_thumb_dist < 0.50)
    thumb_extended = norm_thumb_dist > 0.60

    four_closed = (not index_open) and (not middle_open) and (not ring_open) and (not pinky_open)
    four_open = index_open and middle_open and ring_open and pinky_open

    # 1. Cử chỉ CONFIRM (3 ngón: Trỏ + Giữa + Áp út duỗi thẳng, ngón út co)
    if index_open and middle_open and ring_open and not pinky_open:
        return "CONFIRM"

    # 2. Cử chỉ MODE (2 ngón V-sign: Trỏ + Giữa duỗi, ngón cái không mở rộng)
    elif index_open and middle_open and not ring_open and not pinky_open and not thumb_extended:
        return "MODE"

    # 3. Cử chỉ SPACE (Spider-man: Trỏ + Út duỗi, giữa + áp út co)
    elif index_open and pinky_open and not middle_open and not ring_open:
        return "SPACE"

    # 4. Cử chỉ CYCLE (Chữ L: Ngón cái mở rộng ngang hoặc lên + ngón trỏ duỗi, 3 ngón còn lại co)
    elif thumb_extended and index_open and not middle_open and not ring_open and not pinky_open:
        return "CYCLE"

    # 5. Cử chỉ WRITING (Viết chữ: Chỉ ngón trỏ duỗi, ngón cái khép vào tay)
    elif index_open and not middle_open and not ring_open and not pinky_open:
        return "WRITING"

    # 6. Cử chỉ DELETE (Nắm đấm hoàn toàn: cả 4 ngón co và ngón cái không mở rộng)
    elif four_closed and not thumb_extended:
        return "DELETE"

    else:
        # Xòe 5 ngón hoặc các tư thế khác: IDLE (nghỉ ngơi an toàn, không kích hoạt nhầm)
        return "IDLE"


class HandTracker:
    def __init__(self, min_detection_con=0.7, min_tracking_con=0.5, pinch_threshold=40, stabilization_frames=5):
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
        self.pinch_threshold = pinch_threshold # Giữ tham số tương thích cấu hình cũ
        self.stabilizer = GestureStabilizer(window_size=stabilization_frames)

    def process_frame(self, frame):
        """
        Xử lý frame bằng MediaPipe Tasks API, nhận diện gesture ổn định và trả về tọa độ ngón trỏ.
        Trả về: (frame, stable_gesture, fingertip_pos)
        stable_gesture: "WRITING", "SPACE", "DELETE", "MODE", "IDLE", "NO_HAND"
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = self.detector.detect(mp_image)
        
        raw_gesture = "NO_HAND"
        fingertip_pos = None
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = frame.shape
                
                # 1. Anti-aliased hand skeleton connections
                for connection in HAND_CONNECTIONS:
                    pt1 = hand_landmarks[connection[0]]
                    pt2 = hand_landmarks[connection[1]]
                    x1, y1 = int(pt1.x * w), int(pt1.y * h)
                    x2, y2 = int(pt2.x * w), int(pt2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (235, 235, 235), 2, cv2.LINE_AA)
                
                # 2. Anti-aliased landmark joints
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, (255, 200, 0), -1, cv2.LINE_AA)
                
                # 3. Index Fingertip #8 (Sub-pixel float precision)
                lm8 = hand_landmarks[8]
                px8, py8 = float(lm8.x * w), float(lm8.y * h)
                fingertip_pos = (px8, py8)

                # Glowing target marker on index fingertip (convert to int only for drawing)
                px8_i, py8_i = int(round(px8)), int(round(py8))
                cv2.circle(frame, (px8_i, py8_i), 10, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (px8_i, py8_i), 4, (0, 0, 255), -1, cv2.LINE_AA)

                # Phân loại gesture thô từ landmarks
                raw_gesture = classify_hand_gesture(hand_landmarks)

        # Ổn định gesture bằng Majority Vote qua các frames
        stable_gesture = self.stabilizer.update(raw_gesture)
                    
        return frame, stable_gesture, fingertip_pos


