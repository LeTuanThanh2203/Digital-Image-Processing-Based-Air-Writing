# main.py
import cv2
import config
from processing.preprocessing import preprocess_frame
from processing.segmentation import segment_hsv
from processing.morphology import apply_morphology, find_and_draw_contours
from tracking.hand_tracker import HandTracker
from trajectory.trajectory_manager import TrajectoryManager
from reconstruction.handwriting import HandwritingCanvas
from recognition.classifier import CharacterClassifier
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CANVAS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CANVAS_HEIGHT)

    tracker = HandTracker(
        min_detection_con=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_con=config.MIN_TRACKING_CONFIDENCE,
        pinch_threshold=config.PINCH_THRESHOLD
    )
    
    trajectory_mgr = TrajectoryManager(smoothing_window=config.SMOOTHING_WINDOW_SIZE)
    
    canvas_mgr = HandwritingCanvas(
        width=config.CANVAS_WIDTH,
        height=config.CANVAS_HEIGHT,
        bg_color=config.CANVAS_BG_COLOR,
        draw_color=config.DRAWING_COLOR,
        thickness=config.DRAWING_THICKNESS
    )

    # Khởi tạo CNN Character Classifier (Chỉ load 1 lần duy nhất)
    classifier = None
    try:
        classifier = CharacterClassifier(config.MODEL_PATH, class_labels=config.CLASS_LABELS)
    except FileNotFoundError as e:
        print(f"[Warning] {e}")
        print("[Notice] Hệ thống vẫn chạy ở chế độ Air-Writing Tracking. Vui lòng nạp file model vào 'models/airwriting_cnn.keras' để bật CNN.")
    except Exception as e:
        print(f"[Error] Không thể khởi tạo CNN Classifier: {e}")

    print("-----------------------------------------")
    print("Air-Writing System Initialized!")
    print("Hướng dẫn:")
    print("- Chạm ngón trỏ và ngón cái (Pinch) để viết.")
    print("- Thả ra để dừng nét chữ (Tracking mode) & Nhận diện CNN.")
    print("- Phím 'm': Bật/Tắt chế độ sử dụng ROI (Sequential Pipeline).")
    print("- Phím 'd': Bật/Tắt chế độ Debug cửa sổ CNN input 64x64.")
    print("- Phím 'c': Xóa chữ (Clear).")
    print("- Phím 's': Lưu chữ thành file ảnh (Save).")
    print("- Phím 'q': Thoát chương trình (Quit).")
    print("-----------------------------------------")

    previous_state = "No Hand"
    use_roi_mode = config.USE_ROI_MODE
    show_cnn_debug = config.SHOW_RECOGNITION_DEBUG
    
    last_recognized_char = None
    last_confidence = 0.0

    while True:
        success, raw_frame = cap.read()
        if not success:
            print("Không thể đọc frame từ camera!")
            break
            
        # 1. Tiền xử lý
        frame, blurred_frame = preprocess_frame(raw_frame, config.BLUR_KERNEL_SIZE)
        
        # 2 & 3 & 4. Xử lý OpenCV DIP để tìm Bounding Box của vùng màu da
        hsv_mask = segment_hsv(blurred_frame, config.HSV_LOWER, config.HSV_UPPER)
        cleaned_mask = apply_morphology(hsv_mask, config.MORPH_KERNEL_SIZE)
        
        dip_debug_frame = frame.copy()
        dip_debug_frame, best_bbox = find_and_draw_contours(cleaned_mask, dip_debug_frame)
        
        # Biến lưu trữ khung ảnh sẽ đưa vào MediaPipe
        processing_frame = frame.copy()
        roi_x, roi_y = 0, 0
        is_using_roi = False
        
        # 5. Xác định ROI và Crop (Nếu bật Mode và có Bounding Box hợp lệ)
        if use_roi_mode and best_bbox is not None:
            bx, by, bw, bh = best_bbox
            p = config.ROI_PADDING
            cx, cy = bx + bw // 2, by + bh // 2
            side = max(bw, bh) + 2 * p
            half_side = side // 2
            
            x1 = max(0, cx - half_side)
            y1 = max(0, cy - half_side)
            x2 = min(frame.shape[1], cx + half_side)
            y2 = min(frame.shape[0], cy + half_side)
            
            # Cắt ảnh
            processing_frame = frame[y1:y2, x1:x2].copy()
            roi_x, roi_y = x1, y1
            is_using_roi = True
            
            # Vẽ viền báo hiệu vùng Crop ROI trên debug frame (Màu Đỏ)
            cv2.rectangle(dip_debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(dip_debug_frame, "ROI Cropped", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 6. MediaPipe Hands Tracking
        # MediaPipe sẽ xử lý và vẽ skeleton trực tiếp lên processing_frame
        processing_frame, state, fingertip_pos = tracker.process_frame(processing_frame)
        
        # 7. Coordinate Mapping (Chuyển đổi hệ tọa độ nếu dùng ROI)
        if fingertip_pos is not None:
            # Map tọa độ điểm fingertip về khung hình gốc
            mapped_fingertip_pos = (fingertip_pos[0] + roi_x, fingertip_pos[1] + roi_y)
        else:
            mapped_fingertip_pos = None

        # Overlay processing_frame ngược lại vào frame gốc để hiển thị skeleton
        if is_using_roi:
            h, w = processing_frame.shape[:2]
            frame[roi_y:roi_y+h, roi_x:roi_x+w] = processing_frame
        else:
            frame = processing_frame

        # 8. Xử lý Logic Trajectory dựa trên trạng thái đã map tọa độ
        writing_just_finished = False
        if state == "Writing":
            trajectory_mgr.add_point(mapped_fingertip_pos)
            if mapped_fingertip_pos:
                cv2.circle(frame, mapped_fingertip_pos, 8, (0, 0, 255), cv2.FILLED)
        elif state == "Tracking":
            if previous_state == "Writing":
                trajectory_mgr.finish_stroke()
                writing_just_finished = True
            if mapped_fingertip_pos:
                cv2.circle(frame, mapped_fingertip_pos, 8, (255, 0, 0), cv2.FILLED)
        else:
            if previous_state == "Writing":
                trajectory_mgr.finish_stroke()
                writing_just_finished = True
                
        previous_state = state
        
        # Lấy Trajectory đã được làm mượt và cập nhật vào Canvas
        smoothed_strokes = trajectory_mgr.get_smoothed_strokes()
        canvas_mgr.update_canvas(smoothed_strokes)
        current_canvas = canvas_mgr.get_canvas()
        
        # Nhận diện CNN khi cử chỉ viết kết thúc (writing_just_finished)
        if writing_just_finished and classifier is not None:
            char, conf = classifier.predict(current_canvas)
            if char is not None:
                last_recognized_char = char
                last_confidence = conf
                print(f"[CNN Recognition] Nhận diện nét chữ: {char} ({conf:.2f}%)")
        
        # Hiển thị các nét chữ hiện tại lên frame camera (chỉ preview)
        for stroke in smoothed_strokes:
            for i in range(1, len(stroke)):
                cv2.line(frame, stroke[i-1], stroke[i], (0, 255, 255), 2)
                
        # Hiển thị thông tin lên UI
        mode_text = "Mode B: DIP ROI -> MediaPipe" if use_roi_mode else "Mode A: Full Frame -> MediaPipe"
        cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Hiển thị kết quả nhận diện từ CNN (nếu có)
        if last_recognized_char is not None:
            recogn_text = f"Recognized: {last_recognized_char} ({last_confidence:.1f}%)"
            cv2.putText(frame, recogn_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
        # 9. Hiển thị cửa sổ
        cv2.imshow("Air-Writing System (Main View)", frame)
        cv2.imshow("DIP Pipeline Debug (Mask & Contours)", dip_debug_frame)
        cv2.imshow("Handwriting Canvas (Result)", current_canvas)
        
        if show_cnn_debug and classifier is not None:
            _, debug_64x64 = classifier.preprocess(current_canvas)
            if debug_64x64 is not None:
                cv2.imshow("CNN Input Debug (64x64)", cv2.resize(debug_64x64, (200, 200), interpolation=cv2.INTER_NEAREST))
            else:
                blank_debug = np.zeros((200, 200), dtype=np.uint8)
                cv2.imshow("CNN Input Debug (64x64)", blank_debug)
        
        # Bắt phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            use_roi_mode = not use_roi_mode
            print(f"Đã chuyển sang {mode_text}")
        elif key == ord('d'):
            show_cnn_debug = not show_cnn_debug
            if not show_cnn_debug:
                cv2.destroyWindow("CNN Input Debug (64x64)")
            print(f"Chế độ debug CNN: {'Bật' if show_cnn_debug else 'Tắt'}")
        elif key == ord('c'):
            trajectory_mgr.clear()
            last_recognized_char = None
            last_confidence = 0.0
            print("Đã xóa Canvas.")
        elif key == ord('s'):
            canvas_mgr.save_image()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
