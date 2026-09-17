# main.py
import cv2
import config
from processing.preprocessing import preprocess_frame
from processing.segmentation import segment_hsv
from processing.morphology import apply_morphology, find_and_draw_contours
from tracking.hand_tracker import HandTracker
from trajectory.trajectory_manager import TrajectoryManager
from reconstruction.handwriting import HandwritingCanvas

def main():
    cap = cv2.VideoCapture(0)
    
    # Set độ phân giải nếu có thể, dùng default (640x480)
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

    print("-----------------------------------------")
    print("Air-Writing System Initialized!")
    print("Hướng dẫn:")
    print("- Chạm ngón trỏ và ngón cái (Pinch) để viết.")
    print("- Thả ra để dừng nét chữ (Tracking mode).")
    print("- Phím 'c': Xóa chữ (Clear).")
    print("- Phím 's': Lưu chữ thành file ảnh (Save).")
    print("- Phím 'q': Thoát chương trình (Quit).")
    print("-----------------------------------------")

    previous_state = "No Hand"

    while True:
        success, raw_frame = cap.read()
        if not success:
            print("Không thể đọc frame từ camera!")
            break
            
        # 1. Tiền xử lý
        frame, blurred_frame = preprocess_frame(raw_frame, config.BLUR_KERNEL_SIZE)
        
        # 2 & 3 & 4. Xử lý OpenCV DIP (Minh họa cho yêu cầu của project)
        # Bóc tách màu HSV (ở đây dùng skin color default), xử lý hình thái học và tìm contours
        hsv_mask = segment_hsv(blurred_frame, config.HSV_LOWER, config.HSV_UPPER)
        cleaned_mask = apply_morphology(hsv_mask, config.MORPH_KERNEL_SIZE)
        
        # Vẽ contour của mask lên một bản sao frame để kiểm tra
        dip_debug_frame = frame.copy()
        dip_debug_frame = find_and_draw_contours(cleaned_mask, dip_debug_frame)
        
        # 5. MediaPipe Hands Tracking & Trạng thái viết
        frame, state, fingertip_pos = tracker.process_frame(frame)
        
        # 6. Xử lý Logic Trajectory dựa trên trạng thái
        if state == "Writing":
            trajectory_mgr.add_point(fingertip_pos)
            # Vẽ điểm tròn màu đỏ biểu thị đang viết trực tiếp lên camera
            if fingertip_pos:
                cv2.circle(frame, fingertip_pos, 8, (0, 0, 255), cv2.FILLED)
        elif state == "Tracking":
            # Nếu vừa chuyển từ Writing -> Tracking thì chốt nét chữ
            if previous_state == "Writing":
                trajectory_mgr.finish_stroke()
            # Vẽ điểm tròn màu xanh biểu thị đang track nhưng không vẽ
            if fingertip_pos:
                cv2.circle(frame, fingertip_pos, 8, (255, 0, 0), cv2.FILLED)
        else: # No Hand
            if previous_state == "Writing":
                trajectory_mgr.finish_stroke()
                
        previous_state = state
        
        # Cập nhật thông tin lên màn hình chính
        cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Lấy Trajectory đã được làm mượt và cập nhật vào Canvas
        smoothed_strokes = trajectory_mgr.get_smoothed_strokes()
        canvas_mgr.update_canvas(smoothed_strokes)
        
        # Hiển thị các nét chữ hiện tại lên frame camera (chỉ preview)
        for stroke in smoothed_strokes:
            for i in range(1, len(stroke)):
                cv2.line(frame, stroke[i-1], stroke[i], (0, 255, 255), 2) # Nét màu vàng
                
        # 7. Hiển thị cửa sổ
        cv2.imshow("Air-Writing System (Main View)", frame)
        cv2.imshow("DIP Pipeline Debug (Mask & Contours)", dip_debug_frame)
        cv2.imshow("Handwriting Canvas (Result)", canvas_mgr.get_canvas())
        
        # Bắt phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            trajectory_mgr.clear()
            print("Đã xóa Canvas.")
        elif key == ord('s'):
            canvas_mgr.save_image()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
