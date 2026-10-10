# main.py
import cv2
import math
import time
import config
from processing.pipeline_processor import render_trajectory_to_64x64
from tracking.webcam import Webcam
from tracking.hand_tracker import HandTracker
from trajectory.trajectory_manager import TrajectoryManager
from reconstruction.handwriting import HandwritingCanvas
from reconstruction.ui_dashboard import DashboardUI
from recognition.hybrid_classifier import HybridDatasetClassifier
import numpy as np


# -----------------------------------------------------------------
#  Writing state machine:
#
#   IDLE --> WRITING --> PENDING --> RECOGNIZING --> IDLE
#                ^           | (resumed writing)
#                +-----------+
#
#  - IDLE       : Khong viet, cho gesture WRITING.
#  - WRITING    : Dang thu trajectory tu ngon tro.
#  - PENDING    : Ngung WRITING, cho PENDING_DURATION giay (timestamp).
#                 Neu nguoi dung bat dau viet lai -> quay ve WRITING.
#  - RECOGNIZING: Da cho du, goi CNN roi ve IDLE.
# -----------------------------------------------------------------
STATE_IDLE        = "IDLE"
STATE_WRITING     = "WRITING"
STATE_PENDING     = "PENDING"
STATE_RECOGNIZING = "RECOGNIZING"


def print_controls():
    """In huong dan thao tac gesture khi khoi dong chuong trinh."""
    print("=" * 52)
    print("  AIR-WRITING CONTROLS")
    print("=" * 52)
    print()
    print("  WRITING  : Index UP, other fingers CLOSED")
    print("             -> Write with index fingertip")
    print()
    print("  SPACE    : Index + Pinky UP, Middle + Ring CLOSED")
    print("             -> Hold ~0.5s to add SPACE")
    print()
    print("  DELETE   : All fingers CLOSED (fist)")
    print("             -> Delete last character")
    print()
    print("  MODE     : Index + Middle UP (V-sign)")
    print("             -> Switch LETTER / NUMBER mode")
    print()
    print("  IMPORTANT:")
    print("  * PINCH is NO LONGER used for writing.")
    print("  * Keep the index finger FREE while writing.")
    print("  * Gestures require several stable frames.")
    print("  * SPACE requires holding ~0.5s to avoid accidents.")
    print("  * CNN fires ~{:.1f}s after you stop writing.".format(config.PENDING_DURATION))
    print()
    print("  KEYBOARD SHORTCUTS:")
    print("  [d] Toggle CNN 64x64 debug window")
    print("  [c] Clear canvas")
    print("  [s] Save canvas to file")
    print("  [q] Quit")
    print("=" * 52)
    print()


def _count_trajectory_points(trajectory_mgr) -> int:
    """Count total trajectory points across all strokes (including current)."""
    total = sum(len(s) for s in trajectory_mgr.raw_strokes)
    total += len(trajectory_mgr.current_stroke)
    return total


def main():
    with Webcam(camera_index=0, width=config.CANVAS_WIDTH, height=config.CANVAS_HEIGHT) as webcam:
        tracker = HandTracker(
            min_detection_con=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_con=config.MIN_TRACKING_CONFIDENCE,
            pinch_threshold=config.PINCH_THRESHOLD,
            stabilization_frames=config.GESTURE_STABILIZATION_FRAMES,
        )

        trajectory_mgr = TrajectoryManager(
            smoothing_window=config.SMOOTHING_WINDOW_SIZE,
            max_jump_distance=config.MAX_JUMP_DISTANCE,
            min_stroke_points=config.MIN_STROKE_POINTS,
        )

        canvas_mgr = HandwritingCanvas(
            width=config.CANVAS_WIDTH,
            height=config.CANVAS_HEIGHT,
            bg_color=config.CANVAS_BG_COLOR,
            draw_color=config.DRAWING_COLOR,
            thickness=config.DRAWING_THICKNESS,
        )

        dashboard_ui = DashboardUI(width=1280, height=720)

        # CNN + Custom Dataset Hybrid Classifier (load once)
        classifier = None
        try:
            classifier = HybridDatasetClassifier(config.MODEL_PATH, class_labels=config.CLASS_LABELS)
        except FileNotFoundError as e:
            print(f"[Warning] {e}")
            print("[Notice] System running in tracking-only mode. "
                  "Load model to 'models/airwriting_cnn.keras' to enable CNN.")
        except Exception as e:
            print(f"[Error] Cannot init Hybrid Classifier: {e}")

        print_controls()
        print("[System] Air-Writing System Initialized with Hybrid Dataset Engine!")
        print()

        # Runtime state
        writing_state  = STATE_IDLE     # IDLE | WRITING | PENDING | RECOGNIZING
        pending_start  = None           # time.time() when PENDING begins
        prev_fingertip = None           # fingertip position from previous frame

        prev_gesture   = None           # last printed gesture (spam prevention)
        prev_log_state = None           # last printed writing state
        current_mode   = "LETTER"       # LETTER | NUMBER
        
        # Target character selection for custom dataset collection
        target_char_idx = 1             # Default 'B'
        target_char = "B"
        collect_mode = True
        last_rendered_img_64 = None

        # Per-gesture cooldowns (frame counters)
        mode_cooldown   = 0
        delete_cooldown = 0
        space_cooldown  = 0
        space_hold_ctr  = 0
        delete_hold_ctr = 0
        pending_resume_ctr = 0

        show_cnn_debug = config.SHOW_RECOGNITION_DEBUG

        last_recognized_char = None
        last_confidence = 0.0
        recognized_text = ""

        # Main loop
        while True:
            success, raw_frame = webcam.read_frame()
            if not success:
                print("Cannot read frame from camera!")
                break

            # 1. Flip frame for mirror view
            frame = cv2.flip(raw_frame, 1)

            # 2. MediaPipe tracking + gesture recognition
            frame, stable_gesture, mapped_fingertip = tracker.process_frame(frame)

            # Log gesture changes (no spam)
            if stable_gesture != prev_gesture:
                if stable_gesture not in ("IDLE", "NO_HAND"):
                    print(f"[GESTURE] {stable_gesture}")
                prev_gesture = stable_gesture

            # Decrement cooldowns every frame
            if mode_cooldown   > 0: mode_cooldown   -= 1
            if delete_cooldown > 0: delete_cooldown -= 1
            if space_cooldown  > 0: space_cooldown  -= 1
            if stable_gesture != "SPACE": space_hold_ctr = 0
            if stable_gesture != "DELETE": delete_hold_ctr = 0
            if stable_gesture != "WRITING": pending_resume_ctr = 0

            # GESTURE CONTROL LOGIC
            if stable_gesture == "MODE" and mode_cooldown == 0:
                current_mode = "NUMBER" if current_mode == "LETTER" else "LETTER"
                mode_cooldown = 30
                print(f"[MODE] {current_mode}")

            elif stable_gesture == "SPACE" and space_cooldown == 0:
                space_hold_ctr += 1
                if space_hold_ctr >= config.SPACE_HOLD_FRAMES:
                    if writing_state == STATE_WRITING:
                        trajectory_mgr.finish_stroke()
                    writing_state = STATE_IDLE
                    pending_start = None
                    recognized_text += " "
                    space_cooldown = 30
                    space_hold_ctr = 0
                    print("[GESTURE] SPACE")

            elif stable_gesture == "DELETE" and delete_cooldown == 0:
                delete_hold_ctr += 1
                if delete_hold_ctr >= config.DELETE_HOLD_FRAMES:
                    if writing_state == STATE_WRITING:
                        trajectory_mgr.finish_stroke()
                    writing_state = STATE_IDLE
                    pending_start = None
                    if recognized_text:
                        recognized_text = recognized_text[:-1]
                    trajectory_mgr.clear()
                    last_recognized_char = None
                    last_confidence = 0.0
                    delete_cooldown = 20
                    delete_hold_ctr = 0
                    print("[GESTURE] DELETE")

            elif stable_gesture == "WRITING":
                if writing_state == STATE_IDLE:
                    writing_state = STATE_WRITING
                    pending_start = None

                elif writing_state == STATE_PENDING:
                    pending_resume_ctr += 1
                    if pending_resume_ctr >= config.GESTURE_STABILIZATION_FRAMES:
                        writing_state = STATE_WRITING
                        pending_start = None
                        pending_resume_ctr = 0

                if writing_state == STATE_WRITING:
                    if mapped_fingertip:
                        pt_i = (int(round(mapped_fingertip[0])), int(round(mapped_fingertip[1])))
                        cv2.circle(frame, pt_i, 6, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
                    trajectory_mgr.add_point(mapped_fingertip)

            else:
                if writing_state == STATE_WRITING:
                    trajectory_mgr.finish_stroke()
                    writing_state = STATE_PENDING
                    pending_start = time.time()
                    print("[WRITING] Stopped -> PENDING ({:.1f}s)".format(config.PENDING_DURATION))

                if mapped_fingertip:
                    pt_i = (int(round(mapped_fingertip[0])), int(round(mapped_fingertip[1])))
                    cv2.circle(frame, pt_i, 6, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)

            if writing_state == STATE_PENDING and pending_start is not None:
                elapsed = time.time() - pending_start
                if elapsed >= config.PENDING_DURATION:
                    writing_state = STATE_RECOGNIZING

            # RECOGNIZING: call Hybrid Classifier once, then reset to IDLE
            if writing_state == STATE_RECOGNIZING:
                total_pts = _count_trajectory_points(trajectory_mgr)
                traj_stats = trajectory_mgr.get_trajectory_stats()
                print(f"[DEBUG] RECOGNIZING: total_pts={total_pts}, classifier={'OK' if classifier else 'None'}")

                if classifier is not None and total_pts >= config.MIN_TRAJECTORY_POINTS:
                    smoothed_strokes = trajectory_mgr.get_smoothed_strokes()
                    canvas_mgr.update_canvas(smoothed_strokes)

                    img_64 = render_trajectory_to_64x64(
                        smoothed_strokes,
                        target_size=config.CNN_INPUT_SIZE,
                        padding=6,
                        line_thickness=2
                    )
                    non_zero = cv2.countNonZero(img_64) if img_64 is not None else 0
                    last_rendered_img_64 = img_64.copy() if img_64 is not None else None
                    density = round((non_zero / (64 * 64)) * 100.0, 2)

                    char, conf, info = classifier.predict_hybrid(
                        img_64,
                        mode=current_mode,
                        confidence_threshold=config.CONFIDENCE_THRESHOLD,
                        traj_stats=traj_stats
                    )

                    print("-" * 55)
                    print(" [TRAJECTORY DEBUG LOG]")
                    print(f"  Total points       : {traj_stats['total_pts']} (Strokes: {traj_stats['num_strokes']})")
                    print(f"  Bounding Box       : {traj_stats['bbox_w']}x{traj_stats['bbox_h']} (Aspect Ratio: {traj_stats['aspect_ratio']})")
                    print(f"  Step Distances     : Median={traj_stats['median_step_dist']}px, Max={traj_stats['max_step_dist']}px (Ratio: {traj_stats['ratio_max_to_median']}x)")
                    print(f"  Canvas Occupancy   : {non_zero} px ({density}%)")
                    print(f"  Top-1 Class & Conf : '{info['top1_char']}' ({info['top1_conf']}%)")
                    print(f"  Top-2 Class & Conf : '{info['top2_char']}' ({info['top2_conf']}%)")
                    print(f"  Top Margin (T1-T2) : {info['margin']}%")
                    if info.get('hybrid_boosted'):
                        print(f"  Hybrid Custom Match: YES -> '{info.get('custom_best_label')}' ({info.get('custom_best_conf')}%)")
                    print(f"  Validation Status  : {'REJECT / UNKNOWN' if info['is_unknown'] else 'ACCEPTED'}")
                    if info['reject_reason']:
                        print(f"  Reject Reason      : {info['reject_reason']}")
                    print("-" * 55)

                    if char is not None and char != "UNKNOWN":
                        last_recognized_char = char
                        last_confidence = conf
                        recognized_text += char
                        print(f"[WRITING] END -> HYBRID RECOGNITION: '{char}' ({conf:.2f}%)")
                    elif char == "UNKNOWN":
                        last_recognized_char = "UNKNOWN"
                        last_confidence = conf
                        print(f"[WRITING] END -> UNKNOWN (Trajectory anomaly or low margin)")
                    else:
                        print(f"[WRITING] END -> CNN: unrecognizable or confidence below {config.CONFIDENCE_THRESHOLD}%")
                elif classifier is None:
                    print(f"[WRITING] END -> Classifier not loaded! Cannot recognize.")
                elif total_pts < config.MIN_TRAJECTORY_POINTS:
                    print(f"[WRITING] END -> Skipped CNN ({total_pts} pts < min {config.MIN_TRAJECTORY_POINTS})")

                trajectory_mgr.clear()
                writing_state = STATE_IDLE
                pending_start = None

            prev_fingertip = mapped_fingertip

            if writing_state != prev_log_state:
                if writing_state == STATE_WRITING:
                    print("[STATE] -> WRITING")
                prev_log_state = writing_state

            smoothed_strokes = trajectory_mgr.get_smoothed_strokes()
            canvas_mgr.update_canvas(smoothed_strokes)
            current_canvas = canvas_mgr.get_canvas()

            for stroke in smoothed_strokes:
                for i in range(1, len(stroke)):
                    pt1 = (int(round(float(stroke[i - 1][0]))), int(round(float(stroke[i - 1][1]))))
                    pt2 = (int(round(float(stroke[i][0]))), int(round(float(stroke[i][1]))))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 3, cv2.LINE_AA)

            custom_count = len(classifier.custom_samples.get(target_char, [])) if classifier else 0

            master_dashboard = dashboard_ui.create_dashboard(
                camera_frame=frame,
                canvas_img=current_canvas,
                stable_gesture=stable_gesture,
                writing_state=writing_state,
                current_mode=current_mode,
                recognized_text=recognized_text,
                last_recognized_char=last_recognized_char,
                last_confidence=last_confidence,
                pending_start=pending_start,
                pending_duration=config.PENDING_DURATION,
                show_cnn_debug=show_cnn_debug,
                collect_mode=collect_mode,
                target_char=target_char,
                custom_samples_count=custom_count
            )

            cv2.imshow("Air-Writing System (Dual-Panel Dashboard)", master_dashboard)

            if show_cnn_debug and classifier is not None:
                smoothed_strokes = trajectory_mgr.get_smoothed_strokes()
                debug_64x64 = render_trajectory_to_64x64(
                    smoothed_strokes,
                    target_size=config.CNN_INPUT_SIZE,
                    padding=6,
                    line_thickness=2
                )
                cv2.imshow("CNN Input Debug (64x64)",
                           cv2.resize(debug_64x64, (200, 200), interpolation=cv2.INTER_NEAREST))

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                target_char_idx = (target_char_idx + 1) % 26
                target_char = chr(ord('A') + target_char_idx)
                print(f"[Target] Selected Target Character for Custom Dataset: '{target_char}'")
            elif key == ord('r'):
                smoothed_strokes = trajectory_mgr.get_smoothed_strokes()
                img_to_save = None
                if smoothed_strokes:
                    img_to_save = render_trajectory_to_64x64(
                        smoothed_strokes,
                        target_size=config.CNN_INPUT_SIZE,
                        padding=6,
                        line_thickness=2
                    )
                elif last_rendered_img_64 is not None:
                    img_to_save = last_rendered_img_64

                if classifier is not None and img_to_save is not None:
                    saved_path = classifier.save_custom_sample(target_char, img_to_save)
                    print(f"[Custom Dataset] SAVED custom sample for '{target_char}' -> '{saved_path}'")
                else:
                    print(f"[Custom Dataset] Cannot save: No trajectory drawn on canvas!")
            elif key == ord('d'):
                show_cnn_debug = not show_cnn_debug
                if not show_cnn_debug:
                    cv2.destroyWindow("CNN Input Debug (64x64)")
                print(f"[Debug] CNN debug window: {'ON' if show_cnn_debug else 'OFF'}")
            elif key == ord('c'):
                trajectory_mgr.clear()
                last_recognized_char = None
                last_confidence = 0.0
                recognized_text = ""
                writing_state = STATE_IDLE
                pending_start = None
                space_hold_ctr = 0
                print("[System] Canvas cleared.")
            elif key == ord('s'):
                canvas_mgr.save_image()


if __name__ == "__main__":
    main()
