# main.py
import sys
import os
import ctypes

# Enable Windows High-DPI Awareness (Per-Monitor v2) to prevent Windows DWM from blurring the window
if sys.platform.startswith("win"):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

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
from recognition.trie_search import load_dictionary, get_current_word_prefix, apply_word_completion
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
    """In huong dan thao tac gesture va ban phim khi khoi dong chuong trinh."""
    print("=" * 60)
    print("  AIR-WRITING CONTROLS & DUAL DATASET (LETTERS & NUMBERS)")
    print("=" * 60)
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
    print("  MODE     : Index + Middle UP (V-sign) OR press [m]")
    print("             -> Switch LETTER / NUMBER mode")
    print()
    print("  IMPORTANT:")
    print("  * PINCH is NO LONGER used for writing.")
    print("  * Keep the index finger FREE while writing.")
    print("  * Gestures require several stable frames.")
    print("  * SPACE requires holding ~0.5s to avoid accidents.")
    print("  * CNN fires ~{:.1f}s after you stop writing.".format(config.PENDING_DURATION))
    print()
    print("  DATASET TRAINING SHORTCUTS (CHU CAI & CHU SO):")
    print("  * [t]     : Chuyen doi muc tieu thu thap (A-Z khi LETTER, 0-9 khi NUMBER)")
    print("  * [0-9]   : Chon nhanh chu so 0-9 va tu dong chuyen sang NUMBER mode")
    print("  * [m]     : Chuyen doi che do LETTER <-> NUMBER")
    print("  * [r]     : Luu net ve lam mau Custom Dataset cho muc tieu dang chon")
    print()
    print("  WORD AUTOCOMPLETE / TRIE (GOI Y TU TIENG ANH):")
    print("  * Khi viet chu cai, top 5 tu tieng Anh pho bien nhat se hien thi.")
    print("  * CU CHI TAY: Chi dau ngon tay vao o tu 0.8s (hoac chum ngon Air-Pinch) de CHON!")
    print("  * Ngoai ra co the bam [1] - [5] (hoac Tab) hoac CLICK CHUOT vao nut de chon tu.")
    print("  * Chon xong he thong TU DONG THEM DAU CACH (Space) de viet tiep!")
    print()
    print("  SYSTEM SHORTCUTS:")
    print("  * [d]     : Bat/tat cua so CNN 64x64 debug")
    print("  * [c]     : Xoa canvas")
    print("  * [s]     : Luu canvas thanh file anh")
    print("  * [q]     : Thoat chuong trinh")
    print("=" * 60)
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

        # Trie Dictionary for English Word Suggestions (500,000 words)
        trie = None
        if getattr(config, 'ENABLE_WORD_SUGGESTIONS', True):
            try:
                dict_path = getattr(config, 'DICTIONARY_PATH', 'frequency_dictionary_en_500_000.txt')
                max_words = getattr(config, 'MAX_DICTIONARY_WORDS', 200000)
                trie = load_dictionary(str(dict_path), max_words=max_words)
            except Exception as e:
                print(f"[Warning] Cannot load Trie word suggestions: {e}")

        # Mouse click listener on window for selecting word suggestions
        clicked_word_holder = [None]
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                word = dashboard_ui.handle_click(x, y)
                if word:
                    clicked_word_holder[0] = word

        window_name = "Air-Writing System (Dual-Panel Dashboard)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        cv2.setMouseCallback(window_name, on_mouse_click)

        # CNN + Custom Dataset Hybrid Classifier (load once)
        classifier = None
        try:
            classifier = HybridDatasetClassifier(
                letters_model_path=config.LETTERS_MODEL_PATH,
                digits_model_path=config.DIGITS_MODEL_PATH,
                base_model_path=config.MODEL_PATH if config.MODEL_PATH.exists() else None,
                class_labels=config.CLASS_LABELS
            )
        except FileNotFoundError as e:
            print(f"[Warning] {e}")
            print("[Notice] System running in tracking-only mode. "
                  "Place 'airwriting_letters_cnn.keras' and 'airwriting_digits_cnn.keras' in 'models/' to enable CNN.")
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
        
        # Target character/number selection for custom dataset collection (Letters & Digits)
        LETTER_TARGETS = [chr(ord('A') + i) for i in range(26)]
        NUMBER_TARGETS = [str(i) for i in range(10)]
        target_letter_idx = 0           # Default 'A'
        target_number_idx = 0           # Default '0'
        target_char = "A"               # Ký tự/chữ số đang chọn để train mẫu
        collect_mode = True
        last_rendered_img_64 = None

        # Per-gesture cooldowns (frame counters)
        mode_cooldown    = 0
        delete_cooldown  = 0
        space_cooldown   = 0
        cycle_cooldown   = 0
        confirm_cooldown = 0
        space_hold_ctr   = 0
        delete_hold_ctr  = 0
        cycle_hold_ctr   = 0
        confirm_hold_ctr = 0
        active_suggestion_idx = 0
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

            # Word suggestions calculation via Trie
            current_suggestions = []
            active_word_prefix = ""
            if trie is not None and current_mode == "LETTER":
                active_word_prefix = get_current_word_prefix(recognized_text)
                if active_word_prefix:
                    current_suggestions = trie.search_prefix(
                        active_word_prefix,
                        limit=getattr(config, 'MAX_WORD_SUGGESTIONS', 5)
                    )
            if current_suggestions:
                active_suggestion_idx = active_suggestion_idx % len(current_suggestions)
            else:
                active_suggestion_idx = 0

            # Gesture activation constraints:
            # 1. When recognized_text is empty, only WRITING and MODE are allowed (disable SPACE, DELETE)
            if not recognized_text and stable_gesture in ("SPACE", "DELETE"):
                stable_gesture = "IDLE"

            # 2. CYCLE and CONFIRM only activate when word suggestions are actually available!
            # After choosing a word, after a space, in NUMBER mode, or when no suggestions exist,
            # CYCLE and CONFIRM are strictly disabled (treated as IDLE).
            if not current_suggestions and stable_gesture in ("CYCLE", "CONFIRM"):
                stable_gesture = "IDLE"

            # Log gesture changes (no spam)
            if stable_gesture != prev_gesture:
                if stable_gesture not in ("IDLE", "NO_HAND"):
                    print(f"[GESTURE] {stable_gesture}")
                prev_gesture = stable_gesture

            # Decrement cooldowns every frame
            if mode_cooldown    > 0: mode_cooldown    -= 1
            if delete_cooldown  > 0: delete_cooldown  -= 1
            if space_cooldown   > 0: space_cooldown   -= 1
            if cycle_cooldown   > 0: cycle_cooldown   -= 1
            if confirm_cooldown > 0: confirm_cooldown -= 1
            if stable_gesture != "SPACE": space_hold_ctr = 0
            if stable_gesture != "DELETE": delete_hold_ctr = 0
            if stable_gesture != "CYCLE": cycle_hold_ctr = 0
            if stable_gesture != "CONFIRM": confirm_hold_ctr = 0
            if stable_gesture != "WRITING": pending_resume_ctr = 0

            # GESTURE CONTROL LOGIC
            gesture_chosen_word = None

            if stable_gesture == "MODE" and mode_cooldown == 0:
                current_mode = "NUMBER" if current_mode == "LETTER" else "LETTER"
                mode_cooldown = 30
                if current_mode == "NUMBER":
                    target_char = NUMBER_TARGETS[target_number_idx]
                else:
                    target_char = LETTER_TARGETS[target_letter_idx]
                custom_count = len(classifier.custom_samples.get(target_char, [])) if classifier else 0
                print(f"[MODE] {current_mode} | Target: '{target_char}' ({custom_count} samples)")

            # CỬ CHỈ CYCLE (Chữ L: Ngón cái + ngón trỏ duỗi vuông góc, 3 ngón còn lại co)
            # Chuyển đổi qua lại giữa các lựa chọn 1 - 5 khi giữ khoảng 10 frames
            elif stable_gesture == "CYCLE" and cycle_cooldown == 0:
                cycle_hold_ctr += 1
                if cycle_hold_ctr >= 10:  # Giữ cử chỉ ~10 frames theo đúng yêu cầu
                    if current_suggestions and len(current_suggestions) > 1 and current_mode == "LETTER":
                        active_suggestion_idx = (active_suggestion_idx + 1) % len(current_suggestions)
                        cycle_cooldown = 12  # Ngăn nhảy quá nhanh nếu tiếp tục giữ tay
                        cycle_hold_ctr = 0
                        print(f"[GESTURE] CYCLE (Chữ L) -> Đang trỏ vào từ [{active_suggestion_idx + 1}]: '{current_suggestions[active_suggestion_idx].upper()}'")

            # CỬ CHỈ CONFIRM (3 ngón: Trỏ + Giữa + Áp út duỗi thẳng, ngón út co)
            # Chốt chọn từ đang được highlight
            elif stable_gesture == "CONFIRM" and confirm_cooldown == 0:
                confirm_hold_ctr += 1
                if confirm_hold_ctr >= 8:  # Giữ ~8 frames (~0.25s) để chốt chọn
                    if current_suggestions and current_mode == "LETTER":
                        gesture_chosen_word = current_suggestions[active_suggestion_idx]
                        confirm_cooldown = 30
                        confirm_hold_ctr = 0
                        print(f"[GESTURE] CONFIRM (3 ngón) -> Đã chốt chọn từ: '{gesture_chosen_word.upper()}'")

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
                    print("  Engine Mode        : HYBRID (70% Custom Dataset + 30% Base CNN)")
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
                        print(f"[WRITING] END -> Unrecognizable or confidence below {config.CONFIDENCE_THRESHOLD}%")
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

            # Safely check if user closed the window ('X' button)
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            # Dynamic responsive window dimensions with safe exception handling
            cur_win_w, cur_win_h = 1600, 900
            try:
                win_rect = cv2.getWindowImageRect(window_name)
                if win_rect is not None and len(win_rect) >= 4 and win_rect[2] > 600 and win_rect[3] > 400:
                    cur_win_w, cur_win_h = int(win_rect[2]), int(win_rect[3])
            except Exception:
                pass

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
                custom_samples_count=custom_count,
                suggestions=current_suggestions,
                active_prefix=active_word_prefix,
                selected_idx=active_suggestion_idx,
                width=cur_win_w,
                height=cur_win_h
            )

            cv2.imshow(window_name, master_dashboard)

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

            # Keyboard & Mouse shortcuts
            key = cv2.waitKey(1) & 0xFF

            # 1. Word suggestion selection check (Gesture CONFIRM, Mouse click, or Keys 1-5 / Tab)
            chosen_word = gesture_chosen_word
            if chosen_word is None and clicked_word_holder[0] is not None:
                chosen_word = clicked_word_holder[0]
                clicked_word_holder[0] = None

            if chosen_word is None and current_suggestions and current_mode == "LETTER" and ord('1') <= key <= ord('5'):
                idx = key - ord('1')
                if idx < len(current_suggestions):
                    chosen_word = current_suggestions[idx]
            elif chosen_word is None and current_suggestions and current_mode == "LETTER" and key == 9:  # Tab
                chosen_word = current_suggestions[0]

            if chosen_word is not None:
                recognized_text = apply_word_completion(recognized_text, chosen_word)
                last_recognized_char = chosen_word.upper()
                last_confidence = 100.0
                trajectory_mgr.clear()
                canvas_mgr.update_canvas([])
                writing_state = STATE_IDLE
                pending_start = None
                active_suggestion_idx = 0
                cycle_hold_ctr = 0
                confirm_hold_ctr = 0
                cycle_cooldown = 20
                confirm_cooldown = 30
                selection_method = "CONFIRM Gesture (3 Fingers)" if gesture_chosen_word else "Keyboard/Mouse"
                print(f"[Word Suggestion - {selection_method}] Selected '{chosen_word.upper()}' -> New Text: '{recognized_text}'")

            elif key == ord('q'):
                break
            elif key == ord('m'):
                current_mode = "NUMBER" if current_mode == "LETTER" else "LETTER"
                if current_mode == "NUMBER":
                    target_char = NUMBER_TARGETS[target_number_idx]
                else:
                    target_char = LETTER_TARGETS[target_letter_idx]
                custom_count = len(classifier.custom_samples.get(target_char, [])) if classifier else 0
                print(f"[Mode] Switched to {current_mode} | Target: '{target_char}' ({custom_count} samples)")
            elif key == ord('t'):
                if current_mode == "NUMBER":
                    target_number_idx = (target_number_idx + 1) % len(NUMBER_TARGETS)
                    target_char = NUMBER_TARGETS[target_number_idx]
                else:
                    target_letter_idx = (target_letter_idx + 1) % len(LETTER_TARGETS)
                    target_char = LETTER_TARGETS[target_letter_idx]
                custom_count = len(classifier.custom_samples.get(target_char, [])) if classifier else 0
                print(f"[Target] Selected {'Digit' if target_char.isdigit() else 'Letter'} for Custom Dataset: '{target_char}' ({custom_count} samples)")
            elif ord('0') <= key <= ord('9'):
                digit = chr(key)
                current_mode = "NUMBER"
                target_number_idx = int(digit)
                target_char = digit
                custom_count = len(classifier.custom_samples.get(target_char, [])) if classifier else 0
                print(f"[Quick Select] Mode NUMBER | Target Digit: '{target_char}' ({custom_count} samples)")
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
                    custom_count = len(classifier.custom_samples.get(target_char, []))
                    item_type = "NUMBER" if target_char.isdigit() else "LETTER"
                    print(f"[Custom Dataset] SAVED {item_type} sample for '{target_char}' -> '{saved_path}' (Total: {custom_count} samples)")
                else:
                    print(f"[Custom Dataset] Cannot save: No trajectory drawn on canvas!")
            elif key == ord('d'):
                show_cnn_debug = not show_cnn_debug
                if not show_cnn_debug:
                    cv2.destroyWindow("CNN Input Debug (64x64)")
                print(f"[Debug] CNN debug window: {'ON' if show_cnn_debug else 'OFF'}")
            elif key == ord('c'):
                trajectory_mgr.clear()
                canvas_mgr.update_canvas([])
                last_recognized_char = None
                last_confidence = 0.0
                recognized_text = ""
                writing_state = STATE_IDLE
                pending_start = None
                active_suggestion_idx = 0
                space_hold_ctr = 0
                delete_hold_ctr = 0
                cycle_hold_ctr = 0
                confirm_hold_ctr = 0
                print("[System] All text and canvas cleared. Only WRITING and MODE are active.")
            elif key == ord('s'):
                canvas_mgr.save_image()


if __name__ == "__main__":
    main()
