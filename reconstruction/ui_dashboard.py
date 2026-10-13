# reconstruction/ui_dashboard.py
"""
Dual-Panel Modern Dashboard UI Renderer for Air-Writing System.
Provides a sleek, professional dark slate dashboard layout:
- Left Panel: Live Camera Feed with Hand Skeleton & Trajectory Overlay
- Right Panel: System Status, Gesture Badges, Canvas Preview, CNN Result & Confidence Meter
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Optional


class DashboardUI:
    """Renders dual-panel dashboard UI (1280x720) combining camera stream and status panel."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        
        # Color Palette (BGR)
        self.BG_DARK = (24, 28, 36)          # Dark Slate background
        self.CARD_BG = (34, 40, 52)          # Dark Card background
        self.CARD_BORDER = (55, 65, 82)      # Card border color
        self.TEXT_PRIMARY = (245, 245, 245)  # Bright white text
        self.TEXT_SECONDARY = (160, 175, 200)# Muted grayish text
        self.ACCENT_CYAN = (255, 215, 0)     # Cyan/Yellow accent
        
        # State Colors
        self.COLOR_IDLE = (120, 120, 120)
        self.COLOR_WRITING = (76, 209, 55)    # Emerald Green
        self.COLOR_PENDING = (0, 165, 255)    # Orange
        self.COLOR_RECOGNIZING = (255, 120, 0)# Deep Cyan

        # Tối ưu: Pre-allocate mảng background template 1 lần
        self.dashboard_template = np.full((self.height, self.width, 3), self.BG_DARK, dtype=np.uint8)
        self._last_canvas_id = None
        self._cached_canvas_preview = None

        # Bounding boxes cho các nút gợi ý từ có thể click chuột [(x1, y1, x2, y2, word)]
        self.suggestion_rects: List[Tuple[int, int, int, int, str]] = []

    def handle_click(self, mouse_x: int, mouse_y: int) -> Optional[str]:
        """Kiểm tra toạ độ click chuột có trúng nút gợi ý từ nào không."""
        for (x1, y1, x2, y2, word) in self.suggestion_rects:
            if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                return word
        return None

    def create_dashboard(
        self,
        camera_frame: np.ndarray,
        canvas_img: np.ndarray,
        stable_gesture: str,
        writing_state: str,
        current_mode: str,
        recognized_text: str,
        last_recognized_char: Optional[str],
        last_confidence: float,
        pending_start: Optional[float],
        pending_duration: float = 1.0,
        show_cnn_debug: bool = False,
        collect_mode: bool = False,
        target_char: str = "A",
        custom_samples_count: int = 0,
        suggestions: Optional[List[str]] = None,
        active_prefix: str = "",
        cnn_enabled: bool = False
    ) -> np.ndarray:
        """Constructs and returns the full 1280x720 Dual-Panel Dashboard frame."""
        
        # 1. Background Canvas (Fast copy từ pre-allocated template)
        dashboard = self.dashboard_template.copy()

        # =========================================================================
        # LEFT PANEL: LIVE CAMERA STREAM (640x480 at x=20, y=110)
        # =========================================================================
        cam_x, cam_y, cam_w, cam_h = 20, 110, 640, 480
        
        # Draw Left Panel Card Background & Header
        self._draw_card(dashboard, 15, 15, 650, 690, title="LIVE CAMERA FEED", is_live=True)
        
        # Resize camera frame if necessary
        if camera_frame.shape[0] != cam_h or camera_frame.shape[1] != cam_w:
            cam_display = cv2.resize(camera_frame, (cam_w, cam_h))
        else:
            cam_display = camera_frame.copy()

        # Embed Camera Feed into Dashboard
        dashboard[cam_y : cam_y + cam_h, cam_x : cam_x + cam_w] = cam_display

        # =========================================================================
        # RIGHT PANEL: CONTROL DASHBOARD & METRICS (x=685 to 1265)
        # =========================================================================
        right_x = 680
        
        # 1. App Title Header Card
        self._draw_header_card(dashboard, right_x, 15, 585, 80, cnn_enabled=cnn_enabled)
        
        # 2. Status & Mode Grid
        self._draw_status_grid(
            dashboard, right_x, 105, 585, 95,
            writing_state, stable_gesture, current_mode,
            pending_start, pending_duration
        )

        # 3. Canvas Preview Card
        self._draw_canvas_card(dashboard, right_x, 210, 585, 275, canvas_img)

        # 4. Recognition Result & Confidence Card (Kèm gợi ý từ Trie)
        self._draw_result_card(
            dashboard, right_x, 495, 585, 130,
            recognized_text, last_recognized_char, last_confidence,
            suggestions=suggestions, active_prefix=active_prefix
        )

        # 5. Shortcuts & System Info Footer Card
        self._draw_footer_card(
            dashboard, right_x, 635, 585, 70, show_cnn_debug,
            collect_mode, target_char, custom_samples_count, cnn_enabled=cnn_enabled
        )

        return dashboard

    def _draw_card(self, img: np.ndarray, x: int, y: int, w: int, h: int, title: str = "", is_live: bool = False):
        """Draws a rounded card container with title."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)
        
        if title:
            cv2.putText(img, title, (x + 15, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_PRIMARY, 2, cv2.LINE_AA)
            if is_live:
                # Live pulsing red dot
                cv2.circle(img, (x + w - 30, y + 25), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(img, "LIVE", (x + w - 75, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_header_card(self, img: np.ndarray, x: int, y: int, w: int, h: int, cnn_enabled: bool = False):
        """Draws top title card."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)
        
        cv2.putText(img, "AIR-WRITING SYSTEM", (x + 20, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2, cv2.LINE_AA)
        sub_title = "DIP Trajectory & Hybrid 7:3 (CNN ON)" if cnn_enabled else "DIP Trajectory & 100% Custom Dataset (CNN OFF)"
        sub_color = (0, 255, 120) if not cnn_enabled else self.TEXT_SECONDARY
        cv2.putText(img, sub_title, (x + 20, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sub_color, 1, cv2.LINE_AA)

    def _draw_status_grid(
        self, img: np.ndarray, x: int, y: int, w: int, h: int,
        state: str, gesture: str, mode: str,
        pending_start: Optional[float], pending_duration: float
    ):
        """Draws Status, Mode, Gesture badges."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)

        # Columns
        col_w = w // 3

        # 1. State Column
        cv2.putText(img, "STATE", (x + 15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        state_color = self.COLOR_IDLE
        if state == "WRITING": state_color = self.COLOR_WRITING
        elif state == "PENDING": state_color = self.COLOR_PENDING
        elif state == "RECOGNIZING": state_color = self.COLOR_RECOGNIZING

        # State Badge Box
        cv2.rectangle(img, (x + 15, y + 35), (x + col_w - 10, y + 65), state_color, -1)
        cv2.putText(img, state, (x + 22, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # PENDING Progress Bar
        if state == "PENDING" and pending_start is not None:
            elapsed = time.time() - pending_start
            ratio = min(1.0, elapsed / pending_duration)
            bar_w = int(ratio * (col_w - 25))
            cv2.rectangle(img, (x + 15, y + 72), (x + 15 + bar_w, y + 80), (0, 165, 255), -1)
            cv2.rectangle(img, (x + 15, y + 72), (x + col_w - 10, y + 80), (100, 100, 100), 1)

        # 2. Mode Column
        cv2.putText(img, "MODE", (x + col_w + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        mode_bg = (180, 80, 0) if mode == "LETTER" else (0, 150, 180)
        cv2.rectangle(img, (x + col_w + 10, y + 35), (x + col_w * 2 - 10, y + 65), mode_bg, -1)
        cv2.putText(img, mode, (x + col_w + 20, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 3. Gesture Column
        cv2.putText(img, "GESTURE", (x + col_w * 2 + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        cv2.putText(img, gesture, (x + col_w * 2 + 5, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)

    def _draw_canvas_card(self, img: np.ndarray, x: int, y: int, w: int, h: int, canvas_img: np.ndarray):
        """Draws handwriting canvas preview card."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)
        
        cv2.putText(img, "HANDWRITING CANVAS PREVIEW", (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_PRIMARY, 2, cv2.LINE_AA)

        # Downscale canvas to 320x210 for preview
        cv_w, cv_h = 320, 210
        canvas_preview = cv2.resize(canvas_img, (cv_w, cv_h), interpolation=cv2.INTER_AREA)

        # Center canvas in card
        cx = x + (w - cv_w) // 2
        cy = y + 45
        img[cy : cy + cv_h, cx : cx + cv_w] = canvas_preview
        cv2.rectangle(img, (cx, cy), (cx + cv_w, cy + cv_h), (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_result_card(
        self, img: np.ndarray, x: int, y: int, w: int, h: int,
        text: str, last_char: Optional[str], confidence: float,
        suggestions: Optional[List[str]] = None,
        active_prefix: str = ""
    ):
        """Draws recognition result, confidence meter, and interactive Trie word suggestions."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)

        # 1. Recognized Sentence
        display_text = text if text else "(Write on camera...)"
        if len(display_text) > 22:
            display_text = "..." + display_text[-20:]

        cv2.putText(img, "RECOGNIZED TEXT:", (x + 15, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        cv2.putText(img, display_text, (x + 15, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Last Character & Confidence Bar
        if last_char is not None:
            char_str = f"Last: '{last_char}'"
            conf_str = f"{confidence:.1f}%"
            cv2.putText(img, char_str, (x + 360, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(img, conf_str, (x + 480, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)

            bar_x = x + 360
            bar_y = y + 33
            bar_max_w = 190
            bar_w = int((confidence / 100.0) * bar_max_w)
            
            bar_color = (0, 220, 100) if confidence >= 70.0 else (0, 200, 255)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), bar_color, -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 12), (100, 100, 100), 1)

        # Separator Line
        cv2.line(img, (x + 15, y + 62), (x + w - 15, y + 62), (48, 58, 76), 1)

        # 3. Word Suggestions (Trie Autocompletion)
        self.suggestion_rects.clear()
        if suggestions and len(suggestions) > 0:
            header_str = f"WORD SUGGESTIONS for '{active_prefix.upper()}' (Press [1-{len(suggestions)}] or Click):"
            cv2.putText(img, header_str, (x + 15, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)

            num_sug = len(suggestions)
            gap = 6
            available_w = w - 30
            btn_w = int((available_w - (max(1, num_sug - 1) * gap)) / max(1, num_sug))
            btn_y1 = y + 87
            btn_y2 = y + 119

            for i, word in enumerate(suggestions):
                btn_x1 = x + 15 + i * (btn_w + gap)
                btn_x2 = btn_x1 + btn_w
                self.suggestion_rects.append((btn_x1, btn_y1, btn_x2, btn_y2, word))

                # Button container
                cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), (45, 54, 72), -1)
                cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), (75, 95, 125), 1, cv2.LINE_AA)

                # Number badge & Word
                w_disp = word.upper()
                if len(w_disp) > 6:
                    w_disp = w_disp[:5] + "."
                lbl = f"[{i+1}] {w_disp}"
                cv2.putText(img, lbl, (btn_x1 + 6, btn_y1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            if active_prefix:
                cv2.putText(img, f"WORD SUGGESTIONS: No matching words for prefix '{active_prefix.upper()}'", (x + 15, y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "WORD SUGGESTIONS: (Write letters to see 500k dictionary autocompletions)", (x + 15, y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 135, 155), 1, cv2.LINE_AA)

    def _draw_footer_card(
        self, img: np.ndarray, x: int, y: int, w: int, h: int, show_debug: bool,
        collect_mode: bool = False, target_char: str = "A", custom_samples_count: int = 0,
        cnn_enabled: bool = False
    ):
        """Draws shortcut badges and footer information."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)

        cv2.putText(img, "KEYBOARD SHORTCUTS & DATASET TRAIN:", (x + 15, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)

        # Line 1 Badges
        target_type = "NUM" if target_char.isdigit() else "LET"
        cv2.putText(img, f"[t] Target: '{target_char}' ({target_type})", (x + 15, y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"[r] Save Sample", (x + 175, y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(img, f"Samples: {custom_samples_count}", (x + 325, y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 50), 1, cv2.LINE_AA)
        
        cnn_badge = "[n] CNN: ON" if cnn_enabled else "[n] CNN: OFF"
        cnn_color = (0, 255, 120) if cnn_enabled else (0, 180, 255)
        cv2.putText(img, cnn_badge, (x + 440, y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cnn_color, 1, cv2.LINE_AA)

        # Line 2 Badges
        cv2.putText(img, "[1-5] Word", (x + 15, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "[m] Mode", (x + 115, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        cv2.putText(img, "[c] Clear", (x + 195, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
        cv2.putText(img, "[s] Save", (x + 275, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
        dbg_text = "[d] Debug: ON" if show_debug else "[d] Debug: OFF"
        cv2.putText(img, dbg_text, (x + 355, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255) if show_debug else (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "[q] Quit", (x + 475, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
