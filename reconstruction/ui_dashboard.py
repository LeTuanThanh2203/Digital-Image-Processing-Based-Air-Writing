# reconstruction/ui_dashboard.py
"""
Dual-Panel Modern Dashboard UI Renderer for Air-Writing System.
Provides a sleek, professional dark slate dashboard layout:
- Header: Institutional Logo, Full Project Name, Student IDs & Names (Vietnamese Unicode)
- Left Panel: Live Camera Feed with Hand Skeleton & Trajectory Overlay (Responsive Scaling)
- Right Panel: System Status, Handwriting Canvas Preview, Recognition Result & Trie Word Autocomplete, Shortcuts
- Responsive: Dynamically adapts to window resize and maximize (Full HD / 2K / 4K)
"""

import cv2
import os
from pathlib import Path
import numpy as np
import time
from typing import Tuple, List, Optional

# PIL for high-fidelity Vietnamese text rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class DashboardUI:
    """Renders responsive dual-panel dashboard UI combining camera stream and status panel."""

    def __init__(self, width: int = 1280, height: int = 720, logo_path: Optional[str] = None):
        self.width = width
        self.height = height

        # Resolve logo path
        if logo_path is not None:
            self.logo_path = Path(logo_path)
        else:
            self.logo_path = Path(__file__).resolve().parent.parent / "assets" / "logo_ute.jpg"

        # Color Palette (BGR)
        self.BG_DARK = (24, 28, 36)          # Dark Slate background
        self.CARD_BG = (34, 40, 52)          # Dark Card background
        self.CARD_BORDER = (55, 65, 82)      # Card border color
        self.TEXT_PRIMARY = (245, 245, 245)  # Bright white text
        self.TEXT_SECONDARY = (160, 175, 200)# Muted grayish text
        self.ACCENT_CYAN = (255, 215, 0)     # Cyan/Yellow accent
        self.ACCENT_MINT = (180, 255, 0)     # Mint accent

        # State Colors
        self.COLOR_IDLE = (120, 120, 120)
        self.COLOR_WRITING = (76, 209, 55)    # Emerald Green
        self.COLOR_PENDING = (0, 165, 255)    # Orange
        self.COLOR_RECOGNIZING = (255, 120, 0)# Deep Cyan

        # Font paths for PIL Unicode rendering
        self.font_regular_path = "C:/Windows/Fonts/segoeui.ttf"
        self.font_bold_path = "C:/Windows/Fonts/segoeuib.ttf"
        if not os.path.exists(self.font_regular_path):
            self.font_regular_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(self.font_bold_path):
            self.font_bold_path = "C:/Windows/Fonts/arialbd.ttf"

        # Load Logo
        self.raw_logo = None
        if self.logo_path.exists():
            try:
                self.raw_logo = cv2.imread(str(self.logo_path))
            except Exception as e:
                print(f"[DashboardUI] Cannot read logo at {self.logo_path}: {e}")

        # Dynamic cache
        self.current_w = None
        self.current_h = None
        self.dashboard_template = None
        self.cached_header_w = None
        self.cached_header_h = None
        self.cached_header_img = None

        # Interactive suggestion button bounding boxes: [(x1, y1, x2, y2, word)]
        self.suggestion_rects: List[Tuple[int, int, int, int, str]] = []

    def handle_click(self, mouse_x: int, mouse_y: int) -> Optional[str]:
        """Check if mouse click falls within any suggestion button."""
        for (x1, y1, x2, y2, word) in self.suggestion_rects:
            if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                return word
        return None

    def _render_header(self, w: int, h: int) -> np.ndarray:
        """
        Renders full-width top header with institutional logo badge, project name,
        and student team members (IDs and Vietnamese full names).
        """
        header_card = np.full((h, w, 3), self.CARD_BG, dtype=np.uint8)
        cv2.rectangle(header_card, (0, 0), (w - 1, h - 1), self.CARD_BORDER, 1)

        # 1. Logo Badge (Top-left corner)
        badge_right = 15
        if self.raw_logo is not None:
            logo_h = h - 16
            logo_w = int(logo_h * self.raw_logo.shape[1] / max(1, self.raw_logo.shape[0]))
            logo_resized = cv2.resize(self.raw_logo, (logo_w, logo_h), interpolation=cv2.INTER_AREA)

            badge_pad_x = 4
            badge_pad_y = 2
            badge_w = logo_w + 2 * badge_pad_x
            badge_h = logo_h + 2 * badge_pad_y
            badge_x = 12
            badge_y = (h - badge_h) // 2

            # Clean white badge frame for institutional seal
            header_card[badge_y : badge_y + badge_h, badge_x : badge_x + badge_w] = (255, 255, 255)
            cv2.rectangle(
                header_card,
                (badge_x, badge_y),
                (badge_x + badge_w - 1, badge_y + badge_h - 1),
                (200, 210, 225),
                1
            )

            # Insert logo inside badge
            header_card[badge_y + badge_pad_y : badge_y + badge_pad_y + logo_h,
                        badge_x + badge_pad_x : badge_x + badge_pad_x + logo_w] = logo_resized

            badge_right = badge_x + badge_w

        # 2. Text Drawing using PIL for pristine Vietnamese Unicode accents
        if HAS_PIL and os.path.exists(self.font_bold_path):
            header_rgb = cv2.cvtColor(header_card, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(header_rgb)
            draw = ImageDraw.Draw(pil_img)

            # Responsive font sizes based on width
            scale_factor = min(1.35, max(0.95, w / 1280.0))
            title_font_size = int(15.5 * scale_factor)
            sub_font_size = int(12.0 * scale_factor)
            team_title_size = int(12.5 * scale_factor)
            team_mem_size = int(15.0 * scale_factor)  # To, rõ ràng và đậm nét

            font_title_bold = ImageFont.truetype(self.font_bold_path, title_font_size)
            font_title_sub  = ImageFont.truetype(self.font_regular_path, sub_font_size)
            font_team_title = ImageFont.truetype(self.font_bold_path, team_title_size)
            font_team_mem   = ImageFont.truetype(self.font_bold_path, team_mem_size)

            # Project Title (Left/Center next to logo)
            title_x = badge_right + 14
            draw.text(
                (title_x, int(16 * scale_factor)),
                "DIGITAL IMAGE PROCESSING-BASED AIR-WRITING",
                font=font_title_bold,
                fill=(0, 230, 255)
            )
            draw.text(
                (title_x, int(48 * scale_factor)),
                "Trajectory Extraction and Handwriting Reconstruction",
                font=font_title_sub,
                fill=(215, 225, 240)
            )

            # Project Team Members (Right side of header)
            team_box_w = int(420 * scale_factor)
            team_x = w - team_box_w - 15

            # Subtle vertical separator before team info
            draw.line([(team_x - 16, 10), (team_x - 16, h - 10)], fill=(55, 65, 82), width=1)

            draw.text((team_x, int(6 * scale_factor)), "NHÓM SINH VIÊN THỰC HIỆN:", font=font_team_title, fill=(0, 255, 180))

            members = [
                ("24110075", "Phạm Phước Anh"),
                ("24110127", "Ca Ngọc Tân"),
                ("24110128", "Lê Tuấn Thành"),
            ]
            y_base = int(27 * scale_factor)
            y_gap = int(21 * scale_factor)
            id_offset = int(105 * scale_factor)
            for i, (m_id, m_name) in enumerate(members):
                cur_y = y_base + i * y_gap
                draw.text((team_x, cur_y), f"• {m_id}", font=font_team_mem, fill=(0, 230, 255))
                draw.text((team_x + id_offset, cur_y), m_name, font=font_team_mem, fill=(255, 255, 255))

            header_card = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # Fallback to OpenCV putText (ASCII unaccented text)
            title_x = badge_right + 14
            cv2.putText(header_card, "DIGITAL IMAGE PROCESSING-BASED AIR-WRITING",
                        (title_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(header_card, "Trajectory Extraction and Handwriting Reconstruction",
                        (title_x, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 225, 240), 1, cv2.LINE_AA)

            team_x = w - 420
            cv2.line(header_card, (team_x - 16, 10), (team_x - 16, h - 10), (55, 65, 82), 1)
            cv2.putText(header_card, "NHOM SINH VIEN THUC HIEN:", (team_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 2, cv2.LINE_AA)
            cv2.putText(header_card, "- 24110075  Pham Phuoc Anh", (team_x, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(header_card, "- 24110127  Ca Ngoc Tan", (team_x, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(header_card, "- 24110128  Le Tuan Thanh", (team_x, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        return header_card

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
        selected_idx: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> np.ndarray:
        """
        Constructs and returns responsive Dual-Panel Dashboard frame.
        Supports dynamic window dimensions (e.g. when user resizes or maximizes window).
        """
        target_w = width if width is not None and width >= 800 else self.width
        target_h = height if height is not None and height >= 500 else self.height

        # Update cache template if dimensions changed
        if (target_w != self.current_w) or (target_h != self.current_h):
            self.current_w = target_w
            self.current_h = target_h
            self.dashboard_template = np.full((target_h, target_w, 3), self.BG_DARK, dtype=np.uint8)
            self.cached_header_w = None

        dashboard = self.dashboard_template.copy()

        # =========================================================================
        # 1. TOP HEADER BANNER (Full width across dashboard)
        # =========================================================================
        pad = 12
        header_x = pad
        header_y = pad
        header_w = target_w - 2 * pad
        header_h = 92

        if (self.cached_header_w != header_w) or (self.cached_header_h != header_h) or (self.cached_header_img is None):
            self.cached_header_img = self._render_header(header_w, header_h)
            self.cached_header_w = header_w
            self.cached_header_h = header_h

        dashboard[header_y : header_y + header_h, header_x : header_x + header_w] = self.cached_header_img

        # =========================================================================
        # 2. CONTENT AREA (Left: Camera, Right: Dashboard Panels)
        # =========================================================================
        content_y = header_y + header_h + pad
        content_h = target_h - content_y - pad

        # Layout Split: Left panel ~52%, Right panel ~48%
        left_w = int((target_w - 3 * pad) * 0.52)
        right_w = target_w - 3 * pad - left_w
        left_x = pad
        right_x = left_x + left_w + pad

        # --- LEFT PANEL: CAMERA CARD ---
        self._draw_card(dashboard, left_x, content_y, left_w, content_h, title="LIVE CAMERA FEED", is_live=True)

        # Scale camera keeping aspect ratio inside left card
        cam_pad_x = 10
        cam_pad_top = 38
        cam_pad_bottom = 10
        avail_cam_w = left_w - 2 * cam_pad_x
        avail_cam_h = content_h - cam_pad_top - cam_pad_bottom

        cf_h, cf_w = camera_frame.shape[:2]
        scale = min(avail_cam_w / float(cf_w), avail_cam_h / float(cf_h))
        disp_w = max(10, int(cf_w * scale))
        disp_h = max(10, int(cf_h * scale))

        disp_x = left_x + (left_w - disp_w) // 2
        disp_y = content_y + cam_pad_top + (avail_cam_h - disp_h) // 2

        cam_display = cv2.resize(camera_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        dashboard[disp_y : disp_y + disp_h, disp_x : disp_x + disp_w] = cam_display

        # --- RIGHT PANEL: 4 STACKED CARDS ---
        status_h = 76
        footer_h = 82
        result_h = 145
        gap = 8
        canvas_h = content_h - (status_h + result_h + footer_h + 3 * gap)
        if canvas_h < 130:
            canvas_h = 130

        c1_y = content_y
        c2_y = c1_y + status_h + gap
        c3_y = c2_y + canvas_h + gap
        c4_y = c3_y + result_h + gap

        # Render 4 Cards
        self._draw_status_grid(
            dashboard, right_x, c1_y, right_w, status_h,
            writing_state, stable_gesture, current_mode,
            pending_start, pending_duration
        )

        self._draw_canvas_card(dashboard, right_x, c2_y, right_w, canvas_h, canvas_img)

        self._draw_result_card(
            dashboard, right_x, c3_y, right_w, result_h,
            recognized_text, last_recognized_char, last_confidence,
            suggestions=suggestions, active_prefix=active_prefix,
            selected_idx=selected_idx
        )

        self._draw_footer_card(
            dashboard, right_x, c4_y, right_w, footer_h, show_cnn_debug,
            collect_mode, target_char, custom_samples_count
        )

        return dashboard

    def _draw_card(self, img: np.ndarray, x: int, y: int, w: int, h: int, title: str = "", is_live: bool = False):
        """Draws card container with header title."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)

        if title:
            cv2.putText(img, title, (x + 15, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_PRIMARY, 2, cv2.LINE_AA)
            if is_live:
                cv2.circle(img, (x + w - 25, y + 22), 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(img, "LIVE", (x + w - 65, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_status_grid(
        self, img: np.ndarray, x: int, y: int, w: int, h: int,
        state: str, gesture: str, mode: str,
        pending_start: Optional[float], pending_duration: float
    ):
        """Draws Status, Mode, Gesture badges."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)

        col_w = w // 3

        # 1. State Column
        cv2.putText(img, "STATE", (x + 12, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        state_color = self.COLOR_IDLE
        if state == "WRITING": state_color = self.COLOR_WRITING
        elif state == "PENDING": state_color = self.COLOR_PENDING
        elif state == "RECOGNIZING": state_color = self.COLOR_RECOGNIZING

        badge_w = col_w - 18
        cv2.rectangle(img, (x + 12, y + 26), (x + 12 + badge_w, y + 52), state_color, -1)
        cv2.putText(img, state, (x + 18, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # Progress bar under State when PENDING
        if state == "PENDING" and pending_start is not None:
            elapsed = time.time() - pending_start
            ratio = min(1.0, elapsed / pending_duration)
            bar_w = int(ratio * badge_w)
            cv2.rectangle(img, (x + 12, y + 58), (x + 12 + bar_w, y + 65), (0, 165, 255), -1)
            cv2.rectangle(img, (x + 12, y + 58), (x + 12 + badge_w, y + 65), (80, 80, 80), 1)

        # 2. Mode Column
        cv2.putText(img, "MODE", (x + col_w + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        mode_bg = (180, 80, 0) if mode == "LETTER" else (0, 150, 180)
        cv2.rectangle(img, (x + col_w + 10, y + 26), (x + col_w + 10 + badge_w, y + 52), mode_bg, -1)
        cv2.putText(img, mode, (x + col_w + 18, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # 3. Gesture Column
        cv2.putText(img, "GESTURE", (x + col_w * 2 + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        cv2.putText(img, gesture, (x + col_w * 2 + 5, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 2, cv2.LINE_AA)

    def _draw_canvas_card(self, img: np.ndarray, x: int, y: int, w: int, h: int, canvas_img: np.ndarray):
        """Draws handwriting canvas preview card."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)

        cv2.putText(img, "HANDWRITING CANVAS PREVIEW", (x + 15, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.TEXT_PRIMARY, 2, cv2.LINE_AA)

        avail_w = w - 30
        avail_h = h - 36
        if avail_w < 50 or avail_h < 50:
            return

        # Maintain 16:9 canvas aspect ratio
        canv_aspect = 16.0 / 9.0
        scale = min(avail_w / canv_aspect, float(avail_h))
        cv_h = int(scale * 0.95)
        cv_w = int(cv_h * canv_aspect)
        if cv_w > avail_w:
            cv_w = avail_w
            cv_h = int(cv_w / canv_aspect)

        canvas_preview = cv2.resize(canvas_img, (cv_w, cv_h), interpolation=cv2.INTER_AREA)

        cx = x + (w - cv_w) // 2
        cy = y + 30 + (avail_h - cv_h) // 2
        img[cy : cy + cv_h, cx : cx + cv_w] = canvas_preview
        cv2.rectangle(img, (cx, cy), (cx + cv_w, cy + cv_h), (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_result_card(
        self, img: np.ndarray, x: int, y: int, w: int, h: int,
        text: str, last_char: Optional[str], confidence: float,
        suggestions: Optional[List[str]] = None,
        active_prefix: str = "",
        selected_idx: int = 0
    ):
        """Draws recognized text, confidence meter, and word suggestions buttons."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)

        # 1. Recognized Text
        if text:
            display_text = text.replace(' ', ' _ ')
            max_chars = max(24, int(w / 18))
            if len(display_text) > max_chars:
                display_text = "..." + display_text[-(max_chars - 3):]
        else:
            display_text = "(Ready. Start writing on camera...)"

        cv2.putText(img, "RECOGNIZED TEXT:", (x + 15, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        cv2.putText(img, display_text, (x + 15, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Last Character & Confidence Bar
        if last_char is not None:
            if last_char in (" ", "SPACE"):
                char_str = "Last: [SPACE]"
                conf_str = "100%"
            else:
                char_str = f"Last: '{last_char}'"
                conf_str = f"{confidence:.1f}%"

            info_x = max(x + 280, x + w - 240)
            cv2.putText(img, char_str, (info_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(img, conf_str, (info_x + 120, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 2, cv2.LINE_AA)

            bar_max_w = min(200, w - (info_x - x) - 25)
            if bar_max_w > 40:
                bar_y = y + 31
                bar_w = int((confidence / 100.0) * bar_max_w)
                bar_color = (0, 220, 100) if confidence >= 70.0 else (0, 200, 255)
                cv2.rectangle(img, (info_x, bar_y), (info_x + bar_w, bar_y + 10), bar_color, -1)
                cv2.rectangle(img, (info_x, bar_y), (info_x + bar_max_w, bar_y + 10), (90, 90, 90), 1)

        # Separator Line
        cv2.line(img, (x + 15, y + 58), (x + w - 15, y + 58), (48, 58, 76), 1)

        # 3. Word Suggestions
        self.suggestion_rects.clear()
        if suggestions and len(suggestions) > 0:
            header_str = f"WORD SUGGESTIONS for '{active_prefix.upper()}':"
            cv2.putText(img, header_str, (x + 15, y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)

            num_sug = len(suggestions)
            gap = 6
            available_w = w - 30
            btn_w = int((available_w - (max(1, num_sug - 1) * gap)) / max(1, num_sug))
            btn_y1 = y + 86
            btn_y2 = y + 128

            for i, word in enumerate(suggestions):
                btn_x1 = x + 15 + i * (btn_w + gap)
                btn_x2 = btn_x1 + btn_w
                self.suggestion_rects.append((btn_x1, btn_y1, btn_x2, btn_y2, word))

                is_active = (i == selected_idx)
                btn_bg = (25, 75, 45) if is_active else (45, 54, 72)
                border_col = (0, 255, 150) if is_active else (75, 95, 125)
                border_th = 2 if is_active else 1

                cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_bg, -1)
                cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), border_col, border_th, cv2.LINE_AA)

                w_disp = word.upper()
                lbl = f">[{i+1}]{w_disp}" if is_active else f"[{i+1}] {w_disp}"

                font_scale = 0.42
                max_text_w = btn_w - 6
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                if tw > max_text_w:
                    font_scale = max(0.26, font_scale * (max_text_w / (tw + 1e-6)))
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                text_x = btn_x1 + max(2, (btn_w - tw) // 2)
                text_y = btn_y1 + 27
                txt_col = (0, 255, 220) if is_active else (220, 220, 220)
                cv2.putText(img, lbl, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_col, 1, cv2.LINE_AA)
        else:
            if active_prefix:
                cv2.putText(img, f"WORD SUGGESTIONS: No matching words for prefix '{active_prefix.upper()}'", (x + 15, y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "WORD SUGGESTIONS: (Write letters to see 500k dictionary autocompletions)", (x + 15, y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 135, 155), 1, cv2.LINE_AA)

    def _draw_footer_card(
        self, img: np.ndarray, x: int, y: int, w: int, h: int, show_debug: bool,
        collect_mode: bool = False, target_char: str = "A", custom_samples_count: int = 0
    ):
        """Draws shortcut badges and footer information."""
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BG, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.CARD_BORDER, 1, cv2.LINE_AA)

        cv2.putText(img, "KEYBOARD SHORTCUTS & DATASET TRAIN:", (x + 15, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_SECONDARY, 1, cv2.LINE_AA)

        # Line 1: Dataset targets
        target_type = "NUM" if target_char.isdigit() else "LET"
        cv2.putText(img, f"[t] Target: '{target_char}' ({target_type})", (x + 15, y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"[r] Save Sample", (x + 180, y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(img, f"Samples: {custom_samples_count}", (x + 330, y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 50), 1, cv2.LINE_AA)

        # Line 2: Control keys
        cv2.putText(img, "[1-5] Word", (x + 15, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "[m] Mode", (x + 115, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        cv2.putText(img, "[c] Clear", (x + 195, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
        cv2.putText(img, "[s] Save", (x + 275, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
        dbg_text = "[d] Debug: ON" if show_debug else "[d] Debug: OFF"
        cv2.putText(img, dbg_text, (x + 355, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255) if show_debug else (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "[q] Quit", (x + 475, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)
