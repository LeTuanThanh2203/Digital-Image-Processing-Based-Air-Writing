# recognition/hybrid_classifier.py
"""
Hybrid Dataset & Classifier Module.
Combines 6DMG Pre-trained CNN model with User Custom Dataset Samples (dataset/custom/<label>/)
using DIP Structural Similarity (SSIM) and Ensemble Probability Fusion.
"""

import os
import time
import glob
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
import cv2
import numpy as np
from .classifier import CharacterClassifier


def compute_ssim_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Tính chỉ số tương đồng hình thái SSIM đơn giản giữa 2 ảnh nhị phân 64x64.
    """
    if img1.shape != (64, 64):
        img1 = cv2.resize(img1, (64, 64))
    if img2.shape != (64, 64):
        img2 = cv2.resize(img2, (64, 64))

    # Normalized float [0, 1]
    f1 = img1.astype(np.float32) / 255.0
    f2 = img2.astype(np.float32) / 255.0

    mu1 = cv2.GaussianBlur(f1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(f2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(f1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(f2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(f1 * f2, (11, 11), 1.5) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CUSTOM_DATASET_DIR = PROJECT_ROOT / "dataset" / "custom"


class HybridDatasetClassifier:
    """
    Bộ phân loại kết hợp (Hybrid Ensemble Classifier):
    - CNN Model gốc (huấn luyện trên 6DMG/Digits).
    - Custom Dataset (người dùng tự lưu vào dataset/custom/<Label>/ cho cả chữ và số).
    """

    def __init__(
        self,
        base_model_path: Union[str, Path],
        custom_dataset_dir: Optional[Union[str, Path]] = None,
        class_labels: Optional[list] = None
    ):
        self.base_classifier = CharacterClassifier(base_model_path, class_labels=class_labels)
        if custom_dataset_dir is None:
            self.custom_dataset_dir = DEFAULT_CUSTOM_DATASET_DIR
        else:
            p = Path(custom_dataset_dir)
            self.custom_dataset_dir = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

        self.custom_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Đảm bảo cấu trúc thư mục đầy đủ cho chữ cái (A-Z) và chữ số (0-9)
        all_classes = [chr(ord('A') + i) for i in range(26)] + [str(i) for i in range(10)]
        for c in all_classes:
            (self.custom_dataset_dir / c).mkdir(parents=True, exist_ok=True)

        # Structure: {'A': [img1, img2...], '0': [img1...]}
        self.custom_samples: Dict[str, List[np.ndarray]] = {}
        self.reload_custom_dataset()

    def reload_custom_dataset(self):
        """Đọc và nạp lại toàn bộ các ảnh mẫu trong thư mục dataset/custom/ (hỗ trợ cả chữ A-Z và số 0-9)."""
        self.custom_samples.clear()
        total_count = 0
        letters_count = 0
        numbers_count = 0
        
        valid_exts = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
        for label_dir in self.custom_dataset_dir.iterdir():
            if label_dir.is_dir():
                label = label_dir.name.upper()
                images = []
                img_paths = []
                for ext in valid_exts:
                    img_paths.extend(glob.glob(str(label_dir / ext)))

                for img_path in sorted(set(img_paths)):
                    try:
                        img_data = np.fromfile(img_path, dtype=np.uint8)
                        img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is not None:
                        if img.shape != (64, 64):
                            img = cv2.resize(img, (64, 64))
                        images.append(img)
                        total_count += 1
                        if label.isdigit():
                            numbers_count += 1
                        else:
                            letters_count += 1

                if images:
                    self.custom_samples[label] = images

        print(f"[Hybrid Classifier] Loaded {total_count} custom samples ({letters_count} letters, {numbers_count} digits) across {len(self.custom_samples)} classes.")

    def save_custom_sample(self, label: str, img_64: np.ndarray) -> str:
        """
        Lưu một ảnh mẫu 64x64 nét chữ hoặc chữ số của người dùng vào dataset/custom/<label>/.
        """
        label = label.upper().strip()
        label_dir = self.custom_dataset_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        filename = f"sample_{timestamp}.png"
        filepath = label_dir / filename

        is_success, im_buf = cv2.imencode(".png", img_64)
        if is_success:
            im_buf.tofile(str(filepath))
        else:
            cv2.imwrite(str(filepath), img_64)
        
        # Nạp lại bộ mẫu cá nhân
        self.reload_custom_dataset()
        return str(filepath)

    def get_custom_stats(self) -> Dict[str, int]:
        """Trả về thống kê số lượng mẫu cá nhân đã lưu cho từng ký tự."""
        stats = {}
        for lbl, imgs in self.custom_samples.items():
            stats[lbl] = len(imgs)
        return stats

    def predict_hybrid(
        self,
        input_data: np.ndarray,
        mode: Optional[str] = None,
        confidence_threshold: float = 40.0,
        traj_stats: Optional[dict] = None
    ) -> Tuple[Optional[str], float, dict]:
        """
        Dự đoán ký tự/chữ số kết hợp giữa CNN Model và Custom Dataset của người dùng.
        """
        # 1. Lấy kết quả sơ bộ từ CNN base classifier
        char_base, conf_base, info = self.base_classifier.predict(
            input_data,
            mode=mode,
            confidence_threshold=confidence_threshold,
            traj_stats=traj_stats
        )

        if input_data is None or input_data.size == 0:
            return None, 0.0, info

        # Nếu chưa có custom sample nào, dùng 100% CNN Base
        if not self.custom_samples:
            return char_base, conf_base, info

        # Đảm bảo input_data là ảnh 64x64 nhị phân
        img_64 = input_data
        if len(input_data.shape) == 3:
            img_64 = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
        if img_64.shape != (64, 64):
            img_64 = cv2.resize(img_64, (64, 64))

        # 2. Tính điểm tương đồng SSIM của input_data với tất cả các lớp trong Custom Dataset
        custom_scores: Dict[str, float] = {}
        for label, samples in self.custom_samples.items():
            # Lọc theo mode nếu cần: LETTER chỉ so khớp với chữ cái, NUMBER chỉ so khớp với chữ số
            if mode == "LETTER" and not label.isalpha():
                continue
            if mode == "NUMBER" and not label.isdigit():
                continue

            scores = [compute_ssim_similarity(img_64, sample) for sample in samples]
            custom_scores[label] = max(scores) if scores else 0.0

        if not custom_scores:
            return char_base, conf_base, info

        # Tìm nhãn có độ tương đồng Custom cao nhất
        best_custom_label = max(custom_scores, key=custom_scores.get)
        best_custom_ssim = custom_scores[best_custom_label]
        best_custom_conf = round(best_custom_ssim * 100.0, 2)

        # 3. Hybrid Weighted Fusion (70% Custom Dataset + 30% Base Model)
        weight_custom = 0.70
        weight_base = 0.30

        info['custom_best_label'] = best_custom_label
        info['custom_best_conf'] = best_custom_conf

        # Lấy tất cả danh sách nhãn phù hợp với mode
        valid_labels = self.base_classifier.class_labels
        if mode == "LETTER":
            valid_labels = [lbl for lbl in valid_labels if lbl.isalpha()]
            if not valid_labels:
                valid_labels = [chr(ord('A') + i) for i in range(26)]
        elif mode == "NUMBER":
            valid_labels = [lbl for lbl in valid_labels if lbl.isdigit()]
            if not valid_labels:
                valid_labels = [str(i) for i in range(10)]

        tensor, _ = self.base_classifier.preprocess(input_data)
        base_preds = None
        if tensor is not None and hasattr(self.base_classifier, 'model') and self.base_classifier.model is not None:
            try:
                base_preds = self.base_classifier.model.predict(tensor, verbose=0)[0]
            except Exception:
                base_preds = None

        hybrid_scores: Dict[str, float] = {}
        for lbl in valid_labels:
            base_conf = 0.0
            if base_preds is not None and lbl in self.base_classifier.class_labels:
                idx = self.base_classifier.class_labels.index(lbl)
                base_conf = float(base_preds[idx]) * 100.0
            
            c_ssim = custom_scores.get(lbl, 0.0)
            c_conf = c_ssim * 100.0
            
            # Tính điểm kết hợp 70% Custom + 30% Base
            if base_preds is not None:
                score = weight_custom * c_conf + weight_base * base_conf
            else:
                score = c_conf

            # Nếu đây là lớp có Custom Match cao nhất (>= 50%), thưởng ưu tiên để luôn xếp Top-1
            if lbl == best_custom_label and c_ssim >= 0.50:
                score = max(c_conf, score)

            hybrid_scores[lbl] = round(score, 2)

        if not hybrid_scores:
            return char_base, conf_base, info

        # Trích xuất Top-1 và Top-2 từ điểm số Hybrid đã ưu tiên Custom Match
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top1_char, top1_conf = sorted_hybrid[0]
        top2_char, top2_conf = sorted_hybrid[1] if len(sorted_hybrid) > 1 else (top1_char, 0.0)

        info['top1_char'] = top1_char
        info['top1_conf'] = top1_conf
        info['top2_char'] = top2_char
        info['top2_conf'] = top2_conf
        info['margin'] = round(top1_conf - top2_conf, 2)
        info['weight_custom'] = weight_custom
        info['weight_base'] = weight_base
        info['hybrid_boosted'] = True
        info['is_unknown'] = False
        info['reject_reason'] = None

        return top1_char, top1_conf, info
