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


class HybridDatasetClassifier:
    """
    Bộ phân loại kết hợp (Hybrid Ensemble Classifier):
    - CNN Model gốc (huấn luyện trên 6DMG).
    - Custom Dataset (người dùng tự lưu vào dataset/custom/<Label>/).
    """

    def __init__(
        self,
        base_model_path: Union[str, Path],
        custom_dataset_dir: Union[str, Path] = "dataset/custom",
        class_labels: Optional[list] = None
    ):
        self.base_classifier = CharacterClassifier(base_model_path, class_labels=class_labels)
        self.custom_dataset_dir = Path(custom_dataset_dir)
        self.custom_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Structure: {'A': [img1, img2...], 'B': [img1...]}
        self.custom_samples: Dict[str, List[np.ndarray]] = {}
        self.reload_custom_dataset()

    def reload_custom_dataset(self):
        """Đọc và nạp lại toàn bộ các ảnh mẫu trong thư mục dataset/custom/."""
        self.custom_samples.clear()
        total_count = 0
        
        for label_dir in self.custom_dataset_dir.iterdir():
            if label_dir.is_dir():
                label = label_dir.name.upper()
                images = []
                for img_path in glob.glob(str(label_dir / "*.png")):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        if img.shape != (64, 64):
                            img = cv2.resize(img, (64, 64))
                        images.append(img)
                        total_count += 1
                if images:
                    self.custom_samples[label] = images

        print(f"[Hybrid Classifier] Loaded {total_count} custom dataset samples for {len(self.custom_samples)} classes.")

    def save_custom_sample(self, label: str, img_64: np.ndarray) -> str:
        """
        Lưu một ảnh mẫu 64x64 nét chữ của người dùng vào dataset/custom/<label>/.
        """
        label = label.upper()
        label_dir = self.custom_dataset_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        filename = f"sample_{timestamp}.png"
        filepath = label_dir / filename

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
        Dự đoán ký tự kết hợp giữa 6DMG CNN Model và Custom Dataset của người dùng.
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
            # Lọc theo mode nếu cần
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
        elif mode == "NUMBER":
            valid_labels = [lbl for lbl in valid_labels if lbl.isdigit()]

        tensor, _ = self.base_classifier.preprocess(input_data)
        base_preds = self.base_classifier.model.predict(tensor, verbose=0)[0]

        hybrid_scores: Dict[str, float] = {}
        for lbl in valid_labels:
            idx = self.base_classifier.class_labels.index(lbl) if lbl in self.base_classifier.class_labels else 0
            base_conf = float(base_preds[idx]) * 100.0
            
            c_ssim = custom_scores.get(lbl, 0.0)
            c_conf = c_ssim * 100.0
            
            # Tính điểm kết hợp 70% Custom + 30% Base
            score = weight_custom * c_conf + weight_base * base_conf

            # Nếu đây là lớp có Custom Match cao nhất (>= 50%), thưởng ưu tiên để luôn xếp Top-1
            if lbl == best_custom_label and c_ssim >= 0.50:
                score = max(c_conf, score)

            hybrid_scores[lbl] = round(score, 2)

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
