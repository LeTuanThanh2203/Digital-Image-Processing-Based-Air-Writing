# recognition/classifier.py
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import numpy as np

class NumPyCNNModel:
    """
    Trình khởi chạy suy luận CNN thuần NumPy + h5py.
    Cho phép load mô hình .keras và predict trực tiếp mà không cần TensorFlow hay CMake.
    """
    def __init__(self, keras_file_path: Path):
        import h5py
        z = zipfile.ZipFile(str(keras_file_path))
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extract('model.weights.h5', tmpdir)
            h5_path = os.path.join(tmpdir, 'model.weights.h5')
            with h5py.File(h5_path, 'r') as f:
                self.w_c1 = f['layers/conv2d/vars/0'][:]
                self.b_c1 = f['layers/conv2d/vars/1'][:]
                self.w_c2 = f['layers/conv2d_1/vars/0'][:]
                self.b_c2 = f['layers/conv2d_1/vars/1'][:]
                self.w_c3 = f['layers/conv2d_2/vars/0'][:]
                self.b_c3 = f['layers/conv2d_2/vars/1'][:]
                self.w_d1 = f['layers/dense/vars/0'][:]
                self.b_d1 = f['layers/dense/vars/1'][:]
                self.w_d2 = f['layers/dense_1/vars/0'][:]
                self.b_d2 = f['layers/dense_1/vars/1'][:]

        self.input_shape = (None, 64, 64, 1)
        self.output_shape = (None, self.w_d2.shape[1])

    def predict(self, tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
        # tensor shape: (1, 64, 64, 1)
        x = tensor[0] # (64, 64, 1)
        
        # Conv1 + MaxPool1 (32 filters)
        x1 = np.maximum(0, self._conv2d_same(x, self.w_c1, self.b_c1))
        x1_p = self._maxpool2d(x1)
        
        # Conv2 + MaxPool2 (64 filters)
        x2 = np.maximum(0, self._conv2d_same(x1_p, self.w_c2, self.b_c2))
        x2_p = self._maxpool2d(x2)
        
        # Conv3 + MaxPool3 (128 filters)
        x3 = np.maximum(0, self._conv2d_same(x2_p, self.w_c3, self.b_c3))
        x3_p = self._maxpool2d(x3)
        
        # Flatten + Dense1 (128) + Dense2 (26 Softmax)
        flat = x3_p.reshape(-1)
        dense1 = np.maximum(0, np.dot(flat, self.w_d1) + self.b_d1)
        logits = np.dot(dense1, self.w_d2) + self.b_d2
        
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / np.sum(e_x)
        return np.expand_dims(probs, axis=0)

    @staticmethod
    def _maxpool2d(x: np.ndarray) -> np.ndarray:
        h, w, c = x.shape
        return x.reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))

    @staticmethod
    def _conv2d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        kh, kw, cin, cout = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        x_padded = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        out_h, out_w = x.shape[0], x.shape[1]
        out = np.zeros((out_h, out_w, cout), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                patch = x_padded[i:i+out_h, j:j+out_w, :]
                w_sub = kernel[i, j, :, :]
                out += np.tensordot(patch, w_sub, axes=([2], [0]))
        out += bias
        return out


class CharacterClassifier:
    """
    Module nhận diện chữ cái tiếng Anh (A-Z) từ ảnh Canvas Air-Writing 
    sử dụng mô hình CNN được huấn luyện trên bộ dữ liệu 6DMG (input 64x64x1).
    """
    def __init__(self, model_path: Union[str, Path], class_labels: Optional[list] = None):
        self.model_path = Path(model_path)
        self.custom_labels = class_labels
        self.class_labels = []
        
        # 1. Kiểm tra sự tồn tại của file model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[CNN Classifier Error] Không tìm thấy file mô hình tại: '{self.model_path}'. "
                "Vui lòng lưu file mô hình đã train vào thư mục 'models/airwriting_cnn.keras'."
            )
            
        # 2. Tải mô hình: Ưu tiên Keras/TensorFlow, tự động dùng NumPy engine nếu không có TF
        self.model = None
        try:
            import keras
            self.model = keras.models.load_model(str(self.model_path))
            print(f"[CNN Classifier] Loaded CNN model using Keras from '{self.model_path}'.")
        except Exception:
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(self.model_path))
                print(f"[CNN Classifier] Loaded CNN model using TensorFlow from '{self.model_path}'.")
            except Exception:
                # Dùng NumPy + HDF5 fallback engine siêu nhẹ
                try:
                    self.model = NumPyCNNModel(self.model_path)
                    print(f"[CNN Classifier] Loaded CNN model using NumPy Engine from '{self.model_path}'.")
                except Exception as e:
                    raise RuntimeError(
                        f"[CNN Classifier Error] Failed to load model from '{self.model_path}': {e}"
                    )
            
        # 3. Kiểm tra tính tương thích cấu trúc Input/Output và khởi tạo danh sách nhãn (class_labels)
        self._validate_model_shapes()
        
    def _validate_model_shapes(self):
        """Kiểm tra shape của input/output và tự động ánh xạ danh sách nhãn (0-9, A-Z)."""
        input_shape = self.model.input_shape
        output_shape = self.model.output_shape
        num_classes = output_shape[-1]
        
        # Kiểm tra chiều input
        if len(input_shape) != 4 or input_shape[1:3] != (64, 64) or input_shape[3] != 1:
            print(f"[Warning] Model input shape ({input_shape}) khác với kích thước chuẩn 64x64x1.")
            
        # Tự động khởi tạo bảng nhãn ký tự dựa trên số lượng lớp output
        if self.custom_labels is not None:
            if len(self.custom_labels) != num_classes:
                raise ValueError(
                    f"[CNN Classifier Error] custom_labels truyền vào có {len(self.custom_labels)} phần tử, "
                    f"nhưng mô hình yêu cầu {num_classes} phân lớp."
                )
            self.class_labels = [str(lbl) for lbl in self.custom_labels]
        elif num_classes == 26:
            # 26 lớp chuẩn A-Z
            self.class_labels = [chr(ord('A') + i) for i in range(26)]
        elif num_classes == 10:
            # 10 lớp chữ số 0-9
            self.class_labels = [str(i) for i in range(10)]
        elif num_classes == 36:
            # 36 lớp: Chữ cái A-Z và chữ số 0-9 (hoặc ngược lại)
            # Mặc định: 0-9 (10 lớp đầu) + A-Z (26 lớp sau)
            self.class_labels = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
        else:
            # Fallback cho số phân lớp khác
            self.class_labels = [str(i) for i in range(num_classes)]
            
        print(f"[CNN Classifier] Auto-configured {num_classes} classes mapping: {self.class_labels[:5]}...")

    def preprocess(self, canvas: np.ndarray, pad: int = 15) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Tiền xử lý ảnh Canvas runtime (640x480 BGR nền trắng nét đen) thành tensor (1, 64, 64, 1) nền đen nét trắng chuẩn hóa [0, 1].
        
        Trả về:
            - tensor: (1, 64, 64, 1) float32 trong dải [0, 1] (None nếu canvas trống).
            - debug_img: (64, 64) uint8 [0, 255] nét chữ trên nền đen (để hiển thị debug window).
        """
        if canvas is None or not isinstance(canvas, np.ndarray) or canvas.size == 0:
            return None, None
            
        # 1. Chuyển BGR sang Grayscale
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        
        # 2. Binary Threshold Inverse: Nền trắng (255) -> 0 (Đen), Nét đen (0) -> 255 (Trắng)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Kiểm tra canvas có nét vẽ hay không (Empty Canvas Check)
        pts = cv2.findNonZero(mask)
        if pts is None:
            return None, None
            
        # 4. Xác định Bounding Box của vùng chữ viết và thêm Padding
        x, y, w, h = cv2.boundingRect(pts)
        h_img, w_img = mask.shape
        
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        
        crop = mask[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch == 0 or cw == 0:
            return None, None
            
        # 5. Bảo toàn tỷ lệ khía cạnh (Preserve aspect ratio) bằng cách đặt crop vào khung hình vuông
        max_dim = max(ch, cw)
        square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
        off_y = (max_dim - ch) // 2
        off_x = (max_dim - cw) // 2
        square_img[off_y:off_y + ch, off_x:off_x + cw] = crop
        
        # 6. Resize về kích thước chuẩn 64x64
        resized = cv2.resize(square_img, (64, 64), interpolation=cv2.INTER_AREA)
        
        # 7. Chuẩn hóa pixel về float32 dải 0.0 -> 1.0 và mở rộng chiều Batch & Channel
        normalized = resized.astype(np.float32) / 255.0
        tensor = np.expand_dims(normalized, axis=(0, -1)) # Shape: (1, 64, 64, 1)
        
        return tensor, resized

    def predict(self, canvas: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Dự đoán chữ cái từ ảnh Canvas.
        
        Trả về:
            - character: Ký tự chữ cái viết hoa ('A' đến 'Z') hoặc None nếu canvas trống.
            - confidence: Độ tin cậy dưới dạng phần trăm (ví dụ: 96.72). Trả về 0.0 nếu không dự đoán được.
        """
        tensor, _ = self.preprocess(canvas)
        
        # Nếu canvas rỗng (không tìm thấy nét vẽ)
        if tensor is None:
            return None, 0.0
            
        try:
            # 1. Chạy dự đoán từ CNN model
            preds = self.model.predict(tensor, verbose=0)[0]
            
            # 2. Lấy chỉ số có xác suất cao nhất
            class_idx = int(np.argmax(preds))
            
            # 3. Map chỉ số (class_idx) sang ký tự chữ/số tương ứng từ danh sách nhãn (class_labels)
            character = self.class_labels[class_idx] if class_idx < len(self.class_labels) else str(class_idx)
            
            # 4. Tính xác suất phần trăm
            confidence = round(float(preds[class_idx]) * 100.0, 2)
            
            return character, confidence
        except Exception as e:
            print(f"[CNN Classifier Error] Lỗi trong quá trình predict: {e}")
            return None, 0.0
