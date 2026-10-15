import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import numpy as np
from recognition.hybrid_classifier import HybridDatasetClassifier

print("=== TEST 1: New Dual Model Initialization ===")
hybrid_clf = HybridDatasetClassifier(
    letters_model_path=config.LETTERS_MODEL_PATH,
    digits_model_path=config.DIGITS_MODEL_PATH
)

dummy_img = np.zeros((64, 64), dtype=np.uint8)
# Draw a simple vertical stroke like 'I' or '1'
dummy_img[10:54, 30:34] = 255

print("\n=== TEST 2: Predict LETTER Mode ===")
char_l, conf_l, info_l = hybrid_clf.predict_hybrid(dummy_img, mode='LETTER')
print(f"LETTER Result: {char_l} ({conf_l}%) | Top2: {info_l.get('top2_char')} ({info_l.get('top2_conf')}%) | Margin: {info_l.get('margin')}%")

print("\n=== TEST 3: Predict NUMBER Mode ===")
char_n, conf_n, info_n = hybrid_clf.predict_hybrid(dummy_img, mode='NUMBER')
print(f"NUMBER Result: {char_n} ({conf_n}%) | Top2: {info_n.get('top2_char')} ({info_n.get('top2_conf')}%) | Margin: {info_n.get('margin')}%")

print("\n=== TEST 4: Predict None Mode (Any) ===")
char_a, conf_a, info_a = hybrid_clf.predict_hybrid(dummy_img, mode=None)
print(f"ANY Result:    {char_a} ({conf_a}%) | Top2: {info_a.get('top2_char')} ({info_a.get('top2_conf')}%) | Margin: {info_a.get('margin')}%")

print("\n=== TEST 5: Backward Compatibility (Calling with non-existent single model path) ===")
legacy_clf = HybridDatasetClassifier(config.MODEL_PATH)
print("Legacy fallback base_classifier loaded:", legacy_clf.base_classifier is not None)
char_leg, conf_leg, _ = legacy_clf.predict_hybrid(dummy_img, mode='LETTER')
print(f"Legacy LETTER Result: {char_leg} ({conf_leg}%)")

print("\n=== ALL TESTS PASSED! ===")
