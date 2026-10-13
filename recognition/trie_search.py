# recognition/trie_search.py
"""
Fast Trie-based Word Suggestion & Autocompletion Engine.
Loaded from 500,000 frequent English words dictionary (frequency_dictionary_en_500_000.txt).
Provides instant O(L) prefix search and ranking for Air-Writing recognition.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple


class TrieNode:
    """Nút biểu diễn ký tự trong cấu trúc cây tiền tố Trie."""
    def __init__(self):
        self.children = {}
        self.is_word = False
        # Lưu trước danh sách các từ phổ biến nhất đi qua node này để tìm kiếm O(L) cực nhanh
        self.top_words = []


class Trie:
    """Cây tiền tố Trie hỗ trợ tra cứu và gợi ý từ vựng theo tần suất."""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        """Thêm một từ vào cây Trie."""
        node = self.root
        w = word.lower().strip()
        if not w:
            return

        for char in w:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
            # Cache tối đa 10 từ xuất hiện nhiều nhất (theo thứ tự insert từ file tần suất)
            if len(node.top_words) < 10 and w not in node.top_words:
                node.top_words.append(w)

        node.is_word = True

    def search_prefix(self, prefix: str, limit: int = 5) -> List[str]:
        """
        Tìm kiếm các từ bắt đầu bằng tiền tố prefix (tối đa limit từ).
        Ưu tiên các từ phổ biến nhất theo tần suất.
        """
        node = self.root
        prefix = prefix.lower().strip()
        if not prefix:
            return []

        # Đi đến nút đại diện cho prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # 1. Trả về ngay danh sách top_words đã được tối ưu hóa theo tần suất
        if hasattr(node, 'top_words') and node.top_words:
            return node.top_words[:limit]

        # 2. Duyệt nhánh dự phòng (Fallback DFS) nếu danh sách cache chưa đủ
        results = []

        def collect(current, path):
            if len(results) >= limit:
                return

            if current.is_word and path not in results:
                results.append(path)

            for char, child in sorted(current.children.items()):
                if len(results) >= limit:
                    break
                collect(child, path + char)

        collect(node, prefix)
        return results


def load_dictionary(file_path: str, max_words: Optional[int] = None) -> Trie:
    """
    Nạp từ điển tần suất vào cây Trie.
    Hỗ trợ xử lý UTF-8 có hoặc không có BOM (\ufeff).
    """
    trie = Trie()
    path = Path(file_path)
    if not path.exists():
        print(f"[Warning] Dictionary file not found at: '{file_path}'")
        return trie

    count = 0
    with open(str(path), "r", encoding="utf-8-sig") as file:
        for line in file:
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split()
            word = parts[0].lower() if parts else ""

            if word.isalpha():
                trie.insert(word)
                count += 1
                if max_words is not None and count >= max_words:
                    break

    print(f"[Trie Dictionary] Successfully loaded {count} English words from '{path.name}'.")
    return trie


def get_current_word_prefix(sentence: str) -> str:
    """
    Trích xuất tiền tố của từ hiện tại đang được viết.
    Nếu câu kết thúc bằng khoảng trắng, coi như từ đã hoàn thành -> trả về chuỗi rỗng.
    """
    if not sentence or sentence.endswith(" "):
        return ""
    
    # Lấy token cuối cùng
    tokens = sentence.split(" ")
    last_token = tokens[-1]
    
    # Chỉ lấy phần ký tự chữ cái
    prefix = "".join([c for c in last_token if c.isalpha()])
    return prefix.lower()


def apply_word_completion(sentence: str, chosen_word: str) -> str:
    """
    Thay thế tiền tố từ đang viết bằng từ được chọn và tự động thêm dấu cách (Space).
    Ví dụ:
        sentence = "I WAN TH", chosen_word = "the" -> "I WAN THE "
        sentence = "HEL", chosen_word = "hello" -> "HELLO "
    """
    if not chosen_word:
        return sentence

    chosen_clean = chosen_word.strip().upper()
    
    if not sentence or sentence.endswith(" "):
        return sentence + chosen_clean + " "

    # Tách các từ trước đó và thay thế từ cuối cùng
    last_space_idx = sentence.rfind(" ")
    if last_space_idx == -1:
        # Chỉ có 1 từ duy nhất đang viết
        return chosen_clean + " "
    else:
        prefix_part = sentence[:last_space_idx + 1]
        return prefix_part + chosen_clean + " "


if __name__ == "__main__":
    dict_file = Path(__file__).resolve().parent.parent / "frequency_dictionary_en_500_000.txt"
    trie = load_dictionary(str(dict_file))

    while True:
        try:
            prefix = input("Prefix (q để thoát): ").strip()
            if prefix.lower() == "q":
                break
            suggestions = trie.search_prefix(prefix, limit=5)
            print("Gợi ý từ (tối đa 5 từ):", suggestions)
            if suggestions:
                completed = apply_word_completion("I AM WRITING " + prefix, suggestions[0])
                print(f"Ví dụ tự động điền: '{completed}'")
        except (KeyboardInterrupt, EOFError):
            break
