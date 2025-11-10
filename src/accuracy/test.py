# core.py
import argparse
from pathlib import Path
from typing import List, Tuple

from model import DocumentPair

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "test_data"

def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def evaluate_pairs(pairs: List[Tuple[str, str]]) -> None:
    for idx, (ref_path, hyp_path) in enumerate(pairs, start=1):
        gt = read_text_file(ref_path)
        pr = read_text_file(hyp_path)
        dp = DocumentPair(gt, pr)
        wer = dp.compute_wer()
        cer = dp.compute_cer()
        print(f"Doc {idx}:")
        print(f"  WER: {wer:.6f}")
        print(f"  CER: {cer:.6f}")
        print()

def main():
    pairs = [
        (DOCS_DIR / "Contract_AEOMPI25.txt", DOCS_DIR / "AEOMPI25_signed_by_2_parts[1]_p001.txt"),
        (DOCS_DIR / "Contract_AEOMPI25_part2.txt", DOCS_DIR / "AEOMPI25_signed_by_2_parts[1]_p002.txt"),
    ]
    evaluate_pairs(pairs)

if __name__ == "__main__":
    main()
