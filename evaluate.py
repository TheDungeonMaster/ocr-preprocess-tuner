# ocr_metrics.py
# ------------------------------------------------------------
# OCR evaluation utilities: WER (with edit ops) and CER
# - Unicode + whitespace normalization
# - Remove punctuation
# - CER ignores whitespace errors (remove all spaces)
# - Macro & Micro aggregates across many docs
# - Edge-case safe
# ------------------------------------------------------------

from __future__ import annotations
import math
import os
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import jiwer

from src.accuracy.test import OcrEvaluator
from src.accuracy.model import DocMeasures, AggregateResult, DocPaths


def read_text_file(path: str) -> str:
    """
    Read text safely:
    - utf-8-sig: handles BOM if present
    - strip trailing/leading whitespace to stabilize scoring
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read().strip()



def aggregate_results(per_doc: List[DocMeasures]) -> AggregateResult:
    # Filter docs with valid denominators
    valid_wer_docs = [d for d in per_doc if d.ref_word_count > 0 and not math.isnan(d.wer)]
    valid_cer_docs = [d for d in per_doc if d.ref_char_count > 0 and not math.isnan(d.cer)]

    # Macro means (unweighted)
    macro_wer = float("nan") if not valid_wer_docs else sum(d.wer for d in valid_wer_docs) / len(valid_wer_docs)
    macro_cer = float("nan") if not valid_cer_docs else sum(d.cer for d in valid_cer_docs) / len(valid_cer_docs)

    # Micro WER: sum(S+D+I) / sum(ref_words)
    total_subs = sum(d.subs for d in valid_wer_docs)
    total_dels = sum(d.dels for d in valid_wer_docs)
    total_ins  = sum(d.ins  for d in valid_wer_docs)
    total_ref_words = sum(d.ref_word_count for d in valid_wer_docs)

    if total_ref_words == 0:
        micro_wer = float("nan")
    else:
        micro_wer = (total_subs + total_dels + total_ins) / total_ref_words

    # Micro CER: we only have per-doc CER, but we can compute exact micro CER if we also know
    # ref char counts and the numerator (S+D+I). We don't have char-level ops, so we compute
    # the weighted average by denominator (equivalent to micro if CER is computed consistently):
    total_ref_chars = sum(d.ref_char_count for d in valid_cer_docs)
    if total_ref_chars == 0:
        micro_cer = float("nan")
    else:
        # Weighted mean by ref_char_count
        micro_cer = sum(d.cer * d.ref_char_count for d in valid_cer_docs) / total_ref_chars

    return AggregateResult(
        macro_wer=macro_wer,
        macro_cer=macro_cer,
        micro_wer=micro_wer,
        micro_cer=micro_cer,
        total_ref_words=total_ref_words,
        total_ref_chars=total_ref_chars,
        total_subs=total_subs,
        total_dels=total_dels,
        total_ins=total_ins,
        per_doc=per_doc
    )


# ----------------------------
# Convenience runner
# ----------------------------

def evaluate_from_paths(pairs: List[DocPaths]) -> AggregateResult:
    """
    Evaluate a list of (ref_path, hyp_path) pairs.
    Returns per-doc measures + macro/micro aggregates.
    Safe on missing/empty files; those docs are excluded from macro/micro denominators.
    """
    evaluator = OcrEvaluator()
    per_doc: List[DocMeasures] = []

    for i, p in enumerate(pairs, start=1):
        doc_id = f"doc_{i}"
        try:
            ref_text = read_text_file(p.reference_path)
        except Exception as e:
            ref_text = ""
        try:
            hyp_text = read_text_file(p.hypothesis_path)
        except Exception as e:
            hyp_text = ""

        measures = evaluator.evaluate_document(doc_id, ref_text, hyp_text)
        per_doc.append(measures)

    return aggregate_results(per_doc)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Replace with your actual files
    input_pairs = [
        DocPaths(
            reference_path="docs/doc1/Contract_AEOMPI25.txt",
            hypothesis_path="docs/doc1/AEOMPI25_signed_by_2_parts[1]_p001.txt"
        ),
        DocPaths(
            reference_path="docs/doc2/Contract_AEOMPI25_part2.txt",
            hypothesis_path="docs/doc2/AEOMPI25_signed_by_2_parts[1]_p002.txt"
        ),
    ]

    results = evaluate_from_paths(input_pairs)

    # Print per-doc breakdown
    for d in results.per_doc:
        print(f"[{d.doc_id}] WER={d.wer:.4f} (H={d.hits} S={d.subs} D={d.dels} I={d.ins}, N_words={d.ref_word_count}) "
              f"| CER={d.cer:.4f} (N_chars={d.ref_char_count})")

    # Print aggregates
    print("\n--- Aggregates ---")
    print(f"Macro WER:  {results.macro_wer:.4f}" if not math.isnan(results.macro_wer) else "Macro WER:  NaN")
    print(f"Micro WER:  {results.micro_wer:.4f}" if not math.isnan(results.micro_wer) else "Micro WER:  NaN")
    print(f"Macro CER:  {results.macro_cer:.4f}" if not math.isnan(results.macro_cer) else "Macro CER:  NaN")
    print(f"Micro CER:  {results.micro_cer:.4f}" if not math.isnan(results.micro_cer) else "Micro CER:  NaN")

    print(f"\nTotals -> Ref words: {results.total_ref_words}, Ref chars: {results.total_ref_chars}, "
          f"Subs: {results.total_subs}, Dels: {results.total_dels}, Ins: {results.total_ins}")
