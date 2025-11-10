# wer.py
from typing import Optional
import jiwer
from normalization import transform_word as default_transform_word


def compute_wer(
    ground_truth: str,
    prediction: str,
    reference_transform: Optional[jiwer.Compose] = None,
    hypothesis_transform: Optional[jiwer.Compose] = None,
) -> float:
    """
    Compute Word Error Rate (WER).

    Guard rails:
    - If ground_truth is empty after normalization, return 0.0 when prediction is also empty,
      otherwise return 1.0 (max error). This avoids NaN in degenerate cases.
    """
    ref_t = reference_transform or default_transform_word
    hyp_t = hypothesis_transform or default_transform_word

    # Detect degenerate case to avoid NaN
    ref_lists = ref_t(ground_truth)
    hyp_lists = hyp_t(prediction)
    ref_tokens = sum(len(seq) for seq in ref_lists)  # number of words

    if ref_tokens == 0:
        return 0.0 if sum(len(seq) for seq in hyp_lists) == 0 else 1.0

    return jiwer.wer(
        ground_truth,
        prediction,
        reference_transform=ref_t,
        hypothesis_transform=hyp_t,
    )
