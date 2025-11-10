# cer.py
from typing import Optional
import jiwer
from normalization import transform_character as default_transform_character


def compute_cer(
    ground_truth: str,
    prediction: str,
    reference_transform: Optional[jiwer.Compose] = None,
    hypothesis_transform: Optional[jiwer.Compose] = None,
) -> float:
    """
    Compute Character Error Rate (CER).

    Guard rails mirror wer.compute_wer to avoid NaN when reference becomes empty.
    """
    ref_t = reference_transform or default_transform_character
    hyp_t = hypothesis_transform or default_transform_character

    ref_lists = ref_t(ground_truth)
    hyp_lists = hyp_t(prediction)
    ref_tokens = sum(len(seq) for seq in ref_lists)  # number of chars

    if ref_tokens == 0:
        return 0.0 if sum(len(seq) for seq in hyp_lists) == 0 else 1.0

    return jiwer.cer(
        ground_truth,
        prediction,
        reference_transform=ref_t,
        hypothesis_transform=hyp_t,
    )
