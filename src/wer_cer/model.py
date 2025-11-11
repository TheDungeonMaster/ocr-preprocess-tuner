# model.py
from dataclasses import dataclass

from wer import compute_wer
from cer import compute_cer
from normalization import transform_word, transform_character


@dataclass
class DocumentPair:
    ground_truth: str
    prediction: str

    def compute_wer(self) -> float:
        return compute_wer(
            self.ground_truth.strip(),
            self.prediction.strip(),
            reference_transform=transform_word,
            hypothesis_transform=transform_word,
        )

    def compute_cer(self) -> float:
        return compute_cer(
            self.ground_truth.strip(),
            self.prediction.strip(),
            reference_transform=transform_character,
            hypothesis_transform=transform_character,
        )
