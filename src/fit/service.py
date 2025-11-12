from __future__ import annotations

from typing import Tuple, List, Dict, Any, Iterable

from itertools import product
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.ocr.service import OCRService
from src.ocr.tesseract_ocr import TesseractOCREngine

# Updated paths to match the new structure using the dataclass + class-based service


from src.parser.service import ParserService
from src.evaluate import DocumentPair
from src.preprocess.model import PreprocessConfig
from src.preprocess.pillow_preprocesser import PillowImagePreprocessor
from src.preprocess.service import PreprocessService


def _expand_spec(values: Any) -> Iterable[Any]:
    """
    Expand a parameter spec into a concrete iterable of values.

    Supported formats for each param:
      - Numeric range: (min, max, step)  -> inclusive grid via np.linspace
      - Discrete list: [v1, v2, ...]     -> used as-is (can include bools, None, tuples)
      - Single value:  v                  -> wrapped as [v]
    """
    # (mn, mx, step) numeric range
    if isinstance(values, tuple) and len(values) == 3 and all(isinstance(x, (int, float)) for x in values):
        mn, mx, step = values
        if step <= 0:
            raise ValueError("Range step must be > 0")
        n_steps = int(round((mx - mn) / step)) + 1
        return np.linspace(mn, mx, n_steps)
    # discrete list (including tuples / None / bools)
    if isinstance(values, list):
        return values
    # single literal
    return [values]


class FitImageUseCase:
    def __init__(self):
        pass

    def pipeline(self, ground_truth_path: str, image_path: str, param_map: Dict[str, Any]) -> Tuple[str, str]:
        """
        Runs preprocess -> OCR for a single (gt, image) pair using a param map
        compatible with PreprocessConfig(**param_map).
        Returns (ground_truth_text, predicted_text).
        """
        # 1) Read GT
        with open(ground_truth_path, "r", encoding="utf-8") as file:
            ground_truth_text = file.read().strip()

        # 2) Build config (dataclass validates all fields)
        #    param_map may omit fields; PreprocessConfig will use defaults.
        config = PreprocessConfig(**param_map)

        # 3) Preprocess
        preprocessor_service = PreprocessService(PillowImagePreprocessor)
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
        processed_img_bytes = preprocessor_service.run(raw_bytes, config)

        # 4) OCR
        engine = TesseractOCREngine(lang="eng+kaz+rus")
        ocr_service = OCRService(engine)
        predicted_text = ocr_service.recognize(processed_img_bytes)

        return ground_truth_text, predicted_text

    def generate_grid(self, params_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Builds a full Cartesian product grid from a param spec.

        params_spec: dict like
          {
            "contrast": (0.9, 1.5, 0.2),
            "denoise": [False, True],
            "denoise_strength": [10, 15, 20],
            "denoise_template_window_size": [3, 5, 7],   # odd ints >=3
            "deskew": [False, True],
            "upscale_factor": [None, 1.5, 2.0],          # Optional
            "use_clahe": [False, True],
            "clahe_clip_limit": (1.0, 3.0, 1.0),
            "clahe_tile_grid_size": [(8, 8), (12, 12)],  # tuple choices
          }

        Returns: list of dicts mapping param name -> value for each combination.
        """
        keys = list(params_spec.keys())
        value_lists: List[Iterable[Any]] = [list(_expand_spec(params_spec[k])) for k in keys]

        grid: List[Dict[str, Any]] = []
        for combo in product(*value_lists):
            combo_map = {k: v for k, v in zip(keys, combo)}
            grid.append(combo_map)
        return grid

    def evaluate_pair(self, grid: List[Dict[str, Any]], document_pair: List[str]) -> Tuple[List[float], List[float]]:
        """
        Calculates WER and CER for a single (gt_path, image_path) pair across the grid.
        document_pair: [ground_truth_path, image_path]
        Returns (wers, cers) aligned to grid order.
        """
        gt_path, img_path = document_pair[0], document_pair[1]
        wers: List[float] = []
        cers: List[float] = []

        for param_map in grid:
            ground_truth_text, predicted_text = self.pipeline(gt_path, img_path, param_map)
            doc = DocumentPair(ground_truth_text, predicted_text)
            wer = doc.compute_wer()
            cer = doc.compute_cer()
            wers.append(round(wer, 4))
            cers.append(round(cer, 4))

        return wers, cers

    def fit(self, dataset_path: str, params_spec: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        For each parsed (gt, image) pair, searches the grid and returns the best params per image.
        Primary sort: CER, secondary tie-breaker: WER.
        Returns a list of (image_path, best_param_map).
        """
        parser = ParserService()
        document_pairs = parser.parse(dataset_path)

        grid = self.generate_grid(params_spec)

        image_and_best_parameters: List[Tuple[str, Dict[str, Any]]] = []

        # Example keeps only first item as in your original code (i == 1 break):
        # Remove the early-break if you want to process all pairs.
        for i, document_pair in enumerate(document_pairs):
            if i == 1:
                break

            wers, cers = self.evaluate_pair(grid, document_pair)

            # lexsort uses last key as primary, so (wers, cers) -> CER primary, WER secondary
            order = np.lexsort((np.array(wers), np.array(cers)))
            best_idx = int(order[0])

            image_path = document_pair[1]
            best_params = grid[best_idx]

            image_and_best_parameters.append((image_path, best_params))

        return image_and_best_parameters
