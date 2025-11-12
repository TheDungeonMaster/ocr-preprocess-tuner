# src/usecases/fit_image_usecase.py
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Iterable, Optional
from itertools import product
from pathlib import Path
import time
import numpy as np

from src.fit.reporter import Reporter, NoOpReporter
from src.ocr.service import OCRService
from src.ocr.tesseract_ocr import TesseractOCREngine
from src.parser.service import ParserService
from src.evaluate import DocumentPair

from src.preprocess.model import PreprocessConfig
from src.preprocess.pillow_preprocesser import PillowImagePreprocessor
from src.preprocess.service import PreprocessService


def _expand_spec(values: Any):
    """
    Expands a single parameter specification into a list of concrete values.

    Supported formats:
        - (min, max, step): numeric range → np.linspace(min, max, step)
        - [v1, v2, ...]: discrete list of values
        - single value: converted into [value]
    """
    if isinstance(values, tuple) and len(values) == 3 and all(isinstance(x, (int, float)) for x in values):
        mn, mx, step = values
        if step <= 0:
            raise ValueError("Range step must be > 0")
        n_steps = int(round((mx - mn) / step)) + 1
        return np.linspace(mn, mx, n_steps)
    if isinstance(values, list):
        return values
    return [values]


class FitImageUseCase:
    """
    Orchestrates parameter search for optimal image preprocessing configuration.

    Responsibilities:
        - Iterates over parameter combinations (grid search)
        - Runs preprocessing → OCR → evaluation
        - Reports progress and best-performing configurations per image
    """

    def __init__(self, reporter: Optional[Reporter] = None):
        """
        Initialize the use case with an optional reporter.

        Args:
            reporter: Optional progress reporter (e.g. SimpleLoggerReporter).
                      Defaults to a no-op reporter if not provided.
        """
        self.reporter: Reporter = reporter or NoOpReporter()

    # -------------------------------------------------------------------------
    # Core pipeline (single run)
    # -------------------------------------------------------------------------
    def pipeline(self, ground_truth_path: str, image_path: str, param_map: Dict[str, Any]) -> Tuple[str, str]:
        """
        Executes the full pipeline for a single parameter set:
        1. Load ground truth
        2. Preprocess the image with specified parameters
        3. Run OCR
        4. Return (ground_truth_text, predicted_text)
        """
        # --- Load reference text (ground truth) ---
        with open(ground_truth_path, "r", encoding="utf-8") as file:
            ground_truth_text = file.read().strip()

        # --- Build validated config ---
        config = PreprocessConfig(**param_map)

        # --- Run preprocessing ---
        preprocessor_service = PreprocessService(PillowImagePreprocessor)
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
        processed_img_bytes = preprocessor_service.run(raw_bytes, config)

        # --- Perform OCR ---
        engine = TesseractOCREngine(lang="eng+kaz+rus")
        ocr_service = OCRService(engine)
        predicted_text = ocr_service.recognize(processed_img_bytes)

        return ground_truth_text, predicted_text

    # -------------------------------------------------------------------------
    # Grid generation
    # -------------------------------------------------------------------------
    def generate_grid(self, params_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Builds a Cartesian product of parameter combinations.

        Args:
            params_spec: dictionary mapping parameter name → values, ranges, or lists

        Example:
            {
                "contrast": (0.9, 1.5, 0.2),
                "denoise": [False, True],
                "upscale_factor": [None, 1.5, 2.0]
            }

        Returns:
            List of parameter dictionaries (one per combination).
        """
        keys = list(params_spec.keys())
        value_lists: List[Iterable[Any]] = [list(_expand_spec(params_spec[k])) for k in keys]
        return [{k: v for k, v in zip(keys, combo)} for combo in product(*value_lists)]

    # -------------------------------------------------------------------------
    # Evaluation (single image)
    # -------------------------------------------------------------------------
    def evaluate_pair(self, grid: List[Dict[str, Any]], document_pair: List[str], image_name: str) -> Tuple[List[float], List[float]]:
        """
        Evaluates all parameter combinations for one image.

        Args:
            grid: list of parameter dictionaries
            document_pair: [ground_truth_path, image_path]
            image_name: human-readable image identifier (for reporting)

        Returns:
            Two lists (WERs, CERs), aligned to grid order.
        """
        gt_path, img_path = document_pair[0], document_pair[1]
        wers: List[float] = []
        cers: List[float] = []
        n_combos = len(grid)

        # Notify reporter before starting image evaluation
        self.reporter.on_image_start(image_name, n_combos)

        for idx, param_map in enumerate(grid, start=1):
            t0 = time.perf_counter()

            # --- Run pipeline with current params ---
            gt_text, pred_text = self.pipeline(gt_path, img_path, param_map)

            # --- Evaluate performance ---
            doc = DocumentPair(gt_text, pred_text)
            wer = float(doc.compute_wer())
            cer = float(doc.compute_cer())
            dt = time.perf_counter() - t0

            # --- Record and report ---
            wers.append(round(wer, 6))
            cers.append(round(cer, 6))
            self.reporter.on_run_result(image_name, idx, n_combos, param_map, wer, cer, dt)

        return wers, cers

    # -------------------------------------------------------------------------
    # Fit procedure (dataset-level)
    # -------------------------------------------------------------------------
    def fit(self, dataset_path: str, params_spec: Dict[str, Any], limit: Optional[int] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Executes a full grid search over preprocessing parameters.

        Workflow:
            1. Parse dataset into (ground_truth, image) pairs
            2. Build parameter grid
            3. Evaluate each image for all combinations
            4. Select the best-performing config per image

        Args:
            dataset_path: path to dataset root (parsed by ParserService)
            params_spec: parameter specification dict
            limit: optional limit on number of document pairs to process

        Returns:
            List of tuples: (image_path, best_param_dict)
        """
        # --- Step 1: Parse dataset ---
        parser = ParserService()
        document_pairs = parser.parse(dataset_path)
        if limit is not None:
            document_pairs = document_pairs[:limit]

        # Report parsed images
        images = [dp[1] for dp in document_pairs]
        self.reporter.on_parsed(images)

        # --- Step 2: Generate parameter grid ---
        grid = self.generate_grid(params_spec)
        total_runs = len(grid) * max(len(document_pairs), 1)
        self.reporter.on_grid_built(len(grid), total_runs)

        # --- Step 3: Run evaluations ---
        results: List[Tuple[str, Dict[str, Any]]] = []
        for document_pair in document_pairs:
            image_path = document_pair[1]
            image_name = Path(image_path).name

            wers, cers = self.evaluate_pair(grid, document_pair, image_name)

            # --- Step 4: Select best configuration (lowest CER → WER) ---
            order = np.lexsort((np.array(wers), np.array(cers)))
            best_idx = int(order[0])
            best_params = grid[best_idx]
            best_wer = float(wers[best_idx])
            best_cer = float(cers[best_idx])

            self.reporter.on_image_best(image_path, best_params, best_wer, best_cer)
            results.append((image_path, best_params))

        # Notify completion
        self.reporter.on_finish()
        return results
