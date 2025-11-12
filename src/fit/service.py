# src/usecases/fit_image_usecase.py
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Iterable, Optional
from itertools import product
from pathlib import Path
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import json
import csv
import shutil

from src.fit.reporter import Reporter, NoOpReporter
from src.ocr.service import OCRService
from src.ocr.tesseract_ocr import TesseractOCREngine
from src.parser.service import ParserService
from src.evaluate import DocumentPair

from src.preprocess.model import PreprocessConfig
from src.preprocess.pillow_preprocesser import PillowImagePreprocessor
from src.preprocess.service import PreprocessService


def _expand_spec(values: Any):
    if isinstance(values, tuple) and len(values) == 3 and all(isinstance(x, (int, float)) for x in values):
        mn, mx, step = values
        if step <= 0:
            raise ValueError("Range step must be > 0")
        n_steps = int(round((mx - mn) / step)) + 1
        return np.linspace(mn, mx, n_steps)
    if isinstance(values, list):
        return values
    return [values]


# --------------------------- Worker function (picklable) ---------------------------

def _evaluate_combo_worker(idx: int, param_map: Dict[str, Any], gt_text: str, img_bytes: bytes) -> Tuple[int, float, float, float]:
    """
    Runs preprocess -> OCR -> metrics for a single parameter map.
    Returns (idx, wer, cer, dt).
    Executed in a separate process.
    """
    t0 = time.perf_counter()

    # Build config and preprocess
    config = PreprocessConfig(**param_map)
    preprocessor_service = PreprocessService(PillowImagePreprocessor)
    processed_img_bytes = preprocessor_service.run(img_bytes, config)

    # OCR
    engine = TesseractOCREngine(lang="eng+kaz+rus")
    ocr_service = OCRService(engine)
    predicted_text = ocr_service.recognize(processed_img_bytes)

    # Evaluate
    doc = DocumentPair(gt_text, predicted_text)
    wer = float(doc.compute_wer())
    cer = float(doc.compute_cer())

    dt = time.perf_counter() - t0
    return idx, wer, cer, dt


class FitImageUseCase:
    """
    Orchestrates parameter search for optimal image preprocessing configuration.
    """

    def __init__(self, reporter: Optional[Reporter] = None, n_jobs: Optional[int] = None, chunksize: int = 1):
        """
        Args:
            reporter: optional progress reporter
            n_jobs: number of worker processes (default: os.cpu_count() or 1)
            chunksize: future submission chunk size (kept for API symmetry if you later switch to map())
        """
        self.reporter: Reporter = reporter or NoOpReporter()
        self.n_jobs = max(1, int(n_jobs or (os.cpu_count() or 1)))
        self.chunksize = max(1, int(chunksize))

    # -------------------------------------------------------------------------
    # Core pipeline (single run) - kept for synchronous execution when needed
    # -------------------------------------------------------------------------
    def pipeline(self, ground_truth_path: str, image_path: str, param_map: Dict[str, Any]) -> Tuple[str, str]:
        with open(ground_truth_path, "r", encoding="utf-8") as file:
            ground_truth_text = file.read().strip()

        config = PreprocessConfig(**param_map)

        preprocessor_service = PreprocessService(PillowImagePreprocessor)
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
        processed_img_bytes = preprocessor_service.run(raw_bytes, config)

        engine = TesseractOCREngine(lang="eng+kaz+rus")
        ocr_service = OCRService(engine)
        predicted_text = ocr_service.recognize(processed_img_bytes)

        return ground_truth_text, predicted_text

    # -------------------------------------------------------------------------
    # Grid generation
    # -------------------------------------------------------------------------
    def generate_grid(self, params_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        keys = list(params_spec.keys())
        value_lists: List[Iterable[Any]] = [list(_expand_spec(params_spec[k])) for k in keys]
        return [{k: v for k, v in zip(keys, combo)} for combo in product(*value_lists)]

    # -------------------------------------------------------------------------
    # Evaluation (single image) - PARALLEL
    # -------------------------------------------------------------------------
    def evaluate_pair(self, grid: List[Dict[str, Any]], document_pair: List[str], image_name: str) -> Tuple[List[float], List[float], List[float]]:
        """
        Evaluates all parameter combinations for one image, in parallel.
        Returns three lists (WERs, CERs, DTs) aligned to grid order.
        """
        gt_path, img_path = document_pair[0], document_pair[1]
        n_combos = len(grid)

        # Notify reporter before starting
        self.reporter.on_image_start(image_name, n_combos)

        # Read inputs once in parent and pass to workers to reduce disk contention
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        wers: List[Optional[float]] = [None] * n_combos
        cers: List[Optional[float]] = [None] * n_combos
        dts:  List[Optional[float]] = [None] * n_combos

        # Submit tasks
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for idx, param_map in enumerate(grid):
                fut = executor.submit(_evaluate_combo_worker, idx, param_map, gt_text, img_bytes)
                futures.append((idx, param_map, fut))

            # As each finishes, collect and report
            for idx, param_map, fut in futures:
                i, wer, cer, dt = fut.result()
                wers[i] = round(wer, 6)
                cers[i] = round(cer, 6)
                dts[i]  = round(dt, 6)
                # Safe to report from main proc
                self.reporter.on_run_result(image_name, i + 1, n_combos, param_map, wer, cer, dt)

        # Type narrowing
        return [float(x) for x in wers], [float(x) for x in cers], [float(x) for x in dts]

    # -------------------------------------------------------------------------
    # Fit procedure (dataset-level)
    # -------------------------------------------------------------------------
    def fit(
        self,
        dataset_path: str,
        params_spec: Dict[str, Any],
        limit: Optional[int] = None,
        debug_dir: Optional[str] = None,  # <--- NEW
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Runs the grid search across the dataset.

        If debug_dir is provided, for each image we save:
          - raw input image (as original filename)
          - best processed image (processed_best.png)
          - ground_truth.txt
          - predicted.txt (for best params)
          - best.json (params, WER/CER)
          - grid_stats.csv (row per combo: idx, wer, cer, dt_sec, <params...>)
        """
        parser = ParserService()
        document_pairs = parser.parse(dataset_path)
        if limit is not None:
            document_pairs = document_pairs[:limit]

        images = [dp[1] for dp in document_pairs]
        self.reporter.on_parsed(images)

        grid = self.generate_grid(params_spec)
        total_runs = len(grid) * max(len(document_pairs), 1)
        self.reporter.on_grid_built(len(grid), total_runs)

        results: List[Tuple[str, Dict[str, Any]]] = []

        # Prepare debug root once
        debug_root = Path(debug_dir).resolve() if debug_dir else None
        if debug_root:
            debug_root.mkdir(parents=True, exist_ok=True)

        for document_pair in document_pairs:
            gt_path, image_path = document_pair
            image_name = Path(image_path).name
            image_stem = Path(image_path).stem

            wers, cers, dts = self.evaluate_pair(grid, document_pair, image_name)

            # Select best: lowest CER, then WER
            order = np.lexsort((np.array(wers), np.array(cers)))
            best_idx = int(order[0])
            best_params = grid[best_idx]
            best_wer = float(wers[best_idx])
            best_cer = float(cers[best_idx])

            self.reporter.on_image_best(image_path, best_params, best_wer, best_cer)
            results.append((image_path, best_params))

            # -------------------- DEBUG ARTIFACTS --------------------
            if debug_root:
                per_img_dir = debug_root / image_stem
                per_img_dir.mkdir(parents=True, exist_ok=True)

                # Copy raw input image and ground truth
                try:
                    # raw image
                    shutil.copy2(image_path, per_img_dir / Path(image_name))
                except Exception:
                    # fallback: read/write
                    with open(image_path, "rb") as rf, open(per_img_dir / Path(image_name), "wb") as wf:
                        wf.write(rf.read())

                # ground truth text
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_text = f.read().strip()
                (per_img_dir / "ground_truth.txt").write_text(gt_text, encoding="utf-8")

                # Re-run preprocessing+OCR for the best params to save artifacts deterministically
                with open(image_path, "rb") as f:
                    raw_bytes = f.read()
                config = PreprocessConfig(**best_params)
                preprocessor_service = PreprocessService(PillowImagePreprocessor)
                processed_img_bytes = preprocessor_service.run(raw_bytes, config)

                engine = TesseractOCREngine(lang="eng+kaz+rus")
                ocr_service = OCRService(engine)
                predicted_text = ocr_service.recognize(processed_img_bytes)

                # Save processed image (PNG) & predicted text
                (per_img_dir / "processed_best.png").write_bytes(processed_img_bytes)
                (per_img_dir / "predicted.txt").write_text(predicted_text, encoding="utf-8")

                # Save compact best summary
                best_summary = {
                    "image": image_name,
                    "best_index": best_idx,
                    "best_params": best_params,
                    "best_wer": best_wer,
                    "best_cer": best_cer,
                }
                (per_img_dir / "best.json").write_text(json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8")

                # Save full grid stats as CSV
                stats_csv = per_img_dir / "grid_stats.csv"
                fieldnames = ["index", "wer", "cer", "dt_sec"] + list(grid[0].keys())
                with open(stats_csv, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for i, (w, c, dt) in enumerate(zip(wers, cers, dts)):
                        row = {"index": i, "wer": w, "cer": c, "dt_sec": dt}
                        row.update(grid[i])
                        writer.writerow(row)
                # -------------------------------------------------------

        self.reporter.on_finish()
        return results
