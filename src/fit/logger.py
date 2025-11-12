# src/reporting/simple_logger.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import logging

from .reporter import Reporter

_log = logging.getLogger("FitImageUseCase")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

def _fmt(pm: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={pm[k]!r}" for k in sorted(pm.keys()))

class SimpleLoggerReporter(Reporter):
    def __init__(self):
        _log.setLevel(logging.INFO)

    def on_parsed(self, images: List[str]) -> None:
        _log.info(f"Parsed {len(images)} document pair(s).")

    def on_grid_built(self, n_grid: int, total_runs: int) -> None:
        _log.info(f"Parameter combinations: {n_grid}")
        _log.info(f"Total runs (pairs × combos): {total_runs}")

    def on_finish(self) -> None:
        _log.info("Fitting finished.")

    def on_image_start(self, image_name: str, n_combos: int) -> None:
        _log.info(f"Processing {Path(image_name).name} with {n_combos} combinations...")

    def on_run_result(self, image_name: str, run_idx: int, n_combos: int,
                      params: Dict[str, Any], wer: float, cer: float, seconds: float) -> None:
        _log.info(
            f"[{Path(image_name).name}] run {run_idx}/{n_combos} "
            f"WER={wer:.6f} CER={cer:.6f} time={seconds:.2f}s | {_fmt(params)}"
        )

    def on_image_best(self, image_path: str, best_params: Dict[str, Any],
                      best_wer: float, best_cer: float) -> None:
        _log.info(
            f"[best] {Path(image_path).name}: CER={best_cer:.6f}, WER={best_wer:.6f} | {_fmt(best_params)}"
        )
