from __future__ import annotations
from typing import Dict, Any, List, Protocol

class Reporter(Protocol):
    def on_parsed(self, images: List[str]) -> None: ...
    def on_grid_built(self, n_grid: int, total_runs: int) -> None: ...
    def on_finish(self) -> None: ...

    def on_image_start(self, image_name: str, n_combos: int) -> None: ...
    def on_run_result(self, image_name: str, run_idx: int, n_combos: int,
                      params: Dict[str, Any], wer: float, cer: float, seconds: float) -> None: ...
    def on_image_best(self, image_path: str, best_params: Dict[str, Any],
                      best_wer: float, best_cer: float) -> None: ...

class NoOpReporter:
    def on_parsed(self, images: List[str]) -> None: pass
    def on_grid_built(self, n_grid: int, total_runs: int) -> None: pass
    def on_finish(self) -> None: pass
    def on_image_start(self, image_name: str, n_combos: int) -> None: pass
    def on_run_result(self, image_name: str, run_idx: int, n_combos: int,
                      params: Dict[str, Any], wer: float, cer: float, seconds: float) -> None: pass
    def on_image_best(self, image_path: str, best_params: Dict[str, Any],
                      best_wer: float, best_cer: float) -> None: pass
