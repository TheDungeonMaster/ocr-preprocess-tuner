# core/preprocess_service.py
from __future__ import annotations

from src.logging import get_logger
from src.preprocess.ports import ImagePreprocessorPort

logger = get_logger("PreprocessorService")

class PreprocessService:
    """
    Core-level use case for image preprocessing.
    Wraps a specific preprocessor (adapter) implementing ImagePreprocessorPort.
    """

    def __init__(self, preprocessor: ImagePreprocessorPort):
        self.preprocessor = preprocessor

    def run(self, image_bytes: bytes) -> bytes:
        """
        Runs preprocessing pipeline on a single image (bytes → bytes).
        """
        try:
            return self.preprocessor.preprocess(image_bytes)
        except Exception as e:
            logger.error(f"PreprocessService.run failed: {e}")
            return image_bytes
