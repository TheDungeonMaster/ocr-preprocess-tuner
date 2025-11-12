from __future__ import annotations

from src.logging import get_logger
from src.preprocess.model import PreprocessConfig
from src.preprocess.ports import ImagePreprocessorPort

logger = get_logger("PreprocessorService")


class PreprocessService:
    """
    Core-level use case for image preprocessing.
    Wraps a specific preprocessor (adapter) implementing ImagePreprocessorPort.
    """

    def __init__(self, preprocessor: type[ImagePreprocessorPort]):
        """
        Accepts the preprocessor *class* (adapter type), not instance,
        so we can construct it dynamically with config.
        """
        self._preprocessor_cls = preprocessor

    def run(self, image_bytes: bytes, config: PreprocessConfig) -> bytes:
        """
        Runs preprocessing pipeline on a single image (bytes → bytes).
        Accepts a PreprocessConfig object that controls parameters.
        """
        try:
            preprocessor = self._preprocessor_cls(**config.to_kwargs())
            return preprocessor.preprocess(image_bytes)
        except Exception as e:
            logger.error(f"PreprocessService.run failed: {e}")
            return image_bytes
