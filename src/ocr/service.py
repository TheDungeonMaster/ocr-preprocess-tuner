# core/ocr_service.py
from __future__ import annotations
from typing import Optional

from src import logging
from src.ocr.ports import OCREngine
logger = logging.get_logger(__name__)


class OCRService:
    """
    Minimal OCR service: accepts image bytes and returns recognized text.
    No batching, no thread pools; just delegates to the injected engine.
    """

    def __init__(self, engine: OCREngine, *, cap_native_threads: bool = True):
        """
        :param engine: Any implementation of OCREngine (e.g., TesseractOCREngine)
        :param cap_native_threads: If True, try to reduce native lib threads (OpenMP/OpenCV)
                                   to avoid oversubscription in larger apps.
        """
        self.engine = engine

        if cap_native_threads:
            # These are best-effort, harmless if unavailable.
            import os
            os.environ.setdefault("OMP_THREAD_LIMIT", "1")
            try:
                import cv2 as _cv2
                _cv2.setNumThreads(1)
            except Exception:
                pass

    def recognize(self, image_bytes: bytes, params: list) -> str:
        """
        Do OCR on a single image (bytes) and return text.
        Errors are logged and result is an empty string.
        """
        try:
            return self.engine.recognize(image_bytes, params)
        except Exception as e:
            logger.error(f"OCRService.recognize failed: {e}")
            return ""
