# adapters/tesseract_ocr.py
from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import cv2
import pytesseract
from pytesseract import Output

from src import logging

logger = logging.get_logger(__name__)

class TesseractOCREngine:
    """
    Tesseract-backed OCR engine that implements the OCREngine port.
    - Accepts image bytes
    - Returns a single string with line breaks
    - Internally groups words into lines using (block, paragraph, line) keys
    """

    def __init__(self, lang: str = "eng+kaz+rus"):
        self.lang = lang
        logger.info(f"TesseractOCREngine initialized with lang={self.lang}")

    def recognize(self, image_bytes: bytes) -> str:
        """
        Run OCR and return text only (line-broken).
        """
        try:
            # Decode bytes → OpenCV image (BGR)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image from bytes")
            
       
            # Word-level data
            data = pytesseract.image_to_data(image,  output_type=Output.DICT, lang=self.lang)



            lines = defaultdict(list)
            n = len(data["text"])

            for i in range(n):
                txt = data["text"][i].strip()
                if not txt:
                    continue
                key: Tuple[int, int, int] = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines[key].append(txt)

            # Sort lines by (block, paragraph, line) to ensure deterministic order
            ordered_keys = sorted(lines.keys())
            line_texts: List[str] = [" ".join(lines[k]) for k in ordered_keys]

            # Final plain text, separated by newlines
            return "\n".join(line_texts).strip()

        except Exception as e:
            logger.error(f"Tesseract recognize() failed: {e}")
            return ""
