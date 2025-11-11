# ports/preprocessor_port.py
from typing import Protocol


class ImagePreprocessorPort(Protocol):
    """
    Preprocessing capability: takes image bytes, returns preprocessed image bytes.
    """
    def preprocess(self, image_bytes: bytes) -> bytes: ...
