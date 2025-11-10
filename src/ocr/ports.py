from typing import Protocol


class OCREngine(Protocol):
    """
    OCR Engine interface (port).
    Implementations must accept image bytes and return plain text (no bboxes).
    """
    def recognize(self, image_bytes: bytes) -> str:
        ...
