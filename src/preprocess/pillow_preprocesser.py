# adapters/pillow_preprocessor.py
import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from src.logging import get_logger

logger = get_logger("PillowImagePreprocessor")


class PillowImagePreprocessor:
    """
    Concrete implementation of ImagePreprocessorPort using PIL + OpenCV.
    Single-image API: bytes → bytes (PNG).
    """

    def __init__(
        self,
        *,
        contrast: float,
        denoise: bool,
        denoise_strength: int,
        denoise_template_window_size: int,
        deskew: bool = False,
        upscale_factor: Optional[float],
        # Optional: CLAHE
        use_clahe: bool = False,
        clahe_clip_limit: float ,
        clahe_tile_grid_size: Tuple[int, int],
    ):
        self.contrast = contrast
        self.denoise = denoise
        self.denoise_strength = denoise_strength
        self.denoise_template_window_size = denoise_template_window_size
        self.deskew = deskew
        self.upscale_factor = upscale_factor

        self.use_clahe = use_clahe
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = tuple(clahe_tile_grid_size)

    # ---------- internal helpers ----------

    def _enhance_contrast(self, pil_image: Image.Image) -> Image.Image:
        if self.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            return enhancer.enhance(self.contrast)
        return pil_image

    def _apply_denoise(self, image: np.ndarray) -> np.ndarray:
        # Works best on single-channel; acceptable per-channel for 3-channel images.
        return cv2.fastNlMeansDenoising(
            image,
            h=self.denoise_strength,
            templateWindowSize=self.denoise_template_window_size,
        )

    def _upscale(self, image: np.ndarray) -> np.ndarray:
        if self.upscale_factor and self.upscale_factor > 1.0:
            orig_h, orig_w = image.shape[:2]
            new_w = int(orig_w * self.upscale_factor)
            new_h = int(orig_h * self.upscale_factor)
            new_pixels = new_w * new_h

            MAX_PIXELS = 150_000_000  # Safety limit
            if new_pixels > MAX_PIXELS:
                logger.warning(
                    f"Skipping upscale: target [{new_w}x{new_h}] exceeds {MAX_PIXELS} pixels"
                )
                return image

            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return image

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on L channel (LAB) for RGB, or directly for grayscale.
        """
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)

        if image.ndim == 2:
            # Grayscale
            return clahe.apply(image)

        if image.ndim == 3 and image.shape[2] == 3:
            # RGB → LAB, apply CLAHE on L
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            L, A, B = cv2.split(lab)
            L_eq = clahe.apply(L)
            lab_eq = cv2.merge([L_eq, A, B])
            rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
            return rgb

        return image

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale from RGB if needed
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        h, w = thresh.shape
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            logger.warning("Insufficient contours for deskew; skipping")
            return image

        angles = np.arange(-10, 10, 0.1)
        max_score = 0.0
        best_angle = 0.0

        for angle in angles:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST)
            proj = np.sum(rotated, axis=1)
            score = float(np.sum(proj ** 2))
            if score > max_score:
                max_score = score
                best_angle = angle

        if abs(best_angle) < 0.5:
            return image

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # ---------- main API ----------

    def preprocess(self, image_bytes: bytes) -> bytes:
        """
        Perform preprocessing on a single image.
        Input: raw image bytes
        Output: preprocessed PNG bytes
        """
        try:
            # Load with PIL (respects EXIF orientation); keep RGB
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Contrast (PIL domain)
            pil_img = self._enhance_contrast(pil_img)

            # PIL → NumPy (RGB)
            img = np.array(pil_img)

            # Upscale
            img = self._upscale(img)

            # CLAHE (optional)
            if self.use_clahe:
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                img = self._apply_clahe(img)

            # Denoise (optional)
            if self.denoise:
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                img = self._apply_denoise(img)

            # Deskew (optional)
            if self.deskew:
                img = self._deskew(img)

            # Encode as PNG (RGB)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, "PNG", compress_level=1, dpi=(300, 300), optimize=False)
            return buf.getvalue()

        except Exception:
            logger.exception("Failed to preprocess image")
            return image_bytes
