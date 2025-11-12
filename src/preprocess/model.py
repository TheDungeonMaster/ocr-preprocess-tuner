from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

@dataclass(frozen=True)
class PreprocessConfig:
    contrast: float = 1.0
    denoise: bool = False
    denoise_strength: int = 10
    denoise_template_window_size: int = 7  # odd, >=3
    deskew: bool = False
    upscale_factor: Optional[float] = None

    # CLAHE
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = field(default_factory=lambda: (8, 8))

    def __post_init__(self) -> None:
        if self.contrast <= 0:
            raise ValueError("contrast must be > 0")
        if self.denoise:
            if self.denoise_strength <= 0:
                raise ValueError("denoise_strength must be > 0 when denoise=True")
            if self.denoise_template_window_size < 3 or self.denoise_template_window_size % 2 == 0:
                raise ValueError("denoise_template_window_size must be odd and >= 3")
        if self.upscale_factor is not None and self.upscale_factor <= 0:
            raise ValueError("upscale_factor must be > 0 when provided")
        if self.use_clahe:
            if self.clahe_clip_limit <= 0:
                raise ValueError("clahe_clip_limit must be > 0 when use_clahe=True")
            if min(self.clahe_tile_grid_size) < 1 or len(self.clahe_tile_grid_size) != 2:
                raise ValueError("clahe_tile_grid_size must be a 2-tuple of ints >= 1")

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(
            contrast=self.contrast,
            denoise=self.denoise,
            denoise_strength=self.denoise_strength,
            denoise_template_window_size=self.denoise_template_window_size,
            deskew=self.deskew,
            upscale_factor=self.upscale_factor,
            use_clahe=self.use_clahe,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid_size=self.clahe_tile_grid_size,
        )
