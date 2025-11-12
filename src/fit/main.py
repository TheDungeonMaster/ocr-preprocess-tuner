# main.py
from pathlib import Path

from src.fit.logger import SimpleLoggerReporter
from src.fit.service import FitImageUseCase


def main():
    base = Path(__file__).resolve().parent.parent.parent
    dataset_path = base / "ocr_train_dataset"

    params_spec = {
        "contrast": [1.2],
        "denoise": [True],
        "denoise_strength": [15],
        "denoise_template_window_size": [7],
        "deskew": [True],
        "upscale_factor": (1.0, 2.0, 0.2),   # <-- 1.0,1.2,1.4,1.6,1.8,2.0
        "use_clahe": [True],
        "clahe_clip_limit": [2.0],
        "clahe_tile_grid_size": [(8, 8)],
    }

    # Set limit=None to process all; set to an int for a quick smoke test
    reporter = SimpleLoggerReporter()  # logs per-run WER/CER/time at INFO
    model = FitImageUseCase(reporter=reporter)
    results = model.fit(str(dataset_path), params_spec, limit=None)

    print("\n=== Best parameters per image ===")
    for image_path, best_params in results:
        print(f"\nImage: {image_path}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
