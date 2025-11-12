# main.py
from pathlib import Path

from src.fit.logger import SimpleLoggerReporter
from src.fit.service import FitImageUseCase


def main():
    base = Path(__file__).resolve().parent.parent.parent
    dataset_path = base / "ocr_train_dataset"

    params_spec = {
        "contrast": [1.2],
        "denoise": [False],
        "denoise_strength": [7],
        "denoise_template_window_size": [7],
        "deskew": [False],
        "upscale_factor": (1.0, 2.0, 0.2),   # 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
        "use_clahe": [False],
        "clahe_clip_limit": [2.0],
        "clahe_tile_grid_size": [(8, 8)],
    }

    # --- Configure reporter and use case ---
    reporter = SimpleLoggerReporter()  # Logs per-run WER/CER/time at INFO
    model = FitImageUseCase(reporter=reporter, n_jobs=10)  # <-- parallel workers set to 6

    # --- Run grid search ---
    results = model.fit(str(dataset_path), params_spec, limit= None, debug_dir="fit_debug")

    # --- Print summary ---
    print("\n=== Best parameters per image ===")
    for image_path, best_params in results:
        print(f"\nImage: {image_path}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
