from pathlib import Path

from src.fit.service import FitImageUseCase


def main():
    # --- dataset folder ---
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_path = base_dir / "ocr_train_dataset"

    # --- parameter grid (only upscale varies from 1.0 to 2.0 step 0.2) ---
    params_spec = {
        "contrast": [1.2],                     # fixed
        "denoise": [True],                     # fixed
        "denoise_strength": [15],              # fixed
        "denoise_template_window_size": [7],   # fixed
        "deskew": [True],                      # fixed
        "upscale_factor": (1.0, 2.0, 0.2),     # this is the one we vary
        "use_clahe": [True],                   # fixed
        "clahe_clip_limit": [2.0],             # fixed
        "clahe_tile_grid_size": [(8, 8)],      # fixed
    }

    # --- run fit ---
    usecase = FitImageUseCase()
    results = usecase.fit(str(dataset_path), params_spec)

    # --- print results ---
    print("\n=== Best parameters per image ===")
    for image_path, best_params in results:
        print(f"\nImage: {image_path}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
