# examples/run_preprocessor_example.py
import os
from pathlib import Path

from src.preprocess.pillow_preprocesser import PillowImagePreprocessor


def main():
    # === 1️⃣ Setup paths ===
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "images"      # Folder with your input images
    output_dir = base_dir / "imgs_temp"  # Where results and pairs will be saved
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 2️⃣ Verify input directory ===
    if not input_dir.exists():
        print(f"[!] Input directory not found: {input_dir}")
        print("💡 Create it and put some PNG/JPG files there.")
        return

    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    if not image_paths:
        print(f"[!] No PNG/JPG images found in {input_dir}")
        return

    print(f"[INFO] Found {len(image_paths)} images to preprocess.")
    print(f"[INFO] Results will be saved to: {output_dir}\n")


    # ===  Initialize the preprocessor ===
    preprocessor = PillowImagePreprocessor(
        contrast=1.4,
        denoise=True,
        deskew=True,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        upscale_factor=2.0,
        save_pairs=True,
        save_dir=str(output_dir),
    )

    # ===  Loop over all images ===
    for img_path in image_paths:
        try:
            print(f"[INFO] Processing: {img_path.name}")
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            processed_bytes = preprocessor.preprocess(image_bytes, save_name=img_path.stem)

            # Optionally save the returned image (postprocessed result)
            result_path = output_dir / f"{img_path.stem}_final.png"
            with open(result_path, "wb") as f:
                f.write(processed_bytes)

        except Exception as e:
            print(f"[ERROR] Failed on {img_path.name}: {e}")

    # ===Summary ===
    print("\n✅ Preprocessing complete!")
    print(f"📂 Results saved to: {output_dir}")

    print("\n🗂 Saved files:")
    for file in sorted(output_dir.iterdir()):
        print(f"  {file.name}")


if __name__ == "__main__":
    main()
