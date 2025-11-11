from pathlib import Path

from src.preprocess.pillow_preprocesser import PillowImagePreprocessor
from src.preprocess.service import PreprocessService


def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "imgs"
    output_dir = base_dir / "imgs_temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[!] Input directory does not exist: {input_dir}")
        return

    exts = {".png", ".jpg", ".jpeg"}
    image_paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not image_paths:
        print(f"[!] No images found in {input_dir} (supported: {sorted(exts)})")
        return

    # --- create the adapter (implementation) ---
    # NOTE: PillowImagePreprocessor no longer supports saving pairs; args updated accordingly.
    pillow_impl = PillowImagePreprocessor(
        contrast=1.4,
        denoise=True,
        deskew=True,
        use_clahe=True,
        # clahe_clip_limit=2.0,           # optional, defaults are fine
        # clahe_tile_grid_size=(8, 8),    # optional
        # upscale_factor=2.0,             # optional
    )

    # --- inject it into the service (use case) ---
    preprocessor_service = PreprocessService(pillow_impl)

    print(f"[INFO] Found {len(image_paths)} images in {input_dir}")

    processed = 0
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                raw_bytes = f.read()

            processed_bytes = preprocessor_service.run(raw_bytes)

            # Save result (always PNG)
            result_path = output_dir / f"{img_path.stem}_final.png"
            with open(result_path, "wb") as f:
                f.write(processed_bytes)

            print(f"[OK] Processed: {img_path.name} -> {result_path.name}")
            processed += 1
        except Exception as e:
            print(f"[ERROR] Failed on {img_path.name}: {e}")

    print(f"\n✅ All preprocessing complete. {processed}/{len(image_paths)} files written to {output_dir}")


if __name__ == "__main__":
    main()
