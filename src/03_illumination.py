from pathlib import Path

import cv2

# Resolve input and output locations relative to the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Final Dataset"
PROCESSED_DIR = BASE_DIR / "results" / "illumination_processed_images"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def correct_illumination(image):
    """Return illumination-corrected and CLAHE-enhanced grayscale images."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    corrected = cv2.divide(gray, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)
    return corrected, enhanced


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    image_paths = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {DATASET_DIR}")

    failures = []
    processed_count = 0
    for image_path in image_paths:
        relative_path = image_path.relative_to(DATASET_DIR)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"Could not read image: {image_path}")
            continue

        corrected, enhanced = correct_illumination(image)
        for stage, processed in (("corrected", corrected), ("clahe_enhanced", enhanced)):
            output_path = PROCESSED_DIR / stage / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), processed):
                failures.append(f"Could not save processed image: {output_path}")
                break
        else:
            processed_count += 1
            print(f"Processed: {relative_path}")

    print(f"\nProcessed {processed_count} of {len(image_paths)} image(s).")
    print(f"Corrected and CLAHE-enhanced images saved under: {PROCESSED_DIR}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} image output(s) could not be processed.")


if __name__ == "__main__":
    main()
