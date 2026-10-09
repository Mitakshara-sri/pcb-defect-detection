from pathlib import Path

import cv2

# Resolve input and output locations relative to the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Final Dataset"
PROCESSED_DIR = BASE_DIR / "results" / "processed_images"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def preprocess_image(image):
    """Convert an image to grayscale and apply median and Gaussian filters."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
    return cv2.GaussianBlur(median, (5, 5), 1)


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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(DATASET_DIR)
        output_path = PROCESSED_DIR / relative_path

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"Could not read image: {image_path}")
            continue

        processed = preprocess_image(image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), processed):
            failures.append(f"Could not save processed image: {output_path}")
            continue

        print(f"Processed: {relative_path}")

    processed_count = len(image_paths) - len(failures)
    print(f"\nProcessed {processed_count} of {len(image_paths)} image(s).")
    print(f"Processed images saved under: {PROCESSED_DIR}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} image(s) could not be processed.")


if __name__ == "__main__":
    main()
