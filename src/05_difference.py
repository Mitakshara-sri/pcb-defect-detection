from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from illumination_utils import normalize_illumination

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PATH = BASE_DIR / "reference.jpg"
ALIGNED_DIR = BASE_DIR / "results" / "aligned_images"
RESULTS_DIR = BASE_DIR / "results" / "difference_images"
PREVIEW_PATH = BASE_DIR / "results" / "difference_preview.jpg"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DIFFERENCE_THRESHOLD = 30


def compute_difference(reference, aligned):
    """Return an illumination-normalized difference and binary defect mask."""
    if reference.shape != aligned.shape:
        raise ValueError(
            f"Aligned image size {aligned.shape} does not match reference "
            f"size {reference.shape}."
        )

    normalized_reference = normalize_illumination(reference)
    normalized_aligned = normalize_illumination(aligned)
    difference_blurred = cv2.GaussianBlur(
        cv2.absdiff(normalized_reference, normalized_aligned), (3, 3), 0
    )
    defect_mask = cv2.threshold(
        difference_blurred, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY
    )[1]
    return difference_blurred, defect_mask


def main():
    reference = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError(f"Reference image not found: {REFERENCE_PATH}")
    if not ALIGNED_DIR.is_dir():
        raise FileNotFoundError(
            f"Aligned images folder not found: {ALIGNED_DIR}. "
            "Run 04_alignment.py first."
        )

    image_paths = sorted(
        path
        for path in ALIGNED_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported aligned images found in {ALIGNED_DIR}. "
            "Run 04_alignment.py first."
        )

    preview_by_folder = {}
    failures = []
    processed_count = 0
    for image_path in image_paths:
        relative_path = image_path.relative_to(ALIGNED_DIR)
        aligned = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if aligned is None:
            failures.append(f"Could not read aligned image: {image_path}")
            continue

        try:
            difference, defect_mask = compute_difference(reference, aligned)
            relative_output = relative_path.with_suffix(".png")
            difference_path = RESULTS_DIR / "difference" / relative_output
            mask_path = RESULTS_DIR / "defect_mask" / relative_output
            difference_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            if not cv2.imwrite(str(difference_path), difference):
                raise OSError(f"Could not save difference image: {difference_path}")
            if not cv2.imwrite(str(mask_path), defect_mask):
                raise OSError(f"Could not save defect mask: {mask_path}")

            preview_by_folder.setdefault(
                relative_path.parent,
                (relative_path.name, aligned, difference, defect_mask),
            )
            processed_count += 1
            print(f"Compared with reference: {relative_path}")
        except (OSError, ValueError, cv2.error) as error:
            failures.append(f"{relative_path}: {error}")
            print(f"ERROR processing {relative_path}: {error}")

    print(f"\nProcessed {processed_count} of {len(image_paths)} aligned image(s).")
    print(f"Difference images saved under: {RESULTS_DIR / 'difference'}")
    print(f"Defect masks saved under: {RESULTS_DIR / 'defect_mask'}")

    if preview_by_folder:
        figure, axes = plt.subplots(
            len(preview_by_folder),
            4,
            figsize=(16, max(4, 3.5 * len(preview_by_folder))),
            squeeze=False,
        )
        for row, (folder, (image_name, aligned, difference, defect_mask)) in enumerate(
            preview_by_folder.items()
        ):
            for axis, image, title in (
                (axes[row, 0], reference, "Reference: reference.jpg"),
                (axes[row, 1], aligned, f"Aligned: {image_name}"),
                (axes[row, 2], difference, "Blurred absolute difference"),
                (axes[row, 3], defect_mask, "Adaptive defect mask"),
            ):
                axis.imshow(image, cmap="gray")
                axis.set_title(title)
                axis.axis("off")
            axes[row, 0].set_ylabel(folder.name)

        figure.tight_layout()
        figure.savefig(PREVIEW_PATH, dpi=150, bbox_inches="tight")
        print(f"Difference preview saved to: {PREVIEW_PATH}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(
            f"{len(failures)} aligned image(s) could not be processed."
        )


if __name__ == "__main__":
    main()
