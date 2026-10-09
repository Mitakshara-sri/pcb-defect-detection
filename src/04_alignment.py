from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Final Dataset"
REFERENCE_PATH = BASE_DIR / "reference.jpg"
RESULTS_DIR = BASE_DIR / "results"
ALIGNED_DIR = RESULTS_DIR / "aligned_images"
PREVIEW_PATH = RESULTS_DIR / "alignment_preview.jpg"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def align_image(reference_gray, reference_keypoints, reference_descriptors, detector, image):
    """Align an input image to the reference using ORB features and RANSAC."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_keypoints, image_descriptors = detector.detectAndCompute(gray_image, None)
    if image_descriptors is None:
        raise RuntimeError("No ORB features found in image.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(reference_descriptors, image_descriptors, k=2)
    good_matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    if len(good_matches) < 10:
        raise RuntimeError(f"Not enough good matches ({len(good_matches)} found).")

    reference_points = np.float32(
        [reference_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    image_points = np.float32(
        [image_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        image_points, reference_points, cv2.RANSAC, 5.0
    )
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if homography is None or inlier_count < 10:
        raise RuntimeError(f"Could not compute reliable homography ({inlier_count} inliers).")

    height, width = reference_gray.shape
    aligned = cv2.warpPerspective(image, homography, (width, height))
    return aligned, len(good_matches), inlier_count


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    reference = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(f"Reference image not found: {REFERENCE_PATH}")

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=5000)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(
        reference_gray, None
    )
    if reference_descriptors is None or len(reference_keypoints) < 4:
        raise RuntimeError(f"Could not find ORB features in reference: {REFERENCE_PATH}")

    rotation_folders = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_dir() and path.name.casefold().endswith("_rotation")
    )
    image_paths = sorted(
        image_path
        for folder in rotation_folders
        for image_path in folder.rglob("*")
        if image_path.is_file()
        and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in '*_Rotation' folders under {DATASET_DIR}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    previews = {}
    failures = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(DATASET_DIR)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"Could not read image: {image_path}")
            continue

        try:
            aligned, match_count, inlier_count = align_image(
                reference_gray,
                reference_keypoints,
                reference_descriptors,
                detector,
                image,
            )
            output_path = ALIGNED_DIR / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), aligned):
                raise OSError(f"Could not save aligned image: {output_path}")

            previews.setdefault(relative_path.parent, (image_path.name, aligned))
            print(
                f"Aligned: {relative_path} "
                f"({match_count} matches, {inlier_count} inliers)"
            )
        except (RuntimeError, OSError, cv2.error) as error:
            failures.append(f"{relative_path}: {error}")
            print(f"ERROR aligning {relative_path}: {error}")

    print(f"\nAligned {len(image_paths) - len(failures)} of {len(image_paths)} image(s).")
    print(f"Aligned images saved under: {ALIGNED_DIR}")

    if previews:
        figure, axes = plt.subplots(
            len(previews),
            2,
            figsize=(14, max(5, 4 * len(previews))),
            squeeze=False,
        )
        for row, (folder, (image_name, aligned)) in enumerate(previews.items()):
            axes[row, 0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
            axes[row, 0].set_title("Reference: reference.jpg")
            axes[row, 1].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
            axes[row, 1].set_title(f"Aligned: {folder.name} / {image_name}")
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")

        figure.tight_layout()
        figure.savefig(PREVIEW_PATH, dpi=150, bbox_inches="tight")
        print(f"Reference/alignment preview saved to: {PREVIEW_PATH}")
        plt.show()

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(
            f"{len(failures)} image(s) could not be aligned. "
            "See the errors printed above."
        )


if __name__ == "__main__":
    main()
