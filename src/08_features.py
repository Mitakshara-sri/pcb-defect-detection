from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_MASK_DIR = BASE_DIR / "results" / "morphology" / "cleaned_masks"
FEATURES_CSV_PATH = BASE_DIR / "results" / "features.csv"
SUMMARY_CSV_PATH = BASE_DIR / "results" / "feature_summary.csv"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FEATURE_COLUMNS = [
    "image_name",
    "dataset_category",
    "label",
    "relative_path",
    "defect_id",
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "width",
    "height",
    "area",
    "aspect_ratio",
    "perimeter",
    "circularity",
    "hu1",
    "hu2",
    "hu3",
    "hu4",
    "hu5",
    "hu6",
    "hu7",
    "severity_pct",
    "severity_label",
]
SUMMARY_COLUMNS = [
    "image_name",
    "dataset_category",
    "label",
    "relative_path",
    "detected_defect_count",
    "highest_severity",
    "inspection_status",
    "error",
]


def dataset_label(category):
    """Map a rotation-folder name to its base PCB defect class."""
    return category.casefold().removesuffix("_rotation")


def extract_features(mask):
    """Calculate geometry, shape descriptors, and severity for each component."""
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    image_area = mask.shape[0] * mask.shape[1]
    features = []

    for component_index in range(1, count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < 20 or area > image_area * 0.5:
            continue

        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        center_x, center_y = centroids[component_index]
        component_mask = np.uint8(labels == component_index)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            (4.0 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
        )
        hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
        hu_log = [
            -np.sign(value) * np.log10(abs(value)) if value != 0 else 0.0
            for value in hu_moments
        ]
        severity_pct = round(min(100.0, area / 5.0), 2)
        if severity_pct < 25.0:
            severity_label = "LOW"
        elif severity_pct < 65.0:
            severity_label = "MEDIUM"
        else:
            severity_label = "HIGH"

        features.append(
            {
                "defect_id": len(features) + 1,
                "centroid_x": round(float(center_x), 2),
                "centroid_y": round(float(center_y), 2),
                "bbox_x": x,
                "bbox_y": y,
                "width": width,
                "height": height,
                "area": area,
                "aspect_ratio": round(width / height, 4) if height else 0.0,
                "perimeter": round(perimeter, 2),
                "circularity": round(float(circularity), 5),
                **{
                    f"hu{index + 1}": round(float(value), 7)
                    for index, value in enumerate(hu_log)
                },
                "severity_pct": severity_pct,
                "severity_label": severity_label,
            }
        )
    return features


def main():
    if not CLEAN_MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Cleaned masks folder not found: {CLEAN_MASK_DIR}. "
            "Run 06_morphology.py first."
        )

    mask_paths = sorted(
        path
        for path in CLEAN_MASK_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not mask_paths:
        raise FileNotFoundError(f"No cleaned masks found in {CLEAN_MASK_DIR}")

    feature_rows = []
    summary_rows = []
    failures = []
    for mask_path in mask_paths:
        relative_image_path = mask_path.relative_to(CLEAN_MASK_DIR).with_suffix(".jpg")
        category = relative_image_path.parent.name
        label = dataset_label(category)
        summary = {
            "image_name": relative_image_path.name,
            "dataset_category": category,
            "label": label,
            "relative_path": str(relative_image_path),
            "detected_defect_count": 0,
            "highest_severity": "NONE",
            "inspection_status": "SUCCESS",
            "error": "",
        }
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            summary["inspection_status"] = "ERROR"
            summary["error"] = f"Could not read cleaned mask: {mask_path}"
            failures.append(summary["error"])
            summary_rows.append(summary)
            continue

        features = extract_features(mask)
        summary["detected_defect_count"] = len(features)
        if features:
            severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            summary["highest_severity"] = max(
                (feature["severity_label"] for feature in features),
                key=severity_rank.get,
            )
        feature_rows.extend(
            {
                "image_name": relative_image_path.name,
                "dataset_category": category,
                "label": label,
                "relative_path": str(relative_image_path),
                **feature,
            }
            for feature in features
        )
        summary_rows.append(summary)
        print(f"{relative_image_path}: extracted {len(features)} defect feature(s)")

    pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS).to_csv(
        FEATURES_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
        SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    print(
        f"\nExtracted features from {len(summary_rows) - len(failures)} "
        f"of {len(mask_paths)} image(s), producing {len(feature_rows)} defect row(s)."
    )
    print(f"Feature dataset saved to: {FEATURES_CSV_PATH}")
    print(f"Per-image feature summary saved to: {SUMMARY_CSV_PATH}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} mask(s) could not be analyzed.")


if __name__ == "__main__":
    main()
