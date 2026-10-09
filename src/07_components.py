from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_MASK_DIR = BASE_DIR / "results" / "morphology" / "cleaned_masks"
OUTPUT_DIR = BASE_DIR / "results" / "components" / "labeled_masks"
DETAILS_CSV_PATH = BASE_DIR / "results" / "component_details.csv"
SUMMARY_CSV_PATH = BASE_DIR / "results" / "component_summary.csv"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DETAIL_COLUMNS = [
    "image_name",
    "dataset_category",
    "label",
    "relative_path",
    "component_id",
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "width",
    "height",
    "area",
]
SUMMARY_COLUMNS = [
    "image_name",
    "dataset_category",
    "label",
    "relative_path",
    "detected_component_count",
    "inspection_status",
    "error",
]


def dataset_label(category):
    """Map a rotation-folder name to its base PCB defect class."""
    return category.casefold().removesuffix("_rotation")


def analyze_components(mask):
    """Label connected regions and return a color map and component records."""
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    labeled_image = cv2.applyColorMap(
        np.uint8(labels % 256), cv2.COLORMAP_TURBO
    )
    labeled_image[labels == 0] = 0

    height, width = mask.shape
    image_area = height * width
    components = []
    for component_index in range(1, count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < 20 or area > image_area * 0.5:
            continue

        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        box_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        center_x, center_y = centroids[component_index]
        components.append(
            {
                "component_id": len(components) + 1,
                "centroid_x": round(float(center_x), 2),
                "centroid_y": round(float(center_y), 2),
                "bbox_x": x,
                "bbox_y": y,
                "width": box_width,
                "height": box_height,
                "area": area,
            }
        )
        cv2.rectangle(
            labeled_image,
            (x, y),
            (x + box_width, y + box_height),
            (255, 255, 255),
            2,
        )
    return labeled_image, components


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

    details = []
    summaries = []
    failures = []
    for mask_path in mask_paths:
        relative_mask_path = mask_path.relative_to(CLEAN_MASK_DIR)
        relative_image_path = relative_mask_path.with_suffix(".jpg")
        category = relative_mask_path.parent.name
        label = dataset_label(relative_mask_path.parent.name)
        summary = {
            "image_name": relative_image_path.name,
            "dataset_category": category,
            "label": label,
            "relative_path": str(relative_image_path),
            "detected_component_count": 0,
            "inspection_status": "SUCCESS",
            "error": "",
        }

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            summary["inspection_status"] = "ERROR"
            summary["error"] = f"Could not read mask: {mask_path}"
            failures.append(summary["error"])
            summaries.append(summary)
            continue

        labeled_image, components = analyze_components(mask)
        output_path = OUTPUT_DIR / relative_mask_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), labeled_image):
            summary["inspection_status"] = "ERROR"
            summary["error"] = f"Could not save component image: {output_path}"
            failures.append(summary["error"])
            summaries.append(summary)
            continue

        summary["detected_component_count"] = len(components)
        details.extend(
            {
                "image_name": relative_image_path.name,
                "dataset_category": category,
                "label": label,
                "relative_path": str(relative_image_path),
                **component,
            }
            for component in components
        )
        summaries.append(summary)
        print(f"{relative_image_path}: {len(components)} component(s)")

    pd.DataFrame(details, columns=DETAIL_COLUMNS).to_csv(
        DETAILS_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).to_csv(
        SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"\nAnalyzed {len(summaries) - len(failures)} of {len(mask_paths)} mask(s).")
    print(f"Component label images saved under: {OUTPUT_DIR}")
    print(f"Component details saved to: {DETAILS_CSV_PATH}")
    print(f"Per-image component summary saved to: {SUMMARY_CSV_PATH}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} mask(s) could not be analyzed.")


if __name__ == "__main__":
    main()
