from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from illumination_utils import normalize_illumination

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Final Dataset"
REFERENCE_PATH = BASE_DIR / "reference.jpg"
RESULTS_DIR = BASE_DIR / "results"
ANNOTATED_DIR = RESULTS_DIR / "annotated_images"
REPORT_PATH = RESULTS_DIR / "final_inspection_report.xlsx"
CSV_REPORT_PATH = RESULTS_DIR / "final_inspection_report.csv"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SUMMARY_COLUMNS = [
    "image_name",
    "dataset_category",
    "relative_path",
    "image_width",
    "image_height",
    "detected_defect_count",
    "highest_severity",
    "alignment_matches",
    "alignment_inliers",
    "inspection_status",
    "error",
]
DEFECT_COLUMNS = [
    "image_name",
    "dataset_category",
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


# -------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# -------------------------------------------------------------------------
def load_reference():
    """Load the reference PCB and prepare its reusable ORB descriptors."""
    reference_color = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_COLOR)
    if reference_color is None:
        raise FileNotFoundError(f"Reference image not found: {REFERENCE_PATH}")

    reference_gray = cv2.cvtColor(reference_color, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=5000)
    keypoints, descriptors = detector.detectAndCompute(reference_gray, None)
    if descriptors is None or len(keypoints) < 4:
        raise RuntimeError(f"Could not find alignment features in {REFERENCE_PATH}")
    return reference_gray, keypoints, descriptors, detector


def align_image(reference_gray, reference_keypoints, reference_descriptors, detector, image):
    """Register an input image to the reference and return its valid pixel mask."""
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_keypoints, image_descriptors = detector.detectAndCompute(image_gray, None)
    if image_descriptors is None or len(image_keypoints) < 4:
        raise RuntimeError("Not enough image features to align this image.")

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
        raise RuntimeError(
            f"Not enough reliable alignment matches ({len(good_matches)} found)."
        )

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
        raise RuntimeError(
            f"Could not reliably align image ({inlier_count} inlier matches)."
        )

    height, width = reference_gray.shape
    aligned_image = cv2.warpPerspective(image, homography, (width, height))
    valid_source = np.full(image_gray.shape, 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(
        valid_source,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    return aligned_image, valid_mask, len(good_matches), inlier_count


def compute_defect_mask(reference_gray, aligned_gray, valid_mask):
    """Find significant pixel differences while excluding warped border pixels."""
    reference_normalized = normalize_illumination(reference_gray)
    test_normalized = normalize_illumination(aligned_gray)
    difference = cv2.absdiff(reference_normalized, test_normalized)
    difference = cv2.GaussianBlur(difference, (3, 3), 0)

    binary_mask = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)[1]
    binary_mask = cv2.bitwise_and(binary_mask, valid_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)


def extract_defects_and_severity(cleaned_mask, aligned_color):
    """Extract component geometry, shape features, and severity."""
    image_area = cleaned_mask.shape[0] * cleaned_mask.shape[1]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned_mask, connectivity=8
    )
    output_image = aligned_color.copy()
    defects = []
    defect_counter = 0

    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 20 or area > image_area * 0.5:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        center_x, center_y = centroids[component_id]
        component_mask = np.uint8(labels == component_id)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
        )
        hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
        hu_log = [
            -np.sign(value) * np.log10(abs(value)) if value != 0 else 0.0
            for value in hu_moments
        ]

        defect_counter += 1
        severity_pct = round(min(100.0, (area / 500.0) * 100.0), 2)
        if severity_pct < 25.0:
            severity_label = "LOW"
        elif severity_pct < 65.0:
            severity_label = "MEDIUM"
        else:
            severity_label = "HIGH"

        defect = {
            "defect_id": defect_counter,
            "centroid_x": round(float(center_x), 2),
            "centroid_y": round(float(center_y), 2),
            "bbox_x": x,
            "bbox_y": y,
            "width": width,
            "height": height,
            "area": area,
            "aspect_ratio": round(width / height, 3) if height else 0.0,
            "perimeter": round(perimeter, 2),
            "circularity": round(float(circularity), 4),
            **{
                f"hu{index + 1}": round(float(value), 5)
                for index, value in enumerate(hu_log)
            },
            "severity_pct": severity_pct,
            "severity_label": severity_label,
        }
        defects.append(defect)

        cv2.rectangle(
            output_image, (x, y), (x + width, y + height), (0, 0, 255), 3
        )
        cv2.circle(
            output_image,
            (int(center_x), int(center_y)),
            5,
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            output_image,
            f"#{defect_counter}",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output_image, defects


def write_reports(summary_rows, defect_rows):
    """Write an Excel workbook and an Excel-friendly CSV of defect details."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    defects = pd.DataFrame(defect_rows, columns=DEFECT_COLUMNS)

    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Image Summary", index=False)
        defects.to_excel(writer, sheet_name="Defect Details", index=False)
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.iter_cols():
                values = [str(cell.value or "") for cell in column_cells[:100]]
                width = min(max(len(value) for value in values) + 2, 42)
                worksheet.column_dimensions[column_cells[0].column_letter].width = max(
                    width, 12
                )

    defects.to_csv(CSV_REPORT_PATH, index=False, encoding="utf-8-sig")
    return summary, defects


# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
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

    reference_gray, reference_keypoints, reference_descriptors, detector = (
        load_reference()
    )
    summary_rows = []
    defect_rows = []
    failures = 0

    for image_path in image_paths:
        relative_path = image_path.relative_to(DATASET_DIR)
        category = relative_path.parent.name
        summary_row = {
            "image_name": image_path.name,
            "dataset_category": category,
            "relative_path": str(relative_path),
            "image_width": None,
            "image_height": None,
            "detected_defect_count": 0,
            "highest_severity": "NONE",
            "alignment_matches": 0,
            "alignment_inliers": 0,
            "inspection_status": "SUCCESS",
            "error": "",
        }

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Could not read image: {image_path}")
            summary_row["image_height"], summary_row["image_width"] = image.shape[:2]

            aligned_image, valid_mask, matches, inliers = align_image(
                reference_gray,
                reference_keypoints,
                reference_descriptors,
                detector,
                image,
            )
            summary_row["alignment_matches"] = matches
            summary_row["alignment_inliers"] = inliers

            aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
            cleaned_mask = compute_defect_mask(
                reference_gray, aligned_gray, valid_mask
            )
            output_image, defects = extract_defects_and_severity(
                cleaned_mask, aligned_image
            )
            for defect in defects:
                defect_rows.append(
                    {
                        "image_name": image_path.name,
                        "dataset_category": category,
                        "relative_path": str(relative_path),
                        **defect,
                    }
                )

            summary_row["detected_defect_count"] = len(defects)
            if defects:
                severity_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
                summary_row["highest_severity"] = max(
                    (defect["severity_label"] for defect in defects),
                    key=severity_order.get,
                )

            annotated_path = ANNOTATED_DIR / relative_path
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(annotated_path), output_image):
                raise OSError(f"Could not save annotated image: {annotated_path}")
            print(
                f"{relative_path}: {len(defects)} defect(s), "
                f"highest severity {summary_row['highest_severity']}"
            )
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            failures += 1
            summary_row["inspection_status"] = "ERROR"
            summary_row["error"] = str(error)
            print(f"ERROR inspecting {relative_path}: {error}")

        summary_rows.append(summary_row)

    summary, defects = write_reports(summary_rows, defect_rows)
    total_defects = len(defects)
    print(
        f"\nInspected {len(summary)} image(s); detected {total_defects} defect "
        f"component(s) across {len(defects['relative_path'].unique()) if total_defects else 0} "
        f"image(s)."
    )
    print(f"Excel report: {REPORT_PATH}")
    print(f"CSV report: {CSV_REPORT_PATH}")
    print(f"Annotated images: {ANNOTATED_DIR}")
    if failures:
        raise RuntimeError(
            f"{failures} image(s) could not be inspected. See the report's "
            "'Image Summary' sheet for details."
        )


if __name__ == "__main__":
    main()
