from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ALIGNED_DIR = BASE_DIR / "results" / "aligned_images"
FEATURES_CSV_PATH = BASE_DIR / "results" / "features_with_severity.csv"
OUTPUT_DIR = BASE_DIR / "results" / "final_visualizations" / "annotated_images"
SUMMARY_CSV_PATH = BASE_DIR / "results" / "final_visualizations" / "inspection_summary.csv"
PREVIEW_PATH = BASE_DIR / "results" / "final_inspection_preview.jpg"
REFERENCE_PATH = BASE_DIR / "reference.jpg"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REQUIRED_FEATURE_COLUMNS = {
    "relative_path",
    "defect_id",
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "width",
    "height",
    "severity_pct",
    "severity_label",
    "predicted_defect_type",
}
SUMMARY_COLUMNS = [
    "image_name",
    "dataset_category",
    "label",
    "relative_path",
    "detected_defect_count",
    "highest_severity",
    "predicted_defect_type",
    "inspection_status",
    "error",
]


def annotate_image(aligned, image_features):
    """Draw detected defect boxes and labels on an aligned PCB image."""
    output = aligned.copy()
    for defect in image_features.itertuples(index=False):
        x = int(defect.bbox_x)
        y = int(defect.bbox_y)
        width = int(defect.width)
        height = int(defect.height)
        center = (int(float(defect.centroid_x)), int(float(defect.centroid_y)))
        color = (0, 0, 255)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 3)
        cv2.circle(output, center, 5, (0, 255, 0), -1)
        text = (
            f"#{int(defect.defect_id)} {defect.severity_label} "
            f"{float(defect.severity_pct):.1f}%"
        )
        cv2.putText(
            output,
            text,
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def main():
    if not ALIGNED_DIR.is_dir():
        raise FileNotFoundError(
            f"Aligned images folder not found: {ALIGNED_DIR}. "
            "Run 04_alignment.py first."
        )
    if not FEATURES_CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Trained feature report not found: {FEATURES_CSV_PATH}. "
            "Run 08_features.py and 09_train_svm.py first."
        )
    reference = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(f"Reference image not found: {REFERENCE_PATH}")

    features = pd.read_csv(FEATURES_CSV_PATH)
    missing_columns = sorted(REQUIRED_FEATURE_COLUMNS.difference(features.columns))
    if missing_columns:
        raise ValueError(
            f"Feature report is missing required columns: {', '.join(missing_columns)}"
        )
    features_by_image = {
        relative_path: image_features
        for relative_path, image_features in features.groupby("relative_path")
    }

    image_paths = sorted(
        path
        for path in ALIGNED_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No aligned images found in {ALIGNED_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_by_category = {}
    summaries = []
    failures = []
    processed_count = 0
    for image_path in image_paths:
        relative_path = image_path.relative_to(ALIGNED_DIR)
        relative_key = str(relative_path)
        category = relative_path.parent.name
        image_features = features_by_image.get(
            relative_key, features.iloc[0:0]
        )
        summary = {
            "image_name": image_path.name,
            "dataset_category": category,
            "label": (
                str(image_features["label"].iloc[0])
                if "label" in image_features.columns and not image_features.empty
                else category.casefold().removesuffix("_rotation")
            ),
            "relative_path": relative_key,
            "detected_defect_count": len(image_features),
            "highest_severity": "NONE",
            "predicted_defect_type": "NONE",
            "inspection_status": "SUCCESS",
            "error": "",
        }

        if not image_features.empty:
            severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            summary["highest_severity"] = max(
                image_features["severity_label"].astype(str),
                key=severity_rank.get,
            )
            summary["predicted_defect_type"] = str(
                image_features["predicted_defect_type"].mode().iloc[0]
            )

        aligned = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if aligned is None:
            summary["inspection_status"] = "ERROR"
            summary["error"] = f"Could not read aligned image: {image_path}"
            failures.append(summary["error"])
            summaries.append(summary)
            continue
        if aligned.shape[:2] != reference.shape[:2]:
            summary["inspection_status"] = "ERROR"
            summary["error"] = (
                f"Aligned image dimensions {aligned.shape[:2]} do not match "
                f"reference dimensions {reference.shape[:2]}."
            )
            failures.append(f"{relative_key}: {summary['error']}")
            summaries.append(summary)
            continue

        annotated = annotate_image(aligned, image_features)
        output_path = OUTPUT_DIR / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            summary["inspection_status"] = "ERROR"
            summary["error"] = f"Could not save annotated image: {output_path}"
            failures.append(summary["error"])
            summaries.append(summary)
            continue

        preview_by_category.setdefault(
            category, (image_path.name, aligned, annotated)
        )
        summaries.append(summary)
        processed_count += 1
        print(
            f"Visualized {relative_key}: {summary['detected_defect_count']} "
            f"defect(s), predicted class {summary['predicted_defect_type']}"
        )

    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).to_csv(
        SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"\nCreated annotated output for {processed_count} of {len(image_paths)} images.")
    print(f"Annotated images saved under: {OUTPUT_DIR}")
    print(f"Per-image inspection summary saved to: {SUMMARY_CSV_PATH}")

    if preview_by_category:
        figure, axes = plt.subplots(
            len(preview_by_category),
            3,
            figsize=(15, max(4, 3.5 * len(preview_by_category))),
            squeeze=False,
        )
        for row, (category, (image_name, aligned, annotated)) in enumerate(
            preview_by_category.items()
        ):
            for axis, image, title in (
                (axes[row, 0], reference, "Reference: reference.jpg"),
                (axes[row, 1], aligned, f"Aligned: {image_name}"),
                (axes[row, 2], annotated, "Detected defects"),
            ):
                axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axis.set_title(title)
                axis.axis("off")
            axes[row, 0].set_ylabel(category)

        figure.tight_layout()
        figure.savefig(PREVIEW_PATH, dpi=150, bbox_inches="tight")
        print(f"Final inspection preview saved to: {PREVIEW_PATH}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} image(s) could not be visualized.")


if __name__ == "__main__":
    main()
