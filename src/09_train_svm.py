from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV_PATH = BASE_DIR / "results" / "features.csv"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "svm_defect_classifier.joblib"
PREDICTIONS_CSV_PATH = RESULTS_DIR / "features_with_severity.csv"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "svm_classification_report.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "svm_confusion_matrix.csv"
FEATURE_COLUMNS = [
    "area",
    "width",
    "height",
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
]


def create_model():
    """Build the scaled RBF support-vector classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, random_state=42)),
        ]
    )


def main():
    if not FEATURES_CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_CSV_PATH}. "
            "Run 08_features.py first."
        )

    features = pd.read_csv(FEATURES_CSV_PATH)
    required_columns = {"relative_path", "label", *FEATURE_COLUMNS}
    missing_columns = sorted(required_columns.difference(features.columns))
    if missing_columns:
        raise ValueError(
            f"Feature dataset is missing required columns: {', '.join(missing_columns)}"
        )
    if features.empty:
        raise ValueError(
            "No defect features are available for training. "
            "Check the generated masks and component filtering."
        )
    if features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature dataset contains missing classifier values.")

    image_labels = features[["relative_path", "label"]].drop_duplicates()
    conflicting_labels = image_labels["relative_path"].duplicated(keep=False)
    if conflicting_labels.any():
        raise ValueError("An image has conflicting defect-class labels.")
    label_counts = image_labels["label"].value_counts()
    if len(label_counts) < 2 or label_counts.min() < 2:
        raise ValueError(
            "Training requires at least two labeled images in each of two classes."
        )

    train_images, test_images = train_test_split(
        image_labels,
        test_size=0.2,
        random_state=42,
        stratify=image_labels["label"],
    )
    train_paths = set(train_images["relative_path"])
    test_paths = set(test_images["relative_path"])
    train_rows = features[features["relative_path"].isin(train_paths)]
    test_rows = features[features["relative_path"].isin(test_paths)]
    if train_rows.empty or test_rows.empty:
        raise ValueError(
            "Image-level train/test split produced an empty feature partition."
        )

    evaluation_model = create_model()
    evaluation_model.fit(train_rows[FEATURE_COLUMNS], train_rows["label"])
    predictions = evaluation_model.predict(test_rows[FEATURE_COLUMNS])
    classes = sorted(features["label"].unique())

    report = classification_report(
        test_rows["label"],
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        test_rows["label"], predictions, labels=classes
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(
        CLASSIFICATION_REPORT_PATH, encoding="utf-8-sig"
    )
    pd.DataFrame(
        confusion,
        index=[f"actual_{label}" for label in classes],
        columns=[f"predicted_{label}" for label in classes],
    ).to_csv(CONFUSION_MATRIX_PATH, encoding="utf-8-sig")

    final_model = create_model()
    final_model.fit(features[FEATURE_COLUMNS], features["label"])
    features["predicted_defect_type"] = final_model.predict(
        features[FEATURE_COLUMNS]
    )
    joblib.dump(final_model, MODEL_PATH)
    features.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="utf-8-sig")

    print(
        f"Trained on {len(train_paths)} image(s); evaluated on {len(test_paths)} "
        f"held-out image(s)."
    )
    print(f"Classifier classes: {', '.join(classes)}")
    print(f"Evaluation report saved to: {CLASSIFICATION_REPORT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Final model saved to: {MODEL_PATH}")
    print(f"Features and predictions saved to: {PREDICTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()
