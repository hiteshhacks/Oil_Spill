import os
import cv2
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIG
# ============================================================

DATASET = "DATA/sar_dataset"

TRAIN_DIR = os.path.join(DATASET, "train")
VAL_DIR = os.path.join(DATASET, "val")
TEST_DIR = os.path.join(DATASET, "test")


# ============================================================
# 1. GLCM FEATURE EXTRACTION
# ============================================================

def extract_glcm_features(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # Reduce grayscale levels from 256 to 32.
    # This makes GLCM computation much smaller.
    image = (image / 8).astype(np.uint8)

    distances = [1]
    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )

    features = []

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    for prop in properties:

        values = graycoprops(
            glcm,
            prop
        )

        # Average across 4 directions
        features.append(
            values.mean()
        )

    return features


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset(directory):

    X = []
    y = []

    for label in [0, 1]:

        class_dir = os.path.join(
            directory,
            str(label)
        )

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        print(
            f"Processing {directory}/"
            f"{label}: {len(files)} images"
        )

        for filename in files:

            path = os.path.join(
                class_dir,
                filename
            )

            features = extract_glcm_features(
                path
            )

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


# ============================================================
# 3. EXTRACT FEATURES
# ============================================================

print("\n========================================")
print("EXTRACTING GLCM FEATURES")
print("========================================")

X_train, y_train = load_dataset(
    TRAIN_DIR
)

X_val, y_val = load_dataset(
    VAL_DIR
)

X_test, y_test = load_dataset(
    TEST_DIR
)


print("\nFeature shape:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# 4. COMBINE TRAIN + VALIDATION
# ============================================================

X_train_final = np.concatenate(
    [X_train, X_val],
    axis=0
)

y_train_final = np.concatenate(
    [y_train, y_val],
    axis=0
)


# ============================================================
# 5. SVM PIPELINE
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ============================================================
# 6. TRAIN
# ============================================================

print("\n========================================")
print("TRAINING GLCM + SVM")
print("========================================")

model.fit(
    X_train_final,
    y_train_final
)

print("Training completed.")


# ============================================================
# 7. TEST PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 8. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 9. RESULTS
# ============================================================

print("\n========================================")
print("GLCM + SVM RESULTS")
print("========================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Oil",
            "Oil"
        ],
        digits=4
    )
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["GLCM + SVM"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "ROC_AUC": [roc_auc]
})

results.to_csv(
    "glcm_svm_results.csv",
    index=False
)

print(
    "\nSaved: glcm_svm_results.csv"
)