import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =========================
# 1. Configuration
# =========================

DATA_DIR = "DATA/sar_dataset"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =========================
# 2. Reproducibility
# =========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# 3. Data Augmentation
# =========================
#
# IMPORTANT:
# Augmentation is applied ONLY to training data.
#
# Validation and test data remain unchanged.

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),

    transforms.ToTensor()
])


val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


# =========================
# 4. Dataset
# =========================

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=val_test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


print("Classes:", train_dataset.classes)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# =========================
# 5. CNN MODEL - EXP C
# =========================

class CNNExperimentC(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(64, 1)
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = CNNExperimentC().to(DEVICE)

print(model)


# =========================
# 6. Class Weights
# =========================

n0 = 2586
n1 = 1290

total = n0 + n1

weight_0 = total / (2 * n0)
weight_1 = total / (2 * n1)

print("Class weight 0:", weight_0)
print("Class weight 1:", weight_1)


pos_weight = torch.tensor(
    [weight_1 / weight_0],
    dtype=torch.float32
).to(DEVICE)


criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# =========================
# 7. Optimizer
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 8. Learning Rate Scheduler
# =========================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# =========================
# 9. Training
# =========================

best_val_loss = float("inf")

patience_counter = 0
early_stopping_patience = 4


for epoch in range(EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.float().unsqueeze(1).to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.float().unsqueeze(1).to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()


    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"LR: {current_lr:.6f}"
    )


    # -------------------------
    # Save best model
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "cnn_experiment_c_best.pth"
        )

        patience_counter = 0

        print("Best model saved.")

    else:

        patience_counter += 1

        if patience_counter >= early_stopping_patience:

            print("Early stopping.")

            break


# =========================
# 10. Load Best Model
# =========================

model.load_state_dict(
    torch.load(
        "cnn_experiment_c_best.pth",
        map_location=DEVICE
    )
)

model.eval()


# =========================
# 11. Test Prediction
# =========================

y_true = []
y_prob = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        y_true.extend(
            labels.numpy()
        )

        y_prob.extend(
            probabilities.cpu().numpy().flatten()
        )


y_true = np.array(y_true)
y_prob = np.array(y_prob)

y_pred = (y_prob >= 0.5).astype(int)


# =========================
# 12. Metrics
# =========================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)


tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred
).ravel()


specificity = tn / (tn + fp)


# =========================
# 13. Results
# =========================

print("\n==============================")
print("CNN EXPERIMENT C RESULTS => Bath_normalization + Augumentation")
print("==============================")

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "No Oil",
            "Oil"
        ],
        zero_division=0
    )
)


# =========================
# 14. Save Results
# =========================

results = pd.DataFrame([{

    "Experiment": "C",

    "Architecture":
        "3-block CNN + BatchNorm",

    "Augmentation":
        "Horizontal Flip + Vertical Flip + Rotation",

    "Accuracy":
        accuracy,

    "Precision":
        precision,

    "Recall":
        recall,

    "Specificity":
        specificity,

    "F1":
        f1,

    "ROC_AUC":
        roc_auc,

    "TN":
        tn,

    "FP":
        fp,

    "FN":
        fn,

    "TP":
        tp

}])


results.to_csv(
    "cnn_experiment_c_results.csv",
    index=False
)


print(
    "\nResults saved to "
    "cnn_experiment_c_results.csv"
)