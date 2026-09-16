import zipfile
import io
import hashlib
import os
import shutil

from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_0_ZIP = "DATA/S1SAR_UnBalanced_400by400_Class_0.zip"
CLASS_1_ZIP = "DATA/S1SAR_UnBalanced_400by400_Class_1.zip"

OUTPUT_DIR = "sar_dataset"

RANDOM_STATE = 42


# ============================================================
# 1. Read image files from ZIP
# ============================================================

def get_images_from_zip(zip_path, label):

    records = []

    with zipfile.ZipFile(zip_path, "r") as z:

        for filename in z.namelist():

            if not filename.lower().endswith(
                (".jpg", ".jpeg")
            ):
                continue

            records.append({
                "zip_path": zip_path,
                "filename": filename,
                "label": label
            })

    return records


class0 = get_images_from_zip(
    CLASS_0_ZIP,
    0
)

class1 = get_images_from_zip(
    CLASS_1_ZIP,
    1
)

records = class0 + class1

print("Total images:", len(records))
print("Class 0:", len(class0))
print("Class 1:", len(class1))


# ============================================================
# 2. Check duplicate images using MD5 hash
# ============================================================

print("\nChecking duplicates...")

hashes = {}
duplicates = []

for record in records:

    with zipfile.ZipFile(
        record["zip_path"], "r"
    ) as z:

        image_bytes = z.read(
            record["filename"]
        )

    image_hash = hashlib.md5(
        image_bytes
    ).hexdigest()

    if image_hash in hashes:

        duplicates.append({
            "current": record["filename"],
            "duplicate_of": hashes[image_hash]
        })

    else:

        hashes[image_hash] = record["filename"]


print("Unique image hashes:", len(hashes))
print("Duplicate images:", len(duplicates))


if duplicates:

    print("\nFirst 10 duplicates:")

    for d in duplicates[:10]:
        print(d)


# ============================================================
# 3. Remove duplicate images
# ============================================================

unique_records = []

seen_hashes = set()

for record in records:

    with zipfile.ZipFile(
        record["zip_path"], "r"
    ) as z:

        image_bytes = z.read(
            record["filename"]
        )

    image_hash = hashlib.md5(
        image_bytes
    ).hexdigest()

    if image_hash not in seen_hashes:

        seen_hashes.add(image_hash)
        unique_records.append(record)


print(
    "\nImages after duplicate removal:",
    len(unique_records)
)


# ============================================================
# 4. Create labels
# ============================================================

files = [
    record["filename"]
    for record in unique_records
]

labels = [
    record["label"]
    for record in unique_records
]


# ============================================================
# 5. Train/Test split
# ============================================================

train_files, temp_files, train_labels, temp_labels = (
    train_test_split(
        files,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=RANDOM_STATE
    )
)


# ============================================================
# 6. Validation/Test split
# ============================================================

val_files, test_files, val_labels, test_labels = (
    train_test_split(
        temp_files,
        temp_labels,
        test_size=0.50,
        stratify=temp_labels,
        random_state=RANDOM_STATE
    )
)


print("\n========================================")
print("DATASET SPLIT")
print("========================================")

print(
    "Training   :",
    len(train_files)
)

print(
    "Validation :",
    len(val_files)
)

print(
    "Testing    :",
    len(test_files)
)


# ============================================================
# 7. Helper function
# ============================================================

def count_classes(labels):

    return {
        0: labels.count(0),
        1: labels.count(1)
    }


print("\nClass distribution:")

print(
    "Train:",
    count_classes(train_labels)
)

print(
    "Validation:",
    count_classes(val_labels)
)

print(
    "Test:",
    count_classes(test_labels)
)


# ============================================================
# 8. Create output directories
# ============================================================

directories = [
    "train/0",
    "train/1",
    "val/0",
    "val/1",
    "test/0",
    "test/1"
]

for directory in directories:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            directory
        ),
        exist_ok=True
    )


# ============================================================
# 9. Build lookup
# ============================================================

record_lookup = {}

for record in unique_records:

    record_lookup[
        record["filename"]
    ] = record


# ============================================================
# 10. Copy images
# ============================================================

def copy_images(
    filenames,
    labels,
    split
):

    for filename, label in zip(
        filenames,
        labels
    ):

        record = record_lookup[
            filename
        ]

        destination = os.path.join(
            OUTPUT_DIR,
            split,
            str(label),
            os.path.basename(filename)
        )

        with zipfile.ZipFile(
            record["zip_path"],
            "r"
        ) as z:

            image_bytes = z.read(
                record["filename"]
            )

        with open(
            destination,
            "wb"
        ) as f:

            f.write(image_bytes)


copy_images(
    train_files,
    train_labels,
    "train"
)

copy_images(
    val_files,
    val_labels,
    "val"
)

copy_images(
    test_files,
    test_labels,
    "test"
)


print("\nDataset prepared successfully.")

print(
    f"\nDataset location: {OUTPUT_DIR}/"
)