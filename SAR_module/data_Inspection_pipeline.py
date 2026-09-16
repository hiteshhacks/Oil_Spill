import zipfile
import io
from PIL import Image
import numpy as np


CLASS_1_ZIP = "DATA/S1SAR_UnBalanced_400by400_Class_1.zip"
CLASS_0_ZIP = "DATA/S1SAR_UnBalanced_400by400_Class_0.zip"


def inspect_zip(zip_path, expected_class):

    print("\n" + "=" * 60)
    print(f"CLASS {expected_class}")
    print("=" * 60)

    with zipfile.ZipFile(zip_path, "r") as z:

        files = [
            f for f in z.namelist()
            if f.lower().endswith((".jpg", ".jpeg"))
        ]

        print("Images found:", len(files))

        dimensions = {}
        modes = {}
        corrupted = []
        pixel_stats = []

        for file in files:

            try:
                with z.open(file) as f:
                    image_bytes = f.read()

                image = Image.open(io.BytesIO(image_bytes))
                image.load()

                # Dimensions
                dimensions[image.size] = dimensions.get(
                    image.size, 0
                ) + 1

                # Image mode
                modes[image.mode] = modes.get(
                    image.mode, 0
                ) + 1

                # Convert to grayscale for pixel statistics
                gray = np.array(image.convert("L"))

                pixel_stats.append({
                    "min": gray.min(),
                    "max": gray.max(),
                    "mean": gray.mean(),
                    "std": gray.std()
                })

            except Exception as e:
                corrupted.append((file, str(e)))

        print("\nDimensions:")
        for dimension, count in dimensions.items():
            print(f"  {dimension}: {count}")

        print("\nImage modes:")
        for mode, count in modes.items():
            print(f"  {mode}: {count}")

        print("\nCorrupted images:", len(corrupted))

        if corrupted:
            for item in corrupted[:10]:
                print(item)

        # Pixel statistics
        if pixel_stats:

            mins = [x["min"] for x in pixel_stats]
            maxs = [x["max"] for x in pixel_stats]
            means = [x["mean"] for x in pixel_stats]
            stds = [x["std"] for x in pixel_stats]

            print("\nPixel statistics:")
            print("  Minimum pixel:", min(mins))
            print("  Maximum pixel:", max(maxs))
            print("  Mean intensity:", np.mean(means))
            print("  Mean standard deviation:", np.mean(stds))

        return files


# ============================================================
# Inspect both classes
# ============================================================

class1_files = inspect_zip(
    CLASS_1_ZIP,
    1
)

class0_files = inspect_zip(
    CLASS_0_ZIP,
    0
)


# ============================================================
# Dataset summary
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print("Class 0:", len(class0_files))
print("Class 1:", len(class1_files))
print("Total:", len(class0_files) + len(class1_files))


# ============================================================
# Class imbalance
# ============================================================

total = len(class0_files) + len(class1_files)

print("\nClass distribution:")

print(
    f"Class 0: {len(class0_files)} "
    f"({len(class0_files) / total * 100:.2f}%)"
)

print(
    f"Class 1: {len(class1_files)} "
    f"({len(class1_files) / total * 100:.2f}%)"
)