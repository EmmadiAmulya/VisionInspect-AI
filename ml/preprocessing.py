import argparse
import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from category_utils import get_category_dirs, normalize_category
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

_parser = argparse.ArgumentParser(
    description="Preprocess one image (any MVTec category)."
)
_parser.add_argument("--category", default="bottle")
_parser.add_argument("--image-index", type=int, default=0)
_known, _ = _parser.parse_known_args()
_CATEGORY = (
    normalize_category(_known.category) if _HAS_CAT
    else (str(_known.category).strip().lower() or "bottle")
)


# ==============================
# MVTec AD Dataset Path (dynamic per category)
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _HAS_CAT:
    _DIRS = get_category_dirs(_CATEGORY)
    DATASET_PATH = _DIRS["root"]
    TRAIN_PATH = _DIRS["train_good_dir"]
else:
    DATASET_PATH = os.path.join(
        BASE_DIR, "dataset", "mvtec_anomaly_detection", _CATEGORY
    )

    TRAIN_PATH = os.path.join(
        DATASET_PATH,
        "train",
        "good"
    )


# ==============================
# Process One Image
# ==============================

def preprocess_image(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize image
    image = cv2.resize(
        image,
        (224, 224)
    )

    # Convert to float
    image = image.astype(
        np.float32
    )

    # Normalize pixel values
    image = image / 255.0

    return image


# ==============================
# Test Preprocessing
# ==============================

if __name__ == "__main__":

    if not os.path.exists(TRAIN_PATH):
        print(f"Dataset not found: {TRAIN_PATH}")
        raise SystemExit(1)

    # Get image files
    image_files = [
        file for file in os.listdir(TRAIN_PATH)
        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    if not image_files:
        print("No images found!")
        exit()

    # Select first image
    first_image = image_files[0]

    image_path = os.path.join(
        TRAIN_PATH,
        first_image
    )

    # Preprocess
    processed_image = preprocess_image(
        image_path
    )

    print("PREPROCESSING SUCCESSFUL")
    print("-------------------------")
    print("Original image:", first_image)
    print(
        "Processed shape:",
        processed_image.shape
    )
    print(
        "Minimum pixel value:",
        processed_image.min()
    )
    print(
        "Maximum pixel value:",
        processed_image.max()
    )