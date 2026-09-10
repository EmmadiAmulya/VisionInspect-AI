import cv2
import os
import numpy as np


# ==============================
# MVTec AD Dataset Path
# ==============================

DATASET_PATH = os.path.join(
    "dataset",
    "mvtec_anomaly_detection",
    "bottle"
)


# ==============================
# Training Images
# ==============================

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