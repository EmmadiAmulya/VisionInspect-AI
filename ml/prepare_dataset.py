import os
import cv2


# ==========================================
# MVTec AD Bottle Dataset Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOTTLE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle"
)

TRAIN_DIR = os.path.join(
    BOTTLE_DIR,
    "train",
    "good"
)

TEST_DIR = os.path.join(
    BOTTLE_DIR,
    "test"
)


# ==========================================
# Check dataset folders
# ==========================================

print("========================================")
print("VisionInspect AI - Dataset Preparation")
print("========================================")

print("\nBottle dataset location:")
print(BOTTLE_DIR)

print("\nChecking folders...")


if not os.path.exists(BOTTLE_DIR):
    print("ERROR: Bottle dataset folder not found.")
    exit()


if not os.path.exists(TRAIN_DIR):
    print("ERROR: Training folder not found.")
    exit()


if not os.path.exists(TEST_DIR):
    print("ERROR: Testing folder not found.")
    exit()


print("Dataset folders found successfully!")


# ==========================================
# Count training images
# ==========================================

train_images = [
    file for file in os.listdir(TRAIN_DIR)
    if file.lower().endswith((".png", ".jpg", ".jpeg"))
]

print("\nTraining images:")
print("GOOD images:", len(train_images))


# ==========================================
# Count test images
# ==========================================

print("\nTest images:")

total_test_images = 0

test_categories = os.listdir(TEST_DIR)

for category in test_categories:

    category_path = os.path.join(
        TEST_DIR,
        category
    )

    if os.path.isdir(category_path):

        images = [
            file for file in os.listdir(category_path)
            if file.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"{category}: {len(images)}")

        total_test_images += len(images)


print("\nTotal test images:", total_test_images)


# ==========================================
# Test image loading
# ==========================================

if len(train_images) > 0:

    first_image = train_images[0]

    image_path = os.path.join(
        TRAIN_DIR,
        first_image
    )

    image = cv2.imread(image_path)

    if image is None:
        print("\nERROR: Could not load image.")
        exit()

    print("\nFirst training image:")
    print("Filename:", first_image)
    print("Original shape:", image.shape)


    # Resize image
    resized_image = cv2.resize(
        image,
        (224, 224)
    )

    # Convert BGR → RGB
    rgb_image = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2RGB
    )

    # Normalize pixels
    normalized_image = rgb_image / 255.0

    print("Processed shape:", normalized_image.shape)
    print("Minimum pixel value:", normalized_image.min())
    print("Maximum pixel value:", normalized_image.max())


# ==========================================
# Finished
# ==========================================

print("\n========================================")
print("DATASET PREPARATION SUCCESSFUL")
print("========================================")