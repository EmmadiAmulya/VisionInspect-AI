import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
)

TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "classification_data_clean"

CLASSES = [
    "good",
    "broken_large",
    "broken_small",
    "contamination"
]

# ============================================================
# CLEAN OLD DATASET
# ============================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# ============================================================
# CREATE FOLDERS
# ============================================================

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        folder = OUTPUT_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# COLLECT IMAGES
# ============================================================

all_images = []

for class_name in CLASSES:

    class_dir = TEST_DIR / class_name

    images = sorted(class_dir.glob("*.png"))

    for image_path in images:
        all_images.append((image_path, class_name))

print(f"\nTotal images found: {len(all_images)}")

# ============================================================
# FIRST SPLIT
# 80% temporary train/val
# 20% final test
# ============================================================

paths = [item[0] for item in all_images]
labels = [item[1] for item in all_images]

train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# ============================================================
# SECOND SPLIT
# 75% train
# 25% validation
#
# Overall:
# 60% train
# 20% val
# 20% test
# ============================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths,
    train_val_labels,
    test_size=0.25,
    random_state=42,
    stratify=train_val_labels
)

# ============================================================
# COPY FILES
# ============================================================

def copy_images(paths, labels, split):

    for image_path, label in zip(paths, labels):

        destination = (
            OUTPUT_DIR
            / split
            / label
            / image_path.name
        )

        shutil.copy2(image_path, destination)


copy_images(train_paths, train_labels, "train")
copy_images(val_paths, val_labels, "val")
copy_images(test_paths, test_labels, "test")

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nClassification dataset created.")
print("--------------------------------")

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")

    for class_name in CLASSES:

        folder = OUTPUT_DIR / split / class_name

        count = len(list(folder.glob("*.png")))

        print(f"{class_name:15} : {count}")

print("\nOutput directory:")
print(OUTPUT_DIR)