import argparse
import os
import shutil
import stat
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# WARNING: POTENTIAL DATA LEAKAGE
# ============================================================
# Current source is test/ ONLY (MVTec bottle/test per class).
# Splitting test-only data into train/val/test means the
# classifier never sees truly independent data and reported
# metrics are optimistic.
#
# RECOMMENDED: include train/good (nominal training images) in
# the "good" pool and then do a stratified split, or better,
# keep MVTec train/ vs test/ disjoint and evaluate only on test/.
# Use --include-train to merge train/good into the pool when
# that folder exists. Default is OFF to avoid breaking existing
# trained models, but the code path works when enabled.
# ============================================================

parser = argparse.ArgumentParser(
    description="Build classification dataset from any MVTec category."
)

parser.add_argument(
    "--category",
    default="bottle",
    help="MVTec category (e.g. bottle, cable, capsule).",
)

parser.add_argument(
    "--output-dir",
    default=None,
    help=(
        "Explicit output dir. Default: classification_data_clean for "
        "bottle (legacy), classification_data_{category} otherwise."
    ),
)

parser.add_argument(
    "--include-train",
    action="store_true",
    help=(
        "Also merge dataset/.../<category>/train/good into the "
        "'good' pool before stratified split. Default off for "
        "bottle (legacy weights); recommended ON for new categories."
    ),
)

args = parser.parse_args()

# ============================================================
# PATHS (dynamic per category)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "ml"))
    from category_utils import (
        get_category_dirs as _get_dirs,
        get_test_classes as _get_classes,
        normalize_category as _norm,
        resolve_classification_data_dir as _resolve_data_dir,
    )
    CATEGORY = _norm(args.category)
    _DIRS = _get_dirs(CATEGORY)
    DATASET_DIR = Path(_DIRS["root"])
    TEST_DIR = Path(_DIRS["test_dir"])
    TRAIN_GOOD_DIR = Path(_DIRS["train_good_dir"])
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = Path(_resolve_data_dir(CATEGORY))
    CLASSES = sorted(_get_classes(CATEGORY))
except ImportError:
    CATEGORY = str(args.category).strip().lower() or "bottle"
    DATASET_DIR = (
        PROJECT_ROOT
        / "dataset"
        / "mvtec_anomaly_detection"
        / CATEGORY
    )
    TEST_DIR = DATASET_DIR / "test"
    TRAIN_GOOD_DIR = DATASET_DIR / "train" / "good"
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    elif CATEGORY == "bottle":
        OUTPUT_DIR = PROJECT_ROOT / "classification_data_clean"
    else:
        OUTPUT_DIR = PROJECT_ROOT / f"classification_data_{CATEGORY}"
    CLASSES = [
        d.name for d in sorted(TEST_DIR.iterdir()) if d.is_dir()
    ] or ["good"]

print(f"Category: {CATEGORY}")
print(f"Classes: {CLASSES}")

# ============================================================
# CLEAN OLD DATASET
# ============================================================

if OUTPUT_DIR.exists():
    def _writable(func, path, _exc):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            pass
    shutil.rmtree(OUTPUT_DIR, onerror=_writable)

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

if args.include_train:
    if TRAIN_GOOD_DIR.exists():
        train_good = sorted(TRAIN_GOOD_DIR.glob("*.png"))

        for image_path in train_good:
            all_images.append((image_path, "good"))

        print(
            f"Merged train/good images: {len(train_good)} "
            f"(via --include-train)"
        )
    else:
        print(
            f"WARNING: --include-train requested but not found: "
            f"{TRAIN_GOOD_DIR}"
        )

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

        # shutil.copyfile (not copy/copy2): MVTec sources are read-only
        # and copy/copy2 propagate mode bits, breaking reruns on Windows.
        shutil.copyfile(image_path, destination)


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