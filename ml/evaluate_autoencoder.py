import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from category_utils import (
        get_category_dirs, get_test_classes, normalize_category,
        resolve_autoencoder_path,
    )
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

_parser = argparse.ArgumentParser(description="Evaluate autoencoder (any category).")
_parser.add_argument("--category", default="bottle")
_parser.add_argument("--model", default=None)
_eval_args, _ = _parser.parse_known_args()
CATEGORY = (
    normalize_category(_eval_args.category) if _HAS_CAT
    else (str(_eval_args.category).strip().lower() or "bottle")
)


# ============================================================
# PATHS (dynamic per category)
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _HAS_CAT:
    _DIRS = get_category_dirs(CATEGORY)
    TEST_DIR = _DIRS["test_dir"]
else:
    TEST_DIR = os.path.join(
        BASE_DIR, "dataset", "mvtec_anomaly_detection", CATEGORY, "test"
    )

MODEL_PATH = _eval_args.model or (
    resolve_autoencoder_path(CATEGORY) if _HAS_CAT else os.path.join(
        BASE_DIR, "ml", "models", f"{CATEGORY}_autoencoder.pth"
    )
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("VisionInspect AI - Model Evaluation")
print("========================================")

print("Using device:", device)


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model file not found:")
    print(MODEL_PATH)
    exit()


model = Autoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()

print("Trained model loaded successfully.")


# ============================================================
# CALCULATE ERROR
# ============================================================

def calculate_error(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        reconstructed = model(image)

        error = torch.mean(
            (image - reconstructed) ** 2
        )

    return error.item()


# ============================================================
# TEST CATEGORIES (dynamic per category)
# ============================================================

if _HAS_CAT:
    categories = get_test_classes(CATEGORY)
else:
    categories = [
        d for d in sorted(os.listdir(TEST_DIR))
        if os.path.isdir(os.path.join(TEST_DIR, d))
    ] or [
        "good",
        "broken_large",
        "broken_small",
        "contamination"
    ]


# ============================================================
# PROCESS IMAGES
# ============================================================

all_errors = []
all_labels = []

category_errors = {}


for category in categories:

    folder = os.path.join(
        TEST_DIR,
        category
    )

    if not os.path.exists(folder):

        print("ERROR: Folder not found:")
        print(folder)

        exit()

    image_files = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            image_files.append(filename)

    errors = []

    print()
    print(
        "Processing",
        category,
        ":",
        len(image_files),
        "images"
    )

    for filename in image_files:

        image_path = os.path.join(
            folder,
            filename
        )

        error = calculate_error(
            image_path
        )

        errors.append(error)

        all_errors.append(error)

        if category == "good":

            all_labels.append(0)

        else:

            all_labels.append(1)

    category_errors[category] = errors

    print(
        "Average error:",
        f"{np.mean(errors):.6f}"
    )

    print(
        "Minimum error:",
        f"{np.min(errors):.6f}"
    )

    print(
        "Maximum error:",
        f"{np.max(errors):.6f}"
    )


# ============================================================
# THRESHOLD
# ============================================================

good_errors = category_errors["good"]

mean_good = np.mean(good_errors)

std_good = np.std(good_errors)

threshold = mean_good + (3 * std_good)


print()
print("========================================")
print("ANOMALY THRESHOLD")
print("========================================")

print(
    "Mean GOOD error:",
    f"{mean_good:.6f}"
)

print(
    "GOOD standard deviation:",
    f"{std_good:.6f}"
)

print(
    "Detection threshold:",
    f"{threshold:.6f}"
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = []

for error in all_errors:

    if error > threshold:

        predictions.append(1)

    else:

        predictions.append(0)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    predictions
)

precision = precision_score(
    all_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    predictions,
    zero_division=0
)

matrix = confusion_matrix(
    all_labels,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print()
print("========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    "Accuracy :",
    f"{accuracy:.4f}"
)

print(
    "Precision:",
    f"{precision:.4f}"
)

print(
    "Recall   :",
    f"{recall:.4f}"
)

print(
    "F1 Score :",
    f"{f1:.4f}"
)

print()
print("Confusion Matrix:")
print(matrix)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(
    "Total test images:",
    len(all_labels)
)

print(
    "GOOD images:",
    all_labels.count(0)
)

print(
    "DEFECT images:",
    all_labels.count(1)
)

print()
print("Category Summary:")

for category in categories:

    average_error = np.mean(
        category_errors[category]
    )

    print(
        category,
        "->",
        f"{average_error:.6f}"
    )