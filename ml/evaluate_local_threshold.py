import os
import sys
import cv2
import torch
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from common import (
    IMG_SIZE,
    PERCENTILE,
    GAUSS,
    BORDER,
    AREA_MIN,
    AREA_MAX,
    MAX_WH,
    DEVICE,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "bottle_autoencoder_v2.pth"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test"
)


# ============================================================
# MODEL
# ============================================================

class Autoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                3, 32, 3,
                stride=2,
                padding=1
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                32, 64, 3,
                stride=2,
                padding=1
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                64, 128, 3,
                stride=2,
                padding=1
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                128, 256, 3,
                stride=2,
                padding=1
            ),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                256, 128, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(
                128, 64, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(
                64, 32, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(
                32, 3, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ============================================================
# LOAD MODEL
# ============================================================

device = DEVICE

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model weights not found: {MODEL_PATH}\n"
        f"Train the model first or check the path."
    )

model = Autoencoder().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# CALCULATE LOCAL SCORE
# ============================================================

def get_scores(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    normalized = (
        rgb.astype(np.float32) / 255.0
    )

    tensor = torch.from_numpy(
        normalized.transpose(2, 0, 1)
    ).unsqueeze(0)

    with torch.no_grad():

        reconstructed = model(tensor)

    global_error = torch.mean(
        (tensor - reconstructed) ** 2
    ).item()

    reconstructed = (
        reconstructed
        .squeeze(0)
        .numpy()
        .transpose(1, 2, 0)
    )

    reconstructed = np.clip(
        reconstructed * 255,
        0,
        255
    ).astype(np.uint8)

    difference = cv2.absdiff(
        rgb,
        reconstructed
    )

    gray_difference = cv2.cvtColor(
        difference,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray_difference,
        GAUSS,
        0
    )

    threshold_value = np.percentile(
        blurred,
        PERCENTILE
    )

    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Remove borders
    binary[:BORDER, :] = 0
    binary[-BORDER:, :] = 0
    binary[:, :BORDER] = 0
    binary[:, -BORDER:] = 0

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    best_score = 0

    for i in range(1, num_labels):

        x = stats[
            i,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            i,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            i,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            i,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area < AREA_MIN:
            continue

        if area > AREA_MAX:
            continue

        if w > MAX_WH or h > MAX_WH:
            continue

        pixels = blurred[
            labels == i
        ]

        if len(pixels) == 0:
            continue

        mean_intensity = float(
            np.mean(pixels)
        )

        max_intensity = float(
            np.max(pixels)
        )

        score = (
            mean_intensity * 0.5
            +
            max_intensity * 0.5
        )

        if score > best_score:
            best_score = score

    return global_error, best_score


# ============================================================
# COLLECT DATASET SCORES
# ============================================================

classes = [
    "good",
    "broken_large",
    "broken_small",
    "contamination"
]

all_scores = []
all_labels = []

print()
print("Analyzing MVTec bottle test set...")
print()


for class_name in classes:

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    label = 0 if class_name == "good" else 1

    for filename in files:

        image_path = os.path.join(
            class_dir,
            filename
        )

        result = get_scores(
            image_path
        )

        if result is None:
            continue

        global_error, local_score = result

        all_scores.append(local_score)
        all_labels.append(label)


print(
    f"Images analyzed: {len(all_scores)}"
)


# ============================================================
# FIND BEST LOCAL THRESHOLD
# ============================================================

scores = np.array(
    all_scores
)

labels = np.array(
    all_labels
)

best_threshold = None
best_f1 = -1
best_accuracy = 0
best_precision = 0
best_recall = 0

thresholds = np.arange(
    20,
    201,
    1
)


for threshold in thresholds:

    predictions = (
        scores >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold
        best_accuracy = accuracy
        best_precision = precision
        best_recall = recall


# ============================================================
# FINAL RESULTS
# ============================================================

final_predictions = (
    scores >= best_threshold
).astype(int)

matrix = confusion_matrix(
    labels,
    final_predictions
)


print()
print("========================================")
print("BEST LOCAL ANOMALY THRESHOLD")
print("========================================")

print(
    f"Best threshold: {best_threshold}"
)

print(
    f"Accuracy:  {best_accuracy:.4f}"
)

print(
    f"Precision: {best_precision:.4f}"
)

print(
    f"Recall:    {best_recall:.4f}"
)

print(
    f"F1 Score:  {best_f1:.4f}"
)

print()
print("Confusion Matrix:")
print(matrix)

print()
print("========================================")
print("SCORE STATISTICS")
print("========================================")

good_scores = scores[
    labels == 0
]

defect_scores = scores[
    labels == 1
]

print(
    f"Good images:   {len(good_scores)}"
)

print(
    f"Defect images: {len(defect_scores)}"
)

print()

print(
    f"Good min:  {good_scores.min():.2f}"
)

print(
    f"Good mean: {good_scores.mean():.2f}"
)

print(
    f"Good max:  {good_scores.max():.2f}"
)

print()

print(
    f"Defect min:  {defect_scores.min():.2f}"
)

print(
    f"Defect mean: {defect_scores.mean():.2f}"
)

print(
    f"Defect max:  {defect_scores.max():.2f}"
)

print()
print("Threshold evaluation completed.")