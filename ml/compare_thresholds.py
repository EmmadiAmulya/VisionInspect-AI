import os
import cv2
import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
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
# AUTOENCODER
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

device = torch.device("cpu")

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
# CALCULATE LOCAL ANOMALY SCORE
# ============================================================

def get_local_score(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(
        image,
        (224, 224)
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
        (5, 5),
        0
    )

    threshold_value = np.percentile(
        blurred,
        97
    )

    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Remove border artifacts
    binary[:10, :] = 0
    binary[-10:, :] = 0
    binary[:, :10] = 0
    binary[:, -10:] = 0

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

        # Remove tiny noise
        if area < 15:
            continue

        # Remove huge regions
        if area > 3000:
            continue

        if w > 150 or h > 150:
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

    return best_score


# ============================================================
# ANALYZE ALL TEST IMAGES
# ============================================================

classes = [
    "good",
    "broken_large",
    "broken_small",
    "contamination"
]

scores = []
labels = []

print()
print("Analyzing all 83 test images...")
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

        score = get_local_score(
            image_path
        )

        if score is None:
            continue

        scores.append(score)
        labels.append(label)


scores = np.array(scores)
labels = np.array(labels)

print(
    f"Images analyzed: {len(scores)}"
)


# ============================================================
# TEST MULTIPLE THRESHOLDS
# ============================================================

thresholds = [
    60,
    65,
    66,
    70,
    75,
    80,
    85,
    90,
    95,
    100,
    105,
    110
]


print()
print("==============================================================")
print("THRESHOLD COMPARISON")
print("==============================================================")

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Balanced Acc':<15}"
    f"{'FP':<8}"
)

print("--------------------------------------------------------------")


best_balanced = None


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

    balanced = balanced_accuracy_score(
        labels,
        predictions
    )

    matrix = confusion_matrix(
        labels,
        predictions
    )

    tn, fp, fn, tp = matrix.ravel()

    print(
        f"{threshold:<12}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{balanced:<15.4f}"
        f"{fp:<8}"
    )

    # Choose highest balanced accuracy.
    # If tied, prefer higher F1.
    if (
        best_balanced is None
        or balanced > best_balanced["balanced"]
        or (
            balanced == best_balanced["balanced"]
            and f1 > best_balanced["f1"]
        )
    ):

        best_balanced = {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced": balanced,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }


# ============================================================
# BEST BALANCED THRESHOLD
# ============================================================

print()
print("==============================================================")
print("BEST BALANCED THRESHOLD")
print("==============================================================")

print(
    f"Threshold:        "
    f"{best_balanced['threshold']}"
)

print(
    f"Accuracy:         "
    f"{best_balanced['accuracy']:.4f}"
)

print(
    f"Precision:        "
    f"{best_balanced['precision']:.4f}"
)

print(
    f"Recall:           "
    f"{best_balanced['recall']:.4f}"
)

print(
    f"F1 Score:         "
    f"{best_balanced['f1']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_balanced['balanced']:.4f}"
)

print()
print("Confusion Matrix:")

print(
    np.array([
        [
            best_balanced["tn"],
            best_balanced["fp"]
        ],
        [
            best_balanced["fn"],
            best_balanced["tp"]
        ]
    ])
)

print()
print("Threshold comparison completed.")