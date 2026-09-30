import os
import torch

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import numpy as np


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "classification_data_clean"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ml",
    "models",
    "bottle_classifier.pth"
)


# ============================================================
# 3. TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# # 4. LOAD TEST DATA
# ============================================================

test_dir = os.path.join(
    DATA_DIR,
    "test"
)

dataset = datasets.ImageFolder(
    test_dir,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

classes = dataset.classes

print("\nClasses:")
print(classes)

print(
    "Test images:",
    len(dataset)
)



# ============================================================
# 5. LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model weights not found: {MODEL_PATH}\n"
        f"Train the classifier first or check the path."
    )

# Classifier checkpoint is a dict -> weights_only=False required.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = models.resnet18(
    weights=None
)

number_of_features = (
    model.fc.in_features
)

model.fc = torch.nn.Linear(
    number_of_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []

all_probabilities = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# 7. METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# 8. PRINT OVERALL RESULTS
# ============================================================

print("\n" + "=" * 60)

print("CLASSIFIER EVALUATION")

print("=" * 60)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)

print("PER-CLASS PERFORMANCE")

print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print("=" * 60)

print("CONFUSION MATRIX")

print("=" * 60)

print(matrix)


# ============================================================
# 11. READABLE CONFUSION MATRIX
# ============================================================

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    " " * 20
    +
    " ".join(
        f"{name[:12]:>12}"
        for name in classes
    )
)

for i, row in enumerate(matrix):

    print(
        f"{classes[i]:20}"
        +
        " ".join(
            f"{value:>12}"
            for value in row
        )
    )


# ============================================================
# 12. AVERAGE CONFIDENCE
# ============================================================

average_confidence = np.mean(
    np.max(
        np.array(all_probabilities),
        axis=1
    )
)

print("\n" + "=" * 60)

print(
    f"Average prediction confidence: "
    f"{average_confidence * 100:.2f}%"
)

print("=" * 60)

print("\nEvaluation completed successfully.")