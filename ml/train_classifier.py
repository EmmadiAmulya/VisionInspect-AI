import argparse
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from category_utils import (
        normalize_category, resolve_classification_data_dir,
    )
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

_parser = argparse.ArgumentParser(
    description="Train ResNet18 classifier (any MVTec category)."
)
_parser.add_argument("--category", default="bottle")
_parser.add_argument("--data-dir", default=None)
_parser.add_argument("--epochs", type=int, default=20)
_parser.add_argument("--batch-size", type=int, default=8)
_parser.add_argument("--output", default=None)
_clf_args, _ = _parser.parse_known_args()
CATEGORY = (
    normalize_category(_clf_args.category) if _HAS_CAT
    else (str(_clf_args.category).strip().lower() or "bottle")
)


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

DATA_DIR = _clf_args.data_dir or (
    resolve_classification_data_dir(CATEGORY) if _HAS_CAT else os.path.join(
        PROJECT_ROOT,
        "classification_data_clean" if CATEGORY == "bottle"
        else f"classification_data_{CATEGORY}"
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = _clf_args.output or os.path.join(
    MODEL_DIR,
    f"{CATEGORY}_classifier.pth"
)


# ============================================================
# 3. IMAGE TRANSFORMS
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

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


val_transforms = transforms.Compose([
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
# 4. LOAD DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    os.path.join(
        DATA_DIR,
        "train"
    ),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    os.path.join(
        DATA_DIR,
        "val"
    ),
    transform=val_transforms
)


print("\nClasses:")
print(train_dataset.classes)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


# ============================================================
# 5. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=_clf_args.batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=_clf_args.batch_size,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 6. LOAD PRETRAINED RESNET18
# ============================================================

print("\nLoading ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ============================================================
# 7. FREEZE ALL LAYERS FIRST
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# 8. FINE-TUNE RESNET18 LAYER4
# ============================================================

for parameter in model.layer4.parameters():
    parameter.requires_grad = True


# ============================================================
# 9. REPLACE FINAL CLASSIFIER
# ============================================================

number_of_classes = len(
    train_dataset.classes
)

number_of_features = (
    model.fc.in_features
)

model.fc = nn.Linear(
    number_of_features,
    number_of_classes
)


# Final classifier must be trainable
for parameter in model.fc.parameters():
    parameter.requires_grad = True


model = model.to(device)


# ============================================================
# 10. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    [
        {
            "params": model.layer4.parameters(),
            "lr": 0.0001
        },
        {
            "params": model.fc.parameters(),
            "lr": 0.001
        }
    ]
)


# ============================================================
# 12. TRAINING SETTINGS
# ============================================================

num_epochs = _clf_args.epochs

best_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


print("\nStarting fine-tuning...")
print("=" * 60)


# ============================================================
# 13. TRAINING LOOP
# ============================================================

for epoch in range(num_epochs):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    ) * 100


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0

    val_total = 0

    val_loss_total = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item()
                * images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = (
        val_loss_total / val_total
    )

    val_accuracy = (
        val_correct / val_total
    ) * 100


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


    print(
        f"Epoch [{epoch + 1:02d}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes
    },
    MODEL_PATH
)


# ============================================================
# 15. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)

print("Fine-tuning completed!")

print(
    f"Best validation accuracy: "
    f"{best_accuracy:.2f}%"
)

print(
    "Model saved at:"
)

print(MODEL_PATH)

print("=" * 60)