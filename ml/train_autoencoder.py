import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from category_utils import (
        get_category_dirs, normalize_category,
    )
    from model_defs import Autoencoder as _SharedAE
    _HAS_SHARED = True
except ImportError:
    _HAS_SHARED = False

_parser = argparse.ArgumentParser(
    description="Train convolutional autoencoder (any MVTec category)."
)
_parser.add_argument("--category", default="bottle")
_parser.add_argument("--epochs", type=int, default=20)
_parser.add_argument("--batch-size", type=int, default=16)
_parser.add_argument("--output", default=None)
_train_args, _ = _parser.parse_known_args()
CATEGORY = (
    normalize_category(_train_args.category) if _HAS_SHARED
    else (str(_train_args.category).strip().lower() or "bottle")
)


# ============================================================
# 1. PROJECT PATHS (dynamic per category)
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if _HAS_SHARED:
    _DIRS = get_category_dirs(CATEGORY)
    TRAIN_DIR = _DIRS["train_dir"]
else:
    TRAIN_DIR = os.path.join(
        BASE_DIR,
        "dataset",
        "mvtec_anomaly_detection",
        CATEGORY,
        "train"
    )

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = _train_args.output or os.path.join(
    MODEL_DIR,
    f"{CATEGORY}_autoencoder.pth"
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("VisionInspect AI - Autoencoder Training")
print("========================================")

print("\nUsing device:", device)


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# 4. LOAD GOOD BOTTLE IMAGES
# ============================================================

dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

print("\nTraining dataset:")
print("Number of images:", len(dataset))
print("Classes:", dataset.classes)


# ============================================================
# 5. CREATE DATA LOADER
# ============================================================

batch_size = _train_args.batch_size

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# ============================================================
# 6. CONVOLUTIONAL AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------------
        # Encoder
        # -------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU()
        )


        # -------------------------------
        # Decoder
        # -------------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded


# ============================================================
# 7. CREATE MODEL
# ============================================================

model = Autoencoder()

model = model.to(device)

print("\nAutoencoder created successfully.")


# ============================================================
# 8. LOSS FUNCTION
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# 9. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 10. TRAINING
# ============================================================

epochs = _train_args.epochs

print("\nStarting training...")
print("Epochs:", epochs)
print("Batch size:", batch_size)


for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, _ in train_loader:

        images = images.to(device)

        # Forward pass
        outputs = model(images)

        # Compare original and reconstructed image
        loss = criterion(
            outputs,
            images
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()


    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.6f}"
    )


# ============================================================
# 11. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print("\nModel saved at:")
print(MODEL_PATH)