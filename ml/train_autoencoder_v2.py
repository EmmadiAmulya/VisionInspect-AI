import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "train"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "bottle_autoencoder_v2.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("VisionInspect AI - Autoencoder V2")
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
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

print()
print("Total GOOD images:", len(dataset))
print("Classes:", dataset.classes)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(len(dataset) * 0.8)
validation_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
    generator=generator
)

print()
print("Dataset split:")
print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False
)


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
            nn.ConvTranspose2d(
                256, 128, 3, 2, 1, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64, 3, 2, 1, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32, 3, 2, 1, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3, 3, 2, 1, 1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded


# ============================================================
# CREATE MODEL
# ============================================================

model = Autoencoder()

model.to(device)

print()
print("Autoencoder V2 created successfully.")


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# TRAINING
# ============================================================

epochs = 30

best_validation_loss = float("inf")

print()
print("Starting training...")
print("Epochs:", epochs)
print("Batch size:", batch_size)


for epoch in range(epochs):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for images, _ in train_loader:

        images = images.to(device)

        reconstructed = model(images)

        loss = criterion(
            reconstructed,
            images
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for images, _ in validation_loader:

            images = images.to(device)

            reconstructed = model(images)

            loss = criterion(
                reconstructed,
                images
            )

            validation_loss += loss.item()

    validation_loss = (
        validation_loss /
        len(validation_loader)
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Validation Loss: {validation_loss:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("========================================")
print("V2 TRAINING COMPLETE")
print("========================================")

print(
    "Best validation loss:",
    f"{best_validation_loss:.6f}"
)

print()
print("Model saved at:")
print(MODEL_PATH)