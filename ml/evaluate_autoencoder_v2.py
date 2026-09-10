import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# -----------------------------
# 1. Autoencoder model
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# -----------------------------
# 2. Settings
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "train"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "bottle_autoencoder_v2.pth"
)


# -----------------------------
# 3. Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -----------------------------
# 4. Load model
# -----------------------------
device = torch.device("cpu")

model = Autoencoder().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("V2 model loaded successfully.")


# -----------------------------
# 5. Reconstruction error
# -----------------------------
def calculate_error(image):

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        reconstructed = model(image)

        error = torch.mean(
            (image - reconstructed) ** 2
        ).item()

    return error


# -----------------------------
# 6. Recreate validation split
# -----------------------------
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

train_size = int(0.8 * len(train_dataset))
validation_size = len(train_dataset) - train_size

generator = torch.Generator().manual_seed(42)

_, validation_dataset = random_split(
    train_dataset,
    [train_size, validation_size],
    generator=generator
)


print()
print("Validation images:", len(validation_dataset))


# -----------------------------
# 7. Calculate validation errors
# -----------------------------
validation_errors = []

for image, label in validation_dataset:

    error = calculate_error(image)

    validation_errors.append(error)


validation_errors = np.array(validation_errors)

mean_validation = np.mean(validation_errors)
std_validation = np.std(validation_errors)

threshold = mean_validation + 3 * std_validation


print()
print("Validation Mean Error:", mean_validation)
print("Validation Std Error:", std_validation)
print("Detection Threshold:", threshold)


# -----------------------------
# 8. Load test dataset
# -----------------------------
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)


print()
print("Test classes:", test_dataset.classes)
print("Total test images:", len(test_dataset))


# -----------------------------
# 9. Evaluate each category
# -----------------------------
category_errors = {}

for class_name in test_dataset.classes:

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    errors = []

    for filename in os.listdir(class_dir):

        file_path = os.path.join(
            class_dir,
            filename
        )

        try:

            image = transform(
                datasets.folder.default_loader(file_path)
            )

            error = calculate_error(image)

            errors.append(error)

        except Exception:
            pass

    category_errors[class_name] = errors


# -----------------------------
# 10. Display category results
# -----------------------------
print()
print("========== V2 RESULTS ==========")

for category, errors in category_errors.items():

    if len(errors) == 0:
        continue

    print()
    print(category)

    print(
        "Images:",
        len(errors)
    )

    print(
        "Average Error:",
        np.mean(errors)
    )

    print(
        "Minimum Error:",
        np.min(errors)
    )

    print(
        "Maximum Error:",
        np.max(errors)
    )


# -----------------------------
# 11. Calculate predictions
# -----------------------------
y_true = []
y_pred = []


for category, errors in category_errors.items():

    for error in errors:

        if category == "good":

            y_true.append(0)

        else:

            y_true.append(1)


        if error > threshold:

            y_pred.append(1)

        else:

            y_pred.append(0)


# -----------------------------
# 12. Calculate metrics
# -----------------------------
y_true = np.array(y_true)
y_pred = np.array(y_pred)

tp = np.sum(
    (y_true == 1) & (y_pred == 1)
)

tn = np.sum(
    (y_true == 0) & (y_pred == 0)
)

fp = np.sum(
    (y_true == 0) & (y_pred == 1)
)

fn = np.sum(
    (y_true == 1) & (y_pred == 0)
)


accuracy = (tp + tn) / len(y_true)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)


print()
print("========== PERFORMANCE ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print()
print("Confusion Matrix")
print("-----------------")
print("True Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)