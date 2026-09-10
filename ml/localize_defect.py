import os
import cv2
import torch
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "bottle_autoencoder_v2.pth"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test",
    "broken_small",
    "000.png"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# AUTOENCODER MODEL
# ============================================================

class Autoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                256, 128, 3, stride=2,
                padding=1, output_padding=1
            ),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(
                128, 64, 3, stride=2,
                padding=1, output_padding=1
            ),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(
                64, 32, 3, stride=2,
                padding=1, output_padding=1
            ),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(
                32, 3, 3, stride=2,
                padding=1, output_padding=1
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

print("V2 model loaded.")


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image:\n{IMAGE_PATH}"
    )

image = cv2.resize(image, (224, 224))

original = image.copy()

# OpenCV uses BGR.
# Convert to RGB for PyTorch.
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalize
input_image = rgb.astype(np.float32) / 255.0

# Convert HWC -> CHW
input_tensor = torch.from_numpy(
    input_image.transpose(2, 0, 1)
).unsqueeze(0).to(device)


# ============================================================
# RECONSTRUCTION
# ============================================================

with torch.no_grad():

    reconstructed = model(input_tensor)

reconstructed = reconstructed.squeeze(0).cpu().numpy()

reconstructed = reconstructed.transpose(1, 2, 0)

reconstructed = np.clip(
    reconstructed * 255,
    0,
    255
).astype(np.uint8)


# ============================================================
# DIFFERENCE MAP
# ============================================================

original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

difference = cv2.absdiff(
    original_rgb,
    reconstructed
)

# Convert RGB difference to grayscale
gray_difference = cv2.cvtColor(
    difference,
    cv2.COLOR_RGB2GRAY
)


# ============================================================
# REMOVE VERY SMALL NOISE
# ============================================================

# Small Gaussian blur reduces pixel-level noise.
blurred = cv2.GaussianBlur(
    gray_difference,
    (5, 5),
    0
)


# ============================================================
# THRESHOLD
# ============================================================

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

print(
    f"Localization threshold: "
    f"{threshold_value:.2f}"
)


# ============================================================
# REMOVE IMAGE BORDER
# ============================================================

# Ignore pixels very close to the image border.
# These are often caused by resizing/background differences.

binary[:10, :] = 0
binary[-10:, :] = 0
binary[:, :10] = 0
binary[:, -10:] = 0


# ============================================================
# MORPHOLOGICAL CLEANING
# ============================================================

kernel = np.ones((3, 3), np.uint8)

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


# ============================================================
# FIND CONNECTED COMPONENTS
# ============================================================

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary,
    connectivity=8
)

regions = []

for i in range(1, num_labels):

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]

    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    area = stats[i, cv2.CC_STAT_AREA]

    # Ignore extremely tiny noise.
    if area < 15:
        continue

    # Ignore extremely large regions.
    if area > 3000:
        continue

    # Ignore regions that are almost the entire image.
    if w > 150 or h > 150:
        continue

    # Calculate average anomaly intensity.
    region_pixels = blurred[labels == i]

    if len(region_pixels) == 0:
        continue

    mean_intensity = float(
        np.mean(region_pixels)
    )

    max_intensity = float(
        np.max(region_pixels)
    )

    # Score combines size and intensity.
    score = (
        mean_intensity * 0.5
        +
        max_intensity * 0.5
    )

    regions.append({
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "mean": mean_intensity,
        "max": max_intensity,
        "score": score
    })


# ============================================================
# SORT REGIONS
# ============================================================

regions = sorted(
    regions,
    key=lambda r: r["score"],
    reverse=True
)

print(
    f"Candidate regions detected: "
    f"{len(regions)}"
)


# ============================================================
# DRAW RESULT
# ============================================================

result = original.copy()

if len(regions) > 0:

    # Select strongest candidate
    best = regions[0]

    x = best["x"]
    y = best["y"]
    w = best["w"]
    h = best["h"]

    area = best["area"]
    score = best["score"]

    # Add a small padding around the detected region.
    padding = 8

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(224, x + w + padding)
    y2 = min(224, y + h + padding)

    final_width = x2 - x1
    final_height = y2 - y1

    # Draw bounding box.
    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

    # Label.
    cv2.putText(
        result,
        "Potential defect",
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    print()
    print("========== DEFECT LOCALIZATION ==========")
    print("Status: Potential defect detected")
    print(f"X: {x1}")
    print(f"Y: {y1}")
    print(f"Width: {final_width}")
    print(f"Height: {final_height}")
    print(f"Region area: {area}")
    print(f"Defect score: {score:.2f}")

else:

    print()
    print("========== DEFECT LOCALIZATION ==========")
    print("Status: No significant defect region detected")


# ============================================================
# SAVE OUTPUT
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "localized_defect_v2.png"
)

cv2.imwrite(
    output_path,
    result
)

print()
print("Localization image saved:")
print(output_path)