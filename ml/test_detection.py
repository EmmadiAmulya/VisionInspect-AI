import os
import cv2
import torch
import numpy as np


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


GOOD_IMAGE = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test",
    "good",
    "000.png"
)

DEFECT_IMAGE = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test",
    "broken_small",
    "000.png"
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
# ANALYZE IMAGE
# ============================================================

def analyze_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            image_path
        )

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

        reconstructed = model(
            tensor
        )

    # --------------------------------------------------------
    # GLOBAL RECONSTRUCTION ERROR
    # --------------------------------------------------------

    reconstruction_error = torch.mean(
        (tensor - reconstructed) ** 2
    ).item()

    # --------------------------------------------------------
    # DIFFERENCE IMAGE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOCAL ANOMALY REGIONS
    # --------------------------------------------------------

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

    # Remove border noise
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

    regions = []

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

        if area < 15:
            continue

        if area > 3000:
            continue

        if w > 150 or h > 150:
            continue

        pixels = blurred[
            labels == i
        ]

        if len(pixels) == 0:
            continue

        mean_value = float(
            np.mean(pixels)
        )

        max_value = float(
            np.max(pixels)
        )

        score = (
            mean_value * 0.5
            +
            max_value * 0.5
        )

        regions.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area": int(area),
            "score": float(score)
        })

    regions.sort(
        key=lambda r: r["score"],
        reverse=True
    )

    return (
        reconstruction_error,
        regions
    )


# ============================================================
# TEST GOOD IMAGE
# ============================================================

good_error, good_regions = analyze_image(
    GOOD_IMAGE
)

print()
print("========================================")
print("GOOD IMAGE")
print("========================================")

print(
    f"Global reconstruction error: "
    f"{good_error:.6f}"
)

print(
    f"Candidate regions: "
    f"{len(good_regions)}"
)

if good_regions:

    print(
        "Strongest local region:"
    )

    print(
        good_regions[0]
    )

else:

    print(
        "No local region detected."
    )


# ============================================================
# TEST DEFECT IMAGE
# ============================================================

defect_error, defect_regions = analyze_image(
    DEFECT_IMAGE
)

print()
print("========================================")
print("DEFECT IMAGE")
print("========================================")

print(
    f"Global reconstruction error: "
    f"{defect_error:.6f}"
)

print(
    f"Candidate regions: "
    f"{len(defect_regions)}"
)

if defect_regions:

    print(
        "Strongest local region:"
    )

    print(
        defect_regions[0]
    )

else:

    print(
        "No local region detected."
    )


# ============================================================
# COMPARISON
# ============================================================

print()
print("========================================")
print("COMPARISON")
print("========================================")

if good_regions:
    good_local_score = good_regions[0]["score"]
else:
    good_local_score = 0

if defect_regions:
    defect_local_score = defect_regions[0]["score"]
else:
    defect_local_score = 0

print(
    f"Good global score:   {good_error:.6f}"
)

print(
    f"Defect global score: {defect_error:.6f}"
)

print(
    f"Good local score:    {good_local_score:.2f}"
)

print(
    f"Defect local score:  {defect_local_score:.2f}"
)

print()
print("Detection test completed.")