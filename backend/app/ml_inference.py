import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"ML device: {DEVICE}")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(BASE_DIR)
)

AUTOENCODER_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ml",
    "models",
    "bottle_autoencoder_v2.pth"
)

CLASSIFIER_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ml",
    "models",
    "bottle_classifier.pth"
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
# LOAD AUTOENCODER
# ============================================================

autoencoder = Autoencoder().to(DEVICE)

autoencoder.load_state_dict(
    torch.load(
        AUTOENCODER_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

autoencoder.eval()

print(
    "Autoencoder model loaded successfully"
)


# ============================================================
# LOAD RESNET18 CLASSIFIER
# ============================================================

classifier_checkpoint = torch.load(
    CLASSIFIER_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


CLASS_NAMES = classifier_checkpoint.get(
    "classes",
    [
        "broken_large",
        "broken_small",
        "contamination",
        "good"
    ]
)


classifier = models.resnet18(
    weights=None
)


classifier.fc = nn.Linear(
    classifier.fc.in_features,
    len(CLASS_NAMES)
)


classifier.load_state_dict(
    classifier_checkpoint[
        "model_state_dict"
    ]
)

classifier = classifier.to(DEVICE)

classifier.eval()

print(
    "Classifier model loaded successfully"
)

print(
    "Classifier classes:",
    CLASS_NAMES
)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

AUTOENCODER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


CLASSIFIER_TRANSFORM = transforms.Compose([
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
# DETECTION SETTINGS
# ============================================================

LOCAL_ANOMALY_THRESHOLD = 90.0

# Minimum classifier confidence used for
# final classification decision.
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# GET AUTOENCODER RECONSTRUCTION
# ============================================================

def get_reconstruction(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original_size = image.size

    tensor = AUTOENCODER_TRANSFORM(
        image
    ).unsqueeze(
        0
    ).to(DEVICE)

    with torch.no_grad():

        reconstruction = autoencoder(
            tensor
        )

    reconstruction = (
        reconstruction
        .squeeze(0)
        .cpu()
        .numpy()
    )

    original = (
        tensor
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return (
        original,
        reconstruction,
        original_size
    )


# ============================================================
# LOCAL ANOMALY DETECTION
# ============================================================

def detect_local_anomaly(
    original,
    reconstruction
):

    # Convert CHW → HWC

    original = np.transpose(
        original,
        (1, 2, 0)
    )

    reconstruction = np.transpose(
        reconstruction,
        (1, 2, 0)
    )


    # Calculate pixel difference

    difference = np.abs(
        original - reconstruction
    )


    # Convert to grayscale

    difference_gray = np.mean(
        difference,
        axis=2
    )


    # Convert to 0-255

    difference_uint8 = cv2.normalize(
        difference_gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(
        np.uint8
    )


    # Reconstruction error

    reconstruction_error = float(
        np.mean(difference)
    )


    # Otsu threshold

    otsu_threshold, _ = cv2.threshold(
        difference_uint8,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )


    # 97th percentile

    percentile_threshold = np.percentile(
        difference_uint8,
        97
    )


    threshold = max(
        otsu_threshold,
        percentile_threshold
    )


    # Create mask

    _, mask = cv2.threshold(
        difference_uint8,
        threshold,
        255,
        cv2.THRESH_BINARY
    )


    # Remove image border

    border = 10

    mask[
        :border,
        :
    ] = 0

    mask[
        -border:,
        :
    ] = 0

    mask[
        :,
        :border
    ] = 0

    mask[
        :,
        -border:
    ] = 0


    # Morphological operations

    kernel = np.ones(
        (3, 3),
        np.uint8
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # Connected components

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )


    candidate_regions = []


    for i in range(
        1,
        num_labels
    ):

        x = int(
            stats[
                i,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                i,
                cv2.CC_STAT_TOP
            ]
        )

        width = int(
            stats[
                i,
                cv2.CC_STAT_WIDTH
            ]
        )

        height = int(
            stats[
                i,
                cv2.CC_STAT_HEIGHT
            ]
        )

        area = int(
            stats[
                i,
                cv2.CC_STAT_AREA
            ]
        )


        # Ignore tiny regions

        if area < 15:
            continue


        # Ignore extremely large regions

        if area > 3000:
            continue


        # Ignore huge width/height

        if width > 150 or height > 150:
            continue


        region_values = difference_uint8[
            y:y + height,
            x:x + width
        ]


        if region_values.size == 0:
            continue


        mean_intensity = float(
            np.mean(
                region_values
            )
        )


        max_intensity = float(
            np.max(
                region_values
            )
        )


        # Local anomaly score

        local_score = (
            0.5 * mean_intensity
            +
            0.5 * max_intensity
        )


        candidate_regions.append({

            "x": x,

            "y": y,

            "width": width,

            "height": height,

            "area": area,

            "score": local_score
        })


    # No candidate

    if not candidate_regions:

        return {

            "reconstruction_error":
                reconstruction_error,

            "local_anomaly_score":
                0.0,

            "defect_region":
                None
        }


    # Select strongest region

    best_region = max(
        candidate_regions,
        key=lambda r: r["score"]
    )


    return {

        "reconstruction_error":
            reconstruction_error,

        "local_anomaly_score":
            float(
                best_region["score"]
            ),

        "defect_region": {

            "x":
                best_region["x"],

            "y":
                best_region["y"],

            "width":
                best_region["width"],

            "height":
                best_region["height"]
        }
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_defect(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")


    tensor = CLASSIFIER_TRANSFORM(
        image
    ).unsqueeze(
        0
    ).to(DEVICE)


    with torch.no_grad():

        outputs = classifier(
            tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        confidence, predicted_index = (
            torch.max(
                probabilities,
                dim=1
            )
        )


    predicted_index = int(
        predicted_index.item()
    )


    confidence = float(
        confidence.item()
    )


    defect_type = CLASS_NAMES[
        predicted_index
    ]


    return {

        "defect_type":
            defect_type,

        "classification_confidence":
            confidence
    }


# ============================================================
# MAIN INSPECTION FUNCTION
# ============================================================

def inspect_image(
    image_path
):

    # Check image exists

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # ========================================================
    # AUTOENCODER
    # ========================================================

    (
        original,
        reconstruction,
        original_size
    ) = get_reconstruction(
        image_path
    )


    # ========================================================
    # LOCAL ANOMALY
    # ========================================================

    anomaly_result = detect_local_anomaly(
        original,
        reconstruction
    )


    reconstruction_error = (
        anomaly_result[
            "reconstruction_error"
        ]
    )


    local_anomaly_score = (
        anomaly_result[
            "local_anomaly_score"
        ]
    )


    defect_region = (
        anomaly_result[
            "defect_region"
        ]
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    classification_result = (
        classify_defect(
            image_path
        )
    )


    defect_type = (
        classification_result[
            "defect_type"
        ]
    )


    classification_confidence = (
        classification_result[
            "classification_confidence"
        ]
    )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    #
    # The classifier identifies whether the image
    # belongs to "good" or one of the defect classes.
    #
    # The autoencoder provides supporting anomaly
    # information and localization.
    #


    if (
        defect_type == "good"
        and
        classification_confidence
        >= CLASSIFICATION_CONFIDENCE_THRESHOLD
    ):

        defect_detected = False

        status = "PASS"

        final_defect_type = "good"


    elif (
        defect_type != "good"
        and
        classification_confidence
        >= CLASSIFICATION_CONFIDENCE_THRESHOLD
        and
        local_anomaly_score
        >= LOCAL_ANOMALY_THRESHOLD
    ):

        defect_detected = True

        status = "FAIL"

        final_defect_type = (
            defect_type
        )


    elif (
        defect_type != "good"
        and
        local_anomaly_score
        >= LOCAL_ANOMALY_THRESHOLD
    ):

        # Anomaly is strong but classifier
        # confidence is below 50%.
        #
        # For the prototype, we still mark
        # this as a defect because the
        # anomaly detector provides evidence.

        defect_detected = True

        status = "FAIL"

        final_defect_type = (
            defect_type
        )


    else:

        defect_detected = False

        status = "PASS"

        final_defect_type = "good"


    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "status":
            status,

        "defect_detected":
            defect_detected,

        "defect_type":
            final_defect_type,

        "classification_confidence":
            round(
                classification_confidence,
                4
            ),

        "reconstruction_error":
            round(
                reconstruction_error,
                6
            ),

        "local_anomaly_score":
            round(
                local_anomaly_score,
                2
            ),

        "threshold":
            LOCAL_ANOMALY_THRESHOLD,

        "classification_threshold":
            CLASSIFICATION_CONFIDENCE_THRESHOLD,

        "defect_region":
            defect_region
    }


    return result