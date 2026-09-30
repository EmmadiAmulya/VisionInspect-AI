import os
import cv2
import logging
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"ML device: {DEVICE}")


# ============================================================
# PROJECT PATHS (local + Docker fallback) + MULTI-CATEGORY REGISTRY
# Phase 2: dynamic per-category checkpoints ml/models/{category}_*.pth
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(BASE_DIR)
)

# Canonical MVTec categories (superset; availability = weights on disk).
KNOWN_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]

DEFAULT_CATEGORY = "bottle"


def _model_search_dirs():
    dirs = [
        os.path.join(PROJECT_ROOT, "ml", "models"),
        "/app/models",
        "/models",
        os.path.join(BASE_DIR, "models"),
    ]
    env_dir = os.getenv("MODEL_DIR", "")
    if env_dir:
        dirs.insert(0, env_dir)
    return [d for d in dirs if d]


def _find_model_file(filename: str):
    for d in _model_search_dirs():
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_model_search_dirs()[0], filename)


def _resolve_model_path(*parts: str) -> str:
    primary = os.path.join(PROJECT_ROOT, *parts)
    if os.path.exists(primary):
        return primary
    # Docker: models mounted at /models or /app/models
    for alt in (
        os.path.join("/models", parts[-1]),
        os.path.join("/app/models", parts[-1]),
        os.path.join(BASE_DIR, "models", parts[-1]),
        os.getenv("MODEL_DIR", ""),
    ):
        candidate = os.path.join(alt, "") if False else alt
        if candidate and os.path.exists(candidate):
            return candidate
    return primary


def normalize_category(category) -> str:
    if category is None:
        return DEFAULT_CATEGORY
    name = str(category).strip().lower()
    return name or DEFAULT_CATEGORY


def _autoencoder_filenames(category: str):
    name = normalize_category(category)
    return [f"{name}_autoencoder_v2.pth", f"{name}_autoencoder.pth"]


def _classifier_filenames(category: str):
    name = normalize_category(category)
    names = [f"{name}_classifier.pth"]
    if name == "bottle":
        names.append("bottle_classifier_clean_v1.pth")
    return names


def resolve_autoencoder_path(category: str) -> str:
    # Env override applies to the default (bottle) category only,
    # preserving backward compatibility for existing deployments.
    if normalize_category(category) == DEFAULT_CATEGORY:
        env_path = os.getenv("AUTOENCODER_MODEL_PATH", "")
        if env_path and os.path.exists(env_path):
            return env_path
    for filename in _autoencoder_filenames(category):
        found = _find_model_file(filename)
        if os.path.exists(found):
            return found
    if normalize_category(category) == DEFAULT_CATEGORY:
        legacy = os.getenv(
            "AUTOENCODER_MODEL_PATH",
            _resolve_model_path("ml", "models", "bottle_autoencoder_v2.pth"),
        )
        return legacy
    return _find_model_file(_autoencoder_filenames(category)[0])


def resolve_classifier_path(category: str) -> str:
    if normalize_category(category) == DEFAULT_CATEGORY:
        env_path = os.getenv("CLASSIFIER_MODEL_PATH", "")
        if env_path and os.path.exists(env_path):
            return env_path
    for filename in _classifier_filenames(category):
        found = _find_model_file(filename)
        if os.path.exists(found):
            return found
    if normalize_category(category) == DEFAULT_CATEGORY:
        legacy = os.getenv(
            "CLASSIFIER_MODEL_PATH",
            _resolve_model_path("ml", "models", "bottle_classifier.pth"),
        )
        return legacy
    return _find_model_file(_classifier_filenames(category)[0])


def _category_threshold_path(category: str):
    name = normalize_category(category)
    for d in _model_search_dirs():
        candidate = os.path.join(d, f"{name}_threshold.json")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_model_search_dirs()[0], f"{name}_threshold.json")


def load_category_threshold(category: str, default: float) -> float:
    path = _category_threshold_path(category)
    try:
        if os.path.exists(path):
            import json as _json
            with open(path, "r") as fh:
                data = _json.load(fh)
            return float(data.get("anomaly_threshold", default))
    except (OSError, ValueError, TypeError):
        pass
    return default


AUTOENCODER_MODEL_PATH = os.getenv(
    "AUTOENCODER_MODEL_PATH",
    _resolve_model_path("ml", "models", "bottle_autoencoder_v2.pth"),
)

CLASSIFIER_MODEL_PATH = os.getenv(
    "CLASSIFIER_MODEL_PATH",
    _resolve_model_path("ml", "models", "bottle_classifier.pth"),
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
# LAZY MODEL LOADING — PER-CATEGORY REGISTRY
# (import-safe: API stays up if models missing)
# Backward compat: bottle globals mirror registry entries.
# ============================================================

import threading as _threading

autoencoder = None
classifier = None
CLASS_NAMES = [
    "broken_large",
    "broken_small",
    "contamination",
    "good",
]
_MODELS_LOADED = False
_MODEL_ERROR = None

_REGISTRY = {}        # category -> {"autoencoder": ..., "classifier": ...}
_CLASS_NAMES = {}     # category -> [classes]
_REGISTRY_ERRORS = {}  # category -> str
_REGISTRY_LOCK = _threading.Lock()


def _load_category_models(category: str):
    name = normalize_category(category)
    with _REGISTRY_LOCK:
        if name in _REGISTRY:
            return _REGISTRY[name]
    ae_path = resolve_autoencoder_path(name)
    clf_path = resolve_classifier_path(name)
    if not os.path.exists(ae_path):
        raise FileNotFoundError(
            f"Autoencoder weights not found for '{name}': {ae_path}"
        )
    if not os.path.exists(clf_path):
        raise FileNotFoundError(
            f"Classifier weights not found for '{name}': {clf_path}"
        )
    _ae = Autoencoder().to(DEVICE)
    _ae.load_state_dict(
        torch.load(ae_path, map_location=DEVICE, weights_only=True)
    )
    _ae.eval()
    checkpoint = torch.load(
        clf_path, map_location=DEVICE, weights_only=False
    )
    classes = list(checkpoint.get("classes", CLASS_NAMES))
    _clf = models.resnet18(weights=None)
    _clf.fc = nn.Linear(_clf.fc.in_features, len(classes))
    _clf.load_state_dict(checkpoint["model_state_dict"])
    _clf = _clf.to(DEVICE)
    _clf.eval()
    entry = {
        "autoencoder": _ae,
        "classifier": _clf,
        "autoencoder_path": ae_path,
        "classifier_path": clf_path,
    }
    with _REGISTRY_LOCK:
        _REGISTRY[name] = entry
        _CLASS_NAMES[name] = classes
        _REGISTRY_ERRORS.pop(name, None)
    # Keep legacy bottle globals in sync.
    if name == DEFAULT_CATEGORY:
        globals()["autoencoder"] = _ae
        globals()["classifier"] = _clf
        globals()["CLASS_NAMES"] = classes
        globals()["_MODELS_LOADED"] = True
        globals()["_MODEL_ERROR"] = None
    logger.info("Loaded models for category '%s': %s", name, classes)
    return entry


def _get_category_models(category: str):
    name = normalize_category(category)
    with _REGISTRY_LOCK:
        if name in _REGISTRY:
            return _REGISTRY[name], list(_CLASS_NAMES.get(name, CLASS_NAMES))
    try:
        entry = _load_category_models(name)
        return entry, list(_CLASS_NAMES.get(name, CLASS_NAMES))
    except Exception as exc:
        with _REGISTRY_LOCK:
            _REGISTRY_ERRORS[name] = str(exc)
        if name == DEFAULT_CATEGORY:
            globals()["_MODEL_ERROR"] = str(exc)
        logger.exception("ML model loading failed for '%s'", name)
        raise RuntimeError(
            f"ML models unavailable for '{name}': {exc}"
        ) from exc


def _load_models() -> None:
    # Legacy entry point: loads the default (bottle) category.
    try:
        _load_category_models(DEFAULT_CATEGORY)
    except Exception as exc:
        globals()["_MODEL_ERROR"] = str(exc)
        logger.exception("ML model loading failed")


def _ensure_models(category=None) -> None:
    _get_category_models(category or DEFAULT_CATEGORY)


def get_supported_categories():
    """Categories with at least one checkpoint file present."""
    seen = set()
    for d in _model_search_dirs():
        if not os.path.isdir(d):
            continue
        try:
            files = os.listdir(d)
        except OSError:
            continue
        for fname in files:
            low = fname.lower()
            for suffix in ("_autoencoder_v2.pth", "_autoencoder.pth",
                           "_classifier.pth"):
                if low.endswith(suffix):
                    seen.add(low[: -len(suffix)])
    # Always advertise the default category (may surface a clear
    # missing-weights error instead of silently hiding it).
    seen.add(DEFAULT_CATEGORY)
    # Restrict to known MVTec names + anything discovered.
    ordered = sorted(seen)
    result = []
    for name in ordered:
        ae = resolve_autoencoder_path(name)
        clf = resolve_classifier_path(name)
        classes = list(_CLASS_NAMES.get(name, []))
        if not classes and os.path.exists(clf):
            try:
                ckpt = torch.load(
                    clf, map_location="cpu", weights_only=False
                )
                classes = list(ckpt.get("classes", []))
            except Exception:
                classes = []
        result.append({
            "name": name,
            "display_name": name.replace("_", " ").title(),
            "autoencoder_available": bool(os.path.exists(ae)),
            "classifier_available": bool(os.path.exists(clf)),
            "classes": classes,
            "anomaly_threshold": load_category_threshold(
                name, LOCAL_ANOMALY_THRESHOLD
                if "LOCAL_ANOMALY_THRESHOLD" in globals()
                else 90.0,
            ),
        })
    return result


def get_model_info(category=None):
    name = normalize_category(category)
    return {
        "category": name,
        "autoencoder_path": resolve_autoencoder_path(name),
        "classifier_path": resolve_classifier_path(name),
        "classes": list(_CLASS_NAMES.get(name, [])),
        "loaded": name in _REGISTRY,
        "error": _REGISTRY_ERRORS.get(name),
    }


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
# DETECTION SETTINGS (overridable via env; DB settings documented as TODO)
# ============================================================

LOCAL_ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "90.0"))

# Minimum classifier confidence used for
# final classification decision.
CLASSIFICATION_CONFIDENCE_THRESHOLD = float(
    os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.50")
)

# High-confidence classifier override (no local evidence required).
# A defect prediction at/above this confidence FAILs the part even when
# the local anomaly pipeline found no region (score 0.0), which is the
# norm for diffuse texture defects (carpet color, grid bent) whose
# difference maps never survive region filtering. Chosen on VALIDATION
# splits (never test): val wrong-class calls sit at 0.44-0.78 and stay
# PASS, while confident defect calls flip to FAIL; val goods produced
# zero false-defect predictions.
HIGH_CONFIDENCE_DEFECT_OVERRIDE = 0.90

MODEL_INPUT_SIZE = 224


# ============================================================
# GET AUTOENCODER RECONSTRUCTION
# ============================================================

def get_reconstruction(image_path, category=None):

    _ensure_models(category or DEFAULT_CATEGORY)
    entry, _ = _get_category_models(category or DEFAULT_CATEGORY)
    _ae = entry["autoencoder"]

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

        reconstruction = _ae(
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
    image_path,
    category=None,
):

    _ensure_models(category or DEFAULT_CATEGORY)
    entry, _names = _get_category_models(category or DEFAULT_CATEGORY)
    _clf = entry["classifier"]

    image = Image.open(
        image_path
    ).convert("RGB")


    tensor = CLASSIFIER_TRANSFORM(
        image
    ).unsqueeze(
        0
    ).to(DEVICE)


    with torch.no_grad():

        outputs = _clf(
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


    _names = _names or CLASS_NAMES

    defect_type = _names[
        predicted_index
    ] if 0 <= predicted_index < len(_names) else CLASS_NAMES[
        predicted_index
    ] if 0 <= predicted_index < len(CLASS_NAMES) else "good"


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
    image_path,
    category=None,
    anomaly_threshold=None,
    confidence_threshold=None,
):

    # Backward compat: legacy callers passed anomaly_threshold positionally
    # as the 2nd arg (inspect_image(path, 90.0)). Detect numeric category.
    _category = normalize_category(category)
    if isinstance(category, (int, float)):
        try:
            anomaly_threshold = (
                float(category)
                if anomaly_threshold is None
                else anomaly_threshold
            )
        except (TypeError, ValueError):
            pass
        _category = DEFAULT_CATEGORY
    elif isinstance(category, str):
        try:
            # Numeric string like "90" or "90.0" is a legacy threshold.
            _as_float = float(category)
            if anomaly_threshold is None and _category not in (
                set(KNOWN_CATEGORIES) | {"bottle"}
            ):
                anomaly_threshold = _as_float
                _category = DEFAULT_CATEGORY
        except (TypeError, ValueError):
            pass

    # Per-category calibrated default: {category}_threshold.json wins over
    # env default when no explicit DB/env override is supplied.
    _env_default = LOCAL_ANOMALY_THRESHOLD
    _calibrated = load_category_threshold(_category, _env_default)

    # Per-call overrides for DB-driven SystemSettings. When None,
    # fall back to per-category calibration, then module constants
    # (env defaults). Locals only — globals are never mutated.
    try:
        _anomaly_threshold = (
            _calibrated
            if anomaly_threshold is None
            else float(anomaly_threshold)
        )
    except (TypeError, ValueError):
        _anomaly_threshold = _calibrated

    try:
        _confidence_threshold = (
            CLASSIFICATION_CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else float(confidence_threshold)
        )
    except (TypeError, ValueError):
        _confidence_threshold = CLASSIFICATION_CONFIDENCE_THRESHOLD

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
        image_path,
        category=_category,
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

    # Rescale 224x224 detection coords back to original image size
    # so the frontend overlay is accurate for non-square uploads.
    if defect_region and original_size:
        try:
            orig_w, orig_h = original_size
            if orig_w and orig_h:
                sx = orig_w / MODEL_INPUT_SIZE
                sy = orig_h / MODEL_INPUT_SIZE
                defect_region = {
                    "x": int(defect_region["x"] * sx),
                    "y": int(defect_region["y"] * sy),
                    "width": max(1, int(defect_region["width"] * sx)),
                    "height": max(1, int(defect_region["height"] * sy)),
                    "orig_width": int(orig_w),
                    "orig_height": int(orig_h),
                }
        except (KeyError, TypeError, ZeroDivisionError):
            pass


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    classification_result = (
        classify_defect(
            image_path,
            category=_category,
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
    # Scale note: local_anomaly_score lives on a 0-255 intensity
    # scale (see detect_local_anomaly), while the calibrated
    # {category}_threshold.json value is an MSE-scale global error
    # statistic. Comparing the two is dimensionally wrong, so local
    # evidence is judged against the intensity-scale threshold here
    # (explicit caller override when supplied, else the module
    # default) — never against the MSE-scale file value. A score of
    # 0.0 means "no region survived filtering", i.e. local evidence
    # is UNAVAILABLE, not "no anomaly".
    #

    if anomaly_threshold is None:
        _local_threshold = LOCAL_ANOMALY_THRESHOLD
    else:
        try:
            _local_threshold = float(anomaly_threshold)
        except (TypeError, ValueError):
            _local_threshold = LOCAL_ANOMALY_THRESHOLD

    _has_local_evidence = (
        defect_region is not None
        and local_anomaly_score >= _local_threshold
    )


    if (
        defect_type == "good"
        and
        classification_confidence
        >= _confidence_threshold
    ):

        defect_detected = False

        status = "PASS"

        final_defect_type = "good"


    elif (
        defect_type != "good"
        and
        classification_confidence
        >= _confidence_threshold
        and
        _has_local_evidence
    ):

        defect_detected = True

        status = "FAIL"

        final_defect_type = (
            defect_type
        )


    elif (
        defect_type != "good"
        and
        classification_confidence
        >= HIGH_CONFIDENCE_DEFECT_OVERRIDE
    ):

        # Confident classifier defect without local evidence
        # (diffuse texture defects leave no local region).
        # Trust the classifier rather than discarding its
        # prediction as the old fall-through did.

        defect_detected = True

        status = "FAIL"

        final_defect_type = (
            defect_type
        )


    elif (
        defect_type != "good"
        and
        _has_local_evidence
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
            _anomaly_threshold,

        "classification_threshold":
            _confidence_threshold,

        "defect_region":
            defect_region,

        "product_category":
            _category,
    }


    return result


# ============================================================
# PHASE 3 — ANOMALY VISUALIZATION (heatmap / mask overlays)
# Returns base64 PNGs for Raw ↔ Heatmap ↔ Mask toggle in the UI.
# ============================================================

def get_anomaly_visualization(image_path, category=None):
    """Reconstruct + diff heatmap + binary mask as base64 PNGs."""
    import base64 as _b64

    _category = normalize_category(category)
    original, reconstruction, original_size = get_reconstruction(
        image_path, category=_category
    )
    anomaly = detect_local_anomaly(original, reconstruction)

    orig_hwc = np.transpose(original, (1, 2, 0))
    recon_hwc = np.transpose(reconstruction, (1, 2, 0))
    orig_u8 = np.clip(orig_hwc * 255.0, 0, 255).astype(np.uint8)
    recon_u8 = np.clip(recon_hwc * 255.0, 0, 255).astype(np.uint8)
    # CHW float in [0,1] -> HWC uint8; ensure RGB order for color map.
    diff = np.abs(
        orig_u8.astype(np.int16) - recon_u8.astype(np.int16)
    ).astype(np.uint8)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    def _encode(arr):
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        return _b64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "category": _category,
        "original_size": list(original_size) if original_size else None,
        "reconstruction_error": anomaly["reconstruction_error"],
        "local_anomaly_score": anomaly["local_anomaly_score"],
        "defect_region": anomaly["defect_region"],
        "original_b64": _encode(orig_u8),
        "reconstructed_b64": _encode(recon_u8),
        "heatmap_b64": _encode(heatmap_rgb),
    }