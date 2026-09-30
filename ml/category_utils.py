"""Dynamic MVTec category helpers (single source of truth).

Phase 2: replaces hardcoded 'bottle' paths with discovery + validation.
- list_available_categories(): categories with train/ or test/ data on disk.
- get_test_classes(category): sorted defect class dirs under test/.
- resolve_*_path(category): canonical ml/models/{category}_*.pth locations.
- Threshold JSON: ml/models/{category}_threshold.json (per-category
  calibration written by train_category.py --calibrate).
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MVTEC_ROOT = PROJECT_ROOT / "dataset" / "mvtec_anomaly_detection"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"

DEFAULT_CATEGORY = "bottle"

KNOWN_MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def normalize_category(category) -> str:
    if category is None:
        return DEFAULT_CATEGORY
    name = str(category).strip().lower()
    return name or DEFAULT_CATEGORY


def list_available_categories():
    """Categories with train/ or test/ data present on disk."""
    found = []
    if not MVTEC_ROOT.exists():
        return [DEFAULT_CATEGORY]
    for child in sorted(MVTEC_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if (child / "train").exists() or (child / "test").exists():
            found.append(child.name.lower())
    return found or [DEFAULT_CATEGORY]


def validate_category(category) -> str:
    name = normalize_category(category)
    available = list_available_categories()
    if name not in available and name not in KNOWN_MVTEC_CATEGORIES:
        raise ValueError(
            f"Unknown product category: {category!r}. "
            f"Available: {available}"
        )
    return name


def get_category_dirs(category: str) -> dict:
    name = normalize_category(category)
    root = MVTEC_ROOT / name
    return {
        "category": name,
        "root": str(root),
        "train_dir": str(root / "train"),
        "train_good_dir": str(root / "train" / "good"),
        "test_dir": str(root / "test"),
        "ground_truth_dir": str(root / "ground_truth"),
    }


def get_test_classes(category: str):
    """Sorted class names under <category>/test (dynamic, no hardcode)."""
    test_dir = MVTEC_ROOT / normalize_category(category) / "test"
    if not test_dir.exists():
        return ["good"]
    classes = [
        p.name for p in sorted(test_dir.iterdir()) if p.is_dir()
    ]
    return classes or ["good"]


def autoencoder_candidates(category: str):
    name = normalize_category(category)
    return [
        MODEL_DIR / f"{name}_autoencoder_v2.pth",
        MODEL_DIR / f"{name}_autoencoder.pth",
    ]


def classifier_candidates(category: str):
    name = normalize_category(category)
    cands = [MODEL_DIR / f"{name}_classifier.pth"]
    if name == "bottle":
        # Legacy alias produced by an earlier clean-data run.
        cands.append(MODEL_DIR / "bottle_classifier_clean_v1.pth")
    return cands


def resolve_autoencoder_path(category: str):
    for p in autoencoder_candidates(category):
        if p.exists():
            return str(p)
    return str(autoencoder_candidates(category)[0])


def resolve_classifier_path(category: str):
    for p in classifier_candidates(category):
        if p.exists():
            return str(p)
    return str(classifier_candidates(category)[0])


def threshold_path(category: str) -> Path:
    return MODEL_DIR / f"{normalize_category(category)}_threshold.json"


def load_category_threshold(category: str, default: float = 90.0) -> float:
    path = threshold_path(category)
    try:
        if path.exists():
            data = json.loads(path.read_text())
            return float(data.get("anomaly_threshold", default))
    except (OSError, ValueError, TypeError):
        pass
    return default


def save_category_threshold(
    category: str, anomaly_threshold: float, extra: dict | None = None
) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "category": normalize_category(category),
        "anomaly_threshold": float(anomaly_threshold),
    }
    if extra:
        payload.update(extra)
    path = threshold_path(category)
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def resolve_classification_data_dir(category: str):
    """Training data root for a category (legacy bottle compat).

    - bottle (legacy): classification_data_clean/ (existing weights/data)
    - others: classification_data_{category}/
    Explicit override via env CLASSIFICATION_DATA_DIR_<CATEGORY>.
    """
    name = normalize_category(category)
    env_key = f"CLASSIFICATION_DATA_DIR_{name.upper()}"
    override = os.getenv(env_key)
    if override:
        return override
    if name == "bottle":
        legacy = PROJECT_ROOT / "classification_data_clean"
        if legacy.exists():
            return str(legacy)
        return str(PROJECT_ROOT / "classification_data_bottle")
    return str(PROJECT_ROOT / f"classification_data_{name}")
