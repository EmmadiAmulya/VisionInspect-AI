"""Shared constants and weight-loading helpers for ml/ scripts.

Centralizes magic numbers used across detection / localization scripts
so thresholds and morphology parameters stay consistent.
"""
import os
import torch

# ============================================================
# SHARED CONSTANTS
# ============================================================

IMG_SIZE = 224
PERCENTILE = 97
GAUSS = (5, 5)
BORDER = 10
AREA_MIN = 15
AREA_MAX = 3000
MAX_WH = 150

# Single device definition: prefer CUDA when available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# WEIGHT HELPERS
# ============================================================

def ensure_model_exists(model_path):
    """Raise a friendly error if a weight file is missing."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            f"Train the model first or check the path."
        )
    return model_path


def load_autoencoder_weights(model, model_path, device=None):
    """Load plain autoencoder state_dict (weights_only=True)."""
    device = device if device is not None else DEVICE
    ensure_model_exists(model_path)
    state = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    return model


def load_classifier_checkpoint(model_path, device=None):
    """Load classifier dict checkpoint (weights_only=False required).

    Classifier files store a dict like
    {"model_state_dict": ..., "classes": ...}, not a plain tensor
    state_dict, so weights_only=False is required here.
    """
    device = device if device is not None else DEVICE
    ensure_model_exists(model_path)
    # Classifier checkpoint is a dict -> weights_only=False required.
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    return checkpoint
