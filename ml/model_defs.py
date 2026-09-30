"""Shared PyTorch model definitions (single source of truth).

Phase 2: centralizes the Autoencoder + ResNet18 classifier builders so
train/eval/inference scripts no longer copy-paste the architecture.
Backward compatible: existing scripts can `from model_defs import
Autoencoder, build_classifier`.
"""
import torch
import torch.nn as nn

try:
    from torchvision import models as _tv_models
except Exception:  # pragma: no cover - torchvision always present in prod
    _tv_models = None


class Autoencoder(nn.Module):
    """Convolutional autoencoder: 224x224x3 -> 14x14x256 -> 224x224x3."""

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
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_classifier(num_classes: int, pretrained: bool = False):
    """ResNet18 backbone (frozen except layer4) + new fc head."""
    if _tv_models is None:
        raise RuntimeError("torchvision is required for the classifier")
    weights = _tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    model = _tv_models.resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def build_classifier_for_inference(num_classes: int):
    """Inference-time classifier (no pretrained download)."""
    if _tv_models is None:
        raise RuntimeError("torchvision is required for the classifier")
    model = _tv_models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
