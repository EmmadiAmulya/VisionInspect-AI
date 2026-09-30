"""Pixel-level localization eval vs ground_truth masks (Phase 3).

Metrics: pixel-AUROC, IoU, PRO-proxy (mean recall over connected GT
components). Works for any category with ground_truth/<defect>/*_mask.png.

Usage:
    python ml/evaluate_ground_truth.py --category bottle --limit 0
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_utils import (  # noqa: E402
    get_category_dirs, get_test_classes, normalize_category,
    resolve_autoencoder_path,
)
from model_defs import Autoencoder  # noqa: E402
from common import IMG_SIZE, DEVICE  # noqa: E402

parser = argparse.ArgumentParser(description="Pixel-level GT evaluation.")
parser.add_argument("--category", default="bottle")
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

CATEGORY = normalize_category(args.category)
DIRS = get_category_dirs(CATEGORY)
MODEL_PATH = resolve_autoencoder_path(CATEGORY)
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Weights not found: {MODEL_PATH}")

model = Autoencoder().to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()


def anomaly_map(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        recon = model(t).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    diff = np.abs(rgb - recon).mean(axis=2)
    return (diff * 255).astype(np.float32)


def load_gt_mask(category, defect, stem):
    for ext in (".png", ".PNG"):
        for suffix in (f"{stem}_mask{ext}", f"{stem}{ext}"):
            p = os.path.join(DIRS["ground_truth_dir"], defect, suffix)
            if os.path.exists(p):
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m = cv2.resize(m, (IMG_SIZE, IMG_SIZE))
                    return (m > 127).astype(np.uint8)
    return None


def auroc(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    P, N = tp[-1], fp[-1]
    if P == 0 or N == 0:
        return float("nan")
    tpr = np.concatenate([[0], tp / P])
    fpr = np.concatenate([[0], fp / N])
    return float(np.trapz(tpr, fpr))


all_scores, all_labels = [], []
ious, pro_recalls = [], []
classes = [c for c in get_test_classes(CATEGORY)]

for defect in classes:
    folder = os.path.join(DIRS["test_dir"], defect)
    if not os.path.isdir(folder):
        continue
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if args.limit:
        files = files[: args.limit]
    for fname in files:
        stem = os.path.splitext(fname)[0]
        scores = anomaly_map(os.path.join(folder, fname))
        if defect == "good":
            all_scores.append(scores.ravel())
            all_labels.append(np.zeros_like(scores.ravel()))
            continue
        mask = load_gt_mask(CATEGORY, defect, stem)
        if mask is None:
            continue
        all_scores.append(scores.ravel())
        all_labels.append(mask.ravel().astype(np.uint8))
        thr = float(np.percentile(scores, 97))
        pred = (scores >= thr).astype(np.uint8)
        inter = int(np.logical_and(pred, mask).sum())
        union = int(np.logical_or(pred, mask).sum())
        ious.append(inter / max(1, union))
        n_lab, lab = cv2.connectedComponents(mask, connectivity=8)[:2]
        recs = []
        for i in range(1, n_lab):
            comp = (lab == i)
            recs.append(float((pred[comp] > 0).mean()) if comp.sum() else 0.0)
        if recs:
            pro_recalls.append(float(np.mean(recs)))

S = np.concatenate(all_scores) if all_scores else np.array([])
Y = np.concatenate(all_labels) if all_labels else np.array([])
print(f"Category: {CATEGORY} | pixels={S.size} "
      f"positives={int(Y.sum()) if Y.size else 0}")
print(f"Pixel-AUROC: {auroc(S, Y):.4f}" if S.size else "Pixel-AUROC: n/a")
print(f"Mean IoU @p97: {float(np.mean(ious)):.4f} (n={len(ious)})"
      if ious else "Mean IoU: n/a (no masks)")
print(f"Mean PRO-recall @p97: {float(np.mean(pro_recalls)):.4f} "
      f"(n={len(pro_recalls)})" if pro_recalls else "PRO: n/a")
