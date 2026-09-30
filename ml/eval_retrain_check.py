"""Held-out test evaluation for retrained classifiers (test set only)."""
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_defs import build_classifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def evaluate(category, weight_path, split="test", batch=16):
    data_dir = os.path.join(f"classification_data_{category}", split)
    ds = datasets.ImageFolder(data_dir, transform=TF)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    ckpt = torch.load(weight_path, map_location=DEVICE)
    classes = ckpt["classes"]
    assert classes == ds.classes, f"class mapping mismatch: {classes} vs {ds.classes}"
    net = build_classifier(len(classes), pretrained=False).to(DEVICE)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    yt, yp = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out = net(imgs)
            yp += out.argmax(1).cpu().tolist()
            yt += labels.tolist()
    return {
        "category": category,
        "weights": weight_path,
        "split": split,
        "classes": classes,
        "n": len(yt),
        "accuracy": accuracy_score(yt, yp),
        "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
        "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
        "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
        "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        "y_true": yt,
        "y_pred": yp,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--probe", default=None,
                    help="specific defect dir to probe, e.g. color")
    args = ap.parse_args()
    r = evaluate(args.category, args.weights, args.split)
    yt, yp = r.pop("y_true"), r.pop("y_pred")
    print(json.dumps(r, indent=2))
    if args.probe:
        idx = r["classes"].index(args.probe)
        mask = [t == idx for t in yt]
        got = [yp[i] for i, m in enumerate(mask) if m]
        ok = sum(1 for p in got if p == idx)
        print(f"PROBE {args.category}/{args.probe}: {ok}/{len(got)} correct "
              f"-> {[r['classes'][p] for p in got]}")
