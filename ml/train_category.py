"""Unified per-category training (Phase 2).

Trains autoencoder (V2 w/ val split) + ResNet18 classifier for any MVTec
category into ml/models/{category}_*.pth with train/val hygiene and
per-category threshold calibration.

Usage:
    python ml/train_category.py --category cable --epochs-ae 30 --epochs-clf 20
    python ml/train_category.py --category cable --model autoencoder --calibrate
    python ml/train_category.py --category bottle --model classifier --include-train
"""
import argparse
import copy
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_utils import (
    get_category_dirs,
    list_available_categories,
    normalize_category,
    resolve_autoencoder_path,
    resolve_classifier_path,
    resolve_classification_data_dir,
    save_category_threshold,
    threshold_path,
)
from model_defs import Autoencoder, build_classifier

parser = argparse.ArgumentParser(description="Unified category training.")
parser.add_argument("--category", default="bottle")
parser.add_argument("--model", default="all",
                    choices=["all", "autoencoder", "classifier"])
parser.add_argument("--epochs-ae", type=int, default=30)
parser.add_argument("--epochs-clf", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--batch-size-clf", type=int, default=8)
parser.add_argument("--include-train", action="store_true",
                    help="Merge train/good into classifier 'good' pool.")
parser.add_argument("--data-dir", default=None)
parser.add_argument("--output-ae", default=None,
                    help="Explicit autoencoder output path (default per-category).")
parser.add_argument("--output-clf", default=None,
                    help="Explicit classifier output path (default per-category).")
parser.add_argument("--calibrate", action="store_true",
                    help="Write {category}_threshold.json from val errors.")
parser.add_argument("--list-categories", action="store_true")
args = parser.parse_args()

if args.list_categories:
    print("Available:", list_available_categories())
    raise SystemExit(0)

CATEGORY = normalize_category(args.category)
DIRS = get_category_dirs(CATEGORY)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Category: {CATEGORY} | device: {DEVICE}")

ae_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
])

if args.model in ("all", "autoencoder"):
    print("\n=== Autoencoder ===")
    full = datasets.ImageFolder(DIRS["train_dir"], transform=ae_transform)
    n_train = int(len(full) * 0.8)
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(
        full, [n_train, len(full) - n_train], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False)
    model = Autoencoder().to(DEVICE)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    out = args.output_ae or resolve_autoencoder_path(CATEGORY)
    # resolve_* returns existing file or first candidate; force v2 name.
    if CATEGORY != "bottle":
        out = os.path.join(os.path.dirname(out),
                           f"{CATEGORY}_autoencoder_v2.pth")
    best = float("inf")
    for epoch in range(args.epochs_ae):
        model.train()
        tl = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            loss = crit(model(imgs), imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= max(1, len(train_loader))
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(DEVICE)
                vl += crit(model(imgs), imgs).item()
        vl /= max(1, len(val_loader))
        print(f"Epoch [{epoch+1}/{args.epochs_ae}] "
              f"train={tl:.6f} val={vl:.6f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), out)
    print(f"Saved {out} (best val {best:.6f})")
    if args.calibrate or args.model == "all":
        import numpy as np
        errs = []
        model.eval()
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(DEVICE)
                errs.append(
                    torch.mean((model(imgs) - imgs) ** 2).item())
        errs = __import__("numpy").array(errs)
        thr = float(errs.mean() + 3 * errs.std())
        save_category_threshold(CATEGORY, thr,
                                {"best_val_loss": best,
                                 "val_images": len(val_ds)})
        print(f"Calibrated threshold {thr:.6f} -> {threshold_path(CATEGORY)}")

if args.model in ("all", "classifier"):
    print("\n=== Classifier ===")
    data_dir = args.data_dir or resolve_classification_data_dir(CATEGORY)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.15, 0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                    transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"),
                                  transform=val_tf)
    print(f"Classes: {train_ds.classes} | "
          f"train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size_clf,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_clf,
                            shuffle=False)
    net = build_classifier(len(train_ds.classes), pretrained=True).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([
        {"params": net.layer4.parameters(), "lr": 1e-4},
        {"params": net.fc.parameters(), "lr": 1e-3},
    ])
    best_acc, best_w = 0.0, copy.deepcopy(net.state_dict())
    for epoch in range(args.epochs_clf):
        net.train()
        tot = correct = 0
        rloss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out_ = net(imgs)
            loss = crit(out_, labels)
            loss.backward()
            opt.step()
            rloss += loss.item() * imgs.size(0)
            tot += labels.size(0)
            correct += (out_.argmax(1) == labels).sum().item()
        net.eval()
        vt = vc = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out_ = net(imgs)
                vt += labels.size(0)
                vc += (out_.argmax(1) == labels).sum().item()
        vacc = vc / max(1, vt) * 100
        if vacc > best_acc:
            best_acc = vacc
            best_w = copy.deepcopy(net.state_dict())
        print(f"Epoch [{epoch+1}/{args.epochs_clf}] "
              f"train_acc={correct/max(1,tot)*100:.2f}% val_acc={vacc:.2f}%")
    net.load_state_dict(best_w)
    out = args.output_clf or resolve_classifier_path(CATEGORY)
    if out.endswith("_classifier_clean_v1.pth"):
        out = out.replace("_classifier_clean_v1.pth", "_classifier.pth")
    torch.save({"model_state_dict": net.state_dict(),
                "classes": train_ds.classes}, out)
    print(f"Saved {out} (best val {best_acc:.2f}%)")

print("\nDone.")
