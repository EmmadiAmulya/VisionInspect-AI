"""Unified per-category evaluation (Phase 2).

Wraps backend inference for any category:
    python ml/evaluate_category.py --category cable
    python ml/evaluate_category.py --category bottle --limit 20
"""
import argparse
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend", "app"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))

from ml_inference import inspect_image, get_supported_categories  # noqa: E402
from category_utils import (  # noqa: E402
    get_category_dirs, get_test_classes, normalize_category,
)

parser = argparse.ArgumentParser(description="Evaluate category pipeline.")
parser.add_argument("--category", default="bottle")
parser.add_argument("--limit", type=int, default=0,
                    help="Max images per class (0 = all).")
args = parser.parse_args()

CATEGORY = normalize_category(args.category)
DIRS = get_category_dirs(CATEGORY)
classes = get_test_classes(CATEGORY)
print(f"Category: {CATEGORY} | classes: {classes}")
print(f"Models: {get_supported_categories()}")

tp = tn = fp = fn = 0
total = correct = 0
per_class_total = Counter()
per_class_correct = Counter()

for actual in classes:
    folder = os.path.join(DIRS["test_dir"], actual)
    if not os.path.isdir(folder):
        print(f"Skip missing {folder}")
        continue
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if args.limit:
        files = files[: args.limit]
    print(f"\n{actual}: {len(files)} images")
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            pred = inspect_image(path, category=CATEGORY)
        except Exception as exc:
            print(f"  ERROR {fname}: {exc}")
            continue
        total += 1
        actual_def = actual != "good"
        pred_def = bool(pred["defect_detected"])
        if actual_def and pred_def:
            tp += 1
            correct += 1
        elif not actual_def and not pred_def:
            tn += 1
            correct += 1
        elif not actual_def and pred_def:
            fp += 1
        else:
            fn += 1
        per_class_total[actual] += 1
        if pred["defect_type"] == actual:
            per_class_correct[actual] += 1

acc = correct / max(1, total)
prec = tp / max(1, tp + fp)
rec = tp / max(1, tp + fn)
f1 = 2 * prec * rec / max(1e-9, prec + rec)
print("\n" + "=" * 60)
print(f"Total={total} acc={acc:.4f} prec={prec:.4f} "
      f"rec={rec:.4f} f1={f1:.4f} (TP={tp} TN={tn} FP={fp} FN={fn})")
for c in classes:
    t = per_class_total[c]
    print(f"  {c:25s} {per_class_correct[c]:4d}/{t:4d} "
          f"({per_class_correct[c]/max(1,t):.3f})")
