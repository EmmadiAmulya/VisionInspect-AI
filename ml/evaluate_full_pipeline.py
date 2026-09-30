import os
import sys
from collections import Counter

# ============================================================
# ADD BACKEND APP TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BACKEND_APP = os.path.join(
    PROJECT_ROOT,
    "backend",
    "app"
)

sys.path.insert(0, BACKEND_APP)


# ============================================================
# IMPORT AI INFERENCE
# ============================================================

from ml_inference import inspect_image


# ============================================================
# DATASET PATH
# ============================================================

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test"
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "good",
    "broken_large",
    "broken_small",
    "contamination"
]


# ============================================================
# VARIABLES
# ============================================================

results = []

total_images = 0

correct_detection = 0

false_positives = 0

false_negatives = 0

true_positives = 0

true_negatives = 0


classification_results = []

class_correct = Counter()

class_total = Counter()


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for actual_class in CLASSES:

    class_dir = os.path.join(
        DATASET_DIR,
        actual_class
    )

    if not os.path.exists(class_dir):

        print(
            f"WARNING: Folder not found: {class_dir}"
        )

        continue


    image_files = sorted(
        [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ]
    )


    print()
    print("=" * 60)
    print(
        f"Processing class: {actual_class}"
    )
    print(
        f"Images: {len(image_files)}"
    )
    print("=" * 60)


    for image_file in image_files:

        image_path = os.path.join(
            class_dir,
            image_file
        )


        try:

            prediction = inspect_image(
                image_path
            )


            total_images += 1


            predicted_defect = prediction[
                "defect_detected"
            ]


            actual_defect = (
                actual_class != "good"
            )


            # ------------------------------------------------
            # DETECTION METRICS
            # ------------------------------------------------

            if actual_defect and predicted_defect:

                true_positives += 1

                correct_detection += 1


            elif (
                not actual_defect
                and not predicted_defect
            ):

                true_negatives += 1

                correct_detection += 1


            elif (
                not actual_defect
                and predicted_defect
            ):

                false_positives += 1


            elif (
                actual_defect
                and not predicted_defect
            ):

                false_negatives += 1


            # ------------------------------------------------
            # CLASSIFICATION METRICS
            # Denominator = ALL images of this class
            # (not only predicted-defect), so missed defects
            # (FN) and correct good (TN) count toward accuracy.
            # ------------------------------------------------

            predicted_class = prediction[
                "defect_type"
            ]

            class_total[
                actual_class
            ] += 1


            if predicted_class == actual_class:

                class_correct[
                    actual_class
                ] += 1


            if predicted_defect:

                classification_results.append({
                    "actual": actual_class,
                    "predicted": predicted_class,
                    "confidence": prediction[
                        "classification_confidence"
                    ]
                })


            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            results.append({

                "image": image_file,

                "actual_class": actual_class,

                "actual_defect": actual_defect,

                "predicted_defect": predicted_defect,

                "predicted_class": prediction[
                    "defect_type"
                ],

                "confidence": prediction[
                    "classification_confidence"
                ],

                "reconstruction_error": prediction[
                    "reconstruction_error"
                ],

                "local_anomaly_score": prediction[
                    "local_anomaly_score"
                ]
            })


            print(
                f"{actual_class:15s} "
                f"{image_file:10s} "
                f"-> "
                f"{prediction['status']:4s} "
                f"| "
                f"{prediction['defect_type']:15s} "
                f"| "
                f"score="
                f"{prediction['local_anomaly_score']:.2f}"
            )


        except Exception as e:

            print(
                f"ERROR processing "
                f"{image_path}: {e}"
            )


# ============================================================
# CALCULATE DETECTION METRICS
# ============================================================

if total_images > 0:

    detection_accuracy = (
        correct_detection
        / total_images
    )

else:

    detection_accuracy = 0


if (
    true_positives + false_positives
    > 0
):

    precision = (
        true_positives
        /
        (
            true_positives
            + false_positives
        )
    )

else:

    precision = 0


if (
    true_positives + false_negatives
    > 0
):

    recall = (
        true_positives
        /
        (
            true_positives
            + false_negatives
        )
    )

else:

    recall = 0


if precision + recall > 0:

    f1 = (
        2
        * precision
        * recall
        /
        (
            precision
            + recall
        )
    )

else:

    f1 = 0


# ============================================================
# PRINT DETECTION RESULTS
# ============================================================

print()
print()
print("=" * 70)
print("FULL AI PIPELINE EVALUATION")
print("=" * 70)

print()

print(
    f"Total images:       {total_images}"
)

print(
    f"Correct detection:  {correct_detection}"
)

print(
    f"True positives:     {true_positives}"
)

print(
    f"True negatives:     {true_negatives}"
)

print(
    f"False positives:    {false_positives}"
)

print(
    f"False negatives:    {false_negatives}"
)

print()

print(
    f"Accuracy:           {detection_accuracy:.4f}"
)

print(
    f"Precision:          {precision:.4f}"
)

print(
    f"Recall:             {recall:.4f}"
)

print(
    f"F1 Score:           {f1:.4f}"
)


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

print()
print("=" * 70)
print("DEFECT CLASSIFICATION RESULTS")
print("=" * 70)

print()

print(
    f"Images classified as defects: "
    f"{len(classification_results)}"
)

print()

for class_name in CLASSES:

    total = class_total[
        class_name
    ]

    correct = class_correct[
        class_name
    ]


    if total > 0:

        accuracy = (
            correct
            / total
        )

    else:

        accuracy = 0


    print(
        f"{class_name:20s}"
        f"Correct: {correct:3d}"
        f" / {total:3d}"
        f"  Accuracy: {accuracy:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION CONFUSION MATRIX")
print("=" * 70)

print()

print(
    f"{'Actual':20s}"
    f"{'Predicted':20s}"
    f"{'Count':10s}"
)

print("-" * 50)


confusion = Counter()

# Include ALL images (TN good->good included),
# not only predicted-defect entries.
for item in results:

    key = (
        item["actual_class"],
        item["predicted_class"]
    )

    confusion[key] += 1


for (
    actual,
    predicted
), count in sorted(
    confusion.items()
):

    print(
        f"{actual:20s}"
        f"{predicted:20s}"
        f"{count:10d}"
    )


# ============================================================
# CLASSIFICATION CONFIDENCE
# ============================================================

if classification_results:

    average_confidence = sum(
        item["confidence"]
        for item in classification_results
    ) / len(
        classification_results
    )

else:

    average_confidence = 0


print()

print(
    f"Average classification confidence: "
    f"{average_confidence:.4f}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print()

print(
    "Detection:"
)

print(
    f"  Accuracy  = {detection_accuracy:.4f}"
)

print(
    f"  Precision = {precision:.4f}"
)

print(
    f"  Recall    = {recall:.4f}"
)

print(
    f"  F1 Score  = {f1:.4f}"
)

print()

print(
    "Classification:"
)

print(
    f"  Classified images = "
    f"{len(classification_results)}"
)

print(
    f"  Average confidence = "
    f"{average_confidence:.4f}"
)

print()

print("=" * 70)
print("Evaluation complete.")
print("=" * 70)