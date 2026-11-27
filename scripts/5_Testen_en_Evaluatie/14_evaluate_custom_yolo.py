
from multiprocessing import freeze_support
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ==========================================
# INSTELLINGEN
# ==========================================

MODEL_PATH = (
    r"D:\AI_Lab\Hornet_YOLO\Training"
    r"\hornet_detector_v1\weights\best.pt"
)

DATA_PATH = (
    r"D:\AI_Lab\Hornet_YOLO"
    r"\Hornet_YOLO_ready\data.yaml"
)

OUTPUT = Path(
    r"D:\AI_Lab\Hornet_YOLO\Evaluation"
    r"\independent_test"
)

CONFIDENCE = 0.25


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # ======================================
    # MODEL LADEN EN EVALUEREN
    # ======================================

    model = YOLO(MODEL_PATH)

    print("Starting independent test evaluation...")

    metrics = model.val(
        data=DATA_PATH,
        split="test",
        imgsz=960,
        device=0,
        workers=0,
        conf=CONFIDENCE,
        plots=True,
        project=str(OUTPUT.parent),
        name=OUTPUT.name,
        exist_ok=True
    )

    # ======================================
    # OBJECTDETECTIE CONFUSION MATRIX
    # ======================================

    matrix = np.asarray(
        metrics.confusion_matrix.matrix
    )

    # Eén objectklasse: hornet_nest
    # Rijen = voorspeld
    # Kolommen = werkelijk
    #
    # [TP, FP]
    # [FN, achtergrond (niet gedefinieerd)]

    if matrix.shape != (2, 2):
        raise ValueError(
            f"Onverwachte matrixvorm: {matrix.shape}"
        )

    TP = int(round(matrix[0, 0]))
    FP = int(round(matrix[0, 1]))
    FN = int(round(matrix[1, 0]))

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\n===== CONFUSION MATRIX =====")
    print("True Positives:", TP)
    print("False Positives:", FP)
    print("False Negatives:", FN)
    print("True Negatives: niet gedefinieerd")

    print("\n===== BEREKENDE WAARDEN =====")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # ======================================
    # GRAFIEK 1: CONFUSION MATRIX
    # ======================================

    fig, ax = plt.subplots(figsize=(8, 6))

    display_matrix = np.array([
        [TP, FP],
        [FN, np.nan]
    ], dtype=float)

    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#eeeeee")

    ax.imshow(
        np.ma.masked_invalid(display_matrix),
        cmap=cmap
    )

    labels = [
        ["TP", "FP"],
        ["FN", "N.v.t."]
    ]

    for i in range(2):
        for j in range(2):

            value = display_matrix[i, j]

            text = (
                labels[i][j]
                if np.isnan(value)
                else f"{labels[i][j]}\n{int(value)}"
            )

            ax.text(
                j, i, text,
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold"
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Positive", "Negative"])
    ax.set_yticklabels(["Positive", "Negative"])

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    ax.set_title("YOLO26s - Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        OUTPUT / "confusion_matrix_paper.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # GRAFIEK 2: TP, FP, FN
    # ======================================

    names = [
        "True Positives",
        "False Positives",
        "False Negatives"
    ]

    counts = [TP, FP, FN]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        names,
        counts,
        color=["#2e8b57", "#e5a33d", "#d9534f"]
    )

    ax.bar_label(
        bars,
        padding=4,
        fontsize=12
    )

    ax.set_title(
        "YOLO26s - Detectieresultaten"
    )

    ax.set_ylabel("Aantal detecties")
    ax.set_ylim(0, max(counts + [1]) * 1.15)

    plt.tight_layout()

    plt.savefig(
        OUTPUT / "tp_fp_fn_paper.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # GRAFIEK 3: MODEL PRESTATIES
    # ======================================

    scores = {
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        list(scores.keys()),
        [v * 100 for v in scores.values()],
        color=["#357abd", "#36a98f",
               "#7e62b5", "#dc9849"]
    )

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=4
    )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")

    ax.set_title(
        "YOLO26s - Onafhankelijke testevaluatie"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT / "model_prestaties_paper.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # RESULTATEN OPSLAAN IN CSV
    # ======================================

    results = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": "Niet gedefinieerd",
        "Confidence threshold": CONFIDENCE,
        "Precision (matrix)": precision,
        "Recall (matrix)": recall,
        "F1-score (matrix)": f1,
        "Precision (Ultralytics)": metrics.box.mp,
        "Recall (Ultralytics)": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map
    }

    with open(
        OUTPUT / "resultaten_paper.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow(["Metric", "Waarde"])
        writer.writerows(results.items())

    print("\n===== ULTRALYTICS METRICS =====")

    for name, value in scores.items():
        print(f"{name}: {value:.4f}")

    print("\nGrafieken opgeslagen in:")
    print(OUTPUT)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    freeze_support()
    main()
