
import csv
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import pipeline

# ==========================================
# 1. CHOOSE DATASET FOLDER
# ==========================================

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

selected = filedialog.askdirectory(
    title="Select folder with your hornet photos"
)

root.destroy()

if not selected:
    raise SystemExit("No folder selected.")

INPUT = Path(selected)
OUTPUT = INPUT.parent / (INPUT.name + "_cleaned")

if OUTPUT.resolve() == INPUT.resolve():
    raise SystemExit("Input and output folders must differ.")

for name in ["nests", "no_nests", "review"]:
    (OUTPUT / name).mkdir(parents=True, exist_ok=True)

print("Original dataset:", INPUT)
print("Cleaned dataset:", OUTPUT)

# ==========================================
# 2. CLIP MODEL
# ==========================================

device = 0 if torch.cuda.is_available() else -1

print("Using:", "GPU" if device == 0 else "CPU")
print("Loading AI model...")

classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    device=device
)

# ==========================================
# 3. IMAGE DESCRIPTIONS
# ==========================================

NEST_LABELS = [
    "a large grey paper hornet nest hanging in a tree",
    "a small spherical hornet nest high in a tree",
    "an oval Asian hornet nest among green leaves",
    "a round wasp nest attached to a building",
    "a hornet nest partly hidden by branches",
    "a close-up photograph of a paper wasp nest",
]

OTHER_LABELS = [
    "a close-up photograph of a hornet insect",
    "a yellow-legged hornet sitting on a leaf",
    "a hornet flying without a visible nest",
    "multiple hornets on a branch without a nest",
    "a tree with green leaves and no nest",
    "tree branches without any hornet nest",
    "an insect on a flower",
    "a bird nest made of twigs",
]

LABELS = NEST_LABELS + OTHER_LABELS

# Initial thresholds, not calibrated probabilities.
KEEP_THRESHOLD = 0.75
CROP_THRESHOLD = 0.65
REJECT_THRESHOLD = 0.10

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ==========================================
# 4. GENERATE IMAGE CROPS
# ==========================================

def make_views(image):
    # Keep processing manageable.
    image.thumbnail((1024, 1024))

    width, height = image.size

    crop_width = max(1, int(width * 0.60))
    crop_height = max(1, int(height * 0.60))

    positions = [
        (0, 0),
        (width - crop_width, 0),
        (0, height - crop_height),
        (width - crop_width, height - crop_height),
        (
            (width - crop_width) // 2,
            (height - crop_height) // 2
        )
    ]

    views = [image.copy()]

    for x, y in positions:
        crop = image.crop((
            x, y,
            x + crop_width,
            y + crop_height
        ))
        views.append(crop)

    return views

# ==========================================
# 5. CLASSIFY ALL VIEWS
# ==========================================

def calculate_scores(image):
    views = make_views(image)

    predictions = classifier(
        views,
        candidate_labels=LABELS,
        hypothesis_template="This is a photo of {}.",
        batch_size=6
    )

    scores = []

    for prediction in predictions:
        nest_score = sum(
            item["score"]
            for item in prediction
            if item["label"] in NEST_LABELS
        )
        scores.append(nest_score)

    return scores

# ==========================================
# 6. READ EXISTING METADATA
# ==========================================

metadata_file = INPUT / "metadata.csv"
metadata_by_filename = {}
metadata_fields = []

if metadata_file.exists():
    with open(
        metadata_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)
        metadata_fields = reader.fieldnames or []

        for row in reader:
            filename = row.get("bestand")
            if filename:
                metadata_by_filename[filename] = row

# ==========================================
# 7. FIND IMAGES
# ==========================================

images = sorted(
    path for path in INPUT.rglob("*")
    if path.is_file()
    and path.suffix.lower() in EXTENSIONS
)

print(f"\nFound {len(images)} images.")

# ==========================================
# 8. PROCESS IMAGES
# ==========================================

results = []

for path in tqdm(images, desc="Cleaning dataset"):

    try:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(
                original
            ).convert("RGB")

        scores = calculate_scores(image)

        whole_score = scores[0]
        max_score = max(scores)

        strong_crops = sum(
            score >= CROP_THRESHOLD
            for score in scores[1:]
        )

        # Likely nest
        if (
            whole_score >= KEEP_THRESHOLD
            or strong_crops >= 2
        ):
            category = "nests"

        # Strong evidence of no nest
        elif max_score <= REJECT_THRESHOLD:
            category = "no_nests"

        # Uncertain: review manually
        else:
            category = "review"

        # Preserve subfolders and originals
        relative_path = path.relative_to(INPUT)
        destination = OUTPUT / category / relative_path

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(path, destination)

        results.append({
            "bestand": path.name,
            "relative_path": str(relative_path),
            "category": category,
            "whole_score": round(whole_score, 4),
            "max_score": round(max_score, 4),
            "strong_crops": strong_crops
        })

    except Exception as error:
        print(f"\nError with {path.name}: {error}")

        results.append({
            "bestand": path.name,
            "relative_path": str(path.relative_to(INPUT)),
            "category": "error",
            "whole_score": "",
            "max_score": "",
            "strong_crops": ""
        })

# ==========================================
# 9. SAVE CLASSIFICATION RESULTS
# ==========================================

results_path = OUTPUT / "classification_results.csv"

with open(
    results_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "bestand",
            "relative_path",
            "category",
            "whole_score",
            "max_score",
            "strong_crops"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

# ==========================================
# 10. SAVE METADATA BY CATEGORY
# ==========================================

if metadata_fields:

    for category in ["nests", "no_nests", "review"]:

        selected_rows = []

        for result in results:
            if result["category"] != category:
                continue

            filename = result["bestand"]

            if filename in metadata_by_filename:
                selected_rows.append(
                    metadata_by_filename[filename]
                )

        output_csv = OUTPUT / category / "metadata.csv"

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=metadata_fields,
                extrasaction="ignore"
            )

            writer.writeheader()
            writer.writerows(selected_rows)

# ==========================================
# 11. SUMMARY
# ==========================================

print("\n" + "=" * 45)
print("DATASET CLEANING FINISHED")
print("=" * 45)

for category in ["nests", "no_nests", "review", "error"]:
    count = sum(
        row["category"] == category
        for row in results
    )
    print(f"{category}: {count}")

print("\nResults saved to:", OUTPUT)
print("Classification CSV:", results_path)
print("\nOriginal images are unchanged.")
