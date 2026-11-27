
import csv
import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, ImageOps

# ==========================================
# CONFIGURATION
# ==========================================

SAMPLE_SIZES = {
    "nests": 100,
    "no_nests": 100,
    "review": 150,
}

RANDOM_SEED = 42

SAMPLE_FILE = "review_sample.csv"
LABEL_FILE = "manual_labels.csv"

# ==========================================
# SELECT CLEANED DATASET
# ==========================================

root = tk.Tk()
root.withdraw()

folder = filedialog.askdirectory(
    parent=root,
    title="Select ScrapeResult2_cleaned"
)

if not folder:
    root.destroy()
    raise SystemExit("No folder selected.")

DATASET = Path(folder)

results_file = DATASET / "classification_results.csv"

if not results_file.exists():
    messagebox.showerror(
        "Missing file",
        "classification_results.csv was not found."
    )
    root.destroy()
    raise SystemExit

sample_path = DATASET / SAMPLE_FILE
labels_path = DATASET / LABEL_FILE

# ==========================================
# CREATE OR LOAD RANDOM SAMPLE
# ==========================================

def create_sample():
    with open(
        results_file,
        newline="",
        encoding="utf-8-sig"
    ) as f:
        records = list(csv.DictReader(f))

    rng = random.Random(RANDOM_SEED)
    selected = []

    for category, amount in SAMPLE_SIZES.items():
        candidates = [
            row for row in records
            if row["category"] == category
            and (
                DATASET
                / category
                / row["relative_path"]
            ).is_file()
        ]

        rng.shuffle(candidates)

        chosen = candidates[:amount]

        print(
            f"{category}: selected "
            f"{len(chosen)} of {len(candidates)}"
        )

        for row in chosen:
            selected.append({
                "relative_path": row["relative_path"],
                "ai_category": category,
                "whole_score": row["whole_score"],
                "max_score": row["max_score"],
            })

    rng.shuffle(selected)

    with open(
        sample_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "ai_category",
                "whole_score",
                "max_score"
            ]
        )
        writer.writeheader()
        writer.writerows(selected)

    return selected


if sample_path.exists():
    with open(
        sample_path,
        newline="",
        encoding="utf-8-sig"
    ) as f:
        sample = list(csv.DictReader(f))
else:
    sample = create_sample()

if not sample:
    messagebox.showerror(
        "No images",
        "No valid images were found."
    )
    root.destroy()
    raise SystemExit

# ==========================================
# LOAD EXISTING MANUAL LABELS
# ==========================================

labels = {}

if labels_path.exists():
    with open(
        labels_path,
        newline="",
        encoding="utf-8-sig"
    ) as f:
        for row in csv.DictReader(f):
            key = (
                row["ai_category"],
                row["relative_path"]
            )
            labels[key] = row["manual_label"]


def item_key(item):
    return (
        item["ai_category"],
        item["relative_path"]
    )


def save_labels():
    temp = labels_path.with_suffix(".csv.tmp")

    with open(
        temp,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "relative_path",
            "ai_category",
            "manual_label",
            "whole_score",
            "max_score"
        ])

        for item in sample:
            key = item_key(item)

            if key not in labels:
                continue

            writer.writerow([
                item["relative_path"],
                item["ai_category"],
                labels[key],
                item["whole_score"],
                item["max_score"]
            ])

    temp.replace(labels_path)


# ==========================================
# REVIEW WINDOW
# ==========================================

root.deiconify()
root.title("Vespa velutina - Image Reviewer")
root.geometry("1050x850")
root.minsize(650, 600)

current_index = 0
history = []
photo_reference = None

title = ttk.Label(
    root,
    text="Yellow-legged hornet nest review",
    font=("Arial", 19, "bold")
)
title.pack(pady=(15, 8))

progress_text = tk.StringVar()

progress_label = ttk.Label(
    root,
    textvariable=progress_text,
    font=("Arial", 12)
)
progress_label.pack(pady=4)

progress_bar = ttk.Progressbar(
    root,
    orient="horizontal",
    length=650,
    mode="determinate",
    maximum=len(sample)
)
progress_bar.pack(pady=8)

image_label = tk.Label(
    root,
    text="Loading image...",
    bg="#252525",
    fg="white",
    width=90,
    height=28
)
image_label.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=12
)

filename_text = tk.StringVar()

filename_label = ttk.Label(
    root,
    textvariable=filename_text,
    font=("Arial", 10)
)
filename_label.pack(pady=5)

button_frame = ttk.Frame(root)
button_frame.pack(pady=15)

# ==========================================
# NAVIGATION
# ==========================================

def next_unlabeled(start):
    count = len(sample)

    for offset in range(count):
        index = (start + offset) % count
        if item_key(sample[index]) not in labels:
            return index

    return None


def display_image():
    global photo_reference, current_index

    completed = sum(
        item_key(item) in labels for item in sample
    )

    progress_bar["value"] = completed

    progress_text.set(
        f"Reviewed: {completed} / {len(sample)}"
    )

    if completed == len(sample):
        image_label.config(
            image="",
            text="ALL IMAGES REVIEWED!",
            font=("Arial", 20, "bold")
        )
        image_label.image = None

        filename_text.set(
            f"Results saved to: {labels_path}"
        )
        return

    current_index = next_unlabeled(
        current_index
    )

    if current_index is None:
        return

    item = sample[current_index]

    path = (
        DATASET
        / item["ai_category"]
        / item["relative_path"]
    )

    filename_text.set(
        f"Image: {item['relative_path']}  |  "
        f"AI prediction: {item['ai_category']}"
    )

    try:
        with Image.open(path) as source:
            img = ImageOps.exif_transpose(source)
            img = img.convert("RGB")

        # Fit image to available screen size.
        max_width = min(
            950,
            max(400, root.winfo_screenwidth() - 120)
        )
        max_height = min(
            550,
            max(250, root.winfo_screenheight() - 330)
        )

        img.thumbnail((
            max_width,
            max_height
        ))

        photo_reference = ImageTk.PhotoImage(img)

        image_label.config(
            image=photo_reference,
            text=""
        )
        image_label.image = photo_reference

    except Exception as e:
        image_label.config(
            image="",
            text=f"Could not load image:\n{e}"
        )
        image_label.image = None


def label_image(label):
    global current_index

    if len(labels) >= len(sample):
        return

    item = sample[current_index]
    key = item_key(item)

    old_label = labels.get(key)
    history.append((current_index, old_label))

    labels[key] = label
    save_labels()

    current_index = (
        current_index + 1
    ) % len(sample)

    display_image()


def skip_image():
    global current_index

    current_index = (
        current_index + 1
    ) % len(sample)

    display_image()


def undo_label():
    global current_index

    if not history:
        return

    index, old_label = history.pop()
    key = item_key(sample[index])

    if old_label is None:
        labels.pop(key, None)
    else:
        labels[key] = old_label

    save_labels()

    current_index = index
    display_image()


# ==========================================
# BUTTONS
# ==========================================

ttk.Button(
    button_frame,
    text="1 - Nest",
    command=lambda: label_image("nest"),
    width=18
).grid(row=0, column=0, padx=8)

ttk.Button(
    button_frame,
    text="2 - No nest",
    command=lambda: label_image("no_nest"),
    width=18
).grid(row=0, column=1, padx=8)

ttk.Button(
    button_frame,
    text="3 - Unsure",
    command=lambda: label_image("unsure"),
    width=18
).grid(row=0, column=2, padx=8)

navigation_frame = ttk.Frame(root)
navigation_frame.pack(pady=5)

ttk.Button(
    navigation_frame,
    text="Skip (Space)",
    command=skip_image
).grid(row=0, column=0, padx=10)

ttk.Button(
    navigation_frame,
    text="Undo (Backspace)",
    command=undo_label
).grid(row=0, column=1, padx=10)

# ==========================================
# KEYBOARD SHORTCUTS
# ==========================================

root.bind(
    "1",
    lambda event: label_image("nest")
)

root.bind(
    "2",
    lambda event: label_image("no_nest")
)

root.bind(
    "3",
    lambda event: label_image("unsure")
)

root.bind(
    "<space>",
    lambda event: skip_image()
)

root.bind(
    "<BackSpace>",
    lambda event: undo_label()
)

# ==========================================
# START
# ==========================================

display_image()

print("\nReview tool started.")
print(f"Images to review: {len(sample)}")
print(f"Labels are saved to: {labels_path}")

root.mainloop()
