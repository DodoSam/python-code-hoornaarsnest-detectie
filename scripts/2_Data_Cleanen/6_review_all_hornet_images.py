"""Review every remaining image in ResNet v2 output, in category order.

Run: python review_all_hornet_images.py
Dependencies: pillow (tkinter is included with most Windows Python installations)
Images are NEVER moved or deleted. Progress is saved after every label.
"""

import csv
import os
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

CATEGORIES = ("nests", "no_nests", "review")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABELS = ("nest", "no_nest", "unsure")


def normalized(value):
    return str(value or "").replace("\\", "/").lstrip("./")


def atomic_csv_write(destination, fieldnames, rows):
    temp = destination.with_name(destination.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, destination)


def read_csv(path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summary_text(items, labels, missing):
    counts = Counter((item["category"], labels.get(item["key"], "unlabeled")) for item in items)
    total = len(items)
    done = sum(key in labels for key in (item["key"] for item in items))
    lines = [
        "VESPA VELUTINA - COMPLETE MANUAL REVIEW REPORT",
        "=" * 50,
        f"Images currently present: {total}",
        f"Labeled: {done} | Remaining: {total - done}",
        f"Previously listed but now missing from folders: {len(missing)}",
        "Missing images are NOT assigned any label or included in metrics.",
        "",
        "Per AI category (Nest / No nest / Unsure / Unlabeled):",
    ]
    for category in CATEGORIES:
        n = sum(item["category"] == category for item in items)
        values = [counts[category, lab] for lab in (*LABELS, "unlabeled")]
        lines.append(f"  {category:8s} ({n:4d}): {values[0]:4d} / {values[1]:4d} / {values[2]:4d} / {values[3]:4d}")
    # Binary classification metrics apply to only the 'nests' and 'no_nests' categories.
    tp = counts["nests", "nest"]
    fp = counts["nests", "no_nest"]
    fn = counts["no_nests", "nest"]
    tn = counts["no_nests", "no_nest"]
    lines += [
        "",
        "CONFUSION MATRIX (reviewed, definite labels in nests/no_nests only)",
        f"TP = {tp}, FP = {fp}, TN = {tn}, FN = {fn}",
    ]

    def fmt(n, d):
        return f"{100 * n / d:.1f}%" if d else "N/A (no definite labels)"

    lines.extend([
        f"False-positive proportion in predicted nests = FP / (TP+FP): {fmt(fp, tp + fp)}",
        f"Precision = TP / (TP+FP): {fmt(tp, tp + fp)}",
        f"Recall = TP / (TP+FN): {fmt(tp, tp + fn)}",
        f"False-negative rate = FN / (TP+FN): {fmt(fn, tp + fn)}",
        f"Specificity = TN / (TN+FP): {fmt(tn, tn + fp)}",
        f"False-positive rate = FP / (FP+TN): {fmt(fp, fp + tn)}",
        "",
        f"Review-folder human-confirmed nests: {counts['review', 'nest']}",
        f"Review-folder human-confirmed no-nests: {counts['review', 'no_nest']}",
        "Review folder was an abstention, NOT a binary prediction; it is excluded from binary metrics.",
        "Undecided/unlabeled images are excluded from denominators.",
        "The reported rates describe only reviewed, present images; they can be biased if",
        "previously removed images are missing. This is not an independent test-set score.",
    ])
    return "\n".join(lines)


class Reviewer:
    def __init__(self, root, data):
        self.root = root
        self.data = data
        self.label_path = data / "full_manual_labels.csv"
        self.report_path = data / "full_review_report.txt"
        self.missing_path = data / "missing_from_model_predictions.csv"
        self.items = []
        self.labels = {}
        self.idx = 0
        self.picture = None
        self.last_size = (0, 0)
        self.load_data()
        self.build_ui()
        self.first_unlabeled()
        self.display()

    def load_data(self):
        for category in CATEGORIES:
            folder = self.data / category
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob("*"), key=lambda x: str(x).lower()):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                rel = path.relative_to(folder).as_posix()
                self.items.append({
                    "category": category, "relative_path": rel,
                    "key": (category, rel), "path": path,
                })
        if not self.items:
            raise ValueError("No supported images found in nests/, no_nests/, or review/.")

        old_rows = read_csv(self.label_path)
        existing_keys = {item["key"] for item in self.items}
        for row in old_rows:
            key = (row.get("ai_category", ""), normalized(row.get("relative_path", "")))
            val = row.get("manual_label", "")
            if key in existing_keys and val in LABELS:
                self.labels[key] = val

        # Prediction ledger lets us WARN about already-removed images.
        predictions = read_csv(self.data / "model_predictions.csv")
        missing = {}
        for row in predictions:
            cat = row.get("new_category", "")
            if cat not in CATEGORIES:
                continue
            rel = normalized(row.get("relative_path"))
            if not rel:
                out_rel = normalized(row.get("output_relative_path"))
                if out_rel.startswith(cat + "/"):
                    rel = out_rel[len(cat) + 1:]
            if rel and (cat, rel) not in existing_keys:
                missing[(cat, rel)] = {
                    "ai_category": cat, "relative_path": rel,
                    "note": "Missing from current folder; not automatically labeled",
                }
        self.missing = list(missing.values())
        atomic_csv_write(self.missing_path,
                         ["ai_category", "relative_path", "note"], self.missing)

    def build_ui(self):
        root = self.root
        root.title("Hornet Nest - Full Manual Review")
        root.geometry("1120x850")
        root.minsize(760, 580)
        container = ttk.Frame(root, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Yellow-legged hornet - Full Image Review",
                  font=("Segoe UI", 17, "bold")).pack(anchor="w")
        self.status = tk.StringVar()
        ttk.Label(container, textvariable=self.status).pack(anchor="w", pady=(4, 4))
        self.bar = ttk.Progressbar(container, mode="determinate", maximum=len(self.items))
        self.bar.pack(fill="x", pady=(0, 8))

        self.canvas = tk.Canvas(container, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        self.filename = tk.StringVar()
        ttk.Label(container, textvariable=self.filename, wraplength=1050).pack(pady=7)

        actions = ttk.Frame(container)
        actions.pack(pady=(4, 8))
        for col, (text, value) in enumerate([
            ("1  Nest", "nest"), ("2  No nest", "no_nest"), ("3  Unsure", "unsure"),
        ]):
            ttk.Button(actions, text=text, width=18,
                       command=lambda v=value: self.record(v)).grid(row=0, column=col, padx=7, pady=3)
        ttk.Button(actions, text="← Previous", command=self.previous).grid(row=1, column=0, pady=5)
        ttk.Button(actions, text="→ Next / Skip", command=self.next).grid(row=1, column=1)
        ttk.Button(actions, text="Next unlabeled (U)", command=self.jump_unlabeled).grid(row=1, column=2)
        ttk.Button(container, text="View / save statistics (S)", command=self.show_report).pack(pady=(0, 5))
        ttk.Label(container, text="1 Nest  •  2 No nest  •  3 Unsure  •  ←/→ Navigate  •  U Next unlabeled  •  S Report",
                  foreground="#666666").pack()

        for char, value in [("1", "nest"), ("2", "no_nest"), ("3", "unsure")]:
            root.bind(char, lambda event, v=value: self.record(v))
        root.bind("<Left>", lambda event: self.previous())
        root.bind("<Right>", lambda event: self.next())
        root.bind("<space>", lambda event: self.next())
        root.bind("u", lambda event: self.jump_unlabeled())
        root.bind("s", lambda event: self.show_report())
        root.protocol("WM_DELETE_WINDOW", self.close)

    def first_unlabeled(self):
        for i, item in enumerate(self.items):
            if item["key"] not in self.labels:
                self.idx = i
                return
        self.idx = 0

    def save(self):
        rows = []
        for item in self.items:
            label = self.labels.get(item["key"])
            if label:
                rows.append({
                    "ai_category": item["category"],
                    "relative_path": item["relative_path"],
                    "manual_label": label,
                })
        atomic_csv_write(self.label_path,
                         ["ai_category", "relative_path", "manual_label"], rows)
        self.report_path.write_text(
            summary_text(self.items, self.labels, self.missing), encoding="utf-8")

    def record(self, value):
        item = self.items[self.idx]
        self.labels[item["key"]] = value
        self.save()
        if self.idx < len(self.items) - 1:
            self.idx += 1
        else:
            unreviewed = [i for i, it in enumerate(self.items)
                          if it["key"] not in self.labels]
            if unreviewed:
                self.idx = unreviewed[0]
            else:
                self.show_report()
        self.display()

    def previous(self):
        if self.idx > 0:
            self.idx -= 1
            self.display()

    def next(self):
        if self.idx < len(self.items) - 1:
            self.idx += 1
            self.display()

    def jump_unlabeled(self):
        start = self.idx + 1
        for i in list(range(start, len(self.items))) + list(range(0, start)):
            if self.items[i]["key"] not in self.labels:
                self.idx = i
                self.display()
                return
        messagebox.showinfo("Review complete", "Every remaining image has a manual label.")
        self.show_report()

    def on_resize(self, event):
        new_size = (event.width, event.height)
        if abs(new_size[0] - self.last_size[0]) > 25 or abs(new_size[1] - self.last_size[1]) > 25:
            self.last_size = new_size
            self.draw_picture()

    def draw_picture(self):
        if not self.items:
            return
        item = self.items[self.idx]
        self.canvas.delete("all")
        width = max(100, self.canvas.winfo_width() - 15)
        height = max(100, self.canvas.winfo_height() - 15)
        try:
            with Image.open(item["path"]) as source:
                img = ImageOps.exif_transpose(source).convert("RGB")
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
                self.picture = ImageTk.PhotoImage(img)
            self.canvas.create_image(self.canvas.winfo_width() // 2,
                                     self.canvas.winfo_height() // 2,
                                     image=self.picture, anchor="center")
        except Exception as exc:
            self.picture = None
            self.canvas.create_text(width // 2, height // 2,
                                    text=f"Unable to display image:\n{exc}",
                                    fill="white", width=width - 30)

    def display(self):
        item = self.items[self.idx]
        done = sum(it["key"] in self.labels for it in self.items)
        self.bar["value"] = done
        current = self.labels.get(item["key"], "UNLABELED")
        self.status.set(
            f"Image {self.idx + 1:,}/{len(self.items):,}  |  "
            f"Reviewed {done:,}/{len(self.items):,}  |  "
            f"AI folder: {item['category']}  |  Your label: {current}"
        )
        self.filename.set(str(item["path"]))
        self.draw_picture()

    def show_report(self):
        self.save()
        text = summary_text(self.items, self.labels, self.missing)
        window = tk.Toplevel(self.root)
        window.title("Review metrics")
        window.geometry("880x630")
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        view = tk.Text(frame, wrap="word", font=("Consolas", 11))
        view.insert("1.0", text)
        view.config(state="disabled")
        view.pack(fill="both", expand=True)
        ttk.Button(frame, text="Close", command=window.destroy).pack(pady=8)

    def close(self):
        self.save()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(
        parent=root,
        title="Choose ScrapeResult2_cleaned_model_v2 (with nests/no_nests/review)"
    )
    root.attributes("-topmost", False)
    if not selected:
        root.destroy()
        return
    try:
        Reviewer(root, Path(selected))
    except Exception as exc:
        messagebox.showerror("Cannot start reviewer", str(exc), parent=root)
        root.destroy()
        raise
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
