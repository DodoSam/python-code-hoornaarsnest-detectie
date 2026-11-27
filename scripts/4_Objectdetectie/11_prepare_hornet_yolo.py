"""Build a train/val/test Ultralytics YOLO detection dataset from a CVAT export.

Only standard-library modules are required. Run via `python prepare_hornet_yolo.py`.
"""
from __future__ import annotations

import csv
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from tkinter import Tk, filedialog, messagebox

SEED = 42
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def get_group(stem: str) -> str:
    match = re.fullmatch(r"(\d+)_\d+", stem)
    return match.group(1) if match else stem


def select_inputs():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    zip_file = filedialog.askopenfilename(
        parent=root,
        title="Select CVAT YOLO annotation ZIP",
        filetypes=[("ZIP archive", "*.zip")],
    )
    if not zip_file:
        raise SystemExit("No CVAT ZIP selected.")
    images = filedialog.askdirectory(
        parent=root,
        title="Select local photo folder (e.g. Hornet_Manual_Verified or ScrapeResult2)",
    )
    if not images:
        raise SystemExit("No photo folder selected.")
    parent = filedialog.askdirectory(
        parent=root,
        title="Choose location to create Hornet_YOLO_ready (not inside photo source)",
    )
    root.destroy()
    if not parent:
        raise SystemExit("No output parent folder selected.")
    return Path(zip_file), Path(images), Path(parent) / "Hornet_YOLO_ready"


def read_export(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.namelist()
        image_lists = [x for x in entries if x == "train.txt" or x.endswith("/train.txt")]
        if len(image_lists) != 1:
            raise ValueError("Expected one CVAT train.txt listing all images.")
        raw_lines = archive.read(image_lists[0]).decode("utf-8-sig").splitlines()
        image_names = []
        for raw in raw_lines:
            raw = raw.strip()
            if not raw:
                continue
            filename = PurePosixPath(raw.replace("\\", "/")).name
            if Path(filename).suffix.lower() not in IMAGE_EXTS:
                raise ValueError(f"Unexpected image extension: {filename}")
            image_names.append(filename)
        if len(set(x.casefold() for x in image_names)) != len(image_names):
            raise ValueError("Duplicate image filenames in CVAT train.txt")
        label_index = {}
        for name in entries:
            if name.startswith("labels/") and name.lower().endswith(".txt"):
                stem = PurePosixPath(name).stem.casefold()
                if stem in label_index:
                    raise ValueError(f"Duplicate label name for {stem}")
                label_index[stem] = name
        # Read and validate all labels before writing any files.
        result = []
        for fname in image_names:
            stem = Path(fname).stem.casefold()
            lines = []
            if stem in label_index:
                text = archive.read(label_index[stem]).decode("utf-8-sig").strip()
                lines = text.splitlines() if text else []
            for line in lines:
                values = line.split()
                if len(values) != 5 or values[0] != "0":
                    raise ValueError(f"Unexpected label in {fname}: {line!r}")
                coords = list(map(float, values[1:]))
                if any(not (0 <= v <= 1) for v in coords) or coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError(f"Invalid YOLO bounding box for {fname}: {line!r}")
            result.append({"filename": fname, "lines": lines, "group": get_group(Path(fname).stem)})
        extra = sorted(set(label_index) - {Path(n).stem.casefold() for n in image_names})
        if extra:
            raise ValueError(f"Unexpected label files without a listed image: {extra[:5]}")
        return result


def index_photos(source: Path, output: Path):
    found = defaultdict(list)
    count = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if output == path or output in path.parents:
            continue
        found[path.name.casefold()].append(path)
        count += 1
    print(f"Indexed {count} local image files")
    return found


def select_photo(paths):
    # Prefer a filename exactly matching the CVAT entry; handled when indexing.
    return sorted(paths, key=lambda p: (len(p.parts), str(p).casefold()))[0]


def assign_splits(items):
    by_group = defaultdict(list)
    for entry in items:
        by_group[entry["group"]].append(entry)
    groups = list(by_group.values())
    rng = random.Random(SEED)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)  # stable sort keeps shuffled ties

    total = len(items)
    n_pos = sum(bool(item["lines"]) for item in items)
    n_neg = total - n_pos
    targets = {
        s: (total * frac, n_pos * frac, n_neg * frac)
        for s, frac in SPLITS.items()
    }
    counts = {s: [0, 0, 0] for s in SPLITS}
    assignment = {}

    for group in groups:
        positive = sum(bool(item["lines"]) for item in group)
        size = len(group)
        amount = [size, positive, size - positive]
        best_split, best_score = None, float("inf")
        for s in SPLITS:
            # Compare the global balance before and after assigning this group.
            score = 0.0
            for j, baseline in enumerate((total, n_pos, n_neg)):
                if not baseline:
                    continue
                for split in SPLITS:
                    next_count = counts[split][j] + (amount[j] if split == s else 0)
                    score += ((next_count - targets[split][j]) / baseline) ** 2
            if score < best_score:
                best_split, best_score = s, score
        for j in range(3):
            counts[best_split][j] += amount[j]
        for entry in group:
            assignment[entry["filename"]] = best_split

    for name in SPLITS:
        if counts[name][0] == 0:
            raise ValueError(f"Empty split {name}; check input dataset")
    return assignment, counts


def main():
    zip_path, source, output = select_inputs()
    if not source.exists():
        raise FileNotFoundError(source)
    if source == output or source in output.parents:
        raise ValueError("Output must be outside image source to avoid mixing datasets")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output already contains files: {output}. Rename or remove it first.")

    print("Reading CVAT annotations...", zip_path)
    items = read_export(zip_path)
    print(f"CVAT image entries: {len(items)}")
    print(f"With boxes: {sum(bool(i['lines']) for i in items)}")
    print(f"Without boxes: {sum(not i['lines'] for i in items)}")
    print(f"Total boxes: {sum(len(i['lines']) for i in items)}")

    index = index_photos(source, output)
    missing = []
    collisions = []
    for item in items:
        matches = index.get(item["filename"].casefold(), [])
        if not matches:
            missing.append(item["filename"])
        elif len(matches) > 1:
            collisions.append((item["filename"], matches))
        else:
            item["source"] = matches[0]
    if collisions:
        print(f"WARNING: {len(collisions)} filenames exist in multiple folders.")
        print("Using the first matching file for each; compare duplicates if needed.")
        for fname, matches in collisions:
            selected = select_photo(matches)
            for item in items:
                if item["filename"] == fname:
                    item["source"] = selected
            print(f"  {fname}: chosen {selected}")
    if missing:
        print(f"STOPPED: {len(missing)} images are missing. Example: {missing[:12]}")
        print("Choose a source that contains all 732 original photographs, then rerun.")
        raise SystemExit(2)

    assignments, counts = assign_splits(items)
    print("Assigned splits:")
    for key, numbers in counts.items():
        print(f"  {key}: {numbers[0]} images ({numbers[1]} with boxes, {numbers[2]} without)")

    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in items:
        split = assignments[item["filename"]]
        img_dst = output / "images" / split / item["filename"]
        label_dst = output / "labels" / split / (Path(item["filename"]).stem + ".txt")
        shutil.copy2(item["source"], img_dst)
        label_dst.write_text("\n".join(item["lines"]) + ("\n" if item["lines"] else ""), encoding="utf-8")
        manifest.append({"filename": item["filename"], "observation_id": item["group"],
                         "split": split, "number_of_boxes": len(item["lines"]),
                         "original_source": str(item["source"])})

    yaml_content = (
        f'path: "{output.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n  0: hornet_nest\n"
    )
    (output / "data.yaml").write_text(yaml_content, encoding="utf-8")
    with (output / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "observation_id", "split", "number_of_boxes", "original_source"])
        writer.writeheader()
        writer.writerows(manifest)
    print("\nSUCCESS: Dataset is ready for Ultralytics YOLO training")
    print(f"Dataset: {output}")
    print(f"Configuration: {output / 'data.yaml'}")
    print("All original images were copied, not moved.")


if __name__ == "__main__":
    main()
