"""Apply trained nest model to 3,951 images; never deletes originals.
Run: python predict_hornet_nests.py
"""
import csv
import shutil
from collections import Counter
from pathlib import Path
from tkinter import Tk, filedialog

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms
from tqdm import tqdm

# Conservative starting thresholds. Validate before using for automated exclusion.
KEEP_IF_AT_LEAST = 0.85
REJECT_IF_BELOW = 0.15
CROP_FRACTION = 0.60
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

root = Tk()
root.withdraw()
root.attributes('-topmost', True)
selected = filedialog.askdirectory(title='Select ScrapeResult2_cleaned (with trained_model folder)')
root.destroy()
if not selected:
    raise SystemExit('No folder selected')
DATA = Path(selected)
MODEL_FILE = DATA / 'trained_model' / 'hornet_nest_resnet18.pt'
if not MODEL_FILE.is_file():
    raise SystemExit(f'Cannot find trained model: {MODEL_FILE}')
OUT = DATA.parent / (DATA.name + '_model_v2')
for folder in ['nests', 'no_nests', 'review']:
    (OUT / folder).mkdir(parents=True, exist_ok=True)

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
# load local checkpoint, with CPU/GPU portability
model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu', weights_only=True))
model = model.to(device).eval()

# Use the original CLIP outputs as the source; they contain all 3,951 images.
images = []
for category in ['nests', 'no_nests', 'review']:
    folder = DATA / category
    if folder.exists():
        images += [(category, p, p.relative_to(folder)) for p in folder.rglob('*')
                   if p.is_file() and p.suffix.lower() in EXTENSIONS]
images.sort(key=lambda item: (item[0], str(item[2])))
print(f'Found {len(images)} images. Using {device}.')

# Retain human labels as authoritative, including 'unsure'.
manual = {}
labels_csv = DATA / 'manual_labels.csv'
if labels_csv.is_file():
    with labels_csv.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            manual[(row['ai_category'], row['relative_path'].replace('\\', '/'))] = row['manual_label']

def views(image):
    image.thumbnail((1024, 1024))
    w, h = image.size
    cw, ch = max(1, int(w * CROP_FRACTION)), max(1, int(h * CROP_FRACTION))
    coords = [(0,0), (w-cw,0), (0,h-ch), (w-cw,h-ch), ((w-cw)//2,(h-ch)//2)]
    return [image] + [image.crop((x,y,x+cw,y+ch)) for x,y in coords]

rows = []
for old_category, path, relative in tqdm(images, desc='Classifying'):
    try:
        with Image.open(path) as source:
            im = ImageOps.exif_transpose(source).convert('RGB')
        batch = torch.stack([validation_transform(v) for v in views(im)]).to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(batch), dim=1)[:,1].cpu().tolist()
        whole, max_score = probs[0], max(probs)
        strong_crops = sum(score >= KEEP_IF_AT_LEAST for score in probs[1:])
        human_label = manual.get((old_category, relative.as_posix()), '')
        if human_label == 'nest':
            category, source_type = 'nests', 'manual'
        elif human_label == 'no_nest':
            category, source_type = 'no_nests', 'manual'
        elif human_label == 'unsure':
            category, source_type = 'review', 'manual'
        elif whole >= KEEP_IF_AT_LEAST or strong_crops >= 2:
            category, source_type = 'nests', 'model'
        elif max_score < REJECT_IF_BELOW:
            category, source_type = 'no_nests', 'model'
        else:
            category, source_type = 'review', 'model'
        # Uniqueness: original download filenames are observationID_photoIndex.
        # Keep category folder relative path to avoid collisions if necessary.
        destination = OUT / category / relative
        # Downloaded photo names should be unique across the three source folders.
        # Existing outputs are overwritten on reruns, not renamed.
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        rows.append([old_category, str(relative), category, source_type,
                     human_label, f'{whole:.4f}', f'{max_score:.4f}', strong_crops, str(destination.relative_to(OUT))])
    except Exception as error:
        print(f'Failed: {path}: {error}')
        rows.append([old_category, str(relative), 'error', '', '', '', '', '', ''])

with (OUT / 'model_predictions.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['original_ai_category', 'relative_path', 'new_category', 'decision_source',
                     'manual_label', 'whole_probability', 'max_view_probability', 'strong_crops', 'output_relative_path'])
    writer.writerows(rows)
print('Results:', dict(Counter(row[2] for row in rows)))
print('Output:', OUT)
print('CAUTION: Probabilities are not calibrated, and crop decisions have not been independently validated.')
print('Review no_nests for missed distant nests before deleting anything.')
