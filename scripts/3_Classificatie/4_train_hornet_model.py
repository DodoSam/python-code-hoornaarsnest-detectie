"""Train a conservative nest/no-nest classifier from manual_labels.csv.
Run in VS Code: python train_hornet_model.py
Requires: pip install torch torchvision pillow
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from tkinter import Tk, filedialog

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

SEED = 42
EPOCHS = 16
BATCH_SIZE = 16
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

random.seed(SEED)
torch.manual_seed(SEED)

root = Tk()
root.withdraw()
root.attributes('-topmost', True)
selected = filedialog.askdirectory(title='Select ScrapeResult2_cleaned (contains manual_labels.csv)')
root.destroy()
if not selected:
    raise SystemExit('No folder selected.')
DATA = Path(selected)
LABELS_FILE = DATA / 'manual_labels.csv'
if not LABELS_FILE.exists():
    raise SystemExit(f'Missing: {LABELS_FILE}\nCopy your manual_labels.csv here first.')
OUTPUT = DATA / 'trained_model'
OUTPUT.mkdir(exist_ok=True)

records = []
with LABELS_FILE.open(newline='', encoding='utf-8-sig') as file:
    for r in csv.DictReader(file):
        label = r.get('manual_label', '').strip()
        if label not in ('nest', 'no_nest'):
            continue
        rel = Path(r['relative_path'])
        category = r['ai_category']
        if category not in ('nests', 'no_nests', 'review') or rel.is_absolute() or '..' in rel.parts:
            continue
        path = DATA / category / rel
        if not path.is_file():
            print(f'Skipping missing image: {path}')
            continue
        # The observation ID is the prefix before the underscore in the downloader filename.
        observation_id = rel.name.split('_')[0]
        records.append({'path': path, 'rel': str(rel), 'category': category,
                        'label': int(label == 'nest'), 'group': observation_id})

counts = Counter(x['label'] for x in records)
print(f'Usable labeled images: {len(records)}; nests: {counts[1]}, no nests: {counts[0]}')
if min(counts.values(), default=0) < 12 or len(records) < 50:
    raise SystemExit('Too few labeled examples in one or both classes.')

# Grouped split: entire iNaturalist observations stay on one side.
groups = defaultdict(list)
for r in records:
    groups[r['group']].append(r)
group_ids = list(groups)
best = None
for trial in range(1000):
    rng = random.Random(SEED + trial)
    shuffled = group_ids.copy()
    rng.shuffle(shuffled)
    val_groups = set(shuffled[:max(1, round(len(shuffled) * 0.25))])
    validation = [r for r in records if r['group'] in val_groups]
    training = [r for r in records if r['group'] not in val_groups]
    vc = Counter(r['label'] for r in validation)
    tc = Counter(r['label'] for r in training)
    if min(vc[0], vc[1], tc[0], tc[1]) < 8:
        continue
    score = (abs(len(validation)/len(records) - 0.25)
             + abs(vc[1]/len(validation) - counts[1]/len(records)))
    if best is None or score < best[0]:
        best = (score, training, validation)
if best is None:
    raise SystemExit('Could not create a valid observation-group split.')
_, training, validation = best
print('Training:', len(training), Counter(r['label'] for r in training))
print('Validation:', len(validation), Counter(r['label'] for r in validation))

with (OUTPUT / 'split.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['relative_path', 'ai_category', 'manual_label', 'observation_id', 'split'])
    for name, rows in [('train', training), ('validation', validation)]:
        for r in rows:
            w.writerow([r['rel'], r['category'], 'nest' if r['label'] else 'no_nest', r['group'], name])

# Keep the whole photograph in view (important for distant nests).
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
    transforms.ToTensor(), normalize,
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), normalize,
])

class NestDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        entry = self.rows[index]
        with Image.open(entry['path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        return self.transform(image), torch.tensor(entry['label'], dtype=torch.long)

train_counts = Counter(r['label'] for r in training)
weights = [1.0 / train_counts[r['label']] for r in training]
sampler = WeightedRandomSampler(weights, num_samples=len(training), replacement=True)
train_loader = DataLoader(NestDataset(training, train_transform), batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0)
val_loader = DataLoader(NestDataset(validation, val_transform), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for parameter in model.parameters():
    parameter.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

best_score = -1
best_epoch = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)

    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for images, labels in val_loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            actual.extend(labels.tolist())
            predicted.extend((probabilities >= 0.5).int().cpu().tolist())
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    precision = tp / (tp + fp) if tp + fp else 0
    balanced_accuracy = (recall + specificity) / 2
    print(f'Epoch {epoch:02d}/{EPOCHS} | loss={total_loss/len(training):.3f} '
          f'| val recall={recall:.3f} precision={precision:.3f} '
          f'balanced_accuracy={balanced_accuracy:.3f} | TP={tp} FP={fp} TN={tn} FN={fn}')
    if balanced_accuracy > best_score:
        best_score, best_epoch = balanced_accuracy, epoch
        torch.save(model.state_dict(), OUTPUT / 'hornet_nest_resnet18.pt')
        (OUTPUT / 'validation_metrics.json').write_text(json.dumps({
            'selected_epoch': epoch, 'validation_images': len(validation),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'recall': recall, 'precision': precision, 'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'threshold': 0.5,
            'note': 'Epoch selected on this validation set; metrics are optimistic. Use a separate test set for final performance claims.'
        }, indent=2), encoding='utf-8')

print('\nBest epoch:', best_epoch)
print('Saved model:', OUTPUT / 'hornet_nest_resnet18.pt')
print('Saved split and validation metrics in:', OUTPUT)
print('NOTE: Validation was used for checkpoint selection; this is not an independent test score.')
