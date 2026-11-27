"""Rebuild a manually verified hornet dataset without changing source images.
Run: python reorganize_verified_hornets.py
Requires only Python standard library (tkinter included on most Windows installs).
"""
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from tkinter import Tk, filedialog

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
LABEL_TO_FOLDER = {'nest': 'nests', 'no_nest': 'no_nests', 'unsure': 'unsure'}

root = Tk()
root.withdraw()
root.attributes('-topmost', True)
label_name = filedialog.askopenfilename(
    parent=root, title='Select full_manual_labels.csv',
    filetypes=[('CSV files', '*.csv')]
)
if not label_name:
    root.destroy()
    raise SystemExit('No labels CSV selected.')
source_name = filedialog.askdirectory(
    parent=root,
    title='Select source photo folder (ideally D:\\AI_Lab\\ScrapeResult2; or D:\\AI_Lab for fallback search)'
)
root.destroy()
if not source_name:
    raise SystemExit('No source folder selected.')
labels_path = Path(label_name).resolve()
source_root = Path(source_name).resolve()
output = source_root.parent / 'Hornet_Manual_Verified' if source_root.name.lower() == 'scraperesult2' else source_root / 'Hornet_Manual_Verified'
output = output.resolve()
if output == source_root or output in labels_path.parents:
    raise SystemExit('Unsafe output location; select a different source folder.')
if output.exists() and any(output.iterdir()):
    raise SystemExit(f'Output folder already exists and is not empty: {output}\nMove/rename it before rerunning to avoid mixing old and new exports.')

with labels_path.open('r', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fields = set(reader.fieldnames or [])
    if not {'relative_path', 'manual_label'}.issubset(fields):
        raise SystemExit('CSV needs relative_path and manual_label columns.')
    records = list(reader)

by_filename = {}
for row in records:
    relative = Path(row['relative_path'].replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts or relative.name != str(relative).replace('\\', '/'):
        raise SystemExit(f'Expected plain filename in CSV, found {row["relative_path"]}')
    name = relative.name
    if name in by_filename:
        raise SystemExit(f'Duplicate image filename in CSV: {name}')
    label = row['manual_label'].strip().lower()
    if label not in LABEL_TO_FOLDER:
        raise SystemExit(f'Unexpected label {label!r} for {name}')
    by_filename[name] = label

print(f'Labels: {len(by_filename)}. Indexing source files (this can take some time)...')
# Prefer photos in original download folder over copies in earlier filter outputs.
def preference(p):
    parts = {x.lower() for x in p.parts}
    if 'scraperesult2' in parts:
        return 0
    if 'scraperesult2_cleaned' in parts:
        return 1
    if 'scraperesult2_cleaned_model_v2' in parts:
        return 2
    return 3

candidates = defaultdict(list)
for path in source_root.rglob('*'):
    if output in path.parents or not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        continue
    if path.name in by_filename:
        candidates[path.name].append(path)

for folder in LABEL_TO_FOLDER.values():
    (output / folder).mkdir(parents=True, exist_ok=True)

export_rows = []
missing = []
counts = Counter()
for i, (filename, label) in enumerate(by_filename.items(), 1):
    matches = candidates.get(filename, [])
    if not matches:
        missing.append((filename, label))
        continue
    chosen = min(matches, key=lambda p: (preference(p), len(p.parts), str(p)))
    dest = output / LABEL_TO_FOLDER[label] / filename
    shutil.copy2(chosen, dest)
    counts[label] += 1
    export_rows.append({
        'filename': filename,
        'manual_label': label,
        'observation_id': filename.split('_')[0],
        'relative_path': f'{LABEL_TO_FOLDER[label]}/{filename}',
        'source_path': str(chosen),
    })
    if i % 500 == 0:
        print(f'Processed {i}/{len(by_filename)} labels...')

with (output / 'verified_labels.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename','manual_label','observation_id','relative_path','source_path'])
    writer.writeheader()
    writer.writerows(export_rows)
with (output / 'missing_images.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'manual_label'])
    writer.writerows(missing)
shutil.copy2(labels_path, output / 'full_manual_labels_original.csv')

print('\nDONE — source images unchanged')
print('Output:', output)
print('Confirmed nests:', counts['nest'])
print('Confirmed no nests:', counts['no_nest'])
print('Unsure:', counts['unsure'])
print('Missing:', len(missing))
if missing:
    print('Review missing_images.csv and try selecting D:\\AI_Lab as the source root.')

