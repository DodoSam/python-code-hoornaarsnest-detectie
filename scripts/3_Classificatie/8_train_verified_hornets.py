"""Train ResNet18 on human-verified photos with observation-grouped train/val/test.
Requires: python -m pip install torch torchvision pillow
Run: python train_verified_hornets.py
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
EPOCHS = 12
BATCH_SIZE = 16
IMAGE_SIZE = 224
random.seed(SEED)
torch.manual_seed(SEED)

root = Tk()
root.withdraw()
root.attributes('-topmost', True)
chosen = filedialog.askdirectory(parent=root, title='Choose Hornet_Manual_Verified folder')
root.destroy()
if not chosen:
    raise SystemExit('No folder selected.')
DATA = Path(chosen)
manifest = DATA / 'verified_labels.csv'
if not manifest.is_file():
    raise SystemExit(f'File not found: {manifest}')
output = DATA / 'trained_model_v3'
output.mkdir(exist_ok=True)

records = []
with manifest.open(encoding='utf-8-sig', newline='') as f:
    for r in csv.DictReader(f):
        if r['manual_label'] not in ('nest', 'no_nest'):
            continue
        rel = Path(r['relative_path'].replace('\\','/'))
        if rel.is_absolute() or '..' in rel.parts:
            continue
        path = DATA / rel
        if path.is_file():
            records.append({'path': path, 'relative_path': rel.as_posix(),
                            'group': r['observation_id'], 'label': int(r['manual_label']=='nest')})
print('Definite labeled images:', len(records), Counter(x['label'] for x in records))
if len(records) < 100 or len(set(x['group'] for x in records)) < 20:
    raise SystemExit('Not enough examples or observation groups.')
groups = defaultdict(list)
for r in records:
    groups[r['group']].append(r)
ids = list(groups)
all_prop = sum(r['label'] for r in records) / len(records)
# Choose grouped split with approximate 70/15/15 image counts and class balance.
best = None
for trial in range(500):
    rng = random.Random(SEED + trial)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = len(ids)
    train_ids = set(shuffled[:round(n*.70)])
    val_ids = set(shuffled[round(n*.70):round(n*.85)])
    test_ids = set(shuffled[round(n*.85):])
    splits = [[r for r in records if r['group'] in group_ids]
              for group_ids in (train_ids,val_ids,test_ids)]
    if any(len(a)==0 or min(Counter(x['label'] for x in a).values()) < 10 or len(set(x['label'] for x in a))<2 for a in splits):
        continue
    score = sum(abs(len(a)/len(records)-target) + abs(sum(x['label'] for x in a)/len(a)-all_prop)
                for a, target in zip(splits, [.70,.15,.15]))
    if best is None or score < best[0]:
        best = (score, splits)
if best is None:
    raise SystemExit('Could not form group-balanced splits.')
train, val, test = best[1]
for name, subset in [('train',train),('validation',val),('test',test)]:
    print(name, len(subset), Counter(x['label'] for x in subset))
with (output/'splits.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f)
    w.writerow(['relative_path','observation_id','manual_label','split'])
    for split, subset in [('train',train),('validation',val),('test',test)]:
        for r in subset:
            w.writerow([r['relative_path'],r['group'],'nest' if r['label'] else 'no_nest',split])

normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
train_t=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.10),
                            transforms.ToTensor(),normalize])
eval_t=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),normalize])
class Images(Dataset):
    def __init__(self,rows,tfm): self.rows,self.tfm=rows,tfm
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['path']) as source:
            image=ImageOps.exif_transpose(source).convert('RGB')
        return self.tfm(image),r['label']
counts=Counter(x['label'] for x in train)
weights=[1/counts[r['label']] for r in train]
sampler=WeightedRandomSampler(weights,len(train),replacement=True)
train_loader=DataLoader(Images(train,train_t),batch_size=BATCH_SIZE,sampler=sampler,num_workers=0)
val_loader=DataLoader(Images(val,eval_t),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
test_loader=DataLoader(Images(test,eval_t),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)
model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters(): p.requires_grad=False
model.fc=nn.Linear(model.fc.in_features,2)
model.to(device)
optim=torch.optim.Adam(model.fc.parameters(),lr=.001)
loss_fn=nn.CrossEntropyLoss()
def evaluate(loader):
    model.eval()
    tp=fp=tn=fn=0
    with torch.inference_mode():
        for x,y in loader:
            predictions=model(x.to(device)).argmax(dim=1).cpu().tolist()
            for predicted,actual in zip(predictions,y.tolist()):
                tp+=int(predicted==1 and actual==1)
                fp+=int(predicted==1 and actual==0)
                tn+=int(predicted==0 and actual==0)
                fn+=int(predicted==0 and actual==1)
    recall=tp/(tp+fn) if tp+fn else 0
    specificity=tn/(tn+fp) if tn+fp else 0
    precision=tp/(tp+fp) if tp+fp else 0
    return {'TP':tp,'FP':fp,'TN':tn,'FN':fn,'recall':recall,
            'precision':precision,'specificity':specificity,
            'balanced_accuracy':(recall+specificity)/2}
best=-1
checkpoint=output/'hornet_nest_resnet18_v3.pt'
for epoch in range(1,EPOCHS+1):
    model.train()
    total=0.0
    for x,y in train_loader:
        x,y=x.to(device),y.to(device)
        optim.zero_grad()
        result=model(x)
        loss=loss_fn(result,y)
        loss.backward()
        optim.step()
        total+=loss.item()*len(y)
    metrics=evaluate(val_loader)
    print(f'Epoch {epoch:02d}/{EPOCHS} | loss={total/len(train):.3f} | val {metrics}')
    if metrics['balanced_accuracy']>best:
        best=metrics['balanced_accuracy']
        best_epoch=epoch
        torch.save(model.state_dict(),checkpoint)
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
model.to(device)
test_metrics=evaluate(test_loader)  # Test exactly once, AFTER checkpoint selection.
results={'best_epoch':best_epoch,'validation_balanced_accuracy':best,
         'test_metrics':test_metrics,'test_images':len(test),
         'note':'Grouped by observation ID. Test withheld until final evaluation; test labels are not used for model tuning.'}
(output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('\nBEST EPOCH:',best_epoch)
print('INDEPENDENT TEST RESULTS:',test_metrics)
print('Saved:',checkpoint)
print('Results:',output/'results.json')
