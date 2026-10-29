import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

coco = json.load(open('cs701-sar-course-data/train/instances.json'))
labels = {r['file_name']: r['label'] for r in csv.DictReader(open('cs701-sar-course-data/train/labels.csv'))}
CLASSES = ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine']

by_img = defaultdict(list)
for a in coco['annotations']:
    by_img[a['image_id']].append(a)
info = {i['id']: i for i in coco['images']}

cand = []
for iid, aa in by_img.items():
    if info[iid]['width'] != 512 or len(aa) < 5:
        continue
    med = sorted((a['bbox'][2] * a['bbox'][3]) ** .5 for a in aa)[len(aa) // 2]
    if med < 25:  # lots of boxes AND small
        cand.append((iid, len(aa), med))
print(len(cand), 'candidates')
iid, n, _ = sorted(cand, key=lambda t: -t[1])[12]
fn = info[iid]['file_name']
sides = sorted((a['bbox'][2] * a['bbox'][3]) ** .5 for a in by_img[iid])
cls = CLASSES[int(labels[fn])]
print(fn, n, 'boxes, class', cls, 'median', sides[n // 2])

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(Image.open('cs701-sar-course-data/train/' + fn), cmap='gray')
for a in by_img[iid]:
    x, y, w, h = a['bbox']
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec='#FF4444', lw=1.2))
ax.set_title(f'{cls} — {n} boxes (median side ~{sides[n // 2]:.0f} px)', fontsize=13)
ax.axis('off')
fig.tight_layout(pad=0.3)
fig.savefig('report/figures/sar_example.png', dpi=150)
print('saved report/figures/sar_example.png')
