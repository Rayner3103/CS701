"""5-fold cross-validation splits of the 9,392 labelled train images (never the Codabench val/test images).
Folds are stratified by image class with a fixed seed. Fold k holds out 1/5 of the images, split into two stratified
halves written as 'val' and 'test' (so train.py / pp_eval.py work unchanged), and trains on the other 4/5.
Layout: <dst>/fold{k}/classification/{train,val,test}.csv, detection/instances_{train,val,test}.json, train -> images.
Usage: python tools/make_folds.py [--src ~/cs701_data/cs701-sar-course-data] [--dst ~/cs701_data/folds]"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', type=Path, default=Path.home() / 'cs701_data' / 'cs701-sar-course-data')
    p.add_argument('--dst', type=Path, default=Path.home() / 'cs701_data' / 'folds')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.src / 'train' / 'labels.csv')))
    coco = json.loads((a.src / 'train' / 'instances.json').read_text())
    anns = defaultdict(list)
    for ann in coco['annotations']:
        anns[ann['image_id']].append(ann)
    images = {im['id']: im for im in coco['images']}
    rng = random.Random(a.seed)
    fold_of = {}
    by_label = defaultdict(list)
    for r in rows:
        by_label[r['label']].append(r)
    for label, group in sorted(by_label.items()):
        rng.shuffle(group)
        for i, r in enumerate(group):
            fold_of[r['image_id']] = i % a.k  # stratified: each class spread evenly over the folds
    for k in range(a.k):
        root = a.dst / f'fold{k}'
        (root / 'classification').mkdir(parents=True, exist_ok=True)
        (root / 'detection').mkdir(parents=True, exist_ok=True)
        if not (root / 'train').exists():
            (root / 'train').symlink_to(a.src / 'train', target_is_directory=True)
        held = [r for r in rows if fold_of[r['image_id']] == k]
        # split the held-out fold into stratified halves ('val', 'test')
        half = {}
        groups = defaultdict(list)
        for r in held:
            groups[r['label']].append(r)
        for label, g in sorted(groups.items()):
            rng.shuffle(g)
            for i, r in enumerate(g):
                half[r['image_id']] = 'val' if i % 2 == 0 else 'test'
        splits = {'train': [r for r in rows if fold_of[r['image_id']] != k],
                  'val': [r for r in held if half[r['image_id']] == 'val'],
                  'test': [r for r in held if half[r['image_id']] == 'test']}
        for split, srows in splits.items():
            srows = sorted(srows, key=lambda r: int(r['image_id']))
            with open(root / 'classification' / f'{split}.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['image', 'label'])
                for r in srows:
                    w.writerow([f"train/{r['file_name']}", r['label']])
            ids = [int(r['image_id']) for r in srows]
            out = {'images': [{**images[i], 'file_name': f"train/{images[i]['file_name']}"} for i in ids],
                   'annotations': [x for i in ids for x in anns[i]], 'categories': coco['categories']}
            (root / 'detection' / f'instances_{split}.json').write_text(json.dumps(out))
        print(f'fold{k}: ' + ', '.join(f'{s} {len(v)}' for s, v in splits.items()))


if __name__ == '__main__':
    main()
