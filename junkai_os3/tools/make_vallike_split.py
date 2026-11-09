"""Rebuild the `vallike` / `trainvl` split of Junkai's report (REPORT.md, "The vallike split").

The original builder was never pushed ("scratchpad code"), so this follows the report's description:
  vallike (992 images, 4,156 boxes): a subset of the old local val + test (940 + 940 images) with a per-cell target taken
          from the official val set's (predicted class, image size) mix, seed 0. No official labels are used.
  trainvl (8,400 images): the rest of the official train pool (local train 7,512 + the 888 unused val/test images).
The per-cell targets below are the ones printed in the report ("Cell targets / available"); the "available" numbers are
checked against the data, so a different pool fails loudly. Which images fill a cell is a seeded random draw, so the images
differ from Junkai's, but the class and size mix (the point of the split) is the same.

Writes, under --data (the SARFact-Course-20K folder of prepare_course_data.py):
  classification/{vallike,trainvl}.csv, detection/instances_{vallike,trainvl}.json, vallike_split.json
Usage: python tools/make_vallike_split.py --data ../dataset/SARFact-Course-20K [--seed 0]"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# (label 0..8, image size bucket) -> (target in vallike, available in local val + test), from the report
# labels: 0 aircraft, 1 airport, 2 bridge, 3 car, 4 harbor, 5 oil_tank, 6 playground, 7 ship, 8 wind_turbine
CELLS = {
    (0, '256'): (4, 11), (0, '512'): (99, 225),
    (1, '512'): (130, 130),
    (2, '256'): (10, 45), (2, '512'): (74, 205),
    (3, '512'): (64, 88),
    (4, '256'): (59, 63), (4, '512'): (24, 209),
    (5, '256'): (68, 111), (5, '512'): (38, 135),
    (6, '512'): (131, 236),
    (7, '256'): (106, 230), (7, '512'): (5, 8),
    (8, '512'): (180, 184),
}
N_VALLIKE, N_TRAINVL = 992, 8400


def bucket(info):
    return '512' if max(info['width'], info['height']) >= 400 else '256'


def read_split(data, name):
    rows = list(csv.DictReader(open(data / 'classification' / f'{name}.csv')))
    coco = json.loads((data / 'detection' / f'instances_{name}.json').read_text())
    return rows, coco


def subset(coco, names):
    images = [im for im in coco['images'] if im['file_name'] in names]
    ids = {im['id'] for im in images}
    return {**{k: v for k, v in coco.items() if k not in ('images', 'annotations')}, 'images': images,
            'annotations': [a for a in coco['annotations'] if a['image_id'] in ids]}


def build(data, seed=0):
    pool, coco_pool, available = [], {'images': [], 'annotations': [], 'categories': None}, defaultdict(list)
    for name in ('val', 'test'):
        rows, coco = read_split(data, name)
        info = {im['file_name']: im for im in coco['images']}
        for r in rows:
            available[(int(r['label']), bucket(info[r['image']]))].append(r['image'])
        pool += rows
        coco_pool['images'] += coco['images']
        coco_pool['annotations'] += coco['annotations']
        coco_pool['categories'] = coco['categories']
    counts = {k: len(v) for k, v in available.items()}
    expected = {k: v[1] for k, v in CELLS.items()}
    if counts != expected:
        raise SystemExit(f'local val+test cells differ from the report\n  found    {sorted(counts.items())}\n  expected {sorted(expected.items())}')
    rng = random.Random(seed)
    chosen = set()
    for cell in sorted(CELLS):
        chosen |= set(rng.sample(sorted(available[cell]), CELLS[cell][0]))
    assert len(chosen) == N_VALLIKE, len(chosen)
    train_rows, coco_train = read_split(data, 'train')
    vallike_rows = [r for r in pool if r['image'] in chosen]
    trainvl_rows = train_rows + [r for r in pool if r['image'] not in chosen]
    assert len(trainvl_rows) == N_TRAINVL, len(trainvl_rows)
    merged = {**coco_pool, 'images': coco_train['images'] + coco_pool['images'],
              'annotations': coco_train['annotations'] + coco_pool['annotations']}
    return vallike_rows, trainvl_rows, subset(coco_pool, chosen), subset(merged, {r['image'] for r in trainvl_rows})


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'label'])
        w.writeheader()
        w.writerows({'image': r['image'], 'label': r['label']} for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    vallike, trainvl, coco_v, coco_t = build(a.data, a.seed)
    for name, rows, coco in (('vallike', vallike, coco_v), ('trainvl', trainvl, coco_t)):
        write_csv(a.data / 'classification' / f'{name}.csv', rows)
        (a.data / 'detection' / f'instances_{name}.json').write_text(json.dumps(coco))
        print(f'{name}: {len(rows)} images, {len(coco["annotations"])} boxes')
    (a.data / 'vallike_split.json').write_text(json.dumps({'seed': a.seed, 'vallike': [r['image'] for r in vallike],
                                                           'trainvl': [r['image'] for r in trainvl]}))


if __name__ == '__main__':
    main()
