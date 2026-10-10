"""Convert the Hugging Face course data into the folder layout train.py expects, holding out part of train
for local val/test (the released val/test have no labels). Images are symlinked, not copied.
Example: python prepare_course_data.py --src ../cs701-sar-course-data --dst ../dataset/SARFact-Course-20K"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', type=Path, default=Path(__file__).resolve().parents[1] / 'cs701-sar-course-data')
    p.add_argument('--dst', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--test-frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    src, dst = args.src.resolve(), args.dst
    with open(src / 'train' / 'labels.csv') as f:
        rows = list(csv.DictReader(f))
    coco = json.loads((src / 'train' / 'instances.json').read_text())

    # Stratified split by image class
    by_label = defaultdict(list)
    for row in rows:
        by_label[row['label']].append(row)
    rng = random.Random(args.seed)
    splits = {'train': [], 'val': [], 'test': []}
    for group in by_label.values():
        rng.shuffle(group)
        n_val, n_test = round(len(group) * args.val_frac), round(len(group) * args.test_frac)
        splits['val'] += group[:n_val]
        splits['test'] += group[n_val:n_val + n_test]
        splits['train'] += group[n_val + n_test:]

    # Paths are relative to dst; dst/train -> src/train, so 'train/images/00001.jpg' resolves
    (dst / 'classification').mkdir(parents=True, exist_ok=True)
    (dst / 'detection').mkdir(parents=True, exist_ok=True)
    link = dst / 'train'
    if not link.exists():
        link.symlink_to(src / 'train', target_is_directory=True)

    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    images = {img['id']: img for img in coco['images']}

    for split, split_rows in splits.items():
        split_rows.sort(key=lambda r: int(r['image_id']))
        with open(dst / 'classification' / f'{split}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image', 'label'])
            for r in split_rows:
                w.writerow([f"train/{r['file_name']}", r['label']])
        ids = [int(r['image_id']) for r in split_rows]
        out = {'images': [{**images[i], 'file_name': f"train/{images[i]['file_name']}"} for i in ids],
               'annotations': [a for i in ids for a in anns_by_image[i]],
               'categories': coco['categories']}
        (dst / 'detection' / f'instances_{split}.json').write_text(json.dumps(out))
        print(f'{split}: {len(split_rows)} images, {len(out["annotations"])} boxes')


if __name__ == '__main__':
    main()
