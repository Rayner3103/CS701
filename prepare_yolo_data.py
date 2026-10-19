"""Write the local splits of prepare_course_data.py in the layout ultralytics reads: DST/<split>/images (symlinks) and
DST/<split>/labels (one 'class cx cy w h' line per box, normalised to the image size), plus data.yaml (train on train)
and data_full.yaml (train on train + val + test: the final submission model, README docs/improvement.md 7).
Example: python prepare_yolo_data.py --src ../dataset/SARFact-Course-20K --dst ../dataset/SARFact-YOLO"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

SPLITS = ('train', 'val', 'test')


def yolo_line(ann, width, height):
    """COCO box [x, y, w, h] in pixels -> 'class cx cy w h' normalised, clipped to the image; None if nothing is left."""
    x, y, w, h = ann['bbox']
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return (f"{ann['category_id'] - 1} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f'{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}')


def main():
    root = Path(__file__).resolve().parents[1] / 'dataset'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', type=Path, default=root / 'SARFact-Course-20K', help='the output of prepare_course_data.py')
    p.add_argument('--dst', type=Path, default=root / 'SARFact-YOLO')
    args = p.parse_args()

    dst = args.dst.resolve()
    for split in SPLITS:
        coco = json.loads((args.src / 'detection' / f'instances_{split}.json').read_text())
        anns = defaultdict(list)
        for ann in coco['annotations']:
            anns[ann['image_id']].append(ann)
        (dst / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst / split / 'labels').mkdir(parents=True, exist_ok=True)
        dropped = 0
        for img in coco['images']:
            source = (args.src / img['file_name']).resolve()
            link = dst / split / 'images' / source.name
            if not link.is_symlink():
                link.symlink_to(source)
            lines = [yolo_line(a, img['width'], img['height']) for a in anns[img['id']]]
            dropped += lines.count(None)
            (dst / split / 'labels' / f'{source.stem}.txt').write_text(''.join(f'{l}\n' for l in lines if l))
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations']) - dropped} boxes"
              f"{f' ({dropped} empty after clipping, dropped)' if dropped else ''}")

    names = {c['id'] - 1: c['name'] for c in coco['categories']}
    data = {'path': str(dst), 'train': 'train/images', 'val': 'val/images', 'test': 'test/images', 'names': names}
    (dst / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    # val stays the validation set, but is trained on: its scores then only show that training works
    full = {**data, 'train': [f'{s}/images' for s in SPLITS]}
    (dst / 'data_full.yaml').write_text(yaml.safe_dump(full, sort_keys=False))
    print(f'wrote {dst / "data.yaml"} and {dst / "data_full.yaml"}')


if __name__ == '__main__':
    main()
