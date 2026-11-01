"""Turn an augmented YOLO folder of augment_small_objects.py into the SARFact-Course-20K layout that train.py reads
(classification/*.csv, detection/instances_*.json), so the torchvision models train on the same oversampled and
copy-pasted images as train_yolo.py.
DST/train is a link to the course train folder (the local val and test rows point into it as well); the extra train images
(DST/aug -> YOLO_AUG/train/images) are the `_osK` files: plain duplicates get the boxes of their original, pasted ones the
boxes of their YOLO label file. The originals' rows and ids are unchanged. val and test are the course files as they are.
Example: python yolo_aug_to_course.py --yolo ../dataset/SARFact-YOLO-aug-os2paste --dst ../dataset/SARFact-Course-20K-aug-os2paste"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

OS = re.compile(r'^(?P<base>.+)_os\d+$')


def yolo_to_coco_boxes(label_file, width, height):
    """'class cx cy w h' lines (normalised) -> COCO [x, y, w, h] in pixels, and category_id = class + 1."""
    out = []
    for line in Path(label_file).read_text().splitlines():
        if not line.strip():
            continue
        c, cx, cy, w, h = line.split()
        c, cx, cy, w, h = int(c), float(cx), float(cy), float(w), float(h)
        out.append((c + 1, [round((cx - w / 2) * width, 2), round((cy - h / 2) * height, 2),
                            round(w * width, 2), round(h * height, 2)]))
    return out


def convert(course, yolo, dst):
    course, yolo, dst = Path(course).resolve(), Path(yolo).resolve(), Path(dst).resolve()
    if dst in (course, yolo) or course in dst.parents or yolo in dst.parents:
        raise SystemExit('--dst must be a folder of its own')
    (dst / 'classification').mkdir(parents=True, exist_ok=True)
    (dst / 'detection').mkdir(parents=True, exist_ok=True)
    for name, target in (('train', course / 'train'), ('aug', yolo / 'train' / 'images')):
        link = dst / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())
    for split in ('val', 'test'):  # unchanged
        shutil.copyfile(course / 'classification' / f'{split}.csv', dst / 'classification' / f'{split}.csv')
        shutil.copyfile(course / 'detection' / f'instances_{split}.json', dst / 'detection' / f'instances_{split}.json')

    rows = list(csv.DictReader(open(course / 'classification' / 'train.csv')))
    coco = json.loads((course / 'detection' / 'instances_train.json').read_text())
    by_stem = {Path(r['image']).stem: r for r in rows}
    image_of = {img['file_name']: img for img in coco['images']}
    anns_of = {}
    for a in coco['annotations']:
        anns_of.setdefault(a['image_id'], []).append(a)
    next_image = max(i['id'] for i in coco['images']) + 1
    next_ann = max(a['id'] for a in coco['annotations']) + 1

    images, annotations, new_rows = list(coco['images']), list(coco['annotations']), []
    copied = pasted = 0
    for f in sorted((yolo / 'train' / 'images').iterdir()):
        m = OS.match(f.stem)
        if not m:
            continue
        base = by_stem[m['base']]
        info = image_of[base['image']]
        label_file, base_label = yolo / 'train' / 'labels' / f'{f.stem}.txt', yolo / 'train' / 'labels' / f"{m['base']}.txt"
        if label_file.read_text() == base_label.read_text():  # a plain duplicate: the original's own boxes
            boxes = [(a['category_id'], a['bbox']) for a in anns_of[info['id']]]
            copied += 1
        else:
            boxes = yolo_to_coco_boxes(label_file, info['width'], info['height'])
            pasted += 1
        images.append({'id': next_image, 'file_name': f'aug/{f.name}', 'width': info['width'], 'height': info['height']})
        new_rows.append({'image': f'aug/{f.name}', 'label': base['label']})
        for category, bbox in boxes:
            annotations.append({'id': next_ann, 'image_id': next_image, 'category_id': category, 'bbox': bbox,
                                'area': round(bbox[2] * bbox[3], 2), 'iscrowd': 0})
            next_ann += 1
        next_image += 1

    with open(dst / 'classification' / 'train.csv', 'w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['image', 'label'])
        writer.writeheader()
        writer.writerows(rows + new_rows)
    (dst / 'detection' / 'instances_train.json').write_text(
        json.dumps({'images': images, 'annotations': annotations, 'categories': coco['categories']}))
    print(f'train: {len(rows)} -> {len(rows) + len(new_rows)} images ({copied} duplicates, {pasted} pasted), '
          f'{len(coco["annotations"])} -> {len(annotations)} boxes; wrote {dst}')


def main():
    root = Path(__file__).resolve().parents[1] / 'dataset'
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--course', type=Path, default=root / 'SARFact-Course-20K', help='the output of prepare_course_data.py')
    p.add_argument('--yolo', type=Path, required=True, help='the output folder of augment_small_objects.py')
    p.add_argument('--dst', type=Path, required=True)
    args = p.parse_args()
    convert(args.course, args.yolo, args.dst)


if __name__ == '__main__':
    main()
