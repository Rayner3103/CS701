"""yolo_aug_to_course.py: the augmented YOLO folder as a SARFact-Course-20K folder that SARMultiTask reads."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import augment_small_objects as aug  # noqa: E402
import yolo_aug_to_course as conv  # noqa: E402

W = H = 128


def build(tmp_path):
    """A course folder (train rows a, b; val row c) and its augmented YOLO folder."""
    course, src = tmp_path / 'course', tmp_path / 'yolo'
    images = course / 'real' / 'images'
    images.mkdir(parents=True)
    (course / 'train').symlink_to(course / 'real')
    for split in ('train', 'val', 'test'):
        (src / split / 'images').mkdir(parents=True)
        (src / split / 'labels').mkdir(parents=True)
    cases = {'a': [(0.2, 0.2, 0.1, 0.1), (0.7, 0.3, 0.1, 0.1)], 'b': [(0.5, 0.5, 0.6, 0.6)]}
    coco = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'x'}, {'id': 4, 'name': 'car'}]}
    rows, aid = [], 1
    for i, (name, boxes) in enumerate(cases.items(), 1):
        img = np.random.default_rng(i).integers(0, 60, (H, W, 3), dtype=np.uint8)
        cv2.imwrite(str(images / f'{name}.png'), img)
        (src / 'train' / 'images' / f'{name}.png').symlink_to(images / f'{name}.png')
        (src / 'train' / 'labels' / f'{name}.txt').write_text(''.join(f'3 {a} {b} {c} {d}\n' for a, b, c, d in boxes))
        coco['images'].append({'id': i, 'file_name': f'train/images/{name}.png', 'width': W, 'height': H})
        for a, b, c, d in boxes:
            coco['annotations'].append({'id': aid, 'image_id': i, 'category_id': 4, 'area': c * d * W * H, 'iscrowd': 0,
                                        'bbox': [(a - c / 2) * W, (b - d / 2) * H, c * W, d * H]})
            aid += 1
        rows.append({'image': f'train/images/{name}.png', 'label': 3})
    (course / 'classification').mkdir()
    (course / 'detection').mkdir()
    for split in ('train', 'val', 'test'):
        with open(course / 'classification' / f'{split}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['image', 'label'])
            w.writeheader()
            w.writerows(rows if split == 'train' else rows[:1])
        (course / 'detection' / f'instances_{split}.json').write_text(json.dumps(coco))
    (course / 'detection' / 'instances_train.json').write_text(json.dumps(coco))
    (src / 'data.yaml').write_text('names: {3: car}\n')
    return course, src


def test_conversion(tmp_path, monkeypatch):
    course, src = build(tmp_path)
    out = tmp_path / 'yolo_aug'
    monkeypatch.setattr(sys, 'argv', ['x', '--src', str(src), '--dst', str(out), '--ref-size', str(W),
                                      '--oversample', '2', '--paste-prob', '1.0'])
    aug.main()
    dst = tmp_path / 'course_aug'
    conv.convert(course, out, dst)

    rows = list(csv.DictReader(open(dst / 'classification' / 'train.csv')))
    coco = json.loads((dst / 'detection' / 'instances_train.json').read_text())
    assert [r['image'] for r in rows[:2]] == ['train/images/a.png', 'train/images/b.png']  # originals first, unchanged
    assert [r['image'] for r in rows[2:]] == ['aug/a_os1.png']                              # only a has small boxes
    assert rows[2]['label'] == '3'
    ids = [i['id'] for i in coco['images']]
    assert len(set(ids)) == len(ids) == 3
    assert len({a['id'] for a in coco['annotations']}) == len(coco['annotations'])
    new = [a for a in coco['annotations'] if a['image_id'] == ids[-1]]
    yolo_lines = (out / 'train' / 'labels' / 'a_os1.txt').read_text().split('\n')
    assert len(new) == len([l for l in yolo_lines if l]) > 2                                # originals + pasted
    assert all(a['category_id'] == 4 and a['bbox'][2] > 0 for a in new)
    for a in new:
        x, y, w, h = a['bbox']
        assert 0 <= x and 0 <= y and x + w <= W + 0.01 and y + h <= H + 0.01
    assert (dst / 'aug' / 'a_os1.png').exists() and (dst / 'train' / 'images' / 'a.png').exists()
    assert (dst / 'classification' / 'val.csv').read_text() == (course / 'classification' / 'val.csv').read_text()


def test_plain_duplicates_copy_the_original_boxes(tmp_path, monkeypatch):
    course, src = build(tmp_path)
    out = tmp_path / 'yolo_aug'
    monkeypatch.setattr(sys, 'argv', ['x', '--src', str(src), '--dst', str(out), '--ref-size', str(W),
                                      '--oversample', '3', '--paste-prob', '0'])
    aug.main()
    dst = tmp_path / 'course_aug'
    conv.convert(course, out, dst)
    coco = json.loads((dst / 'detection' / 'instances_train.json').read_text())
    by_image = {}
    for a in coco['annotations']:
        by_image.setdefault(a['image_id'], []).append(a['bbox'])
    assert len(coco['images']) == 4 and by_image[3] == by_image[1] == by_image[4]  # a, b, a_os1, a_os2 (ids 1, 2, 3, 4)


def test_refuses_dst_inside_source(tmp_path):
    course, src = build(tmp_path)
    with pytest.raises(SystemExit):
        conv.convert(course, src, src / 'x')
