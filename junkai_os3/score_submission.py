"""Score a submission.zip the way Codabench describes it (macro-F1 on classification.csv, COCO mAP on detection.json,
at most 100 detections per image and class) against a labelled local split, and validate its format.
Use it to test the whole submission path on a local holdout:
  python score_submission.py --make-images val             # writes <data>/local_val_images.json
  python predict.py --run RUN --images <data> --images-json local_val_images.json --out local_val.zip ...
  python score_submission.py --zip local_val.zip --split val"""
import argparse
import contextlib
import csv
import io
import json
import math
import zipfile
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sarbench.metrics import classification_metrics

DATA = Path.home() / 'cs701_data' / 'dataset' / 'SARFact-Course-20K'


def check_format(rows, dets, image_ids):
    """The Evaluation page's format rules; raises on the first violation."""
    assert [r['image_id'] for r in rows] and len(rows) == len(image_ids), 'one row per image'
    assert {int(r['image_id']) for r in rows} == set(image_ids), 'classification.csv image ids'
    assert all(0 <= int(r['label']) <= 8 for r in rows), 'labels 0..8'
    for d in dets:
        assert set(d) >= {'image_id', 'category_id', 'bbox', 'score'}, d
        assert d['image_id'] in image_ids and 1 <= d['category_id'] <= 9, d
        x, y, w, h = d['bbox']
        assert all(math.isfinite(v) for v in (x, y, w, h, d['score'])) and w > 0 and h > 0, d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--make-images', help='write an images.json for this labelled split (no labels in it)')
    p.add_argument('--zip', type=Path)
    p.add_argument('--split', default='val')
    a = p.parse_args()
    if a.make_images:
        coco = json.loads((a.data / 'detection' / f'instances_{a.make_images}.json').read_text())
        out = a.data / f'local_{a.make_images}_images.json'
        out.write_text(json.dumps({'images': [{k: im[k] for k in ('id', 'file_name', 'width', 'height')} for im in coco['images']]}))
        print('wrote', out)
        return
    gt_json = a.data / 'detection' / f'instances_{a.split}.json'
    gt_images = json.loads(gt_json.read_text())['images']
    with zipfile.ZipFile(a.zip) as z:
        names = z.namelist()
        cls_name = [n for n in names if n.endswith('classification.csv')]
        det_name = [n for n in names if n.endswith('detection.json')]
        assert len(cls_name) == 1 and len(det_name) == 1, names
        rows = list(csv.DictReader(io.StringIO(z.read(cls_name[0]).decode())))
        dets = json.loads(z.read(det_name[0]))
    image_ids = [im['id'] for im in gt_images]
    check_format(rows, dets, set(image_ids))
    # classification: the image class = the class of its boxes
    labels = {}
    for ann in json.loads(gt_json.read_text())['annotations']:
        labels[ann['image_id']] = ann['category_id'] - 1
    pred = {int(r['image_id']): int(r['label']) for r in rows}
    cm = classification_metrics([labels[i] for i in image_ids], [pred[i] for i in image_ids])
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
        ev = COCOeval(gt, gt.loadRes(dets) if dets else COCO(), 'bbox')
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    print(f'format OK: {len(rows)} rows, {len(dets)} detections ({len(dets) / len(rows):.1f} per image)')
    print(f'macro_f1 {cm["macro_f1"]:.4f} accuracy {cm["accuracy"]:.4f} mAP {ev.stats[0]:.4f} AP50 {ev.stats[1]:.4f}')


if __name__ == '__main__':
    main()
