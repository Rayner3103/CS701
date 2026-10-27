"""Check a submission.zip against the Codabench format rules before uploading it (no labels needed).
Usage: python tools/check_submission.py submission.zip ../cs701-sar-course-data/val/images.json"""
import collections
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_submission import check_format


def main():
    zip_path, images_json = sys.argv[1], sys.argv[2]
    images = json.loads(Path(images_json).read_text())['images']
    ids, size = {im['id'] for im in images}, {im['id']: (im['width'], im['height']) for im in images}
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None, 'corrupt zip'
        names = z.namelist()
        cls = [n for n in names if n.endswith('classification.csv')]
        det = [n for n in names if n.endswith('detection.json')]
        assert len(cls) == 1 and len(det) == 1, f'need exactly one classification.csv and one detection.json: {names}'
        text = z.read(cls[0]).decode()
        rows = list(csv.DictReader(io.StringIO(text)))
        dets = json.loads(z.read(det[0]))
    assert text.split('\n')[0].strip() == 'image_id,label', 'header must be image_id,label'
    assert len({r['image_id'] for r in rows}) == len(rows), 'duplicate image ids'
    check_format(rows, dets, ids)  # one row per image, labels 0..8, category_id 1..9, finite boxes with w, h > 0
    per = collections.Counter((d['image_id'], d['category_id']) for d in dets)
    outside = sum(not (d['bbox'][0] >= -1e-3 and d['bbox'][1] >= -1e-3
                       and d['bbox'][0] + d['bbox'][2] <= size[d['image_id']][0] + 1e-3
                       and d['bbox'][1] + d['bbox'][3] <= size[d['image_id']][1] + 1e-3) for d in dets)
    label = {int(r['image_id']): int(r['label']) for r in rows}
    high = [d for d in dets if d['score'] > 0.3]
    agree = sum(d['category_id'] == label[d['image_id']] + 1 for d in high) / max(1, len(high))
    print(f'OK: {len(rows)} images, {len(dets)} boxes, {len(ids - {d["image_id"] for d in dets})} images without boxes, '
          f'at most {max(per.values()) if per else 0} boxes per image and class (only the top 100 are scored), '
          f'{outside} boxes outside their image, {100 * agree:.1f}% of boxes scored > 0.3 have category_id = label + 1')


if __name__ == '__main__':
    main()
