"""Standalone run provenance and scoring for the F2-based experiment."""
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path


def internal_delta(f1, ap):
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (f1, ap)):
        raise ValueError('Metrics must be finite fractions in [0,1]')
    return 50 * ((f1 - .9331) / .9331 + (ap - .2613) / .2613)


def group_length(index, batches, accumulation):
    start = (index // accumulation) * accumulation
    return min(accumulation, batches - start)


def check_resume(saved, requested):
    ignored = {'resume', 'out', 'name'}
    changed = [k for k in set(saved) | set(requested)
               if k not in ignored and saved.get(k) != requested.get(k)]
    if changed:
        raise ValueError('Resume configuration changed: ' + ', '.join(sorted(changed)))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def validate_partition(manifest, coco):
    train = {int(r['image_id']) for r in manifest['train']}
    val = {int(r['image_id']) for r in manifest['val']}
    if train & val:
        raise ValueError('Train/validation overlap')
    images = {int(i['id']) for i in coco['images']}
    if train | val != images:
        raise ValueError('Partition does not cover labeled images')
    labels = {int(r['image_id']): int(r['label']) for part in ('train','val') for r in manifest[part]}
    counts = defaultdict(int)
    for a in coco['annotations']:
        image = int(a['image_id'])
        if int(a['category_id']) != labels[image] + 1:
            raise ValueError('Annotation class disagrees with image class')
        counts[image] += 1
    if any(not counts[i] for i in images):
        raise ValueError('Labeled image has no boxes')


def source_hashes(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*.py'))
            if '__pycache__' not in p.parts and 'tests' not in p.parts}


def data_fingerprint(root):
    root = Path(root)
    files = sorted((root/'classification').glob('*.csv')) + sorted((root/'detection').glob('*.json'))
    value = {'metadata': {p.relative_to(root).as_posix(): sha256(p) for p in files}, 'images': {}}
    for csv_path in (root/'classification').glob('*.csv'):
        with csv_path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                path = root / row['image']
                value['images'][str(path)] = sha256(path)
    canonical = json.dumps(value, sort_keys=True, separators=(',',':')).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()
