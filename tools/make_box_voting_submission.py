"""Apply single-forward coordinate voting to the archived SC2 e130 submission.

Use original submitted detections as anchors and cached pre-NMS candidates as
voters. Preserve classification bytes and every detection field except bbox,
so cached prediction precision cannot change scores, classes, count or order.
"""
import argparse
import collections
import copy
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import torch
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_box_voting import BASELINE, ROOT, SNAPSHOT, load_postprocessor, sha256, vote_boxes
from score_submission import check_format


def main():
    backup = SNAPSHOT.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=backup / 'weights/SC2_e130_val_raw.pt')
    parser.add_argument('--baseline', type=Path, default=backup / 'submission/SC2_e130_val_shift-0.2.zip')
    parser.add_argument('--images', type=Path,
                        default=Path.home() / 'cs701_data/cs701-sar-course-data/val/images.json')
    parser.add_argument('--iou', type=float, default=0.8)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'submissions/SC2_e130_val_shift-0.2_boxvote_iou0.8_alpha0.10.zip')
    args = parser.parse_args()
    if not 0 < args.iou <= 1 or not 0 <= args.alpha <= 1:
        parser.error('IoU must be in (0, 1], alpha in [0, 1]')
    if args.out.resolve() == args.baseline.resolve() or args.out.exists():
        parser.error('Choose a new output path; existing submissions are never overwritten')
    torch.set_num_threads(1)
    started = time.monotonic()
    images = json.loads(args.images.read_text())['images']
    records = torch.load(args.raw, map_location='cpu', weights_only=False)
    if [r['image_id'] for r in records] != [im['id'] for im in images]:
        raise ValueError('Cached predictions do not match the supplied image list')
    with zipfile.ZipFile(args.baseline) as source:
        if source.testzip() is not None:
            raise ValueError('Corrupt baseline archive')
        classification = source.read('classification.csv')
        baseline = json.loads(source.read('detection.json'))
    rows = list(csv.DictReader(io.StringIO(classification.decode())))
    check_format(rows, baseline, {im['id'] for im in images})
    detections = copy.deepcopy(baseline)
    by_image = collections.defaultdict(list)
    for index, detection in enumerate(detections):
        by_image[detection['image_id']].append(index)
    candidate_config = {**BASELINE, 'nms': 'hard', 'nms_thresh': 1.0,
                        'per_class': None, 'total': None}
    postprocess = load_postprocessor()
    changed, missing_neighbors = 0, 0
    total_motion, max_motion = 0., 0.
    for index, (im, record) in enumerate(zip(images, records)):
        indices = by_image[im['id']]
        if not indices:
            continue
        original_xywh = torch.tensor([baseline[j]['bbox'] for j in indices], dtype=torch.float64)
        input_scale = original_xywh.new_tensor([512 / im['width'], 512 / im['height']] * 2)
        original_xyxy = original_xywh.clone()
        original_xyxy[:, 2:] += original_xyxy[:, :2]
        anchors = (original_xyxy * input_scale).float()
        labels = torch.tensor([baseline[j]['category_id'] for j in indices])
        cb, cs, cl = postprocess(record['raw_boxes'], record['raw_logits'].float(),
                                record['cls_logits'].softmax(0),
                                objectness=record['raw_obj'], stage_inputs=record['raw_stage'],
                                **candidate_config)
        support = ((box_iou(anchors, cb) >= args.iou) & (labels[:, None] == cl[None, :])).any(1)
        missing_neighbors += int((~support).sum())
        voted = vote_boxes(anchors, labels, cb, cs, cl, iou_threshold=args.iou, alpha=args.alpha)
        delta = voted - anchors
        total_motion += float(delta.abs().sum())
        max_motion = max(max_motion, float(delta.abs().max()))
        # Add the displacement to the original double-precision box rather than
        # reserializing float32 anchors: boxes with zero motion remain identical.
        updated = original_xyxy + delta.double() / input_scale
        updated[:, 0::2].clamp_(0, im['width'])
        updated[:, 1::2].clamp_(0, im['height'])
        updated[:, 2:] -= updated[:, :2]
        if not torch.isfinite(updated).all() or not (updated[:, 2:] > 0).all():
            raise ValueError('Voting produced invalid boxes')
        for j, box in zip(indices, updated.tolist()):
            if box != baseline[j]['bbox']:
                changed += 1
            detections[j]['bbox'] = box
        if (index + 1) % 200 == 0:
            print(f'Voted {index + 1}/{len(images)} images ({time.monotonic() - started:.1f}s)', flush=True)
    if any({k: v for k, v in before.items() if k != 'bbox'} !=
           {k: v for k, v in after.items() if k != 'bbox'}
           for before, after in zip(baseline, detections)):
        raise AssertionError('Only bbox may change')
    check_format(rows, detections, {im['id'] for im in images})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr('classification.csv', classification)
        output.writestr('detection.json', json.dumps(detections))
    digest = sha256(args.out)
    metadata = dict(baseline=str(args.baseline), baseline_sha256=sha256(args.baseline),
                    raw=str(args.raw), raw_sha256=sha256(args.raw), images=str(args.images),
                    images_sha256=sha256(args.images), script_sha256=sha256(Path(__file__)),
                    voting_code_sha256=sha256(ROOT / 'tools/evaluate_box_voting.py'),
                    snapshot_postprocess_sha256=sha256(SNAPSHOT / 'sarbench/postprocess.py'),
                    candidate_postprocessing=candidate_config, iou_threshold=args.iou, alpha=args.alpha,
                    anchors='Archived 51.86 submission; shift already applied',
                    voters='Same model/single-forward raw cache; shift=-0.2 applied once',
                    image_count=len(images), detection_count=len(detections), changed_boxes=changed,
                    anchors_without_neighbors=missing_neighbors,
                    mean_abs_coordinate_change_input_px=total_motion / (4 * len(detections)),
                    max_coordinate_change_input_px=max_motion,
                    classification_unchanged=True, scores_classes_count_order_unchanged=True,
                    output=str(args.out), output_sha256=digest,
                    seconds=time.monotonic() - started)
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    args.out.with_suffix('.sha256').write_text(f'{digest}  {args.out.name}\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
