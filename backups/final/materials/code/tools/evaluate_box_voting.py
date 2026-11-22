"""CPU-only box-voting ablation on cached, single-forward SC1 predictions.

Select among four prespecified settings on local val, then compare only that
setting with the baseline on local test. Both are splits of official train.
The frozen SC2 postprocessor supplies both selected boxes and the candidate
pool; only the output coordinates change. No images or model weights are read.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert, box_iou


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'backups/51.86-full150ep-e130-DINOv3-LoRA80-Head1536-Cascade-NWD-SizeIoU-FlipBias-shift0.2/code'
BASELINE = dict(score_thresh=1e-5, min_keep=30, per_class=100, total=100,
                fusion='none', nms='soft', soft_sigma=0.75, obj_pow=0.25,
                shift=-0.2, stage_box='mean')
VARIANTS = [dict(name=f'vote_iou{threshold:.1f}_alpha{alpha:.2f}',
                 iou_threshold=threshold, alpha=alpha)
            for threshold in (0.8, 0.9) for alpha in (0.1, 0.25)]
SIZE_BINS = [('all', 0, 1e5), ('lt8', 0, 8), ('8to16', 8, 16),
             ('lt16', 0, 16), ('16to32', 16, 32), ('ge32', 32, 1e5)]


def vote_boxes(boxes, labels, candidate_boxes, candidate_scores, candidate_labels,
               *, iou_threshold, alpha):
    """Weak score-weighted coordinate voting; class, score and count stay fixed."""
    if not 0 < iou_threshold <= 1 or not 0 <= alpha <= 1:
        raise ValueError('iou_threshold must be in (0, 1], alpha in [0, 1]')
    result = boxes.clone()
    if not len(boxes) or not len(candidate_boxes) or alpha == 0:
        return result
    eligible = ((box_iou(boxes, candidate_boxes) >= iou_threshold)
                & (labels[:, None] == candidate_labels[None, :]))
    weights = eligible * candidate_scores.float().clamp_min(0)[None, :]
    totals = weights.sum(1)
    valid = totals > 0
    targets = weights[valid] @ candidate_boxes.float() / totals[valid, None]
    result[valid] += alpha * (targets - boxes[valid])
    return result


def load_postprocessor():
    spec = importlib.util.spec_from_file_location('frozen_sc2_postprocess', SNAPSHOT / 'sarbench/postprocess.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.detections_from_raw


def image_ids(data, split):
    source = json.loads((data / 'detection' / f'instances_{split}.json').read_text())
    return {im['id'] for im in source['images']}


def validate_holdouts(run_args, data, splits):
    training = run_args['train_split'].split('+')
    if set(training) & set(splits):
        raise ValueError('Evaluation split occurs in checkpoint training split')
    train_ids = set().union(*(image_ids(data, part) for part in training))
    seen = set()
    for split in splits:
        ids = image_ids(data, split)
        if ids & train_ids or ids & seen:
            raise ValueError(f'Image IDs overlap across training/holdouts: {split}')
        seen.update(ids)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def classification_f1(raw):
    matrix = np.zeros((9, 9), dtype=np.int64)
    for truth, lg, roi_lg in zip(raw['labels'], raw['cls_logits'], raw['raw_logits']):
        logq = (roi_lg.float().softmax(-1)[:, 1:].max(0).values + 1e-6).log()
        prediction = int((lg.float().log_softmax(0) + logq).argmax())
        matrix[int(truth), prediction] += 1
    denom = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2 * matrix.diagonal(), denom,
                          out=np.zeros(9, dtype=float), where=denom > 0).mean())


def make_predictions(run, data, split, variants, out_dir):
    started = time.monotonic()
    raw_path = run / f'raw_{split}.pt'
    raw = torch.load(raw_path, map_location='cpu', weights_only=False)
    if len(raw['image_ids']) != len(set(raw['image_ids'])) or set(raw['image_ids']) != image_ids(data, split):
        raise ValueError('Cached image IDs must exactly match the local holdout')
    if not raw.get('raw_obj') or not raw.get('raw_stage'):
        raise ValueError('The baseline requires objectness and Cascade stages')
    postprocess = load_postprocessor()
    predictions = {'baseline': []}
    predictions.update({v['name']: [] for v in variants})
    motion = {v['name']: dict(abs_coordinate_sum=0., coordinates=0, changed_boxes=0, boxes=0,
                             max_coordinate_change=0.) for v in variants}
    gt = json.loads((data / 'detection' / f'instances_{split}.json').read_text())
    sizes = {im['id']: (im['height'], im['width']) for im in gt['images']}
    # Hard NMS with IoU threshold 1 and no caps is an identity selection:
    # it exposes every pre-NMS candidate after the SAME decoding/filtering.
    candidate_config = {**BASELINE, 'nms': 'hard', 'nms_thresh': 1.0,
                        'per_class': None, 'total': None}
    for index, iid in enumerate(raw['image_ids']):
        if tuple(raw['orig_sizes'][index]) != sizes[iid]:
            raise ValueError(f'Cached original size disagrees with GT: {iid}')
        args = (raw['raw_boxes'][index], raw['raw_logits'][index].float(),
                raw['cls_logits'][index].softmax(0))
        extra = dict(objectness=raw['raw_obj'][index], stage_inputs=raw['raw_stage'][index])
        boxes, scores, labels = postprocess(*args, **extra, **BASELINE)
        cb, cs, cl = postprocess(*args, **extra, **candidate_config)
        outputs = {'baseline': boxes}
        for setting in variants:
            name = setting['name']
            moved = vote_boxes(boxes, labels, cb, cs, cl,
                               iou_threshold=setting['iou_threshold'], alpha=setting['alpha'])
            if moved.shape != boxes.shape or not torch.isfinite(moved).all():
                raise ValueError('Voting must preserve count and finite coordinates')
            if not ((moved[:, 2:] >= moved[:, :2]).all() and (moved >= 0).all() and (moved <= 512).all()):
                raise ValueError('Invalid voted box geometry')
            outputs[name] = moved
            delta = (moved - boxes).abs()
            m = motion[name]
            m['abs_coordinate_sum'] += float(delta.sum())
            m['coordinates'] += delta.numel()
            m['changed_boxes'] += int((delta.max(1).values > 1e-6).sum()) if len(delta) else 0
            m['boxes'] += len(boxes)
            m['max_coordinate_change'] = max(m['max_coordinate_change'], float(delta.max()) if delta.numel() else 0.)
        h, w = raw['orig_sizes'][index]
        for name, output_boxes in outputs.items():
            xywh = box_convert(output_boxes * output_boxes.new_tensor([w, h, w, h]) / 512,
                               'xyxy', 'xywh').tolist()
            predictions[name].extend(dict(image_id=iid, category_id=int(label), bbox=box, score=float(score))
                                     for box, label, score in zip(xywh, labels.tolist(), scores.tolist()))
        if (index + 1) % 100 == 0:
            print(f'{split}: postprocessed {index + 1}/{len(raw["image_ids"])} images '
                  f'({time.monotonic() - started:.1f}s)', flush=True)
    paths = {}
    for name, detections in predictions.items():
        path = out_dir / f'{split}_{name}.json'
        path.write_text(json.dumps(detections, separators=(',', ':')))
        paths[name] = path
    for stats in motion.values():
        stats['mean_abs_coordinate_change'] = stats['abs_coordinate_sum'] / max(1, stats['coordinates'])
    return paths, dict(images=len(raw['image_ids']), detections=len(predictions['baseline']),
                       macro_f1=classification_f1(raw), motion=motion,
                       raw_sha256=sha256(raw_path), seconds=time.monotonic() - started)


def evaluate_file(job):
    name, gt_path, prediction_path = job
    started = time.monotonic()
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        images = gt.imgs
        # Match tools/size_ap.py: bin by sqrt(area) on the 512 input, including
        # for images whose original dimensions are 256. IoUs stay original-scale.
        for ann in gt.dataset['annotations']:
            im = images[ann['image_id']]
            ann['area'] = ann['bbox'][2] * ann['bbox'][3] * 512 ** 2 / (im['width'] * im['height'])
        dt = gt.loadRes(str(prediction_path))
        for ann in dt.dataset['annotations']:
            im = images[ann['image_id']]
            ann['area'] = ann['bbox'][2] * ann['bbox'][3] * 512 ** 2 / (im['width'] * im['height'])
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = sorted(images)
        ev.params.areaRng = [[lo * lo, hi * hi] for _, lo, hi in SIZE_BINS]
        ev.params.areaRngLbl = [name for name, _, _ in SIZE_BINS]
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval['precision']
    def average(array):
        valid = array[array > -1]
        return float(valid.mean()) if len(valid) else None
    result = dict(mAP=average(precision[:, :, :, 0, -1]),
                  AP50=average(precision[np.isclose(ev.params.iouThrs, .5), :, :, 0, -1]),
                  AP75=average(precision[np.isclose(ev.params.iouThrs, .75), :, :, 0, -1]),
                  size_AP={label: average(precision[:, :, :, index, -1])
                           for index, (label, _, _) in enumerate(SIZE_BINS) if label != 'all'},
                  per_class_AP={gt.cats[c]['name']: average(precision[:, :, k, 0, -1])
                                for k, c in enumerate(ev.params.catIds)},
                  seconds=time.monotonic() - started)
    return name, result


def evaluate_predictions(paths, gt_path, workers):
    jobs = [(name, gt_path, path) for name, path in paths.items()]
    results = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        for name, result in executor.map(evaluate_file, jobs):
            results[name] = result
            print(f'{gt_path.stem} {name}: mAP {100 * result["mAP"]:.4f}, '
                  f'AP50 {100 * result["AP50"]:.4f}, AP75 {100 * result["AP75"]:.4f}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path.home() / 'cs701_runs/SC1_r80h1536')
    parser.add_argument('--data', type=Path, default=Path.home() / 'cs701_data/dataset/SARFact-Course-20K')
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments/box_voting_sc1')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    torch.set_num_threads(1)
    run_args = json.loads((args.run / 'metrics.json').read_text())['args']
    validate_holdouts(run_args, args.data, ['val', 'test'])
    args.out.mkdir(parents=True, exist_ok=True)
    output = dict(run=str(args.run), data=str(args.data), snapshot=str(SNAPSHOT),
                  baseline=BASELINE, variants=VARIANTS, selection='highest val mAP; baseline wins ties',
                  size_definition='sqrt(box area) at 512x512 input; COCO area endpoints inclusive',
                  classification='unchanged, detector-informed lambda=1',
                  script_sha256=sha256(Path(__file__)),
                  snapshot_postprocess_sha256=sha256(SNAPSHOT / 'sarbench/postprocess.py'))
    output_path = args.out / 'results.json'
    paths, metadata = make_predictions(args.run, args.data, 'val', VARIANTS, args.out)
    val = evaluate_predictions(paths, args.data / 'detection/instances_val.json', args.workers)
    output['val'] = dict(metadata=metadata, metrics=val)
    selected = max(val, key=lambda name: val[name]['mAP'])
    output['selected_on_val'] = selected
    output_path.write_text(json.dumps(output, indent=2))
    print(f'Frozen validation choice: {selected}', flush=True)
    if selected != 'baseline':
        settings = [v for v in VARIANTS if v['name'] == selected]
        paths, metadata = make_predictions(args.run, args.data, 'test', settings, args.out)
        test = evaluate_predictions(paths, args.data / 'detection/instances_test.json', args.workers)
        output['test'] = dict(metadata=metadata, metrics=test)
        output['improves_both_holdouts'] = test[selected]['mAP'] > test['baseline']['mAP']
    else:
        output['improves_both_holdouts'] = False
        output['test'] = {'skipped': 'No setting improved validation mAP; keep baseline.'}
    output_path.write_text(json.dumps(output, indent=2))
    print(f'Results saved to {output_path}', flush=True)


if __name__ == '__main__':
    main()
