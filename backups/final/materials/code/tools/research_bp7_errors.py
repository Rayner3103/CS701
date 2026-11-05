"""CPU-only descriptive audit of existing local BP7 predictions; no model inference."""
import collections
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    root = Path(__file__).resolve().parents[1]
    output = root / 'experiments/autoresearch_20261031'
    output.mkdir(exist_ok=True)
    data = Path('/mnt/hdd/junkaichen/CS701/cs701_data/dataset/SARFact-Course-20K')
    run = Path('/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_LR130_local_s0_20261029_v2/lr1x')
    train_file = data / 'detection/instances_train.json'
    val_file = data / 'detection/instances_val.json'
    predictions_file = run / 'predictions_val.json'
    train, val = [json.loads(p.read_text()) for p in (train_file, val_file)]
    assert not ({im['file_name'] for im in train['images']} & {im['file_name'] for im in val['images']})
    metrics = json.loads((run / 'metrics.json').read_text())
    assert metrics['args']['train_split'] == 'train'
    post = json.loads((run / 'metrics_submission_post_val.json').read_text())
    images = {im['id']: im for im in train['images']}
    categories = {c['id']: c['name'] for c in train['categories']}
    groups = collections.defaultdict(list)
    for ann in train['annotations']:
        if ann.get('iscrowd'):
            continue
        im = images[ann['image_id']]
        w, h = ann['bbox'][2] * 512 / im['width'], ann['bbox'][3] * 512 / im['height']
        if w > 0 and h > 0:
            groups[categories[ann['category_id']]].append((math.sqrt(w*h), max(w/h, h/w)))
    stats = {}
    for name, values in groups.items():
        stats[name] = dict(boxes=len(values), median_sqrt_area_px=statistics.median(v[0] for v in values),
                           fraction_sqrt_area_below16=sum(v[0] < 16 for v in values)/len(values),
                           median_aspect_ratio=statistics.median(v[1] for v in values),
                           fraction_aspect_ratio_over3=sum(v[1] > 3 for v in values)/len(values))
    confusion = post['metrics']['confusion_matrix']
    report = dict(time=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
                  source_run=str(run), data_used='Official train only, partitioned local train/val',
                  training_bbox_stats_at512=stats, post_per_class_AP=post['metrics']['per_class_AP'],
                  post_image_classification_errors=sum(sum(row)-row[i] for i,row in enumerate(confusion)),
                  post_bridge_playground_confusions=confusion[2][6]+confusion[6][2],
                  trainable_parameters=metrics['trainable_parameters'],
                  trainable_budget_remaining=40000000-metrics['trainable_parameters'],
                  schedule_update_counts={'local130':470*130, 'local160':470*160,
                                          'full130':587*130, 'full160':587*160},
                  sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [train_file, val_file, predictions_file, run/'metrics.json',
                                            run/'metrics_submission_post_val.json']})
    (output/'data_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    from tidecv import TIDE, datasets
    from tidecv.data import Data
    # Box-only course annotations omit segmentation; avoid the mask-dependent COCO loader.
    gt = Data('local_val940', max_dets=100)
    for image in val['images']:
        gt.add_image(image['id'], image['file_name'])
    for category in val['categories']:
        gt.add_class(category['id'], category['name'])
    assert not any(a.get('iscrowd', 0) for a in val['annotations'])
    for ann in val['annotations']:
        gt.add_ground_truth(ann['image_id'], ann['category_id'], box=ann['bbox'])
    predictions = datasets.COCOResult(str(predictions_file), name='lr1x_native_eval')
    diagnostics = {}
    for threshold in (0.5, 0.75):
        tide = TIDE()
        result = tide.evaluate(gt, predictions, pos_threshold=threshold, mode=TIDE.BOX)
        errors = tide.get_all_errors()
        expected = metrics['val']['AP50' if threshold == .5 else 'AP75']*100
        diagnostics[str(threshold)] = dict(AP_percent=float(result.ap),
            existing_coco_AP_percent=expected, difference_pp=float(result.ap)-expected, **errors)
        print('THRESHOLD',threshold,json.dumps(diagnostics[str(threshold)]),flush=True)
    (output/'tide_errors.json').write_text(json.dumps(dict(
        time=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(), tide_version='1.0.1',
        diagnostics=diagnostics, source_predictions=str(predictions_file),
        note='Native training-time postprocessing, NOT final submission postprocessing. '
             'Oracle error corrections estimate headroom, not achievable gains. '
             'Each correction is isolated; dAP values cannot be added.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
