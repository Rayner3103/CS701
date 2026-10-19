"""Score a DINO-YOLO run (train_yolo.py) on the local val and test splits with the metrics of train.py: macro-F1 (the
image class from the classification head, or read from the detections), COCO mAP and Δm. Writes metrics.json and
predictions_{split}.json into the run folder, so compare_runs.py lists the run.
Example: python eval_yolo.py --run runs/yolo/dino_yolo12s"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from sarbench.dino_yolo import image_class, parameter_counts, predict
from sarbench.metrics import classification_metrics, detection_metrics
from train import delta_m

DATA = Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K'

def evaluate(model, root, split, imgsz=512, batch=16):
    """Classification and COCO detection metrics on one split of prepare_course_data.py, and the detections in COCO
    result format (original pixels)."""
    with open(root / 'classification' / f'{split}.csv') as f:
        rows = list(csv.DictReader(f))
    gt_json = root / 'detection' / f'instances_{split}.json'
    coco_images = {img['file_name']: img for img in json.loads(gt_json.read_text())['images']}
    labels, predicted, detections, image_ids = [], [], [], []
    paths = [str(root / row['image']) for row in rows]
    for row, (boxes, classes, scores, logits) in zip(rows, predict(model, paths, imgsz=imgsz, batch=batch)):
        image_id = coco_images[row['image']]['id']
        labels.append(int(row['label']))
        predicted.append(image_class(classes, scores, logits))
        image_ids.append(image_id)
        xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
        detections += [{'image_id': image_id, 'category_id': int(c) + 1, 'bbox': b, 'score': float(s)}
                       for b, c, s in zip(xywh.tolist(), classes, scores)]
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(str(gt_json), detections, image_ids)}
    metrics['delta_m'] = delta_m(metrics)
    return metrics, detections


def score_run(run, weights='best.pt', data=DATA, splits=('val', 'test'), imgsz=512, batch=16, trainable=None):
    """Evaluate run/weights/<weights> on `splits`; write metrics.json and predictions_{split}.json into `run`.
    trainable: the trainable parameter count during training (a saved model has gradients off)."""
    model = YOLO(run / 'weights' / weights)
    out = {'weights': weights, 'total_parameters': parameter_counts(model.model)[0], 'trainable_parameters': trainable}
    for split in splits:
        metrics, detections = evaluate(model, data, split, imgsz, batch)
        out[split] = metrics
        (run / f'predictions_{split}.json').write_text(json.dumps(detections))
        print(f"{split}: accuracy {metrics['accuracy']:.4f} macro_f1 {metrics['macro_f1']:.4f} mAP {metrics['mAP']:.4f} "
              f"AP50 {metrics['AP50']:.4f} delta_m {metrics['delta_m']:+.2f}")
    (run / 'metrics.json').write_text(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='a run folder of train_yolo.py (has weights/)')
    p.add_argument('--weights', default='best.pt', help='best.pt (best val epoch) or last.pt')
    p.add_argument('--data', type=Path, default=DATA, help='the output of prepare_course_data.py')
    p.add_argument('--splits', nargs='+', default=['val', 'test'])
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    score_run(args.run, args.weights, args.data, args.splits, args.imgsz, args.batch_size)


if __name__ == '__main__':
    main()
