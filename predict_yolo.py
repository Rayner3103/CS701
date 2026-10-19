"""Predict on the released (unlabelled) val or test images with a DINO-YOLO run and write submission.zip for Codabench.
Example: python predict_yolo.py --run runs/yolo/dino_yolo12s --images ../dataset-v2/val"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from make_submission import write_submission
from sarbench.dino_yolo import image_class, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='a run folder of train_yolo.py (has weights/)')
    p.add_argument('--weights', default='best.pt', help='best.pt (best val epoch) or last.pt')
    p.add_argument('--images', type=Path, required=True, help='the split folder holding images.json')
    p.add_argument('--out', type=Path, default=Path('submission.zip'))
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()

    model = YOLO(args.run / 'weights' / args.weights)
    images = json.loads((args.images / 'images.json').read_text())['images']
    paths = [str(args.images / im['file_name']) for im in images]
    preds, size = {}, 512
    for im, (boxes, classes, scores) in zip(images, predict(model, paths, imgsz=args.imgsz, batch=args.batch_size)):
        # Boxes are in original pixels; write_submission expects them on a size x size input and scales them back
        sx, sy = size / im['width'], size / im['height']
        boxes = boxes * [sx, sy, sx, sy]
        preds[im['file_name']] = (image_class(classes, scores), boxes.tolist(), classes.tolist(), scores.tolist())
        print(f'\r{len(preds)}/{len(images)}', end='', flush=True)

    write_submission(images, preds, size=size, out=args.out)
    n_boxes = sum(len(v[1]) for v in preds.values())
    print(f'\nwrote {args.out}: {len(preds)} images, {n_boxes} boxes')


if __name__ == '__main__':
    main()
