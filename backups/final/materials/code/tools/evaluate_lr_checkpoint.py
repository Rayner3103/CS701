"""Evaluate one LR-trial checkpoint on a local holdout using submission post-processing.

The default is local validation only. Scores are fractions; local_delta_m is the
relative percentage score against the supplied local reference, not a leaderboard score.
This command never trains, exports a submission, or evaluates an official split.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

# Permit both `python tools/evaluate_lr_checkpoint.py` and module imports in tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--weights', default='model.pt')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--reference', type=Path,
                        default=Path.home() / 'cs701_runs/ref_s0/vit_pretrained_lora/metrics.json')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error('--batch-size must be positive and --workers must be nonnegative')
    return args


def validate_holdout(saved, split):
    """Reject known contamination before loading a model or touching a GPU."""
    training_splits = {part.strip() for part in saved.get('train_split', 'train').split('+')}
    if split in training_splits:
        raise ValueError(f'Refusing holdout evaluation: {split!r} is in the checkpoint training split '
                         f'{saved.get("train_split", "train")!r}.')
    return sorted(training_splits)


def local_delta_m(scores, reference):
    """Relative percentage gain, using fraction-valued F1 and AP inputs on both sides."""
    for label, values in [('scores', scores), ('reference', reference)]:
        for key in ('macro_f1', 'mAP'):
            value = float(values[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{label}.{key} must be a finite fraction in [0, 1], got {value}')
            if label == 'reference' and value == 0:
                raise ValueError(f'reference.{key} must be positive')
    return 50 * sum((scores[key] - reference[key]) / reference[key] for key in ('macro_f1', 'mAP'))


def postprocessing_arguments():
    """Use the same defaults as predict.py, with the established explicit -0.2 shift."""
    from pp_eval import FINAL, LAM
    return argparse.Namespace(nms_thresh=0.5, gamma=1.0, topk=9,
                              **{**FINAL, 'quality_pow': FINAL.get('quality_pow', 0.0),
                                 'shift': -0.2, 'cls_lambda': LAM})


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_checkpoint(args):
    import torch
    from torch.utils.data import DataLoader
    from predict import image_class, load_run, saved_args, set_post
    from sarbench.data import SARMultiTask, collate, to_original_xywh
    from sarbench.metrics import classification_metrics, detection_metrics

    run = args.run.resolve()
    saved = saved_args(run)
    training_splits = validate_holdout(saved, args.split)
    reference_file = args.reference.resolve()
    reference = json.loads(reference_file.read_text())[args.split]
    reference_scores = {key: float(reference[key]) for key in ('macro_f1', 'mAP')}
    # Validate units and reference denominators before allocating a model.
    local_delta_m(reference_scores, reference_scores)
    weights = (run / args.weights).resolve()
    weights_sha256 = sha256_file(weights)
    data = args.data.resolve()
    dataset = SARMultiTask(data, args.split, train=False)
    image_ids_expected = [dataset.coco_images[row['image']]['id'] for row in dataset.rows]
    if not image_ids_expected or len(set(image_ids_expected)) != len(image_ids_expected):
        raise ValueError('The evaluated split must contain a nonempty, unique image list.')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, collate_fn=collate)
    post_args = postprocessing_arguments()
    torch.set_num_threads(2)
    model, _ = load_run(run, args.weights, device='cuda')
    set_post(model, post_args)
    predicted, labels, detections, image_ids = [], [], [], []
    start = time.monotonic()
    with torch.inference_mode():
        for images, truth, targets in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, outputs, _ = model(images.cuda(), return_raw=True)
            labels.extend(truth.tolist())
            for target, logits_one, output in zip(targets, logits, outputs):
                image_id = target['image_id']
                image_ids.append(image_id)
                predicted.append(image_class(logits_one, output, post_args.cls_lambda))
                boxes = to_original_xywh(output['boxes'].float(), target['orig_size']).tolist()
                detections.extend({'image_id': image_id, 'category_id': int(category),
                                   'bbox': box, 'score': float(score)}
                                  for box, category, score in zip(boxes, output['labels'].tolist(),
                                                                   output['scores'].tolist()))
            print(f'evaluated {len(image_ids)}/{len(dataset)} images', flush=True)
    if image_ids != image_ids_expected:
        raise RuntimeError('Evaluation image order or coverage differs from the declared holdout.')
    ground_truth = data / 'detection' / f'instances_{args.split}.json'
    scores = {**classification_metrics(labels, predicted),
              **detection_metrics(ground_truth, detections, image_ids)}
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'run': str(run), 'weights': str(weights), 'weights_sha256': weights_sha256,
        'split': args.split, 'data': str(data), 'training_splits': training_splits,
        'training_args': saved, 'evaluation_args': {key: str(value) if isinstance(value, Path) else value
                                                  for key, value in vars(args).items()},
        'post': vars(post_args), 'n_images': len(image_ids), 'n_detections': len(detections),
        'ground_truth_sha256': sha256_file(ground_truth),
        'reference': {'path': str(reference_file), 'sha256': sha256_file(reference_file),
                      'split': args.split, **reference_scores},
        'metric_units': {'classification_rates_and_AP': 'fraction [0, 1]',
                         'confusion_matrix': 'image counts',
                         'local_delta_m': 'relative percentage gain; 50 * (relative F1 gain + relative mAP gain)'},
        'metrics': scores, 'local_delta_m': local_delta_m(scores, reference_scores),
        'seconds': time.monotonic() - start,
        'note': 'Local held-out score; not an official leaderboard score. No submission was generated.',
    }
    write_json_atomic(args.out, result)
    print(json.dumps({'out': str(args.out), 'split': args.split, 'n_images': len(image_ids),
                      'macro_f1': scores['macro_f1'], 'mAP': scores['mAP'],
                      'local_delta_m': result['local_delta_m']}, indent=2), flush=True)
    return result


def main():
    evaluate_checkpoint(parse_args())


if __name__ == '__main__':
    main()
