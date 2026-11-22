"""Check whether a global detection cap discards useful per-class predictions.

Replay the frozen SC2 postprocessing on SC1 caches. Compare total=100 with
total=None; per_class=100 and every other setting remain unchanged. Select on
local val, then confirm the single alternative on local test only if val gains.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torchvision.ops import box_convert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_box_voting import (BASELINE, ROOT, SNAPSHOT, classification_f1,
                                      evaluate_predictions, image_ids,
                                      load_postprocessor, sha256, validate_holdouts)


def make_predictions(run, data, split, out):
    started = time.monotonic()
    path = run / f'raw_{split}.pt'
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if set(raw['image_ids']) != image_ids(data, split) or len(set(raw['image_ids'])) != len(raw['image_ids']):
        raise ValueError('Cached predictions do not exactly match the holdout')
    if not raw.get('raw_obj') or not raw.get('raw_stage'):
        raise ValueError('Missing baseline objectness or stage predictions')
    gt = json.loads((data / 'detection' / f'instances_{split}.json').read_text())
    sizes = {im['id']: (im['height'], im['width']) for im in gt['images']}
    configurations = {'baseline': dict(BASELINE), 'per_class100_no_total_cap': {**BASELINE, 'total': None}}
    predictions = {name: [] for name in configurations}
    image_counts = {name: [] for name in configurations}
    post = load_postprocessor()
    for i, iid in enumerate(raw['image_ids']):
        if tuple(raw['orig_sizes'][i]) != sizes[iid]:
            raise ValueError('Image dimensions disagree with cache')
        inputs = (raw['raw_boxes'][i], raw['raw_logits'][i].float(), raw['cls_logits'][i].softmax(0))
        extra = dict(objectness=raw['raw_obj'][i], stage_inputs=raw['raw_stage'][i])
        outputs = {name: post(*inputs, **extra, **config) for name, config in configurations.items()}
        baseline = outputs['baseline']
        expanded = outputs['per_class100_no_total_cap']
        if any(not torch.equal(a, b[:len(a)]) for a, b in zip(baseline, expanded)):
            raise AssertionError('Removing only the total cap must preserve the exact baseline prefix')
        h, w = raw['orig_sizes'][i]
        for name, (boxes, scores, labels) in outputs.items():
            image_counts[name].append(len(boxes))
            xywh = box_convert(boxes * boxes.new_tensor([w, h, w, h]) / 512, 'xyxy', 'xywh').tolist()
            predictions[name].extend(dict(image_id=iid, category_id=int(c), bbox=b, score=float(s))
                                     for b, c, s in zip(xywh, labels.tolist(), scores.tolist()))
        if (i + 1) % 200 == 0:
            print(f'{split}: {i + 1}/{len(raw["image_ids"])} ({time.monotonic() - started:.1f}s)', flush=True)
    paths = {}
    for name, detections in predictions.items():
        paths[name] = out / f'{split}_{name}.json'
        paths[name].write_text(json.dumps(detections, separators=(',', ':')))
    counts = {name: dict(total=len(predictions[name]), mean=sum(v) / len(v), maximum=max(v),
                         images_at_least_100=sum(n >= 100 for n in v)) for name, v in image_counts.items()}
    return paths, dict(images=len(raw['image_ids']), counts=counts, macro_f1=classification_f1(raw),
                       raw_sha256=sha256(path), baseline_prefix_exact_for_every_image=True,
                       seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path.home() / 'cs701_runs/SC1_r80h1536')
    parser.add_argument('--data', type=Path, default=Path.home() / 'cs701_data/dataset/SARFact-Course-20K')
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments/detection_cap_sc1')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_args = json.loads((args.run / 'metrics.json').read_text())['args']
    validate_holdouts(run_args, args.data, ['val', 'test'])
    args.out.mkdir(parents=True, exist_ok=True)
    results = dict(run=str(args.run), data=str(args.data), baseline=BASELINE,
                   alternative={**BASELINE, 'total': None},
                   selection='Only one alternative; confirm on test only if val mAP increases',
                   script_sha256=sha256(Path(__file__)),
                   metric_code_sha256=sha256(ROOT / 'tools/evaluate_box_voting.py'),
                   snapshot_postprocess_sha256=sha256(SNAPSHOT / 'sarbench/postprocess.py'))
    output = args.out / 'results.json'
    paths, metadata = make_predictions(args.run, args.data, 'val', args.out)
    val = evaluate_predictions(paths, args.data / 'detection/instances_val.json', args.workers)
    results['val'] = dict(metadata=metadata, metrics=val)
    alternative = 'per_class100_no_total_cap'
    results['val_improves'] = val[alternative]['mAP'] > val['baseline']['mAP']
    output.write_text(json.dumps(results, indent=2))
    if results['val_improves']:
        print('Validation improves; confirming the same fixed setting on local test.', flush=True)
        paths, metadata = make_predictions(args.run, args.data, 'test', args.out)
        test = evaluate_predictions(paths, args.data / 'detection/instances_test.json', args.workers)
        results['test'] = dict(metadata=metadata, metrics=test)
        results['improves_both_holdouts'] = test[alternative]['mAP'] > test['baseline']['mAP']
    else:
        results['test'] = dict(skipped='No val improvement; preserve the second holdout')
        results['improves_both_holdouts'] = False
    output.write_text(json.dumps(results, indent=2))
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
