"""Post-processing study on saved raw outputs (train.py --dump-raw / predict.py --dump-raw): re-score the same
boxes under different settings and report COCO mAP on the labelled local split. No retraining, no new forward pass.
Example: python pp_study.py --run ~/cs701_runs/E01 --splits val,test"""
import argparse
import contextlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sarbench.data import to_original_xywh
from sarbench.postprocess import detections_from_raw

DATA = Path.home() / 'cs701_data' / 'dataset' / 'SARFact-Course-20K'

GRID = [
    ('reference (thr .05, top100)', dict(score_thresh=0.05, per_class=None, total=100, fusion='none')),
    ('thr .001, top100', dict(score_thresh=0.001, per_class=None, total=100, fusion='none')),
    ('thr .001, 100/class', dict(score_thresh=0.001, per_class=100, total=None, fusion='none')),
    ('mult g1 top9', dict(score_thresh=0.001, per_class=100, total=None, fusion='mult', gamma=1.0, topk=9)),
    ('mult g0.5 top9', dict(score_thresh=0.001, per_class=100, total=None, fusion='mult', gamma=0.5, topk=9)),
    ('mult g1 top3', dict(score_thresh=0.001, per_class=100, total=None, fusion='mult', gamma=1.0, topk=3)),
    ('agn g1 top9', dict(score_thresh=0.001, per_class=100, total=None, fusion='agn', gamma=1.0, topk=9)),
    ('agn g1 top3', dict(score_thresh=0.001, per_class=100, total=None, fusion='agn', gamma=1.0, topk=3)),
    ('agn g1 top1', dict(score_thresh=0.001, per_class=100, total=None, fusion='agn', gamma=1.0, topk=1)),
    ('hard (argmax class)', dict(score_thresh=0.001, per_class=100, total=None, fusion='hard')),
    ('agn g1 top3 nms.6', dict(score_thresh=0.001, per_class=100, total=None, fusion='agn', gamma=1.0, topk=3, nms_thresh=0.6)),
    ('agn g1 top3 nms.4', dict(score_thresh=0.001, per_class=100, total=None, fusion='agn', gamma=1.0, topk=3, nms_thresh=0.4)),
    ('soft s.75', dict(score_thresh=0.001, per_class=100, total=100, nms='soft', soft_sigma=0.75)),
    ('soft s.5', dict(score_thresh=0.001, per_class=100, total=100, nms='soft', soft_sigma=0.5)),
    ('soft s1', dict(score_thresh=0.001, per_class=100, total=100, nms='soft', soft_sigma=1.0)),
    ('soft s.75 obj^.25', dict(score_thresh=0.001, per_class=100, total=100, nms='soft', soft_sigma=0.75, obj_pow=0.25)),
    ('soft s.75 obj^.5', dict(score_thresh=0.001, per_class=100, total=100, nms='soft', soft_sigma=0.75, obj_pow=0.5)),
    ('hard thr.001 obj^.25', dict(score_thresh=0.001, per_class=None, total=100, obj_pow=0.25)),
]


def coco_map(gt_json, dets, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
        dt = gt.loadRes(dets) if dets else COCO()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    prec = ev.eval['precision'][:, :, :, 0, -1]
    per_class = {gt.cats[c]['name']: float(prec[:, :, k][prec[:, :, k] > -1].mean()) for k, c in enumerate(ev.params.catIds)}
    return float(ev.stats[0]), float(ev.stats[1]), per_class


def run_one(job):
    raw_path, gt_json, name, cfg = job
    raw = torch.load(raw_path)
    if cfg.get('obj_pow') and not raw.get('raw_obj'):
        print(f'WARNING: {raw_path} has no RPN objectness; obj_pow ignored for {name}', flush=True)
    if cfg.get('stage_box') == 'mean' and not raw.get('raw_stage'):
        print(f'WARNING: {raw_path} has no per-stage boxes (older run / not a cascade); stage_box=mean ignored', flush=True)
    dets = []
    for i, image_id in enumerate(raw['image_ids']):
        probs = raw['cls_logits'][i].softmax(0)
        obj = raw['raw_obj'][i] if raw.get('raw_obj') else None
        q = raw['raw_iou'][i] if raw.get('raw_iou') else None
        st = raw['raw_stage'][i] if raw.get('raw_stage') else None
        boxes, scores, labels = detections_from_raw(raw['raw_boxes'][i], raw['raw_logits'][i].float(), probs,
                                                    objectness=obj, quality=q, stage_inputs=st, **cfg)
        xywh = to_original_xywh(boxes, raw['orig_sizes'][i]).tolist()
        dets += [{'image_id': image_id, 'category_id': int(c), 'bbox': b, 'score': float(s)}
                 for b, c, s in zip(xywh, labels.tolist(), scores.tolist())]
    m, ap50, per_class = coco_map(gt_json, dets, list(raw['image_ids']))
    return name, m, ap50, per_class, len(dets) / len(raw['image_ids'])


def cls_study(raw_path, lambdas=(0.0, 0.25, 0.5, 1.0, 2.0)):
    """Macro-F1 when the image class is argmax_c [log p_img(c) + lam * log q(c)], q(c) = the highest detector
    probability of class c over the image's proposals (the same forward pass)."""
    from sarbench.metrics import classification_metrics
    raw = torch.load(raw_path)
    logp = torch.stack([lg.log_softmax(0) for lg in raw['cls_logits']])
    logq = torch.stack([torch.log(rl.float().softmax(-1)[:, 1:].max(0).values + 1e-6) for rl in raw['raw_logits']])
    out = {}
    for lam in lambdas:
        pred = (logp + lam * logq).argmax(1).tolist()
        out[lam] = classification_metrics(raw['labels'], pred)['macro_f1']
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--splits', default='val,test')
    p.add_argument('--workers', type=int, default=24)
    a = p.parse_args()
    torch.set_num_threads(1)
    results = {}
    for split in a.splits.split(','):
        raw_path = a.run / f'raw_{split}.pt'
        jobs = [(raw_path, DATA / 'detection' / f'instances_{split}.json', n, c) for n, c in GRID]
        with ProcessPoolExecutor(min(a.workers, len(jobs))) as ex:
            for name, m, ap50, per_class, per_img in ex.map(run_one, jobs):
                results.setdefault(name, {})[split] = {'mAP': m, 'AP50': ap50, 'per_class': per_class, 'dets_per_img': per_img}
    print(f'{"setting":30s} ' + ' '.join(f'{s + " mAP":>9s} {s + " AP50":>9s}' for s in a.splits.split(',')) + '  dets/img')
    for name, _ in GRID:
        r = results[name]
        print(f'{name:30s} ' + ' '.join(f'{100 * r[s]["mAP"]:9.2f} {100 * r[s]["AP50"]:9.2f}' for s in a.splits.split(','))
              + f'  {np.mean([r[s]["dets_per_img"] for s in r]):.0f}')
    cls = {split: cls_study(a.run / f'raw_{split}.pt') for split in a.splits.split(',')}
    for lam in next(iter(cls.values())):
        print(f'cls fusion lambda {lam:4.2f}: macro-F1 ' + ' '.join(f'{s} {100 * cls[s][lam]:.2f}' for s in cls))
    results['cls_fusion'] = {s: {str(k): v for k, v in d.items()} for s, d in cls.items()}
    (a.run / 'pp_study.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
