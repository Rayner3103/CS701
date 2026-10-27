"""COCO AP by box size (side at the 512 input: <8, 8-16, 16-32, 32-96, >=96 px) and per class, for runs with raw dumps,
under the final post-processing (pp_eval.FINAL). Usage: python tools/size_ap.py RUN_DIR [RUN_DIR ...]"""
import contextlib, io, json, sys
from pathlib import Path
import numpy as np, torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pp_eval import FINAL
from pp_study import DATA
from sarbench.data import to_original_xywh
from sarbench.postprocess import detections_from_raw

BINS = [('<8', 0, 8), ('8-16', 8, 16), ('16-32', 16, 32), ('32-96', 32, 96), ('>=96', 96, 1e5)]


def run(run_dir, split):
    raw = torch.load(Path(run_dir) / f'raw_{split}.pt')
    dets = []
    for i, iid in enumerate(raw['image_ids']):
        obj = raw['raw_obj'][i] if raw.get('raw_obj') else None
        st = raw['raw_stage'][i] if raw.get('raw_stage') else None
        b, s, l = detections_from_raw(raw['raw_boxes'][i], raw['raw_logits'][i].float(), raw['cls_logits'][i].softmax(0), objectness=obj, stage_inputs=st, **FINAL)
        dets += [{'image_id': iid, 'category_id': int(c), 'bbox': bb, 'score': float(sc)} for bb, c, sc in zip(to_original_xywh(b, raw['orig_sizes'][i]).tolist(), l.tolist(), s.tolist())]
    gt_json = DATA / 'detection' / f'instances_{split}.json'
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
        # area ranges are in ORIGINAL pixels; convert the 512-input side bins per image scale: use the 512-equivalent area
        imgs = {im['id']: im for im in gt.dataset['images']}
        for a in gt.dataset['annotations']:
            im = imgs[a['image_id']]
            a['area'] = a['bbox'][2] * 512 / im['width'] * a['bbox'][3] * 512 / im['height']
        gt.createIndex()
        dt = gt.loadRes(dets)
        for d in dt.dataset['annotations']:
            im = imgs[d['image_id']]
            d['area'] = d['bbox'][2] * 512 / im['width'] * d['bbox'][3] * 512 / im['height']
        dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = list(raw['image_ids'])
        ev.params.areaRng = [[0, 1e10]] + [[lo * lo, hi * hi] for _, lo, hi in BINS]
        ev.params.areaRngLbl = ['all'] + [n for n, _, _ in BINS]
        ev.evaluate(); ev.accumulate()
    P = ev.eval['precision']  # [T, R, K, A, M]
    out = {}
    for a, lbl in enumerate(ev.params.areaRngLbl):
        p = P[:, :, :, a, -1]
        out[lbl] = float(p[p > -1].mean()) if (p > -1).any() else float('nan')
    names = [gt.cats[c]['name'] for c in ev.params.catIds]
    out['cls'] = {n[:6]: float(P[:, :, k, 0, -1][P[:, :, k, 0, -1] > -1].mean()) for k, n in enumerate(names)}
    return out


if __name__ == '__main__':
    for rd in sys.argv[1:]:
        r = {s: run(rd, s) for s in ('val', 'test')}
        name = Path(rd).name if Path(rd).name != 'vit_pretrained_lora' else 'E00_ref'
        size = ' '.join(f"{lbl} {100 * (r['val'][lbl] + r['test'][lbl]) / 2:5.1f}" for lbl in ['all'] + [n for n, _, _ in BINS])
        cls = ' '.join(f"{k} {100 * (r['val']['cls'][k] + r['test']['cls'][k]) / 2:4.1f}" for k in r['val']['cls'])
        print(f'{name[:22]:22s} | {size}\n{"":22s} | {cls}')
