"""Score a raw dump (predict.py --split NAME --dump-raw) on a labelled local split with the FINAL recipe of the 56.49 / 61.25
model: agn class-agnostic fusion (gamma 1, topk 9), soft-NMS sigma 0.5, objectness^0, mean of the 3 cascade stages, shift
-0.2 px, RPN top 2000 (set when the raw dump was made), min_keep 100, image label argmax[log p_img + 2 log q], and the
lattice snapping (integer corners for aircraft/car and the 256 px cells of bridge/harbor/oil_tank/ship; a per-image
fractional lattice for wind turbines when the concentration R >= 0.5).
mAP is COCO bbox AP (area all, 100 dets, IoU .50:.95, 101-point): the per-image-matching replica of
tools/boot_eval.py of the report (validated there against pycocotools). Delta-m uses the official VAL references
(F1 93.31, mAP 26.13), so it is comparable between runs on the same split, not with the leaderboard.

Usage: python tools/local_eval.py --raw RUN/raw_vallike.pt --data ../dataset/SARFact-Course-20K --split vallike [--out FILE.json]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbench.data import to_original_xywh  # noqa: E402
from sarbench.postprocess import detections_from_raw  # noqa: E402

REF_F1, REF_MAP = 93.31, 26.13  # official val references (REPORT.md section 1)
FINAL = dict(score_thresh=1e-5, min_keep=100, per_class=100, total=100, nms='soft', soft_sigma=0.5, obj_pow=0.0,
             shift=-0.2, stage_box='mean', fusion='agn', gamma=1.0, topk=9)
CLS_LAMBDA = 2.0
IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0, 1, 101)
MAXDET = 100
# category ids (1 aircraft, 3 bridge, 4 car, 5 harbor, 6 oil_tank, 8 ship, 9 wind_turbine)
INT_CELLS = {(1, 'any'), (4, 'any'), (3, '256'), (5, '256'), (6, '256'), (8, '256')}
WIND = 9


def delta_m(f1, mAP):
    """f1 and mAP in percent."""
    return 50 * ((f1 - REF_F1) / REF_F1 + (mAP - REF_MAP) / REF_MAP)


def size_bucket(w, h):
    return '512' if max(w, h) >= 400 else '256'


# ---- lattice snapping (report eval_tools/snap_eval.py and lattice_eval.py) ----
def snap(dets, sizes):
    """Round the corners of the integer-lattice cells to whole original pixels. sizes[img] = (width, height)."""
    out = {}
    for img, (xywh, sc, lab) in dets.items():
        size = size_bucket(*sizes[img])
        b = xywh.copy()
        m = np.array([(int(c), 'any') in INT_CELLS or (int(c), size) in INT_CELLS for c in lab], dtype=bool)
        if m.any():
            x1, y1, x2, y2 = b[m, 0], b[m, 1], b[m, 0] + b[m, 2], b[m, 1] + b[m, 3]
            nx1, ny1, nx2, ny2 = np.round(x1), np.round(y1), np.round(x2), np.round(y2)
            nx2 = np.maximum(nx2, nx1 + 1e-3)
            ny2 = np.maximum(ny2, ny1 + 1e-3)
            b[m, 0], b[m, 1], b[m, 2], b[m, 3] = nx1, ny1, nx2 - nx1, ny2 - ny1
        out[img] = (b, sc, lab)
    return out


def circular_mean(v):
    z = np.exp(2j * np.pi * (v - np.floor(v))).mean()
    return (np.angle(z) / (2 * np.pi)) % 1.0, abs(z)


def lattice_snap(dets, classes=(WIND,), r0=0.5, smin=0.3, nmin=3):
    """Per-image fractional lattice (ox, oy) from the confident boxes of a class; snap all its boxes when R >= r0."""
    out = {}
    for img, (xywh, sc, lab) in dets.items():
        b = xywh.copy()
        for c in classes:
            m = lab == c
            conf = m & (sc >= smin)
            if not m.any() or conf.sum() < nmin:
                continue
            x1, y1 = b[conf, 0], b[conf, 1]
            ox, rx = circular_mean(np.r_[x1, x1 + b[conf, 2]])
            oy, ry = circular_mean(np.r_[y1, y1 + b[conf, 3]])
            if min(rx, ry) < r0:
                continue
            X1, Y1 = b[m, 0], b[m, 1]
            X2, Y2 = X1 + b[m, 2], Y1 + b[m, 3]
            s = lambda v, o: np.round(v - o) + o  # noqa: E731
            nX1, nY1, nX2, nY2 = s(X1, ox), s(Y1, oy), s(X2, ox), s(Y2, oy)
            nX2, nY2 = np.maximum(nX2, nX1 + 1), np.maximum(nY2, nY1 + 1)
            b[m, 0], b[m, 1], b[m, 2], b[m, 3] = nX1, nY1, nX2 - nX1, nY2 - nY1
        out[img] = (b, sc, lab)
    return out


# ---- COCO AP by per-image matching (report eval_tools/boot_eval.py) ----
def iou_xywh(d, g):
    dx1, dy1, dx2, dy2 = d[:, 0:1], d[:, 1:2], d[:, 0:1] + d[:, 2:3], d[:, 1:2] + d[:, 3:4]
    gx1, gy1, gx2, gy2 = g[None, :, 0], g[None, :, 1], g[None, :, 0] + g[None, :, 2], g[None, :, 1] + g[None, :, 3]
    iw = np.clip(np.minimum(dx2, gx2) - np.maximum(dx1, gx1), 0, None)
    ih = np.clip(np.minimum(dy2, gy2) - np.maximum(dy1, gy1), 0, None)
    inter = iw * ih
    union = d[:, 2:3] * d[:, 3:4] + g[None, :, 2] * g[None, :, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_image(dets, gts, cats):
    """Per category: top-MAXDET detections by score, greedy COCO matching at each IoU threshold -> (scores, tp, n_gt)."""
    xywh, scores, labels = dets
    res = {}
    for c in cats:
        g = gts.get(c, np.zeros((0, 4)))
        sel = np.where(labels == c)[0]
        if len(sel):
            order = sel[np.argsort(-scores[sel], kind='mergesort')][:MAXDET]
            d, s = xywh[order], scores[order]
        else:
            d, s = np.zeros((0, 4)), np.zeros((0,))
        n, m = len(d), len(g)
        tp = np.zeros((len(IOU_THRS), n), dtype=bool)
        if n and m:
            ious = iou_xywh(d, g)
            for ti, t in enumerate(IOU_THRS):
                used, thr = np.zeros(m, dtype=bool), min(t, 1 - 1e-10)
                for di in range(n):
                    cand = np.where((~used) & (ious[di] >= thr))[0]
                    if len(cand):
                        used[cand[np.argmax(ious[di, cand])]] = True
                        tp[ti, di] = True
        res[c] = (s, tp, m)
    return res


def coco_map(matched, image_ids, cats):
    """mAP, AP50 and per-class AP (fractions) from the per-image matches."""
    per_class, ap50 = {}, []
    for c in cats:
        sc, tp, ngt = [], [], 0
        for img in image_ids:
            s, t, m = matched[img][c]
            ngt += m
            if len(s):
                sc.append(s)
                tp.append(t)
        if ngt == 0:
            continue
        if sc:
            sc, tp = np.concatenate(sc), np.concatenate(tp, axis=1)
            order = np.argsort(-sc, kind='mergesort')
            tp = tp[:, order]
        aps = []
        for ti in range(len(IOU_THRS)):
            if len(sc) == 0:
                aps.append(0.0)
                continue
            tps, fps = np.cumsum(tp[ti]), np.cumsum(~tp[ti])
            rc, pr = tps / ngt, tps / np.maximum(tps + fps, np.finfo(float).eps)
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            inds = np.searchsorted(rc, REC_THRS, side='left')
            q = np.zeros(len(REC_THRS))
            ok = inds < len(pr)
            q[ok] = pr[inds[ok]]
            aps.append(q.mean())
        per_class[c] = float(np.mean(aps))
        ap50.append(aps[0])
    return float(np.mean(list(per_class.values()))), float(np.mean(ap50)), per_class


def macro_f1(labels, pred, n_classes=9):
    f1 = []
    for c in range(n_classes):
        tp = ((pred == c) & (labels == c)).sum()
        fp = ((pred == c) & (labels != c)).sum()
        fn = ((pred != c) & (labels == c)).sum()
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return 100 * float(np.mean(f1))


def detections(raw, cfg=None):
    """Post-processed detections {image_id: (xywh, scores, labels)} in original pixels, and the image labels (lambda 2)."""
    cfg = dict(FINAL if cfg is None else cfg)
    dets, pred = {}, []
    for i, image_id in enumerate(raw['image_ids']):
        rb, rl, ro = raw['raw_boxes'][i], raw['raw_logits'][i].float(), raw['raw_obj'][i]
        stage = raw['raw_stage'][i] if raw.get('raw_stage') else None
        quality = raw['raw_iou'][i] if raw.get('raw_iou') else None
        probs = raw['cls_logits'][i].softmax(0)
        b, s, l = detections_from_raw(rb, rl, probs, objectness=ro, quality=quality, stage_inputs=stage, **cfg)
        dets[int(image_id)] = (to_original_xywh(b, raw['orig_sizes'][i]).numpy().astype(np.float64),
                               s.numpy().astype(np.float64), l.numpy().astype(np.int64))
        logq = torch.log(rl.softmax(-1)[:, 1:].max(0).values + 1e-6)
        pred.append(int((raw['cls_logits'][i].log_softmax(0) + CLS_LAMBDA * logq).argmax()))
    return dets, np.array(pred)


def score(raw, gt_json, apply_snap=True):
    gt = json.loads(Path(gt_json).read_text())
    cats = sorted(c['id'] for c in gt['categories'])
    gts = {}
    for a in gt['annotations']:
        gts.setdefault(a['image_id'], {}).setdefault(a['category_id'], []).append(a['bbox'])
    gts = {i: {c: np.array(b, dtype=np.float64) for c, b in d.items()} for i, d in gts.items()}
    sizes = {int(i): (int(o[1]), int(o[0])) for i, o in zip(raw['image_ids'], raw['orig_sizes'])}  # orig_size = (h, w)
    dets, pred = detections(raw)
    if apply_snap:
        dets = lattice_snap(snap(dets, sizes))
    ids = [int(i) for i in raw['image_ids']]
    mAP, ap50, per_class = coco_map({i: match_image(dets[i], gts.get(i, {}), cats) for i in ids}, ids, cats)
    f1 = macro_f1(np.array(raw['labels']), pred)
    names = {c['id']: c['name'] for c in gt['categories']}
    return {'images': len(ids), 'macro_f1': f1, 'mAP': 100 * mAP, 'AP50': 100 * ap50, 'delta_m': delta_m(f1, 100 * mAP),
            'per_class_AP': {names[c]: round(100 * v, 2) for c, v in per_class.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--split', default='vallike')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    raw = torch.load(a.raw)
    gt_json = a.data / 'detection' / f'instances_{a.split}.json'
    result = {'split': a.split, 'post_processing': {**FINAL, 'cls_lambda': CLS_LAMBDA, 'snap': 'lattice'},
              'references': {'F1': REF_F1, 'mAP': REF_MAP},
              'FINAL (with snapping)': score(raw, gt_json, True), 'without snapping': score(raw, gt_json, False)}
    for k in ('FINAL (with snapping)', 'without snapping'):
        r = result[k]
        print(f"{a.split} {k}: macro-F1 {r['macro_f1']:.2f} mAP {r['mAP']:.2f} AP50 {r['AP50']:.2f} delta_m {r['delta_m']:+.2f}")
    print('per-class AP', result['FINAL (with snapping)']['per_class_AP'])
    if a.out:
        a.out.write_text(json.dumps(result, indent=2))
        print('wrote', a.out)


if __name__ == '__main__':
    main()
