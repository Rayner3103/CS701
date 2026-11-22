"""Offline, paired-bootstrap comparison of post-processing variants on one raw dump (local holdout with labels).

Replicates COCO bbox AP (area=all, maxDets=100 per image & category, IoU .50:.95, 101-point interpolation) with
per-image matching precomputed once, so image-level bootstrap resampling is cheap. Validated against pycocotools.
"""
import json, sys, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
ROOT = Path('/mnt/hdd/junkaichen/CS701/CS701'); sys.path.insert(0, str(ROOT))
from sarbench.data import to_original_xywh
from sarbench.postprocess import detections_from_raw
from pp_study import DATA, coco_map

BASE = dict(score_thresh=1e-5, min_keep=30, per_class=100, total=100, nms='soft', soft_sigma=0.5, obj_pow=0.0,
            shift=-0.2, stage_box='mean', fusion='none', gamma=1.0, topk=9)
VARIANTS = {
    'none':        dict(),
    'hard':        dict(fusion='hard'),
    'hard_fused':  dict(fusion='hard', _probs='fused'),
    'agn_k2':      dict(fusion='agn', gamma=1.0, topk=2),
    'agn_k2_fused':dict(fusion='agn', gamma=1.0, topk=2, _probs='fused'),
    'agn_k3':      dict(fusion='agn', gamma=1.0, topk=3),
    'agn_k9':      dict(fusion='agn', gamma=1.0, topk=9),
    'agn_g05_k2':  dict(fusion='agn', gamma=0.5, topk=2),
    'mult_g1':     dict(fusion='mult', gamma=1.0, topk=9),
    'hard_s04':    dict(fusion='hard', soft_sigma=0.4),
    'hard_s06':    dict(fusion='hard', soft_sigma=0.6),
    'hard_sh01':   dict(fusion='hard', shift=-0.1),
    'hard_sh03':   dict(fusion='hard', shift=-0.3),
}
LAM = 2.0
IOU_THRS = np.linspace(0.5, 0.95, 10); REC_THRS = np.linspace(0, 1, 101); MAXDET = 100

def fused_probs(cls_logits, raw_logits, lam=LAM):
    logp = cls_logits.float().log_softmax(0)
    logq = torch.log(raw_logits.float().softmax(-1)[:, 1:].max(0).values + 1e-6)
    return (logp + lam * logq).softmax(0)

def build_dets(args):
    name, raw_path = args
    cfg = dict(BASE); v = dict(VARIANTS[name]); probs_mode = v.pop('_probs', 'cls'); cfg.update(v)
    raw = torch.load(raw_path)
    torch.set_num_threads(1)
    out = {}
    for i, image_id in enumerate(raw['image_ids']):
        probs = raw['cls_logits'][i].softmax(0) if probs_mode == 'cls' else fused_probs(raw['cls_logits'][i], raw['raw_logits'][i])
        boxes, scores, labels = detections_from_raw(raw['raw_boxes'][i], raw['raw_logits'][i].float(), probs,
                                                    objectness=raw['raw_obj'][i], quality=None,
                                                    stage_inputs=raw['raw_stage'][i] if raw.get('raw_stage') else None, **cfg)
        xywh = to_original_xywh(boxes, raw['orig_sizes'][i]).numpy()
        out[int(image_id)] = (xywh.astype(np.float64), scores.numpy().astype(np.float64), labels.numpy().astype(np.int64))
    return name, out

def iou_xywh(d, g):
    # d [n,4], g [m,4] xywh -> [n,m]
    dx1, dy1, dx2, dy2 = d[:, 0:1], d[:, 1:2], d[:, 0:1] + d[:, 2:3], d[:, 1:2] + d[:, 3:4]
    gx1, gy1, gx2, gy2 = g[None, :, 0], g[None, :, 1], g[None, :, 0] + g[None, :, 2], g[None, :, 1] + g[None, :, 3]
    iw = np.clip(np.minimum(dx2, gx2) - np.maximum(dx1, gx1), 0, None); ih = np.clip(np.minimum(dy2, gy2) - np.maximum(dy1, gy1), 0, None)
    inter = iw * ih; union = d[:, 2:3] * d[:, 3:4] + (g[None, :, 2] * g[None, :, 3]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)

def match_image(dets, gts, cats):
    """Per category: dets sorted by score, top MAXDET, greedy COCO matching for each IoU thr -> (scores, tp[T,n]), n_gt."""
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
        T, n, m = len(IOU_THRS), len(d), len(g)
        tp = np.zeros((T, n), dtype=bool)
        if n and m:
            ious = iou_xywh(d, g)
            for ti, t in enumerate(IOU_THRS):
                gm = np.zeros(m, dtype=bool); thr = min(t, 1 - 1e-10)
                for di in range(n):
                    cand = np.where((~gm) & (ious[di] >= thr))[0]
                    if len(cand):
                        bi = cand[np.argmax(ious[di, cand])]
                        gm[bi] = True; tp[ti, di] = True
        res[c] = (s, tp, m)
    return res

def ap_from(scores, tp, npig):
    """COCO AP for one (category, IoU): scores [n], tp [n] bool, npig GTs."""
    if npig == 0: return None
    if len(scores) == 0: return 0.0
    order = np.argsort(-scores, kind='mergesort'); tps = np.cumsum(tp[order]); fps = np.cumsum(~tp[order])
    rc = tps / npig; pr = tps / np.maximum(tps + fps, np.finfo(float).eps)
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]: pr[i - 1] = pr[i]
    inds = np.searchsorted(rc, REC_THRS, side='left')
    q = np.zeros(len(REC_THRS))
    ok = inds < len(pr); q[ok] = pr[inds[ok]]
    return q.mean()

def prep(matched, image_ids, cats):
    """Per category: globally score-sorted detections with their image index and tp flags [T, n], plus GT counts per image."""
    out = {}
    for c in cats:
        sc, tp, im, m = [], [], [], np.zeros(len(image_ids), dtype=np.int64)
        for k, img in enumerate(image_ids):
            s, t, mm = matched[img][c]; m[k] = mm
            if len(s): sc.append(s); tp.append(t); im.append(np.full(len(s), k))
        if sc:
            sc, tp, im = np.concatenate(sc), np.concatenate(tp, axis=1), np.concatenate(im)
            order = np.argsort(-sc, kind='mergesort'); sc, tp, im = sc[order], tp[:, order], im[order]
        else:
            sc, tp, im = np.zeros(0), np.zeros((len(IOU_THRS), 0), bool), np.zeros(0, np.int64)
        out[c] = (sc, tp, im, m)
    return out

def evaluate(prepped, cats, weights=None, n_img=None):
    """mAP, AP50 and per-class AP; weights = image multiplicities (bootstrap)."""
    per_class = {}; ap50 = []
    for c in cats:
        sc, tp, im, m = prepped[c]
        w = np.ones(len(m), dtype=np.int64) if weights is None else weights
        npig = int((w * m).sum())
        if npig == 0: continue
        cnt = w[im].astype(np.float64)
        aps = []
        for ti in range(len(IOU_THRS)):
            tps = np.cumsum(tp[ti] * cnt); fps = np.cumsum((~tp[ti]) * cnt)
            rc = tps / npig; pr = tps / np.maximum(tps + fps, np.finfo(float).eps)
            pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
            inds = np.searchsorted(rc, REC_THRS, side='left'); q = np.zeros(len(REC_THRS)); ok = inds < len(pr); q[ok] = pr[inds[ok]]
            aps.append(q.mean())
        per_class[c] = float(np.mean(aps)); ap50.append(aps[0])
    return float(np.mean(list(per_class.values()))), float(np.mean(ap50)), per_class

def main():
    split = sys.argv[1]; raw_path = sys.argv[2]; nboot = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    gt = json.loads((DATA / 'detection' / f'instances_{split}.json').read_text())
    cats = sorted(c['id'] for c in gt['categories'])
    gts = {}
    for a in gt['annotations']:
        gts.setdefault(a['image_id'], {}).setdefault(a['category_id'], []).append(a['bbox'])
    gts = {i: {c: np.array(b, dtype=np.float64) for c, b in d.items()} for i, d in gts.items()}
    names = list(VARIANTS)
    t0 = time.time()
    with ProcessPoolExecutor(min(6, len(names))) as ex:
        dets = dict(ex.map(build_dets, [(n, raw_path) for n in names]))
    print(f'detections built for {len(names)} variants in {time.time() - t0:.0f} s', flush=True)
    image_ids = sorted(dets['none'])
    # validate the evaluator against pycocotools on the 'none' and 'hard' full sets
    matched = {}
    for n in names:
        matched[n] = {img: match_image(dets[n][img], gts.get(img, {}), cats) for img in image_ids}
    print(f'matching done in {time.time() - t0:.0f} s', flush=True)
    prepped = {n: prep(matched[n], image_ids, cats) for n in names}
    full = {n: evaluate(prepped[n], cats) for n in names}
    for n in ('none', 'hard'):
        coco_dets = [{'image_id': i, 'category_id': int(c), 'bbox': list(map(float, b)), 'score': float(s)}
                     for i in image_ids for b, s, c in zip(*dets[n][i])]
        m, a50, _ = coco_map(DATA / 'detection' / f'instances_{split}.json', coco_dets, image_ids)
        print(f'VALIDATION {n}: mine mAP {100*full[n][0]:.4f} AP50 {100*full[n][1]:.4f} | pycocotools mAP {100*m:.4f} AP50 {100*a50:.4f}', flush=True)
    print(f'\n{split}: full-set results (mAP / AP50), diff vs none and vs hard, in mAP points')
    for n in names:
        print(f'  {n:14s} mAP {100*full[n][0]:7.3f} AP50 {100*full[n][1]:7.3f}  d_none {100*(full[n][0]-full["none"][0]):+6.3f}  d_hard {100*(full[n][0]-full["hard"][0]):+6.3f}')
    # paired bootstrap over images
    rng = np.random.default_rng(0); N = len(image_ids)
    boots = {n: [] for n in names}
    for b in range(nboot):
        idx = rng.integers(0, N, N); w = np.bincount(idx, minlength=N)
        for n in names:
            boots[n].append(evaluate(prepped[n], cats, w)[0])
        if (b + 1) % 100 == 0: print(f'  bootstrap {b+1}/{nboot} ({time.time()-t0:.0f} s)', flush=True)
    boots = {n: np.array(v) for n, v in boots.items()}
    print(f'\n{split}: paired bootstrap ({nboot} resamples of {N} images), mAP points')
    for ref in ('none', 'hard'):
        print(f'  --- difference vs {ref} ---')
        for n in names:
            if n == ref: continue
            d = 100 * (boots[n] - boots[ref])
            print(f'  {n:14s} mean {d.mean():+6.3f}  95% CI [{np.percentile(d,2.5):+6.3f}, {np.percentile(d,97.5):+6.3f}]  P(>0) {np.mean(d>0):.3f}')
    json.dump({'full': {n: {'mAP': full[n][0], 'AP50': full[n][1], 'per_class': full[n][2]} for n in names},
               'boot_mAP': {n: boots[n].tolist() for n in names}}, open(f'boot/{split}_results.json', 'w'))

if __name__ == '__main__':
    main()
