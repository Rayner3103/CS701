"""Integer-lattice snapping: in the training annotations, aircraft, car and every 256-px image (bridge/harbor/oil_tank/ship)
have x, y, w, h exactly integer in original pixels. Snap predicted boxes of those (class, size) cells to the integer grid
(pure post-processing of one forward pass) and measure the effect on local holdouts with a paired bootstrap."""
import sys, json, numpy as np
sys.path.insert(0, '.')
import boot_eval as be
from pp_study import DATA
INT_CELLS = {(1, 'any'), (4, 'any'), (3, '256'), (5, '256'), (6, '256'), (8, '256')}   # category ids: 1 aircraft, 4 car, 3 bridge, 5 harbor, 6 oil_tank, 8 ship

def is_int_cell(c, size):
    return (c, 'any') in INT_CELLS or ((c, size) in INT_CELLS)

def snap(dets, sizes, mode='round', alpha=1.0, cells=None):
    out = {}
    for img, (xywh, sc, lab) in dets.items():
        w0, h0 = sizes[img]; size = '512' if max(w0, h0) >= 400 else '256'
        b = xywh.copy()
        m = np.array([is_int_cell(int(c), size) and (cells is None or int(c) in cells) for c in lab], dtype=bool)
        if m.any():
            x1, y1, x2, y2 = b[m, 0], b[m, 1], b[m, 0] + b[m, 2], b[m, 1] + b[m, 3]
            if mode == 'round':
                nx1, ny1, nx2, ny2 = np.round(x1), np.round(y1), np.round(x2), np.round(y2)
            elif mode == 'floor_ceil':   # conservative: never shrink
                nx1, ny1, nx2, ny2 = np.floor(x1 + 0.5), np.floor(y1 + 0.5), np.ceil(x2 - 0.5), np.ceil(y2 - 0.5)
            nx1, ny1 = x1 + alpha * (nx1 - x1), y1 + alpha * (ny1 - y1); nx2, ny2 = x2 + alpha * (nx2 - x2), y2 + alpha * (ny2 - y2)
            nx2 = np.maximum(nx2, nx1 + 1e-3); ny2 = np.maximum(ny2, ny1 + 1e-3)
            b[m, 0], b[m, 1], b[m, 2], b[m, 3] = nx1, ny1, nx2 - nx1, ny2 - ny1
        out[img] = (b, sc, lab)
    return out

def errors(dets, gts, sizes, cats, names):
    """TP (IoU>=0.5, greedy) corner errors in original px per class/size cell."""
    res = {}
    for img, (xywh, sc, lab) in dets.items():
        w0, h0 = sizes[img]; size = '512' if max(w0, h0) >= 400 else '256'
        for c in cats:
            g = gts.get(img, {}).get(c)
            if g is None: continue
            sel = np.where(lab == c)[0]
            if not len(sel): continue
            order = sel[np.argsort(-sc[sel])]; d = xywh[order]; ious = be.iou_xywh(d, g); used = np.zeros(len(g), bool)
            for di in range(len(d)):
                cand = np.where((~used) & (ious[di] >= 0.5))[0]
                if not len(cand): continue
                gi = cand[np.argmax(ious[di, cand])]; used[gi] = True
                pe = np.r_[d[di, :2], d[di, :2] + d[di, 2:]] - np.r_[g[gi, :2], g[gi, :2] + g[gi, 2:]]
                res.setdefault((names[c], size), []).append(pe)
    return {k: np.array(v) for k, v in res.items()}

if __name__ == '__main__':
    NB = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for split in ('val', 'test'):
        gt = json.loads((DATA / 'detection' / f'instances_{split}.json').read_text()); cats = sorted(c['id'] for c in gt['categories']); names = {c['id']: c['name'] for c in gt['categories']}
        sizes = {im['id']: (im['width'], im['height']) for im in gt['images']}
        gts = {}
        for a in gt['annotations']: gts.setdefault(a['image_id'], {}).setdefault(a['category_id'], []).append(a['bbox'])
        gts = {i: {c: np.array(b, dtype=np.float64) for c, b in d.items()} for i, d in gts.items()}
        base = be.build_dets(('agn_k9', f'boot/W3_raw_{split}.pt'))[1]; ids = sorted(base)
        err = errors(base, gts, sizes, cats, names)
        print(f'\n{split}: TP corner errors (original px): median |e|, share |e|<0.5, mean e (bias) per cell')
        for k in sorted(err):
            e = err[k]; print(f'  {k[0]:13s} {k[1]}: n={len(e):4d} med|e| {np.median(np.abs(e)):.2f} share<0.5 {np.mean(np.abs(e)<0.5):.2f} bias x1,y1,x2,y2 {np.round(e.mean(0),2).tolist()}')
        variants = {'base': base, 'round': snap(base, sizes, 'round'), 'round_a05': snap(base, sizes, 'round', 0.5),
                    'floorceil': snap(base, sizes, 'floor_ceil')}
        for cid in (1, 4, 8, 6, 5, 3):
            variants[f'round_only_{names[cid]}'] = snap(base, sizes, 'round', 1.0, cells={cid})
        prepped = {}
        for n, d in variants.items():
            matched = {i: be.match_image(d[i], gts.get(i, {}), cats) for i in ids}; prepped[n] = be.prep(matched, ids, cats)
        full = {n: be.evaluate(prepped[n], cats) for n in variants}
        print(f'{split}: mAP (and per-class change vs base)')
        for n in variants:
            ch = {names[c]: round(100 * (full[n][2][c] - full['base'][2][c]), 2) for c in cats if abs(full[n][2][c] - full['base'][2][c]) > 1e-6}
            print(f'  {n:22s} mAP {100*full[n][0]:.3f} ({100*(full[n][0]-full["base"][0]):+.3f})  AP50 {100*full[n][1]:.2f}  {ch}')
        rng = np.random.default_rng(0); N = len(ids); dd = {n: [] for n in ('round', 'round_a05', 'floorceil')}
        for b in range(NB):
            w = np.bincount(rng.integers(0, N, N), minlength=N); b0 = be.evaluate(prepped['base'], cats, w)[0]
            for n in dd: dd[n].append(100 * (be.evaluate(prepped[n], cats, w)[0] - b0))
        for n, v in dd.items():
            v = np.array(v); print(f'  bootstrap {n:10s}: {v.mean():+.3f} [{np.percentile(v,2.5):+.3f}, {np.percentile(v,97.5):+.3f}] P>0 {np.mean(v>0):.3f}')
