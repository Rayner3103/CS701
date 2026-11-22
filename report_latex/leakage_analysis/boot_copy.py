"""(1) Paired half-subsample bootstrap of mAP / dm differences on the 'far' tier (and 'all').
(2) A pure-memorisation baseline: label = label of the nearest train image; boxes = that image's boxes mapped through
the inverse flip/rot90, for 'exact' images only."""
import sys, csv, io, json, zipfile, contextlib, copy, numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

D = Path(sys.argv[1]); HF = Path(sys.argv[2]); TR = Path(sys.argv[3]); zips = dict(a.split('=', 1) for a in sys.argv[4:])
r = np.load(D / 'nn.npz'); px = np.load(D / 'pix.npz')['val']
ids = r['val_id']; s = r['val_d4']
tier = np.where(px >= .8, 'exact', np.where(s >= .94, 'near', 'far'))
labels = {int(row['image_id']): int(row['label']) for row in csv.DictReader(open(HF / 'val_labels.csv'))}
F1R, MR = 93.31, 26.13
q = lambda f: contextlib.redirect_stdout(io.StringIO())
with q(None):
    gt = COCO(str(HF / 'val_instances.json'))


def f1(I, pred):
    y = np.array([labels[i] for i in I]); p = np.array([pred[i] for i in I]); fs = []
    for c in np.unique(np.concatenate([y, p])):
        tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
        fs.append(2 * tp / max(2 * tp + fp + fn, 1))
    return 100 * float(np.mean(fs))


def load(zp):
    with zipfile.ZipFile(zp) as z:
        n = z.namelist()
        rows = list(csv.DictReader(io.StringIO(z.read([x for x in n if x.endswith('classification.csv')][0]).decode())))
        dets = json.loads(z.read([x for x in n if x.endswith('detection.json')][0]))
    return {int(x['image_id']): int(x['label']) for x in rows}, dets


def evaluator(dets):
    with q(None):
        ev = COCOeval(gt, gt.loadRes(dets) if dets else COCO(), 'bbox'); ev.evaluate()
    return ev


def mAP(ev, I):
    """mAP over image list I (duplicates allowed): rebuild evalImgs for I instead of COCOeval.accumulate(p),
    whose subset indexing is wrong."""
    full = ev._paramsEval.imgIds; I0 = len(full); pos = {im: n for n, im in enumerate(full)}
    sel = [pos[i] for i in I]; blocks = len(ev.evalImgs) // I0
    e = copy.copy(ev); pe = copy.copy(ev._paramsEval); pe.imgIds = list(range(len(sel)))
    e._paramsEval = pe; e.params = pe
    e.evalImgs = [ev.evalImgs[b * I0 + i] for b in range(blocks) for i in sel]
    with q(None):
        e.accumulate(); e.summarize()
    return 100 * e.stats[0]


M = {}
for k, zp in zips.items():
    pred, dets = load(zp); M[k] = (pred, evaluator(dets))
rng = np.random.default_rng(0)
far = ids[tier == 'far']; allI = ids
import os
pairs = [tuple(x.split('-')) for x in os.environ.get('PAIRS', 'F3-SC2').split(',') if '-' in x]
nonexact = ids[tier != 'exact']
out = {}
for A, B in pairs:
    if A not in M or B not in M: continue
    for gname, G in [g for g in (('far', far), ('nonexact', nonexact)) if g[0] in os.environ.get('GROUPS', 'far,nonexact')]:
        dm_d, map_d = [], []
        for _ in range(int(os.environ.get('REPS', 150))):
            I = rng.choice(G, len(G), replace=True)
            v = {}
            for k in (A, B):
                f, m = f1(I, M[k][0]), mAP(M[k][1], I)
                v[k] = (m, 50 * ((f - F1R) / F1R + (m - MR) / MR))
            map_d.append(v[A][0] - v[B][0]); dm_d.append(v[A][1] - v[B][1])
        dm_d, map_d = np.array(dm_d), np.array(map_d)
        out[f'{A}-{B}-{gname}'] = dict(dm_mean=dm_d.mean(), dm_sd=dm_d.std(), p_dm_pos=(dm_d > 0).mean(), map_mean=map_d.mean(), map_sd=map_d.std(), p_map_pos=(map_d > 0).mean())
        print(flush=True, end='');print(f'{A} - {B} [{gname}] dm {dm_d.mean():+.2f} sd {dm_d.std():.2f} P(>0) {(dm_d > 0).mean():.2f} | mAP {map_d.mean():+.2f} sd {map_d.std():.2f} P(>0) {(map_d > 0).mean():.2f}')

if os.environ.get('NOCOPY'): sys.exit()
# memorisation baseline
tr = json.loads((TR / 'instances.json').read_text())
tr_lab = {int(x[0]): int(x[2]) for x in (l.split(',') for l in (TR / 'labels.csv').read_text().splitlines()[1:])}
tr_size = {im['id']: (im['width'], im['height']) for im in tr['images']}
boxes = {}
for a in tr['annotations']: boxes.setdefault(a['image_id'], []).append(a)
tid = r['train_id']; j = r['val_d4_j']; vw = r['val_d4_view']
vsize = {im['id']: (im['width'], im['height']) for im in gt.dataset['images']}
pred, dets = {}, []
for n, i in enumerate(ids):
    t = int(tid[j[n]]); pred[int(i)] = tr_lab[t]
    if tier[n] != 'exact': continue
    W, H = tr_size[t]; Wq, Hq = vsize[int(i)]; k = vw[n] % 4; fl = vw[n] // 4
    for a in boxes.get(t, []):
        x, y, w, h = a['bbox']; pts = [(x, y), (x + w, y + h)]
        # train = rot90^k(flip^fl(query)) with np.rot90 (CCW). Undo: rotate CW k times, then flip.
        cw, ch = W, H
        for _ in range(k):  # one CW rotation of an image of width cw, height ch: (x, y) -> (ch - y, x)
            pts = [(ch - py, px_) for px_, py in pts]; cw, ch = ch, cw
        if fl: pts = [(cw - px_, py) for px_, py in pts]
        sx, sy = Wq / cw, Hq / ch
        xs = sorted(p[0] * sx for p in pts); ys = sorted(p[1] * sy for p in pts)
        dets.append(dict(image_id=int(i), category_id=a['category_id'], bbox=[xs[0], ys[0], xs[1] - xs[0], ys[1] - ys[0]], score=1.0))
ev = evaluator(dets)
res = {}
for g in ('all', 'exact', 'near', 'far'):
    I = ids if g == 'all' else ids[tier == g]
    f, m = f1(I, pred), mAP(ev, I); res[g] = (f, m, 50 * ((f - F1R) / F1R + (m - MR) / MR))
    print(f'copy-baseline [{g}] F1 {f:.2f} mAP {m:.2f} dm {res[g][2]:.2f}')
json.dump(dict(boot=out, copy=res), open(D / f"boot_{os.environ.get('TAG','x')}.json", 'w'), indent=1, default=float)
