"""Score val submissions on the released val labels, overall and per leakage tier.
Tiers: exact = pixel NCC>=.8 after flip/rot90 alignment to the nearest train image; near = D4 cosine >=.94 (not exact);
far = D4 cosine <.94."""
import sys, csv, io, json, zipfile, contextlib, numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def f1_score(y, p, average='macro'):
    y = np.array(y); p = np.array(p); fs = []
    for c in np.unique(np.concatenate([y, p])):
        tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
        fs.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(fs))

D = Path(sys.argv[1]); HF = Path(sys.argv[2]); zips = sys.argv[3:]
r = np.load(D / 'nn.npz'); px = np.load(D / 'pix.npz')['val']
ids = r['val_id']; s = r['val_d4']
tier = np.where(px >= .8, 'exact', np.where(s >= .94, 'near', 'far'))
sh = np.load(D / 'shift.npz')['val']
strict = (tier == 'far') & (sh < .8)
nearv = (tier == 'near') & (sh >= .8)
groups = {'all': set(ids.tolist())}
for t in ('exact', 'near', 'far'): groups[t] = set(ids[tier == t].tolist())
groups = {'far': groups['far'], 'strict_none': set(ids[strict].tolist()), 'near_verified': set(ids[nearv].tolist()), 'near_unverified': set(ids[(tier == 'near') & (sh < .8)].tolist())}
gt_json = HF / 'val_instances.json'
labels = {int(row['image_id']): int(row['label']) for row in csv.DictReader(open(HF / 'val_labels.csv'))}
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(str(gt_json))
F1R, MR = 93.31, 26.13
print('tier sizes', {k: len(v) for k, v in groups.items()})
res = {}
for zp in zips:
    with zipfile.ZipFile(zp) as z:
        n = z.namelist()
        rows = list(csv.DictReader(io.StringIO(z.read([x for x in n if x.endswith('classification.csv')][0]).decode())))
        dets = json.loads(z.read([x for x in n if x.endswith('detection.json')][0]))
    pred = {int(row['image_id']): int(row['label']) for row in rows}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
    out = {}
    for g, S in groups.items():
        I = sorted(S)
        f1 = 100 * f1_score([labels[i] for i in I], [pred[i] for i in I], average='macro')
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, dt, 'bbox'); ev.params.imgIds = I; ev.evaluate(); ev.accumulate(); ev.summarize()
        mp = 100 * ev.stats[0]
        out[g] = (f1, mp, 50 * ((f1 - F1R) / F1R + (mp - MR) / MR))
    res[Path(zp).name] = out
    print(Path(zp).name, ' | '.join(f'{g}: F1 {v[0]:.2f} mAP {v[1]:.2f} dm {v[2]:.2f}' for g, v in out.items()))
json.dump(res, open(D / 'bands_strict.json', 'w'), indent=1)
