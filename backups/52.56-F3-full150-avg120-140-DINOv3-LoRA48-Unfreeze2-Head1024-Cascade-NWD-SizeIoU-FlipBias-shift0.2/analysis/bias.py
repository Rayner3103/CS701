"""Systematic box bias of confident detections matched to GT (IoU>0.3, same class): per class, dx/dy of centres and
log w/h ratios, plus GT w/h spread (is the class near fixed size?)."""
import sys, json, numpy as np
sys.argv += []
exec(open(__file__.replace('bias.py', 'weak.py')).read().split("run = sys.argv[1]")[0])
from collections import defaultdict
def iou(a, b):
    ax2, ay2, bx2, by2 = a[0]+a[2], a[1]+a[3], b[0]+b[2], b[1]+b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0])); ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    i = iw * ih; return i / (a[2]*a[3] + b[2]*b[3] - i + 1e-9)
run = sys.argv[1]
for split in ['val', 'test']:
    G = gt(split); names = {c['id']: c['name'] for c in G.dataset['categories']}
    preds = defaultdict(list)
    for p in json.load(open(f'{run}/predictions_{split}.json')):
        if p['score'] > 0.3: preds[(p['image_id'], p['category_id'])].append(p['bbox'])
    st = defaultdict(list); gs = defaultdict(list)
    for a in G.dataset['annotations']:
        g = a['bbox']; gs[a['category_id']].append((g[2], g[3]))
        c = preds.get((a['image_id'], a['category_id']), [])
        if not c: continue
        b = max(c, key=lambda b: iou(b, g)); v = iou(b, g)
        if v < 0.3: continue
        st[a['category_id']].append((b[0]+b[2]/2-g[0]-g[2]/2, b[1]+b[3]/2-g[1]-g[3]/2, np.log(b[2]/g[2]), np.log(b[3]/g[3]), v))
    print(f'== {split}  (median over matches; w/h ratio = pred/GT)')
    print('%-13s %5s %6s %6s %6s %6s %6s | GT w med/IQR    h med/IQR' % ('class', 'n', 'dx', 'dy', 'w%', 'h%', 'IoU'))
    for k in sorted(st):
        s = np.array(st[k]); w = np.array(gs[k])
        q = lambda x: '%5.1f/%4.1f' % (np.median(x), np.subtract(*np.percentile(x, [75, 25])))
        print('%-13s %5d %6.2f %6.2f %6.1f %6.1f %6.2f | %s  %s' % (names[k], len(s), *np.median(s[:, :2], 0), *(np.exp(np.median(s[:, 2:4], 0)) - 1) * 100, np.median(s[:, 4]), q(w[:, 0]), q(w[:, 1])))
