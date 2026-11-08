"""Does snapping predicted boxes to the integer pixel grid (where the GT of that class/image size is integer) raise mAP?"""
import sys, json, copy, numpy as np, io, contextlib
exec(open(__file__.replace('snap.py', 'weak.py')).read().split("run = sys.argv[1]")[0])
INT_CLS = {'aircraft', 'car', 'ship'}
run = sys.argv[1]
def mAP(G, preds):
    with contextlib.redirect_stdout(io.StringIO()): P = G.loadRes(preds)
    ap, _ = ev(G, P); return np.nanmean(ap) * 100, ap * 100
for split in ['val', 'test']:
    G = gt(split); names = {c['id']: c['name'] for c in G.dataset['categories']}
    size = {i['id']: i['width'] for i in G.dataset['images']}
    base = json.load(open(f'{run}/predictions_{split}.json'))
    def snap(rule, mode='corner'):
        out = copy.deepcopy(base)
        for p in out:
            n = names[p['category_id']]; s = size[p['image_id']]
            if not rule(n, s): continue
            x, y, w, h = p['bbox']
            if mode == 'corner':
                x1, y1, x2, y2 = round(x), round(y), round(x + w), round(y + h)
                p['bbox'] = [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]
            else:  # wh only, keep centre
                W, H = max(1, round(w)), max(1, round(h)); p['bbox'] = [x + w / 2 - W / 2, y + h / 2 - H / 2, W, H]
        return out
    b, bap = mAP(G, base)
    print(f'== {split} base {b:.2f}')
    for lab, rule, mode in [('int classes', lambda n, s: n in INT_CLS, 'corner'),
                            ('int classes + all 256px imgs', lambda n, s: n in INT_CLS or s == 256, 'corner'),
                            ('everything', lambda n, s: True, 'corner'),
                            ('wind wh only', lambda n, s: n == 'wind_turbine', 'wh')]:
        m, ap = mAP(G, snap(rule, mode))
        print(f'  {lab:30s} {m:.2f} ({m - b:+.2f})  per-class d', {names[k + 1][:6]: round(float(ap[k] - bap[k]), 1) for k in range(len(ap)) if abs(ap[k] - bap[k]) > 0.05})
