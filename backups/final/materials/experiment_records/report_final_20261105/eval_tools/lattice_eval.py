"""Per-image lattice snapping for wind_turbine (and oil_tank/512): many of their training images have all boxes on a shared
integer lattice x = k + ox, y = k + oy with integer w, h. Estimate (ox, oy) from the image's own predictions (circular mean of
fractional parts of x1, x2 / y1, y2 over confident boxes), snap only when the concentration R exceeds r0."""
import sys, json, numpy as np
sys.path.insert(0, '.')
import boot_eval as be
from snap_eval import snap
from pp_study import DATA

def lattice_snap(dets, sizes, classes, r0, smin=0.3, nmin=3, mode='corners'):
    out = {}; stats = [0, 0]
    for img, (xywh, sc, lab) in dets.items():
        b = xywh.copy(); w0, h0 = sizes[img]
        for c in classes:
            m = lab == c
            if not m.any(): continue
            conf = m & (sc >= smin)
            if conf.sum() < nmin: continue
            stats[0] += 1
            x1, y1 = b[conf, 0], b[conf, 1]; x2, y2 = x1 + b[conf, 2], y1 + b[conf, 3]
            def circ(v):
                a = 2 * np.pi * (v - np.floor(v)); z = np.exp(1j * a).mean(); return (np.angle(z) / (2 * np.pi)) % 1.0, abs(z)
            ox, rx = circ(np.r_[x1, x2]); oy, ry = circ(np.r_[y1, y2])
            if min(rx, ry) < r0: continue
            stats[1] += 1
            X1, Y1 = b[m, 0], b[m, 1]; X2, Y2 = X1 + b[m, 2], Y1 + b[m, 3]
            s = lambda v, o: np.round(v - o) + o
            nX1, nY1, nX2, nY2 = s(X1, ox), s(Y1, oy), s(X2, ox), s(Y2, oy)
            nX2 = np.maximum(nX2, nX1 + 1); nY2 = np.maximum(nY2, nY1 + 1)
            b[m, 0], b[m, 1], b[m, 2], b[m, 3] = nX1, nY1, nX2 - nX1, nY2 - nY1
        out[img] = (b, sc, lab)
    return out, stats

if __name__ == '__main__':
    NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for split in ('val', 'test'):
        gt = json.loads((DATA / 'detection' / f'instances_{split}.json').read_text()); cats = sorted(c['id'] for c in gt['categories']); names = {c['id']: c['name'] for c in gt['categories']}
        sizes = {im['id']: (im['width'], im['height']) for im in gt['images']}
        gts = {}
        for a in gt['annotations']: gts.setdefault(a['image_id'], {}).setdefault(a['category_id'], []).append(a['bbox'])
        gts = {i: {c: np.array(b, dtype=np.float64) for c, b in d.items()} for i, d in gts.items()}
        base = snap(be.build_dets(('agn_k9', f'boot/W3_raw_{split}.pt'))[1], sizes)    # integer snap already applied
        ids = sorted(base); variants = {'snap': base}
        for r0 in (0.0, 0.3, 0.5, 0.7):
            variants[f'wt_r{r0}'], st = lattice_snap(base, sizes, [9], r0); print(f'{split} wind_turbine r0={r0}: images considered {st[0]}, snapped {st[1]}')
        for r0 in (0.3, 0.5):
            variants[f'oil_r{r0}'], st = lattice_snap(base, {i: s for i, s in sizes.items()}, [6], r0)
            print(f'{split} oil_tank r0={r0}: considered {st[0]}, snapped {st[1]}')
        prepped = {n: be.prep({i: be.match_image(d[i], gts.get(i, {}), cats) for i in ids}, ids, cats) for n, d in variants.items()}
        full = {n: be.evaluate(prepped[n], cats) for n in variants}
        for n in variants:
            print(f'  {split} {n:10s} mAP {100*full[n][0]:.3f} ({100*(full[n][0]-full["snap"][0]):+.3f})  wind_turbine AP {100*full[n][2][9]:.2f}  oil_tank AP {100*full[n][2][6]:.2f}')
        rng = np.random.default_rng(0); N = len(ids); dd = {n: [] for n in variants if n != 'snap'}
        for b in range(NB):
            w = np.bincount(rng.integers(0, N, N), minlength=N); b0 = be.evaluate(prepped['snap'], cats, w)[0]
            for n in dd: dd[n].append(100 * (be.evaluate(prepped[n], cats, w)[0] - b0))
        for n, v in dd.items():
            v = np.array(v); print(f'  {split} boot {n:10s} vs snap: {v.mean():+.3f} [{np.percentile(v,2.5):+.3f},{np.percentile(v,97.5):+.3f}] P>0 {np.mean(v>0):.3f}')
