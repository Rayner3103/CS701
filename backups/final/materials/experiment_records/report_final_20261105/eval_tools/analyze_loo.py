#!/usr/bin/env python3
"""Score every finished leave-one-out run on vallike with the FINAL pipeline (no averaging: one final checkpoint) and aggregate by variant."""
import glob, json, os, re, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablate_pp as ap
E = '/mnt/hdd/junkaichen/CS701/CS701/experiments/report_final_20261105'
def job(d):
    name = os.path.basename(d); var, seed = re.match(r'RPT_LOO36_(.+)_s(\d+)$', name).groups()
    cfg = dict(ap.FIN)
    if var == 'noFlipBias': cfg['shift'] = -0.4
    r = ap.run((name, 'final', cfg, d, 0))
    r.pop('boot_mAP'); r.pop('boot_F1'); r.update(variant=var, seed=int(seed)); return r
if __name__ == '__main__':
    ds = sorted(d for d in glob.glob(E + '/raws/RPT_LOO36_*') if os.path.exists(d + '/final_vallike.pt'))
    with Pool(4) as p: res = p.map(job, ds, chunksize=1)
    refs = {}
    for f in glob.glob('/mnt/hdd/junkaichen/CS701/cs701_runs/RPT_REF24_s*/run/metrics.json'):
        refs[f] = json.load(open(f))
    out = {}
    for var in sorted({r['variant'] for r in res}):
        rs = [r for r in res if r['variant'] == var]; m = [r['mAP'] for r in rs]; f = [r['F1'] for r in rs]; d = [r['dm'] for r in rs]
        out[var] = dict(n=len(rs), seeds=[r['seed'] for r in rs], mAP=m, F1=f, AP50=[r['AP50'] for r in rs], dm=d, mAP_mean=float(np.mean(m)), mAP_std=float(np.std(m, ddof=1)) if len(m) > 1 else None, F1_mean=float(np.mean(f)), dm_mean=float(np.mean(d)), dm_std=float(np.std(d, ddof=1)) if len(d) > 1 else None,
                        per_class={c: float(np.mean([r['per_class'][c] for r in rs])) for c in rs[0]['per_class']})
    json.dump(dict(runs=res, variants=out), open(E + '/results/loo.json', 'w'), indent=1)
    for v, o in out.items(): print(v, o['n'], round(o['mAP_mean'], 2), round(o['F1_mean'], 2), round(o['dm_mean'], 2))
