#!/usr/bin/env python3
"""Post-processing/averaging ablation of one full-length twin on vallike: cumulative ladder + leave-one-out from FINAL.
Usage: ablate_pp.py <rawdir with avg100-180s10_vallike.pt, e180_vallike.pt, e160_vallike.pt> <out.json> [nboot]"""
import json, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/mnt/hdd/junkaichen/CS701/CS701/experiments/report_final_20261105/eval_tools')
import proxy_eval as pe, boot_eval as be
from pp_study import DATA
AVG, E180, E160 = 'avg100-180s10', 'e180', 'e160'
OLD = dict(score_thresh=1e-5, min_keep=30, per_class=100, total=100, nms='soft', soft_sigma=0.75, obj_pow=0.25, shift=-0.2, stage_box='mean', fusion='none', gamma=1.0, topk=9, _lam=1.0, _topn=1000)
def m(**k): return {**OLD, **k}
L1 = m(soft_sigma=0.5); L2 = m(soft_sigma=0.5, obj_pow=0.0)
L3 = {k: v for k, v in L2.items() if k != '_topn'}; L4 = {**L3, '_lam': 2.0}
HARD = {**L4, 'fusion': 'hard'}; AGN = {**L4, 'fusion': 'agn'}; MK = {**AGN, 'min_keep': 100}; FIN = {**MK, '_snap': True}
LADDER = [('old PP, e180', E180, OLD), ('+ sigma 0.5', E180, L1), ('+ objectness power 0', E180, L2), ('+ RPN 2000', E180, L3), ('+ lambda 2', E180, L4),
          ('+ average e100-180', AVG, L4), ('+ hard fusion', AVG, HARD), ('agn fusion instead', AVG, AGN), ('+ min_keep 100', AVG, MK), ('+ lattice snap (FINAL)', AVG, FIN)]
LOO = [('FINAL without snap', AVG, MK), ('FINAL min_keep 30', AVG, {**FIN, 'min_keep': 30}), ('FINAL no fusion', AVG, {**FIN, 'fusion': 'none'}), ('FINAL hard fusion', AVG, {**FIN, 'fusion': 'hard'}),
       ('FINAL sigma 0.75', AVG, {**FIN, 'soft_sigma': 0.75}), ('FINAL objectness 0.25', AVG, {**FIN, 'obj_pow': 0.25}), ('FINAL RPN 1000', AVG, {**FIN, '_topn': 1000}),
       ('FINAL last-stage box', AVG, {**FIN, 'stage_box': 'last'}), ('FINAL shift 0', AVG, {**FIN, 'shift': 0.0}), ('FINAL shift -0.4', AVG, {**FIN, 'shift': -0.4}), ('FINAL lambda 1', AVG, {**FIN, '_lam': 1.0}),
       ('FINAL single e180', E180, FIN), ('FINAL single e160', E160, FIN)]
GT = json.loads((DATA / 'detection' / 'instances_vallike.json').read_text()); CATS = sorted(c['id'] for c in GT['categories'])
GTS = {}
for a in GT['annotations']: GTS.setdefault(('vallike', a['image_id']), {}).setdefault(a['category_id'], []).append(a['bbox'])
GTS = {i: {c: np.array(b, dtype=np.float64) for c, b in d.items()} for i, d in GTS.items()}
def run(job):
    name, w, cfg, rawdir, nboot = job
    dets, ids, labels, pred, _ = pe.load_split(f'{rawdir}/{w}', 'vallike', dict(cfg))
    prep = be.prep({i: be.match_image(dets[i], GTS.get(i, {}), CATS) for i in ids}, ids, CATS)
    mAP, ap50, pc = be.evaluate(prep, CATS); f1 = pe.macro_f1(labels, pred); N = len(ids)
    rng = np.random.default_rng(0); B = [np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(nboot)]
    bm = [100 * be.evaluate(prep, CATS, x)[0] for x in B]; bf = [pe.macro_f1(labels, pred, x) for x in B]
    return dict(name=name, weights=w, cfg={k: v for k, v in cfg.items()}, mAP=100 * mAP, AP50=100 * ap50, F1=f1, dm=50 * ((f1 - 93.31) / 93.31 + (100 * mAP - 26.13) / 26.13), per_class={c: 100 * v for c, v in pc.items()}, boxes=sum(len(d[0]) for d in dets.values()), boot_mAP=bm, boot_F1=bf)
if __name__ == '__main__':
    rawdir, out = sys.argv[1:3]; nboot = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    jobs = [(n, w, c, rawdir, nboot) for n, w, c in LADDER + LOO]
    with Pool(4) as p: res = p.map(run, jobs, chunksize=1)
    by = {r['name']: r for r in res}
    def diff(a, b):
        d = np.array(by[a]['boot_mAP']) - np.array(by[b]['boot_mAP']); f = np.array(by[a]['boot_F1']) - np.array(by[b]['boot_F1']); s = 1.9135 * d + 0.5358 * f
        return dict(mAP=by[a]['mAP'] - by[b]['mAP'], F1=by[a]['F1'] - by[b]['F1'], dm=by[a]['dm'] - by[b]['dm'], mAP_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], dm_ci=[float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))], p=float((d > 0).mean()))
    names = [n for n, _, _ in LADDER]; steps = {}
    for i in range(1, len(names)):
        prev = names[i - 1] if names[i] != 'agn fusion instead' else '+ hard fusion'
        if names[i] == '+ min_keep 100': prev = 'agn fusion instead'
        steps[names[i]] = dict(vs=prev, **diff(names[i], prev))
    loo = {n: diff('+ lattice snap (FINAL)', n) for n, _, _ in LOO}
    for r in res: r.pop('boot_mAP'); r.pop('boot_F1')
    json.dump(dict(rawdir=rawdir, results=res, ladder_steps=steps, final_minus_variant=loo), open(out, 'w'), indent=1)
    for r in res: print(f"{r['name']:28s} mAP {r['mAP']:.3f} AP50 {r['AP50']:.2f} F1 {r['F1']:.3f} dm {r['dm']:.2f} boxes {r['boxes']}")
