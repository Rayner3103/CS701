"""Local proxy evaluation on the merged holdouts (local val 940 + local test 940 = 1,880 labelled images, all from the
official train pool, never seen by 7,512-train models), with two weightings:
  uniform  : every image once
  val-like : images resampled with probability proportional to p_officialval(class, size) / p_local(class, size)
             (official val class = our 56.49 model's prediction, 95.6% accurate; size = 256 vs 512 bucket)
Reports COCO mAP (exact pycocotools replica), macro-F1 (lambda-2 fused label) and the local Delta-m-like score
1.9135*mAP + 0.5358*F1 (official sensitivities), with image-bootstrap CIs; paired differences between models.

Usage: proxy_eval.py <nboot> <name1>=<rawprefix1> [<name2>=<rawprefix2> ...] [--cfg key=val ...]
  rawprefix: path prefix such that <prefix>_val.pt and <prefix>_test.pt exist (raw dumps with 2000 proposals).
  --cfg overrides post-processing (defaults = final Phase-1 recipe: agn gamma1 topk9, sigma0.5, obj0, shift-0.2, mean).
"""
import sys, json, csv, io, zipfile, collections, numpy as np, torch
S = '/mnt/hdd/junkaichen/CS701/CS701/experiments/report_final_20261105/eval_tools'; sys.path.insert(0, S)
import boot_eval as be
from pp_study import DATA
from sarbench.data import to_original_xywh
from sarbench.postprocess import detections_from_raw
from sarbench.metrics import classification_metrics

FINAL = dict(be.BASE, fusion='agn', gamma=1.0, topk=9)
K_MAP, K_F1 = 1.9135, 0.5358

def parse(argv):
    nboot = int(argv[0]); models, cfg = [], dict(FINAL)
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '--snap':
            cfg['_snap'] = True; i += 1; continue
        if a == '--splits':
            cfg['_splits'] = argv[i + 1].split(','); i += 2; continue
        if a == '--subset':
            cfg['_subset'] = argv[i + 1]; i += 2; continue
        if a == '--cfg':
            i += 1
            while i < len(argv) and '=' in argv[i] and not argv[i].startswith('--'):
                k, v = argv[i].split('=', 1)
                try: v = json.loads(v)
                except Exception: pass
                cfg[k] = v; i += 1
            continue
        n, p = a.split('='); models.append((n, p)); i += 1
    return nboot, models, cfg

def load_split(prefix, split, cfg):
    raw = torch.load(f'{prefix}_{split}.pt')
    topn = cfg.pop('_topn', None) if '_topn' in cfg else None
    dets, ids, labels, logits_pred = {}, [], [], []
    for i, image_id in enumerate(raw['image_ids']):
        rb, rl, ro = raw['raw_boxes'][i], raw['raw_logits'][i].float(), raw['raw_obj'][i]
        st = raw['raw_stage'][i] if raw.get('raw_stage') else None
        if topn:
            rb, rl, ro = rb[:topn], rl[:topn], ro[:topn]; st = st[:, :topn] if st is not None else None
        probs = raw['cls_logits'][i].softmax(0)
        qq = raw['raw_iou'][i] if raw.get('raw_iou') else None
        if qq is not None and topn: qq = qq[:topn]
        b, s, l = detections_from_raw(rb, rl, probs, objectness=ro, quality=qq, stage_inputs=st, **{k: v for k, v in cfg.items() if not k.startswith('_')})
        key = (split, int(image_id))
        dets[key] = (to_original_xywh(b, raw['orig_sizes'][i]).numpy().astype(np.float64), s.numpy().astype(np.float64), l.numpy().astype(np.int64))
        logp = raw['cls_logits'][i].log_softmax(0); logq = torch.log(rl.softmax(-1)[:, 1:].max(0).values + 1e-6)
        logits_pred.append(int((logp + cfg.get('_lam', 2.0) * logq).argmax())); labels.append(int(raw['labels'][i])); ids.append(key)
    sizes = ['512' if max(int(s[0]), int(s[1])) >= 400 else '256' for s in raw['orig_sizes']]
    if cfg.get('_snap'):
        from snap_eval import snap
        from lattice_eval import lattice_snap
        wh = {key: (int(o[1]), int(o[0])) for key, o in zip(ids, raw['orig_sizes'])}   # orig_size = (h, w)
        dets = lattice_snap(snap(dets, wh), wh, [9], 0.5)[0]
    if topn: cfg['_topn'] = topn
    return dets, ids, np.array(labels), np.array(logits_pred), sizes

def val_like_weights(labels, sizes):
    vimgs = json.load(open('/mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/val/images.json'))['images']
    with zipfile.ZipFile('/mnt/hdd/junkaichen/CS701/CS701/submissions/BP7_avg100-180x9_rpn2000_obj0_soft05_cls2_agn_g1_k9.zip') as z:
        cname = [n for n in z.namelist() if n.endswith('.csv')][0]
        pred = {int(r[0]): int(r[1]) for r in list(csv.reader(io.StringIO(z.read(cname).decode())))[1:]}
    bucket = lambda w, h: '512' if max(w, h) >= 400 else '256'
    pv = collections.Counter((pred[im['id']], bucket(im['width'], im['height'])) for im in vimgs)
    pl = collections.Counter(zip(labels.tolist(), sizes))
    w = np.array([(pv[(l, s)] / len(vimgs)) / (pl[(l, s)] / len(labels)) if pl[(l, s)] else 0.0 for l, s in zip(labels.tolist(), sizes)])
    return w / w.sum()

def macro_f1(labels, pred, w=None):
    # weighted macro-F1 over the 9 classes (w = image multiplicities)
    w = np.ones(len(labels)) if w is None else w
    f1s = []
    for c in range(9):
        tp = (w * ((pred == c) & (labels == c))).sum(); fp = (w * ((pred == c) & (labels != c))).sum(); fn = (w * ((pred != c) & (labels == c))).sum()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return 100 * float(np.mean(f1s))

def main():
    nboot, models, cfg = parse(sys.argv[1:])
    splits = cfg.get('_splits', ['val', 'test'])
    gts, cats = {}, None
    for split in splits:
        gt = json.loads((DATA / 'detection' / f'instances_{split}.json').read_text()); cats = sorted(c['id'] for c in gt['categories'])
        for a in gt['annotations']:
            gts.setdefault((split, a['image_id']), {}).setdefault(a['category_id'], []).append(a['bbox'])
    gts = {i: {c: np.array(b, dtype=np.float64) for c, b in d.items()} for i, d in gts.items()}
    keep = None
    if cfg.get('_subset'):
        import csv as _csv
        names = {r['image'] for r in _csv.DictReader(open(DATA / 'classification' / f"{cfg['_subset']}.csv"))}
        keep = set()
        for split in ('val', 'test'):
            for im in json.loads((DATA / 'detection' / f'instances_{split}.json').read_text())['images']:
                if im['file_name'] in names: keep.add((split, im['id']))
        print(f"subset {cfg['_subset']}: {len(keep)} images of the merged holdouts")
    res = {}
    for name, prefix in models:
        dets, ids, labels, pred, sizes = {}, [], [], [], []
        for split in splits:
            d, i, l, p, s = load_split(prefix, split, {k: v for k, v in cfg.items() if k not in ('_subset', '_splits')})
            if keep is not None:
                sel = [j for j, key in enumerate(i) if key in keep]
                d = {i[j]: d[i[j]] for j in sel}; l = l[sel]; p = p[sel]; s = [s[j] for j in sel]; i = [i[j] for j in sel]
            dets.update(d); ids += i; labels.append(l); pred.append(p); sizes += s
        labels, pred = np.concatenate(labels), np.concatenate(pred)
        matched = {i: be.match_image(dets[i], gts.get(i, {}), cats) for i in ids}
        res[name] = dict(prep=be.prep(matched, ids, cats), ids=ids, labels=labels, pred=pred, sizes=sizes)
    ids = res[models[0][0]]['ids']; N = len(ids)
    labels, sizes = res[models[0][0]]['labels'], res[models[0][0]]['sizes']
    pvl = val_like_weights(labels, sizes)
    print(f'proxy set: {N} images; val-like weights ESS {1/np.sum(pvl**2):.0f}; post-processing {json.dumps({k: v for k, v in cfg.items() if k in ("fusion","gamma","topk","soft_sigma","obj_pow","shift","stage_box","min_keep","_topn")})}')
    rng = np.random.default_rng(0)
    draws = {'uniform': [np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(nboot)],
             'val-like': [np.bincount(rng.choice(N, N, p=pvl), minlength=N) for _ in range(nboot)]}
    out = {}
    for name in res:
        r = res[name]; full_u = be.evaluate(r['prep'], cats); f1_u = macro_f1(r['labels'], r['pred'])
        wv = pvl * N; full_v = be.evaluate(r['prep'], cats, None)  # point estimate under val-like = mean of draws below
        boots = {}
        for mode, ws in draws.items():
            m = np.array([be.evaluate(r['prep'], cats, w)[0] * 100 for w in ws]); f = np.array([macro_f1(r['labels'], r['pred'], w) for w in ws])
            boots[mode] = (m, f, K_MAP * m + K_F1 * f)
        out[name] = boots
        print(f'{name:14s} uniform: mAP {100*full_u[0]:.3f} F1 {f1_u:.3f} score {K_MAP*100*full_u[0]+K_F1*f1_u:.3f} | val-like (boot mean): mAP {boots["val-like"][0].mean():.3f} F1 {boots["val-like"][1].mean():.3f} score {boots["val-like"][2].mean():.3f}')
        print('   per-class AP:', {c: round(100*v, 1) for c, v in full_u[2].items()})
    if len(models) > 1:
        ref = models[0][0]
        for name, _ in models[1:]:
            for mode in ('uniform', 'val-like'):
                dm = out[name][mode][0] - out[ref][mode][0]; df = out[name][mode][1] - out[ref][mode][1]; ds = out[name][mode][2] - out[ref][mode][2]
                print(f'  {name} - {ref} [{mode:8s}]: mAP {dm.mean():+.3f} [{np.percentile(dm,2.5):+.3f},{np.percentile(dm,97.5):+.3f}] P>0 {np.mean(dm>0):.3f} | F1 {df.mean():+.3f} | score {ds.mean():+.3f} [{np.percentile(ds,2.5):+.3f},{np.percentile(ds,97.5):+.3f}] P>0 {np.mean(ds>0):.3f}')

if __name__ == '__main__':
    main()
