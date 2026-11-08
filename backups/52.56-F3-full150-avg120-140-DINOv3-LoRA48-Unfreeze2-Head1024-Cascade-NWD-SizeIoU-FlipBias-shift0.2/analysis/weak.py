"""Weakness analysis of a team run on the local seed-0 80/10/10 split: AP by box size, by IoU threshold, per class."""
import csv, json, os, random, sys, io, contextlib
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SRC = os.environ.get('SAR_TRAIN', '../../../../cs701-sar-course-data/train')  # folder with labels.csv + instances.json
rows = list(csv.DictReader(open(f'{SRC}/labels.csv')))
coco = json.load(open(f'{SRC}/instances.json'))
by = defaultdict(list)
for r in rows: by[r['label']].append(r)
rng = random.Random(0); sp = {'train': [], 'val': [], 'test': []}
for g in by.values():
    rng.shuffle(g); nv, nt = round(len(g) * .1), round(len(g) * .1)
    sp['val'] += g[:nv]; sp['test'] += g[nv:nv + nt]; sp['train'] += g[nv + nt:]
anns = defaultdict(list)
for a in coco['annotations']: anns[a['image_id']].append(a)
imgs = {i['id']: i for i in coco['images']}

def gt(split):
    ids = sorted(int(r['image_id']) for r in sp[split])
    d = {'images': [imgs[i] for i in ids], 'annotations': [a for i in ids for a in anns[i]], 'categories': coco['categories']}
    for k, a in enumerate(d['annotations']): a.setdefault('iscrowd', 0); a.setdefault('area', a['bbox'][2] * a['bbox'][3])
    c = COCO(); c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()): c.createIndex()
    return c

def ev(G, P, iou=None, area=None, cats=None):
    E = COCOeval(G, P, 'bbox')
    if iou is not None: E.params.iouThrs = np.array([iou])
    if area: E.params.areaRng = [area]; E.params.areaRngLbl = ['x']
    with contextlib.redirect_stdout(io.StringIO()): E.evaluate(); E.accumulate()
    p = E.eval['precision'][:, :, :, 0, -1]  # T,R,K
    r = E.eval['recall'][:, :, 0, -1]          # T,K
    ap = [np.mean(p[:, :, k][p[:, :, k] > -1]) if (p[:, :, k] > -1).any() else np.nan for k in range(p.shape[2])]
    rc = [np.mean(r[:, k][r[:, k] > -1]) if (r[:, k] > -1).any() else np.nan for k in range(r.shape[1])]
    return np.array(ap), np.array(rc)

run = sys.argv[1]
for split in ['val', 'test']:
    G = gt(split)
    names = [c['name'] for c in sorted(G.dataset['categories'], key=lambda c: c['id'])]
    with contextlib.redirect_stdout(io.StringIO()): P = G.loadRes(f'{run}/predictions_{split}.json')
    print(f'===== {split}: {len(G.dataset["images"])} images, {len(G.dataset["annotations"])} boxes')
    ap, _ = ev(G, P)
    print('mAP %.2f' % (np.nanmean(ap) * 100))
    hdr = '%-13s' % 'class' + ''.join('%8s' % h for h in ['AP', 'AP10', 'AP30', 'AP50', 'AP75', 'R10', 'R50', 'nGT', 'medsz'])
    print(hdr)
    cols = [ev(G, P)[0], ev(G, P, .1), ev(G, P, .3)[0], ev(G, P, .5), ev(G, P, .75)[0]]
    sizes = defaultdict(list)
    for a in G.dataset['annotations']: sizes[a['category_id']].append(np.sqrt(a['bbox'][2] * a['bbox'][3]))
    cids = sorted(G.getCatIds())
    for k, n in enumerate(names):
        print('%-13s' % n + ''.join('%8.1f' % (v * 100) for v in [cols[0][k], cols[1][0][k], cols[2][k], cols[3][0][k], cols[4][k], cols[1][1][k], cols[3][1][k]])
              + '%8d%8.1f' % (len(sizes[cids[k]]), np.median(sizes[cids[k]])))
    print('mean         ' + ''.join('%8.1f' % (np.nanmean(v) * 100) for v in [cols[0], cols[1][0], cols[2], cols[3][0], cols[4], cols[1][1], cols[3][1]]))
    print('AP by sqrt(area) px:')
    for lo, hi in [(0, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 64), (64, 1e5)]:
        nb = sum(lo <= np.sqrt(a['bbox'][2] * a['bbox'][3]) < hi for a in G.dataset['annotations'])
        a_, r_ = ev(G, P, area=[lo * lo, hi * hi])
        print('  %4d-%-5s n=%5d  AP %5.1f  R(.5:.95) %5.1f' % (lo, str(int(hi)) if hi < 1e5 else 'inf', nb, np.nanmean(a_) * 100, np.nanmean(r_) * 100))
