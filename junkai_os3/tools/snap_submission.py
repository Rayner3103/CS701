"""Post-process a submission zip: snap boxes to the annotation lattice learned from the training labels.
- Integer cells (training boxes have integer x, y, w, h in original pixels): aircraft and car at any size; bridge, harbor,
  oil_tank and ship on 256-px images. Every corner is rounded to the nearest integer.
- Wind turbines: many training images put all boxes on one integer lattice shifted by a per-image (ox, oy). The offset is
  estimated from the image's confident predictions (circular mean of corner fractional parts) and used only when the
  concentration R >= 0.5 on both axes.
Scores, categories, order and classification.csv are unchanged. One forward pass; this only edits the written boxes.
Usage: python tools/snap_submission.py IN.zip images.json OUT.zip [--no-lattice]"""
import argparse, collections, io, json, zipfile
import numpy as np

INT_ANY = {1, 4}            # category ids: aircraft, car
INT_256 = {3, 5, 6, 8}      # bridge, harbor, oil_tank, ship on 256-px images
WIND = 9

def circ(v):
    a = 2 * np.pi * (v - np.floor(v)); z = np.exp(1j * a).mean(); return (np.angle(z) / (2 * np.pi)) % 1.0, abs(z)

def main():
    p = argparse.ArgumentParser(); p.add_argument('inp'); p.add_argument('images'); p.add_argument('out')
    p.add_argument('--no-lattice', action='store_true'); p.add_argument('--r0', type=float, default=0.5)
    p.add_argument('--smin', type=float, default=0.3); p.add_argument('--nmin', type=int, default=3)
    a = p.parse_args()
    sizes = {im['id']: (im['width'], im['height']) for im in json.load(open(a.images))['images']}
    with zipfile.ZipFile(a.inp) as z:
        names = z.namelist(); det_name = [n for n in names if n.endswith('.json')][0]; files = {n: z.read(n) for n in names}
    dets = json.loads(files[det_name])
    by_img = collections.defaultdict(list)
    for k, d in enumerate(dets): by_img[d['image_id']].append(k)
    n_int = n_lat = 0
    for img, idx in by_img.items():
        w0, h0 = sizes[img]; small = max(w0, h0) < 400
        for k in idx:
            d = dets[k]; c = d['category_id']
            if c in INT_ANY or (small and c in INT_256):
                x, y, w, h = d['bbox']; x1, y1, x2, y2 = round(x), round(y), round(x + w), round(y + h)
                x1, y1 = min(max(x1, 0), w0 - 1), min(max(y1, 0), h0 - 1); x2, y2 = min(x2, w0), min(y2, h0)
                x2 = max(x2, x1 + 1); y2 = max(y2, y1 + 1)
                d['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]; n_int += 1
        if a.no_lattice:
            continue
        wk = [k for k in idx if dets[k]['category_id'] == WIND]
        conf = [k for k in wk if dets[k]['score'] >= a.smin]
        if len(conf) < a.nmin:
            continue
        b = np.array([dets[k]['bbox'] for k in conf], dtype=np.float64)
        ox, rx = circ(np.r_[b[:, 0], b[:, 0] + b[:, 2]]); oy, ry = circ(np.r_[b[:, 1], b[:, 1] + b[:, 3]])
        if min(rx, ry) < a.r0:
            continue
        s = lambda v, o: float(np.round(v - o) + o)
        for k in wk:
            x, y, w, h = dets[k]['bbox']; x1, y1, x2, y2 = s(x, ox), s(y, oy), s(x + w, ox), s(y + h, oy)
            x1, y1 = min(max(x1, 0.0), w0 - 1.0), min(max(y1, 0.0), h0 - 1.0); x2, y2 = min(x2, float(w0)), min(y2, float(h0))
            x2 = max(x2, x1 + 1); y2 = max(y2, y1 + 1)
            dets[k]['bbox'] = [x1, y1, x2 - x1, y2 - y1]; n_lat += 1
    files[det_name] = json.dumps(dets).encode()
    with zipfile.ZipFile(a.out, 'x', zipfile.ZIP_DEFLATED) as z:
        for n, data in files.items(): z.writestr(n, data)
    print(f'wrote {a.out}: {len(dets)} boxes, {n_int} snapped to the integer grid, {n_lat} wind-turbine boxes snapped to a per-image lattice')

if __name__ == '__main__':
    main()
