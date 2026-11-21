"""Shift-tolerant verification of copy pairs: after undoing the best flip/rotation, match the central 50% crop of
each image inside the other (OpenCV normalised cross-correlation) at a few relative scales. A pair counts as a
verified shifted/cropped copy if the best score >= 0.8. Reports the verified share per tier and per cosine band."""
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image

D = Path(sys.argv[1]); r = np.load(D / 'nn.npz'); px = np.load(D / 'pix.npz')
S = 128


def view(a, k):
    if k // 4: a = a[:, ::-1]
    return np.ascontiguousarray(np.rot90(a, k % 4))


def g(p):
    return np.asarray(Image.open(p).convert('L').resize((S, S), Image.BILINEAR), np.float32)


def best(a, b):
    sc = 0.0
    for src, dst in ((a, b), (b, a)):
        for f in (1.0, 0.8, 1.25):
          for oy, ox in ((S // 4, S // 4), (0, 0), (0, S // 2), (S // 2, 0), (S // 2, S // 2)):
            t = src[oy:oy + S // 2, ox:ox + S // 2]
            if f != 1.0:
                n = int(round(t.shape[0] * f)); t = cv2.resize(t, (n, n))
            if t.std() < 1e-3 or t.shape[0] >= S: continue
            sc = max(sc, float(cv2.matchTemplate(dst, t, cv2.TM_CCOEFF_NORMED).max()))
    return sc


out = {}
for sp in ('val', 'test'):
    s, j, v, P = r[f'{sp}_d4'], r[f'{sp}_d4_j'], r[f'{sp}_d4_view'], r[f'{sp}_path']
    tier = np.where(px[sp] >= .8, 'exact', np.where(s >= .94, 'near', 'none'))
    sc = np.array([best(view(g(P[i]), v[i]), g(r['train_path'][j[i]])) if tier[i] != 'exact' else 1.0 for i in range(len(P))])
    out[sp] = sc
    for t in ('near', 'none'):
        k = tier == t
        print(sp, t, 'n', k.sum(), 'verified(>=.8) %.1f%%' % (100 * (sc[k] >= .8).mean()), ' >=.7 %.1f%%' % (100 * (sc[k] >= .7).mean()))
    for lo, hi in ((.97, 1.01), (.95, .97), (.94, .95), (.92, .94), (.90, .92), (0, .90)):
        k = (tier != 'exact') & (s >= lo) & (s < hi)
        print('   band %.2f-%.2f n=%4d verified %.0f%%' % (lo, hi, k.sum(), 100 * (sc[k] >= .8).mean()))
np.savez(D / 'shift.npz', **out)
# control: random unrelated pairs, to estimate the false-positive rate of the matcher
rng = np.random.default_rng(0); P = r['val_path']; T = r['train_path']
ctl = np.array([best(g(P[rng.integers(len(P))]), g(T[rng.integers(len(T))])) for _ in range(400)])
print('control random pairs: >=.8 %.1f%%  >=.7 %.1f%%' % (100 * (ctl >= .8).mean(), 100 * (ctl >= .7).mean()))
