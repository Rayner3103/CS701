"""Pixel check of D4-aligned pairs + contact sheet of random pairs per similarity band (for threshold calibration)."""
import sys, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

D = Path(sys.argv[1]); r = np.load(D / 'nn.npz')
tp = r['train_path']


def view(a, v):  # same order as emb.py: flip index v//4, rot k=v%4 (torch.rot90 on (H,W) == np.rot90)
    if v // 4: a = a[:, ::-1]
    return np.rot90(a, v % 4)


def g(p, s=64):
    return np.asarray(Image.open(p).convert('L').resize((s, s), Image.BILINEAR), np.float32)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


out = {}
for sp in ('val', 'test'):
    paths, j, v, s = r[f'{sp}_path'], r[f'{sp}_d4_j'], r[f'{sp}_d4_view'], r[f'{sp}_d4']
    px = np.array([ncc(view(g(paths[i]), v[i]), g(tp[j[i]])) for i in range(len(paths))])
    out[sp] = px
    print(sp, 'pixel NCC>=.9 after D4 align: %.1f%%' % (100 * (px >= .9).mean()),
          ' NCC>=.8: %.1f%%' % (100 * (px >= .8).mean()), ' identity view among those: %.1f%%' % (100 * (v[px >= .9] == 0).mean()))
    for lo, hi in ((.99, 1.01), (.97, .99), (.95, .97), (.94, .95), (.92, .94), (.90, .92), (0, .90)):
        k = (s >= lo) & (s < hi)
        print('   band %.2f-%.2f  n=%4d  pixNCC median %.2f  share>=.8 %.0f%%' % (lo, hi, k.sum(), np.median(px[k]), 100 * (px[k] >= .8).mean()))
np.savez(D / 'pix.npz', **out)

# contact sheet, val: 6 random pairs per band
rng = np.random.default_rng(0)
bands = ((.97, .99), (.95, .97), (.94, .95), (.92, .94), (.90, .92), (0, .90))
T = 128; sheet = Image.new('L', (12 * (T + 4), len(bands) * (T + 18)), 255); dr = ImageDraw.Draw(sheet)
s, j, v, paths = r['val_d4'], r['val_d4_j'], r['val_d4_view'], r['val_path']
for bi, (lo, hi) in enumerate(bands):
    idx = np.where((s >= lo) & (s < hi))[0]; idx = rng.choice(idx, min(6, len(idx)), replace=False)
    for c, i in enumerate(idx):
        a = Image.fromarray(np.ascontiguousarray(view(g(paths[i], T), v[i])).astype(np.uint8)); b = Image.fromarray(g(tp[j[i]], T).astype(np.uint8))
        x = c * 2 * (T + 4); y = bi * (T + 18)
        sheet.paste(a, (x, y + 16)); sheet.paste(b, (x + T + 2, y + 16))
        dr.text((x + 2, y + 2), f'{s[i]:.3f} px{out["val"][i]:.2f}', fill=0)
sheet.save(D / 'sheet.png')
