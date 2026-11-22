"""Report figure: (top) val->train pairs for the three tiers; (bottom) mAP per tier for our entries."""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

D = Path(sys.argv[1]); OUT = Path(sys.argv[2])
r = np.load(D / 'nn.npz'); px = np.load(D / 'pix.npz')['val']
s, j, v, vp, tp = r['val_d4'], r['val_d4_j'], r['val_d4_view'], r['val_path'], r['train_path']
NAVY, BLUE, AMBER, GREY, TEAL, INK, RED = '#141B3D', '#2E4A8B', '#F2A900', '#A3ACC0', '#2A8C7E', '#3D4566', '#C8414F'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
def view(a, k):
    if k // 4: a = a[:, ::-1]
    return np.rot90(a, k % 4)
g = lambda p: np.asarray(Image.open(p).convert('L').resize((160, 160), Image.BILINEAR))
VIEW = ['identity', 'rot 90', 'rot 180', 'rot 270', 'flip', 'flip+rot 90', 'flip+rot 180', 'flip+rot 270']
# picks: exact with a non-identity view; near ~.955-.965 with low pixel NCC; far ~.88-.90
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
pick = [rng.choice(np.where((px >= .95) & (v > 0))[0]),
        rng.choice(np.where((s > .955) & (s < .965) & (px < .3))[0]),
        rng.choice(np.where((s > .88) & (s < .90))[0])]
ONLY = len(sys.argv) > 4
fig = plt.figure(figsize=(3.4, 1.05) if ONLY else (3.4, 2.75))
gs = fig.add_gridspec(1, 1, top=0.78, bottom=0.12) if ONLY else fig.add_gridspec(2, 1, height_ratios=[0.62, 1.25], hspace=0.30)
top = gs[0].subgridspec(1, 6, wspace=0.05)
ttl = ['exact copy', 'near copy', 'no copy']
for c, i in enumerate(pick):
    vi = g(vp[i]) if c == 0 else np.ascontiguousarray(view(g(vp[i]), v[i]))
    axs = []
    for k, (img, lab) in enumerate(((vi, 'val' if c == 0 else 'val (aligned)'), (g(tp[j[i]]), 'train'))):
        ax = fig.add_subplot(top[0, 2 * c + k]); axs.append(ax); ax.imshow(img, cmap='gray', vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(GREY)
        ax.set_xlabel(lab, fontsize=6.5, labelpad=1)
    sub = f'cos {s[i]:.3f}' + (f', {VIEW[v[i]]}' if c == 0 else '')
    p0, p1 = axs[0].get_position(), axs[1].get_position(); xc = (p0.x0 + p1.x1) / 2
    fig.text(xc, p0.y1 + (0.15 if ONLY else 0.045), ttl[c], ha='center', fontsize=7.5, weight='bold', color=NAVY)
    fig.text(xc, p0.y1 + (0.035 if ONLY else 0.012), sub, ha='center', fontsize=6.3, color=INK)
if ONLY:
    fig.savefig(OUT, bbox_inches='tight', pad_inches=0.02, dpi=400); sys.exit()
b = json.load(open(D / 'bands_all.json'))
models = [('YOLO11-L', 'submission-val-yolo11l-alltrain96-seed0.zip', GREY), ('ConvNeXt-S', 'submission-val-rcnn-v3-pp-softnms-obj-shift-stagemean-fused.zip', TEAL),
          ('F2', 'F2_full_val_shift-0.2.zip', '#8FA3CC'), ('SC2', 'SC2_e130_val_shift-0.2.zip', BLUE), ('F3', 'F3_avg120-140_val_shift-0.2.zip', AMBER)]
tiers = [('exact', 'exact\n(32%)'), ('near', 'near\n(31%)'), ('far', 'no copy\n(37%)'), ('all', 'all val\n(board)')]
ax = fig.add_subplot(gs[1]); w = 0.16
for m, (nm, key, col) in enumerate(models):
    y = [b[key][t][1] for t, _ in tiers]
    ax.bar(np.arange(4) + (m - 2) * w, y, w * 0.92, color=col, label=nm, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in tiers], fontsize=7)
ax.set_ylabel('mAP on official val'); ax.set_ylim(0, 85); ax.grid(axis='y', color='#E3E6EE', zorder=0)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.legend(ncol=5, frameon=False, fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, 1.13), handlelength=1, columnspacing=0.8)
fig.savefig(OUT, bbox_inches='tight', pad_inches=0.02, dpi=400)
print('picked', [(int(r['val_id'][i]), float(s[i]), int(v[i]), float(px[i])) for i in pick])
