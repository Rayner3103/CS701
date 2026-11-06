"""Report/slide figures: (1) ablation ladder, (2) model architecture block diagram.
Saves PDF + SVG into report/figures/. Numbers from USAGE.md (local holdouts)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, GREY = '#0B5CAB', '#1B7A3D', '#666666'

# ------------------------------------------------------------------ ablation ladder
runs = [('E00\nViT+LoRA r16', 0.0), ('E01\n+DINOv3-B', 8.1), ('E04r\n+LoRA r32 all-lin\n+norms', 18.1),
        ('D1\n+Cascade', 21.0), ('C36\n+NWD+small\nanchors', 31.3), ('C48\n+longer', 38.6)]
fig, ax = plt.subplots(figsize=(5.6, 2.6))
x = range(len(runs))
vals = [v for _, v in runs]
bars = ax.bar(x, vals, color=[GREY] + [BLUE] * 4 + [GREEN], width=0.62, edgecolor='none')
ax.plot(x, vals, color='#333', lw=0.8, marker='o', ms=3, zorder=3)
for xi, v in zip(x, vals):
    ax.text(xi, v + 1.0, f'+{v:.1f}' if v else '0', ha='center', fontsize=7.5)
ax.axhline(13.5, color='r', ls='--', lw=0.9)
ax.text(-0.45, 14.6, 'full FT (over budget) +13.5', color='r', fontsize=7, ha='left')
ax.set_xticks(list(x))
ax.set_xticklabels([n for n, _ in runs], fontsize=6.8)
ax.set_ylabel(r'local $\Delta$m', fontsize=9)
ax.set_ylim(-2, 44)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(labelsize=8)
ax.yaxis.label.set_size(9)
fig.tight_layout()
fig.savefig(OUT / 'ablation_ladder.pdf')
fig.savefig(OUT / 'ablation_ladder.svg')
fig.savefig(OUT / 'ablation_ladder.png', dpi=160)
print('wrote ablation_ladder')

# ------------------------------------------------------------------ architecture
fig, ax = plt.subplots(figsize=(7.4, 3.2))
ax.set_xlim(0, 10.4)
ax.set_ylim(0, 5)
ax.axis('off')


def block(x, y, w, h, text, fc='#EAF1FA', ec=BLUE, fs=8, tc='#111'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.08',
                                linewidth=1.1, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc)


def arrow(x1, y1, x2, y2, style='-|>', color='#444'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 lw=1.1, color=color))


block(0.15, 2.1, 1.15, 1.0, 'SAR image\n512×512\ngray', fc='#F4F4F4', ec='#888')
block(1.75, 1.55, 2.5, 2.1, 'DINOv3 ViT-B/16\nFROZEN\n+ LoRA r32 (all linear)\n+ trained LayerNorms', fc='#EAF1FA', ec=BLUE, fs=8.5)
ax.text(3.0, 3.9, '23.8 M trainable / 109 M total', ha='center', fontsize=7, color=GREY)

block(4.85, 3.05, 2.1, 1.0, 'mean token →\nLayerNorm + Linear', fs=8)
block(4.85, 0.75, 2.1, 1.6, 'ViTDet pyramid\nstrides 4–64\nRPN: max(IoU, NWD)\nanchors 6–181 px\ncell-centred', fs=7.3)
block(7.3, 0.75, 1.5, 1.6, 'Cascade\nR-CNN\n3 stages\nclass-agnostic', fs=7.5)
block(9.15, 0.75, 1.2, 1.6, 'post-\nprocess', fc='#FFF4E5', ec='#C07A00', fs=8)
ax.text(9.75, 0.5, 'score × obj$^{0.25}$\nsoft-NMS σ .75\nstage-mean, −0.4 px', ha='center', fontsize=6, color=GREY)
block(7.3, 3.05, 3.05, 1.0, 'image class (9)\nargmax log $p_{img}$ + log $q_{det}$', fc='#E8F5EC', ec=GREEN, fs=8)
block(9.15, 2.5, 1.2, 0.42, 'boxes,\nscores', fc='#E8F5EC', ec=GREEN, fs=7)

arrow(1.3, 2.6, 1.75, 2.6)
arrow(4.25, 3.0, 4.85, 3.5)
arrow(4.25, 2.3, 4.85, 1.55)
arrow(6.95, 1.55, 7.3, 1.55)
arrow(8.8, 1.55, 9.15, 1.55)
arrow(6.95, 3.55, 7.3, 3.55)
arrow(9.75, 2.35, 9.75, 2.92)   # post-proc up to boxes,scores
arrow(8.05, 2.35, 8.05, 3.05)   # cascade up to fused class (q_det)
fig.tight_layout()
fig.savefig(OUT / 'model_diagram.pdf')
fig.savefig(OUT / 'model_diagram.svg')
fig.savefig(OUT / 'model_diagram.png', dpi=160)
print('wrote model_diagram')
