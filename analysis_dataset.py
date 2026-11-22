"""Dataset statistics for the report: per-class image/box counts, box-size
distribution (the 'half the boxes < 16 px' claim), boxes per image, image sizes.
Run after prepare_course_data.py (needs ../cs701-sar-course-data/train)."""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent / 'cs701-sar-course-data' / 'train'
OUT = Path(__file__).resolve().parent / 'report' / 'figures'


def main():
    rows = list(csv.DictReader(open(SRC / 'labels.csv')))
    coco = json.loads((SRC / 'instances.json').read_text())
    cats = {c['id']: c['name'] for c in coco['categories']}
    images = {im['id']: im for im in coco['images']}
    anns = defaultdict(list)
    for a in coco['annotations']:
        anns[a['image_id']].append(a)

    img_class = Counter(int(r['label']) for r in rows)
    box_class = Counter()
    sizes, sides, per_img = [], [], []
    img_wh = Counter((im['width'], im['height']) for im in coco['images'])
    for iid, al in anns.items():
        per_img.append(len(al))
        for a in al:
            box_class[a['category_id']] += 1
            w, h = a['bbox'][2], a['bbox'][3]
            sides.append(np.sqrt(max(w * h, 0)))
            sizes.append((w, h))

    print(f'images: {len(rows)}, boxes: {len(coco["annotations"])}')
    print('image sizes:', dict(img_wh))
    print(f'{"class":14s} {"images":>7s} {"boxes":>7s}')
    for cid in sorted(cats):
        print(f'{cats[cid]:14s} {img_class[cid - 1]:7d} {box_class[cid]:7d}')

    sides = np.array(sides)
    per_img = np.array(per_img)
    print(f'\nbox sqrt-area px: min {sides.min():.1f} p10 {np.percentile(sides,10):.1f} '
          f'p25 {np.percentile(sides,25):.1f} median {np.median(sides):.1f} '
          f'p75 {np.percentile(sides,75):.1f} p90 {np.percentile(sides,90):.1f} max {sides.max():.1f}')
    for t in (8, 16, 32, 64):
        print(f'  boxes < {t}px: {100 * (sides < t).mean():.1f}%')
    print(f'boxes/image: min {per_img.min()} median {np.median(per_img):.0f} '
          f'mean {per_img.mean():.1f} max {per_img.max()}; single-box {100 * (per_img == 1).mean():.0f}%')

    # --- figure: box-size histogram (SVG for the report) ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        OUT.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
        axes[0].hist(sides, bins=np.arange(0, 300, 5), color='#0B5CAB', edgecolor='none')
        for t, c in ((16, 'r'),):
            axes[0].axvline(t, color=c, lw=1, ls='--')
            axes[0].text(t + 4, axes[0].get_ylim()[1] * 0.85, '16 px', color=c, fontsize=8)
        axes[0].set_xlabel(r'box side $\sqrt{w\cdot h}$ (px)')
        axes[0].set_ylabel('boxes')
        axes[0].set_title('(a) Box-size distribution')
        axes[1].hist(np.clip(per_img, 0, 40), bins=np.arange(0, 42) - .5, color='#1B7A3D', edgecolor='none')
        axes[1].set_xlabel('boxes per image')
        axes[1].set_ylabel('images')
        axes[1].set_title('(b) Objects per image')
        for ax in axes:
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
        fig.tight_layout()
        fig.savefig(OUT / 'box_stats.pdf')
        fig.savefig(OUT / 'box_stats.svg')
        print('\nwrote', OUT / 'box_stats.pdf / .svg')
    except Exception as e:
        print('figure failed:', e)


if __name__ == '__main__':
    main()
