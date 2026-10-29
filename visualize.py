"""Browse a random sample of a labelled split with ground-truth and predicted boxes, side by side.
Example: python visualize.py --run runs/single_seed0_ep100/terramind_pretrained_moelora --class bridge playground

--run can also be a folder of runs (e.g. runs/full_seed0): you then pick one from a list.
Keys: left/right previous/next image, up/down raise/lower the score threshold, l labels on/off, v overlay (GT and
predictions in one panel) on/off, e show only the sampled images with an error on/off, s save the figure as PNG,
q quit. Zoom with the toolbar's magnifier: both panels zoom together.
Box colours: one per class. Solid = matched (IoU >= --iou, same class), dashed = missed GT / false-positive prediction."""
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou

from sarbench.adapters import adapter_kwargs, add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import SARMultiTask, collate
from sarbench.model import MultiTaskModel

COLOURS = plt.get_cmap('tab10').colors
KEYS = {'left', 'right', 'up', 'down', 'l', 'v', 'e', 's'}  # taken from matplotlib's default key bindings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True, help='a run folder of train.py, or a folder of run folders')
    p.add_argument('--weights', default='best_model.pt', help='best_model.pt (default) or model.pt (last epoch)')
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='a local split with labels')
    p.add_argument('--class', dest='classes', nargs='+', metavar='NAME',
                   help='sample only images of these ground-truth classes (e.g. bridge playground)')
    p.add_argument('--n', type=int, default=50, help='sample size')
    p.add_argument('--seed', type=int, default=0, help='which random sample')
    p.add_argument('--score-thresh', type=float, default=0.3, help='initial threshold for shown predictions')
    p.add_argument('--iou', type=float, default=0.5, help='IoU for a prediction to match a ground-truth box')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--save', type=Path, help='no window: write one PNG per sampled image to this folder and exit')
    return p.parse_args()


def pick_run(path):
    """`path` itself if it is a run, else the run the user picks from the runs inside it."""
    if (path / 'metrics.json').exists():
        return path
    runs = sorted(d for d in path.iterdir() if (d / 'metrics.json').exists())
    if not runs:
        raise SystemExit(f'no run folder (with metrics.json) in {path}')
    for i, run in enumerate(runs):
        m = json.loads((run / 'metrics.json').read_text())
        best = m.get('best', {}).get('val', m['val'])
        print(f'[{i}] {run.name:40s} val macro-F1 {best["macro_f1"]:.4f} mAP {best["mAP"]:.4f}')
    return runs[int(input('run number: '))]


def load_model(run, weights, device):
    """The model as the run built it, as in predict.py; pretrained=False: every weight comes from the checkpoint."""
    args = json.loads((run / 'metrics.json').read_text())['args']
    if not (run / weights).exists():
        raise SystemExit(f'{run / weights} does not exist (older runs have only model.pt: --weights model.pt)')
    backbone = build_backbone(args['backbone'], pretrained=False)
    if args['adapt'] != 'full':
        add_adapters(backbone, args['adapt'], **adapter_kwargs(args))
    model = MultiTaskModel(backbone, task_routing=args['adapt'] == 'moelora',
                           spatial_prior=args.get('spatial_prior', False),
                           cascade=args.get('cascade', False))
    model.load_state_dict(torch.load(run / weights, map_location='cpu'))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, dataset, device, batch_size):
    """Per image: class probabilities and detections (boxes XYXY on the 512 px input, labels 1..9, scores)."""
    results = []
    for images, _, _ in DataLoader(dataset, batch_size, collate_fn=collate, num_workers=4):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits, outputs, _ = model(images.to(device))
        for lg, out in zip(logits.float().softmax(-1).cpu(), outputs):
            results.append({'probs': lg, **{k: v.float().cpu() if k != 'labels' else v.cpu() for k, v in out.items()}})
        print(f'\rpredicted {len(results)}/{len(dataset)}', end='', flush=True)
    print()
    return results


def match(pred_boxes, pred_labels, gt_boxes, gt_labels, iou):
    """Greedy matching in descending score order (as COCO): a prediction matches the unmatched GT box of its class
    with the highest IoU >= iou. -> (bool per prediction, bool per GT box)."""
    pred_hit = torch.zeros(len(pred_boxes), dtype=torch.bool)
    gt_hit = torch.zeros(len(gt_boxes), dtype=torch.bool)
    if len(pred_boxes) and len(gt_boxes):
        ious = box_iou(pred_boxes, gt_boxes)
        ious[pred_labels[:, None] != gt_labels[None, :]] = 0
        for i in range(len(pred_boxes)):  # torchvision returns detections sorted by score
            ious[i, gt_hit] = 0
            best = ious[i].argmax()
            if ious[i, best] >= iou:
                pred_hit[i], gt_hit[best] = True, True
    return pred_hit, gt_hit


class Viewer:
    def __init__(self, dataset, results, names, args, title):
        self.dataset, self.results, self.names, self.args, self.title = dataset, results, names, args, title
        self.thresh, self.labels, self.overlay, self.errors_only = args.score_thresh, True, False, False
        self.index = 0
        for key, bound in plt.rcParams.items():  # free our keys from matplotlib's defaults (e.g. s = save dialog)
            if key.startswith('keymap.'):
                plt.rcParams[key] = [k for k in bound if k not in KEYS]
        self.fig = plt.figure(figsize=(16, 8.5))
        self.axes = []
        self.layout()

    def layout(self):
        """Two panels sharing their zoom, or one panel in overlay mode."""
        self.fig.clear()
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.07, wspace=0.03)
        if self.overlay:
            self.axes = [self.fig.add_subplot(1, 1, 1)]
        else:
            left = self.fig.add_subplot(1, 2, 1)
            self.axes = [left, self.fig.add_subplot(1, 2, 2, sharex=left, sharey=left)]
        self.fig.text(0.01, 0.005, '←/→ image   ↑/↓ score threshold   l labels   v overlay   e errors only   '
                      's save   q quit   (zoom: toolbar magnifier)', fontsize=8, color='gray')

    def shown(self):
        """Indices (into the sample) of the images currently browsable."""
        if not self.errors_only:
            return list(range(len(self.dataset)))
        return [i for i in range(len(self.dataset)) if self.has_error(i)]

    def item(self, i):
        """Everything one image needs: the original image, its GT and its predictions above the threshold, with boxes
        in original pixels."""
        _, label, target = self.dataset[i]
        row = self.dataset.dataset.rows[self.dataset.indices[i]]
        image = Image.open(self.dataset.dataset.root / row['image']).convert('L')
        h, w = target['orig_size']
        scale = torch.tensor([w, h, w, h]) / 512  # the input is always 512 x 512 (non-square images stretched)
        r = self.results[i]
        keep = r['scores'] >= self.thresh
        pred_boxes, pred_labels, scores = r['boxes'][keep] * scale, r['labels'][keep], r['scores'][keep]
        gt_boxes, gt_labels = target['boxes'] * scale, target['labels']
        pred_hit, gt_hit = match(pred_boxes, pred_labels, gt_boxes, gt_labels, self.args.iou)
        return dict(image=image, file=row['image'], label=label, probs=r['probs'], gt_boxes=gt_boxes,
                    gt_labels=gt_labels, gt_hit=gt_hit, pred_boxes=pred_boxes, pred_labels=pred_labels,
                    scores=scores, pred_hit=pred_hit)

    def has_error(self, i):
        it = self.item(i)
        return it['probs'].argmax() != it['label'] or not it['gt_hit'].all() or not it['pred_hit'].all()

    def boxes(self, ax, boxes, labels, hits, scores=None, gt=True):
        for k, (box, c, hit) in enumerate(zip(boxes.tolist(), labels.tolist(), hits.tolist())):
            x1, y1, x2, y2 = box
            colour = COLOURS[(c - 1) % len(COLOURS)]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour,
                                   linewidth=1.0 if gt else 1.4, linestyle='-' if hit else '--'))
            if self.labels:
                text = self.names[c] if gt else f'{self.names[c]} {scores[k]:.2f}'
                ax.text(x1, y1 if gt else y2, text, color='white', fontsize=7, va='bottom' if gt else 'top',
                        bbox=dict(facecolor=colour, alpha=0.6, pad=1, linewidth=0))

    def draw(self, keep_view=False):
        shown = self.shown()
        if not shown:
            for ax in self.axes:
                ax.clear()
            self.fig.legends.clear()
            self.fig.suptitle('no image in the sample has an error at this threshold (e: show all)')
            self.fig.canvas.draw_idle()
            return
        self.index %= len(shown)
        view = (self.axes[0].get_xlim(), self.axes[0].get_ylim()) if keep_view else None
        it = self.item(shown[self.index])
        for ax in self.axes:
            ax.clear()
            ax.imshow(it['image'], cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
        gt_ax, pred_ax = self.axes[0], self.axes[-1]
        self.boxes(gt_ax, it['gt_boxes'], it['gt_labels'], it['gt_hit'])
        self.boxes(pred_ax, it['pred_boxes'], it['pred_labels'], it['pred_hit'], it['scores'], gt=False)
        n_gt, n_pred = len(it['gt_boxes']), len(it['pred_boxes'])
        tp = int(it['pred_hit'].sum())
        if self.overlay:
            gt_ax.set_title(f'GT {n_gt} (thin) + predictions {n_pred} (thick): {tp} matched, '
                            f'{n_gt - tp} missed, {n_pred - tp} false positives')
        else:
            gt_ax.set_title(f'ground truth: {n_gt} boxes, {n_gt - int(it["gt_hit"].sum())} missed (dashed)')
            pred_ax.set_title(f'predictions >= {self.thresh:.2f}: {n_pred} boxes, {n_pred - tp} false positives '
                              f'(dashed)')
        if view:
            gt_ax.set_xlim(view[0]), gt_ax.set_ylim(view[1])

        pred_cls = int(it['probs'].argmax())
        correct = pred_cls == it['label']
        top3 = ', '.join(f'{self.names[c + 1]} {p:.2f}' for p, c in zip(*(t.tolist() for t in it['probs'].topk(3))))
        self.fig.suptitle(
            f'{self.title}   [{self.index + 1}/{len(shown)}{" errors" if self.errors_only else ""}]   {it["file"]}  '
            f'{it["image"].width}x{it["image"].height}\nclass: true {self.names[it["label"] + 1]}, predicted '
            f'{self.names[pred_cls + 1]} ({"correct" if correct else "WRONG"})   top 3: {top3}',
            color='black' if correct else 'red', fontsize=11)
        present = sorted(set(it['gt_labels'].tolist()) | set(it['pred_labels'].tolist()))
        handles = [Line2D([], [], color=COLOURS[(c - 1) % len(COLOURS)], label=self.names[c]) for c in present]
        handles += [Line2D([], [], color='gray', label='matched'),
                    Line2D([], [], color='gray', linestyle='--', label='missed / false positive')]
        self.fig.legends.clear()
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(handles), fontsize=9,
                        frameon=False)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ('right', 'left'):
            self.index += 1 if event.key == 'right' else -1
            self.draw()
        elif event.key in ('up', 'down'):
            self.thresh = min(max(self.thresh + (0.05 if event.key == 'up' else -0.05), 0.05), 0.95)
            self.draw(keep_view=not self.errors_only)  # with errors only, the image list itself can change
        elif event.key == 'l':
            self.labels = not self.labels
            self.draw(keep_view=True)
        elif event.key == 'v':
            self.overlay = not self.overlay
            self.layout()
            self.draw()
        elif event.key == 'e':
            self.errors_only, self.index = not self.errors_only, 0
            self.draw()
        elif event.key == 's':
            self.save(Path('.'))

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        name = Path(self.item(self.shown()[self.index])['file']).stem
        path = folder / f'{self.title.replace("/", "_")}_{name}.png'
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'saved {path}')

    def show(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.draw()
        plt.show()


def main():
    args = parse_args()
    run = pick_run(args.run)
    gt = json.loads((args.data / 'detection' / f'instances_{args.split}.json').read_text())
    names = {c['id']: c['name'] for c in gt['categories']}  # category_id 1..9 -> name; image label = id - 1

    dataset = SARMultiTask(args.data, args.split)  # train=False: no flips
    candidates = range(len(dataset))
    if args.classes:
        unknown = set(args.classes) - set(names.values())
        if unknown:
            raise SystemExit(f'unknown class {sorted(unknown)}; choose from {sorted(names.values())}')
        wanted = {cid - 1 for cid, name in names.items() if name in args.classes}
        candidates = [i for i, row in enumerate(dataset.rows) if int(row['label']) in wanted]
    sample = sorted(random.Random(args.seed).sample(list(candidates), min(args.n, len(candidates))))
    subset = Subset(dataset, sample)
    print(f'{run}: {len(subset)} {args.split} images'
          f'{" of class " + ", ".join(args.classes) if args.classes else ""} (seed {args.seed}), {args.weights}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = predict(load_model(run, args.weights, device), subset, device, args.batch_size)

    viewer = Viewer(subset, results, names, args, title=f'{run.parent.name}/{run.name}')
    if args.save:
        for viewer.index in range(len(subset)):
            viewer.draw()
            viewer.save(args.save)
    else:
        viewer.show()


if __name__ == '__main__':
    main()
