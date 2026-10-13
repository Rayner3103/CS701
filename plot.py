"""Plot each run's metrics across training epochs, from its log.txt (so it also works on a run still in progress).
Writes two figures to OUT (default FOLDER/plots), one line per run:
  - val_metrics.png: the validation metrics (macro-F1, accuracy, balanced accuracy, mAP, AP50, Δm) at every evaluation,
    with the best epoch (highest val Δm, the one best_model.pt keeps) ringed;
  - train_loss.png: the training loss and each of its parts, averaged per epoch over the iterations in the log.
Runs of the same configuration share a color; a spatial-prior run (name ending in _sp) is dashed.
Examples:
  python plot.py runs/full_seed0                          # every run in the folder
  python plot.py runs/full_seed0/vit_pretrained_lora      # one run
  python plot.py runs/full_seed0 --runs 'vit_*' --show    # only the ViT runs, also opened in a window"""
import argparse
import fnmatch
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

ITER = re.compile(r'^epoch (\d+) iter (\d+)/\d+ loss ([\d.]+) \((.*)\) lr ')
VAL = re.compile(r'^epoch (\d+) val: (.*?) delta_m ([+-]?[\d.]+)')
VAL_METRICS = [('macro_f1', 'Macro-F1'), ('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced accuracy'),
               ('mAP', 'mAP'), ('AP50', 'AP50'), ('delta_m', 'Δm (%)')]
# Categorical palette, in fixed order: one color per configuration
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']


def parse_log(path):
    """{'train': {epoch: {loss part: mean}}, 'val': {epoch: {metric: value}}} from a run's log.txt.
    A resumed run logs the interrupted epoch again; its later lines replace the earlier ones."""
    iters, val = {}, {}
    for line in path.read_text().splitlines():
        if m := ITER.match(line):
            epoch, step, loss, parts = int(m[1]), int(m[2]), float(m[3]), m[4].split()
            iters[epoch, step] = {'loss': loss, **{k: float(v) for k, v in zip(parts[::2], parts[1::2])}}
        elif m := VAL.match(line):
            values = m[2].split()
            val[int(m[1])] = {**{k: float(v) for k, v in zip(values[::2], values[1::2])}, 'delta_m': float(m[3])}
    by_epoch = defaultdict(list)
    for (epoch, _), losses in iters.items():
        by_epoch[epoch].append(losses)
    train = {epoch: {k: sum(l[k] for l in rows) / len(rows) for k in rows[0]} for epoch, rows in by_epoch.items()}
    return {'train': dict(sorted(train.items())), 'val': dict(sorted(val.items()))}


def find_runs(folder, patterns):
    """{run name: log.txt} for FOLDER itself if it is a run, else for each run folder inside it."""
    logs = [folder / 'log.txt'] if (folder / 'log.txt').exists() else sorted(folder.glob('*/log.txt'))
    return {log.parent.name: log for log in logs
            if not patterns or any(fnmatch.fnmatch(log.parent.name, p) for p in patterns)}


def style(names):
    """{run name: line style}: one color per configuration (the name without _sp), dashed with the spatial prior."""
    configs = list(dict.fromkeys(name.removesuffix('_sp') for name in names))
    if len(configs) > len(COLORS):
        print(f'warning: {len(configs)} configurations but {len(COLORS)} colors; some share one (use --runs)')
    return {name: {'color': COLORS[configs.index(name.removesuffix('_sp')) % len(COLORS)],
                   'linestyle': '--' if name.endswith('_sp') else '-'} for name in names}


def plot_grid(runs, styles, panels, key, title, path, ring_best=False):
    """A 2x3 grid, one panel per (metric, label) of PANELS, one line per run from runs[name][KEY]."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, panels):
        for name, run in runs.items():
            points = {epoch: values[metric] for epoch, values in run[key].items() if metric in values}
            if not points:
                continue
            ax.plot(list(points), list(points.values()), label=name, linewidth=2, marker='o', markersize=4,
                    **styles[name])
            if ring_best:
                best = max(run['val'], key=lambda epoch: run['val'][epoch]['delta_m'])
                if best in points:
                    ax.plot(best, points[best], marker='o', markersize=11, markerfacecolor='none',
                            markeredgewidth=1.5, color=styles[name]['color'])
        ax.set_title(label, fontsize=11, loc='left')
        ax.set_xlabel('epoch', color='#52514e')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(color='#e6e5e0', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#b5b4ae')
        ax.tick_params(colors='#52514e')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=min(len(labels), 4), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13, x=0.01, ha='left')
    fig.savefig(path, dpi=150)
    print(f'saved {path}')
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', type=Path, nargs='?', default=Path('runs'), help='a run folder, or a folder of runs')
    p.add_argument('--runs', nargs='+', help="only the runs whose name matches one of these patterns, e.g. 'vit_*'")
    p.add_argument('--out', type=Path, help='where to save the figures (default: FOLDER/plots)')
    p.add_argument('--show', action='store_true', help='also open the figures in a window')
    args = p.parse_args()
    if not args.show:
        matplotlib.use('Agg')  # no display needed
    import matplotlib.pyplot as plt

    logs = find_runs(args.folder, args.runs)
    if not logs:
        p.error(f'no log.txt in {args.folder} or its subfolders' + (f' matching {args.runs}' if args.runs else ''))
    runs = {name: parse_log(log) for name, log in logs.items()}
    styles = style(runs)
    out = args.out or args.folder / 'plots'
    out.mkdir(parents=True, exist_ok=True)

    if any(run['val'] for run in runs.values()):
        plot_grid(runs, styles, VAL_METRICS, 'val', f'Validation metrics: {args.folder}  (ringed: best epoch)',
                  out / 'val_metrics.png', ring_best=True)
    else:
        print('no validation results logged yet: skipping val_metrics.png')
    parts = list(dict.fromkeys(k for run in runs.values() for losses in run['train'].values() for k in losses))
    plot_grid(runs, styles, [(k, k.removeprefix('loss_').replace('_', ' ')) for k in parts[:6]], 'train',
              f'Training loss, mean per epoch: {args.folder}', out / 'train_loss.png')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
