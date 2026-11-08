"""Collect every finished run into one table (CSV + Markdown) for the report: parameters, compute, raw and
post-processed local scores, and Δm against the local reference runs.
Usage: python tools/summarize_runs.py [--out DIR]"""
import argparse
import csv
import glob
import json
from pathlib import Path

RUNS = Path.home() / 'cs701_runs'
REF_RUNS = ['ref_s0/vit_pretrained_lora']  # the local Δm reference (the reference configuration, seed 0)


def load(run):
    d = RUNS / run
    m = json.loads((d / 'metrics.json').read_text()) if (d / 'metrics.json').exists() else None
    pp = json.loads((d / 'metrics_pp.json').read_text()) if (d / 'metrics_pp.json').exists() else None
    return m, pp


def dm(scores, ref):
    return 50 * ((scores['macro_f1'] - ref['macro_f1']) / ref['macro_f1'] + (scores['mAP'] - ref['mAP']) / ref['mAP'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=RUNS / 'tracker')
    a = p.parse_args()
    refs = [load(r)[0] for r in REF_RUNS]
    ref = {s: {k: sum(m[s][k] for m in refs) / len(refs) for k in ('macro_f1', 'mAP')} for s in ('val', 'test')}
    rows = []
    for mpath in sorted(glob.glob(str(RUNS / '**' / 'metrics.json'), recursive=True)):
        run = str(Path(mpath).parent.relative_to(RUNS))
        if '/code' in run or run.startswith(('bench', 'smoke')):
            continue
        m, pp = load(run)
        if not m or 'val' not in m or 'test' not in m:
            continue
        r = {'run': run, 'total_M': round(m['parameters'] / 1e6, 2), 'trainable_M': round(m['trainable_parameters'] / 1e6, 2),
             'train_h': round(m['train_hours'], 2), 'peak_GB': round(m['peak_gpu_memory_gb'], 1)}
        for s in ('val', 'test'):
            r[f'f1_{s}'], r[f'map_{s}'] = round(100 * m[s]['macro_f1'], 2), round(100 * m[s]['mAP'], 2)
        r['dm'] = round((dm(m['val'], ref['val']) + dm(m['test'], ref['test'])) / 2, 2)
        if pp:
            for s in ('val', 'test'):
                r[f'f1pp_{s}'], r[f'mappp_{s}'] = round(100 * pp[s]['macro_f1'], 2), round(100 * pp[s]['mAP'], 2)
            r['dm_pp'] = round((dm(pp['val'], ref['val']) + dm(pp['test'], ref['test'])) / 2, 2)
        r['args'] = ' '.join(f'--{k.replace("_", "-")} {v}' for k, v in m['args'].items()
                             if k not in ('data', 'out', 'name', 'workers') and v not in (None, False, '', 0, 0.0))
        rows.append(r)
    rows.sort(key=lambda r: -(r.get('dm_pp') or r['dm']))
    keys = ['run', 'total_M', 'trainable_M', 'train_h', 'peak_GB', 'f1_val', 'map_val', 'f1_test', 'map_test', 'dm',
            'f1pp_val', 'mappp_val', 'f1pp_test', 'mappp_test', 'dm_pp', 'args']
    with open(a.out / 'runs_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    md = ['| ' + ' | '.join(keys[:-1]) + ' |', '|' + '---|' * (len(keys) - 1)]
    md += ['| ' + ' | '.join(str(r.get(k, '')) for k in keys[:-1]) + ' |' for r in rows]
    (a.out / 'runs_summary.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'reference (local, mean of {len(refs)} run): val {ref["val"]}, test {ref["test"]}')


if __name__ == '__main__':
    main()
