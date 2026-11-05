"""Evaluate final post-processing with/without image-class fusion, from saved raw predictions."""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pp_eval import FINAL
from pp_study import run_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads((args.run / 'metrics.json').read_text())['args']
    data = Path(saved['data']) / 'detection'
    base = {**FINAL, 'shift': 0.0} if saved.get('flip_bias') else dict(FINAL)
    configs = {'final': base, 'fusion_mult': {**base, 'fusion': 'mult', 'gamma': 1.0}}
    jobs = [(args.run / f'raw_{split}.pt', data / f'instances_{split}.json', f'{name}|{split}', config)
            for name, config in configs.items() for split in ('val', 'test')]
    with Pool(2) as pool:
        outputs = pool.map(run_one, jobs)
    indexed = {name: (m, per_class) for name, m, _, per_class, _ in outputs}
    result = {}
    for name in configs:
        val, test = indexed[f'{name}|val'], indexed[f'{name}|test']
        result[name] = {'mAP_val': 100 * val[0], 'mAP_test': 100 * test[0],
                        'per_class': {c: 50 * (val[1][c] + test[1][c]) for c in val[1]}}
        classes = result[name]['per_class']
        print(f'PPCLS {args.run.name} {name}: mAP val {100*val[0]:.2f} test {100*test[0]:.2f} | '
              f'bridge {classes["bridge"]:.1f} playground {classes["playground"]:.1f}', flush=True)
    ordinary = json.loads((args.run / 'metrics_pp.json').read_text())
    for split in ('val', 'test'):
        if abs(result['final'][f'mAP_{split}'] / 100 - ordinary[split]['mAP']) > 1e-7:
            raise RuntimeError(f'{split}: final post-processing disagrees with metrics_pp.json')
    (args.run / 'ppcls.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
