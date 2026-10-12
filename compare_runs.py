"""Print one line per run with its test macro-F1, mAP and Δm (last epoch and best epoch), to compare runs side by side.
Example: python compare_runs.py runs"""
import glob
import json
import sys

from train import delta_m


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'runs'
    print(f"{'run':34} {'F1':>7} {'mAP':>7} {'Δm':>8}   {'best ep':>7} {'best Δm':>8}")
    for path in sorted(glob.glob(f'{out}/*/metrics.json')):
        metrics = json.load(open(path))
        test, best = metrics['test'], metrics.get('best')  # runs from before best-model saving have no 'best'
        best_cols = f"{best['epoch']:>7} {delta_m(best['test']):+8.2f}" if best else f"{'-':>7} {'-':>8}"
        print(f"{path.split('/')[-2]:34} {test['macro_f1']:7.4f} {test['mAP']:7.4f} {delta_m(test):+8.2f}   {best_cols}")


if __name__ == '__main__':
    main()
