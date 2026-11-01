"""Compare the ViT-S augmentation arms of run_aug_experiments.sh on the local val split and write the winner.
The winner is the arm with the highest val Δm (never test: it is the number to report, not to choose by). The baseline is
the mean of the two existing seeds of the unaugmented run. If no arm beats it, the winner is NONE and the script exits 1.
Example: python aug_pick_winner.py os2paste os2 os3 os2pasteall"""
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent / 'runs'
BASELINES = ['dino_yolo26s_ep24', 'dino_yolo26s_ep24_seed1']


def metrics(name):
    f = RUNS / 'yolo' / name / 'metrics.json'
    return json.loads(f.read_text()) if f.exists() else None


def main():
    arms = sys.argv[1:]
    base = [metrics(b) for b in BASELINES]
    val = sum(m['val']['delta_m'] for m in base) / len(base)
    test = sum(m['test']['delta_m'] for m in base) / len(base)
    print(f"{'arm':<14}{'val Δm':>8}{'test Δm':>9}{'val mAP':>9}{'test mAP':>10}")
    print(f"{'baseline':<14}{val:>8.2f}{test:>9.2f}  (mean of {len(base)} seeds, 7,512 images)")
    scores = {}
    for arm in arms:
        m = metrics(f'aug_{arm}_dino_yolo26s_ep24')
        if m is None:
            print(f'{arm:<14}  no metrics.json (run failed or missing)')
            continue
        scores[arm] = m['val']['delta_m']
        print(f"{arm:<14}{m['val']['delta_m']:>8.2f}{m['test']['delta_m']:>9.2f}{m['val']['mAP']:>9.3f}{m['test']['mAP']:>10.3f}")
    best = max(scores, key=scores.get) if scores else None
    out = RUNS / 'aug' / 'winner.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    if best is None or scores[best] <= val:
        out.write_text('NONE\n')
        print(f'winner: NONE (best augmented arm does not beat the baseline val Δm {val:.2f})')
        sys.exit(1)
    gain = scores[best] - val
    out.write_text(f'{best}\n')
    print(f"winner: {best} (val Δm {scores[best]:.2f}, +{gain:.2f} over the baseline"
          f"{'; inside the ~1.5 seed noise, treat as a tie' if gain < 1.5 else ''})")


if __name__ == '__main__':
    main()
