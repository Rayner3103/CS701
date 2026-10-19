"""Score a run's saved raw outputs with the chosen post-processing (the one predict.py uses for the submission):
per-class Gaussian soft-NMS (sigma 0.75) on scores p_det(c|box) x RPN objectness^0.25, threshold 0.001, at most
100 boxes per image; and detection-informed classification
(argmax_c log p_img(c) + lam * log q(c), q(c) = the highest detector probability of class c, lam = 1).
Writes metrics_pp.json next to the run's metrics.json.
Example: python pp_eval.py --run ~/cs701_runs/E01_dino_ref"""
import argparse
import json
from pathlib import Path

import torch

from pp_study import DATA, cls_study, run_one

FINAL = dict(score_thresh=0.001, per_class=100, total=100, fusion='none', nms='soft', soft_sigma=0.75, obj_pow=0.25)
LAM = 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--splits', default='val,test')
    a = p.parse_args()
    torch.set_num_threads(4)
    out = {}
    for split in a.splits.split(','):
        raw = a.run / f'raw_{split}.pt'
        _, m, ap50, per_class, _ = run_one((raw, DATA / 'detection' / f'instances_{split}.json', 'final', FINAL))
        f1 = cls_study(raw, lambdas=(LAM,))[LAM]
        out[split] = {'macro_f1': f1, 'mAP': m, 'AP50': ap50, 'per_class_AP': per_class}
        print(f'RESULT pp {split}: macro_f1 {f1:.4f} mAP {m:.4f} AP50 {ap50:.4f}')
    (a.run / 'metrics_pp.json').write_text(json.dumps({'post': {**FINAL, 'cls_lambda': LAM}, **out}, indent=1))


if __name__ == '__main__':
    main()
