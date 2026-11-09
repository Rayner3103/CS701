"""Score a run's saved raw outputs with the chosen post-processing (the one predict.py uses for the submission):
per-class Gaussian soft-NMS (sigma 0.75) on scores p_det(c|box) x RPN objectness^0.25, threshold 1e-5 (and at least
30 boxes per image: every image has objects, and low-scored boxes at the end of the ranking can only add recall),
at most 100 boxes per image, Cascade: the mean of the three stages' boxes, boxes translated by -0.4 px on the 512 input (a systematic position bias measured on
the local holdouts); and detection-informed classification
(argmax_c log p_img(c) + lam * log q(c), q(c) = the highest detector probability of class c, lam = 1).
Writes metrics_pp.json next to the run's metrics.json.
Example: python pp_eval.py --run ~/cs701_runs/E01_dino_ref"""
import argparse
import json
from pathlib import Path

import torch

from pp_study import DATA, cls_study, run_one

FINAL = dict(score_thresh=1e-5, min_keep=30, per_class=100, total=100, fusion='none', nms='soft', soft_sigma=0.75, obj_pow=0.25,
             shift=-0.4,  # -0.4 px: measured box-position bias (3 models, both holdouts, +0.4..0.7 mAP)
             stage_box='mean')  # Cascade: mean of the three stages' boxes (C48: +0.53 / +0.54 mAP on val / test)
LAM = 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--splits', default='val,test')
    a = p.parse_args()
    torch.set_num_threads(4)
    out = {}
    saved = json.loads((a.run / 'metrics.json').read_text()).get('args', {}) if (a.run / 'metrics.json').exists() else {}
    data = Path(saved.get('data') or DATA)  # fold runs keep their ground truth under their own --data root
    post = {**FINAL, 'shift': 0.0} if saved.get('flip_bias') else FINAL  # trained on bias-consistent flips
    for split in a.splits.split(','):
        raw = a.run / f'raw_{split}.pt'
        _, m, ap50, per_class, _ = run_one((raw, data / 'detection' / f'instances_{split}.json', 'final', post))
        f1 = cls_study(raw, lambdas=(LAM,))[LAM]
        out[split] = {'macro_f1': f1, 'mAP': m, 'AP50': ap50, 'per_class_AP': per_class}
        print(f'RESULT pp {split}: macro_f1 {f1:.4f} mAP {m:.4f} AP50 {ap50:.4f}')
    (a.run / 'metrics_pp.json').write_text(json.dumps({'post': {**post, 'cls_lambda': LAM}, **out}, indent=1))


if __name__ == '__main__':
    main()
