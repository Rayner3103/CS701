"""Predict with a trained run: either score a labelled local split (and optionally save raw outputs for the
post-processing study), or predict the released Codabench val/test images and write submission.zip.
Examples:
  python predict.py --run ~/cs701_runs/E01 --split val --dump-raw
  python predict.py --run ~/cs701_runs/FINAL --images ~/cs701_data/cs701-sar-course-data/val --out submission.zip
(the defaults apply the final post-processing: soft-NMS 0.75, objectness^0.25, threshold 0.001, 100 boxes per image,
detection-informed classification lambda 1)
Each image goes through the model once, at 512 x 512, with no test-time augmentation."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import train
from make_submission import write_submission
from pp_eval import FINAL, LAM
from sarbench.data import SARImages, SARMultiTask, collate
from sarbench.metrics import classification_metrics, detection_metrics


def load_run(run_dir, weights='model.pt'):
    """Rebuild the run's model from its metrics.json args and load its weights. Runs saved with --save trainable
    store only the trained tensors: the frozen ones are the pretrained weights, so the model is built pretrained."""
    saved = json.loads((run_dir / 'metrics.json').read_text())['args']
    required = ['--backbone', saved['backbone'], '--init', saved['init'], '--adapt', saved['adapt']]
    args = train.parse_args(required)  # defaults for options older runs did not have
    for k, v in saved.items():
        if hasattr(args, k) and k not in ('data', 'out'):
            setattr(args, k, v)
    model = train.build_model(args)
    state = torch.load(run_dir / weights, map_location='cpu')
    state = {k.replace('._orig_mod', ''): v for k, v in state.items()}  # runs trained with --compile
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not unexpected, unexpected
    assert not (set(missing) & trainable), f'trained weights missing from the checkpoint: {sorted(set(missing) & trainable)[:5]}'
    if missing and args.init != 'pretrained':
        raise SystemExit('checkpoint lacks frozen weights of a scratch model')
    return model.cuda().eval(), args


def image_class(logits, out, lam):
    """The image's class: argmax of the classifier, or with lam > 0 of log p_img(c) + lam * log q(c), where q(c) is the
    highest detector probability of class c over the image's proposals (pp_study.cls_study, same forward pass)."""
    if lam <= 0:
        return int(logits.argmax())
    logq = torch.log(out['raw_logits'].float().softmax(-1)[:, 1:].max(0).values + 1e-6)
    return int((logits.float().log_softmax(0) + lam * logq).argmax())


def set_post(model, a):
    model.roi_heads.post.update(score_thresh=a.score_thresh, nms_thresh=a.nms_thresh, per_class=a.per_class or None,
                                total=a.total, fusion=a.fusion, gamma=a.gamma, topk=a.topk, nms=a.nms,
                                soft_sigma=a.soft_sigma, obj_pow=a.obj_pow, shift=a.shift, quality_pow=a.quality_pow)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--weights', default='model.pt')
    p.add_argument('--data', type=Path, default=Path.home() / 'cs701_data' / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--split', help='a labelled local split to score (val, test)')
    p.add_argument('--dump-raw', action='store_true')
    p.add_argument('--images', type=Path, help='a released split folder holding images.json (Codabench val/test)')
    p.add_argument('--images-json', default='images.json', help='file under --images listing the images to predict')
    p.add_argument('--out', type=Path, default=Path('submission.zip'))
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    # post-processing (defaults: the final settings, = pp_eval.FINAL + cls lambda 1; the reference is
    # --score-thresh 0.05 --per-class 0 --nms hard --obj-pow 0 --cls-lambda 0)
    p.add_argument('--score-thresh', type=float, default=FINAL['score_thresh'])
    p.add_argument('--nms-thresh', type=float, default=0.5)
    p.add_argument('--per-class', type=int, default=FINAL['per_class'])
    p.add_argument('--total', type=int, default=FINAL['total'])
    p.add_argument('--fusion', default='none', choices=['none', 'mult', 'agn', 'hard'])
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--topk', type=int, default=9)
    p.add_argument('--nms', default=FINAL['nms'], choices=['hard', 'soft'])
    p.add_argument('--soft-sigma', type=float, default=FINAL['soft_sigma'])
    p.add_argument('--obj-pow', type=float, default=FINAL['obj_pow'], help='multiply box scores by RPN objectness^obj_pow')
    p.add_argument('--shift', type=float, default=None,
                   help='translate boxes by this many px on the 512 input (default: FINAL, or 0 for --flip-bias runs)')
    p.add_argument('--quality-pow', type=float, default=FINAL.get('quality_pow', 0.0), help='re-score boxes by the IoU branch^pow')
    p.add_argument('--cls-lambda', type=float, default=LAM,
                   help='detection-informed classification: argmax log p_img(c) + lambda * log max_box p_det(c) (0: off)')
    a = p.parse_args()

    model, args = load_run(a.run, a.weights)
    if a.shift is None:  # a model trained on bias-consistent flips already predicts the annotation convention
        a.shift = 0.0 if getattr(args, 'flip_bias', 0.0) else FINAL['shift']
    set_post(model, a)
    print(f'parameters: {sum(q.numel() for q in model.parameters()):,} total')
    if a.split:
        loader = DataLoader(SARMultiTask(a.data, a.split), a.batch_size, num_workers=a.workers, collate_fn=collate)
        t0 = time.time()
        m, dets, raw = train.evaluate(model, loader, a.data / 'detection' / f'instances_{a.split}.json', raw=a.dump_raw)
        print(f'{a.split}: {train.headline(m)} ({time.time() - t0:.0f} s)')
        print('per-class AP', {k: round(v, 3) for k, v in m['per_class_AP'].items()})
        if raw is not None:
            torch.save(raw, a.run / f'raw_{a.split}.pt')
        return

    images = json.loads((a.images / a.images_json).read_text())['images']
    loader = DataLoader(SARImages(a.images, images), a.batch_size, num_workers=a.workers,
                        collate_fn=lambda b: (torch.stack([x for x, _ in b]), [i for _, i in b]))
    preds, n_img, t0 = {}, 0, time.time()
    with torch.no_grad():
        for batch, infos in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, outputs, _ = model(batch.cuda(), return_raw=a.cls_lambda > 0)
            for info, lg, out in zip(infos, logits, outputs):
                name = images[n_img]['file_name']
                assert images[n_img]['id'] == info['image_id']
                label = image_class(lg, out, a.cls_lambda)
                preds[name] = (label, out['boxes'].float().tolist(),
                               (out['labels'] - 1).tolist(),  # detector labels 1..9; write_submission adds 1
                               out['scores'].float().tolist())
                n_img += 1
    seconds = time.time() - t0
    write_submission(images, preds, out=str(a.out))
    n_boxes = sum(len(v[1]) for v in preds.values())
    print(f'wrote {a.out}: {len(preds)} images, {n_boxes} boxes, {1000 * seconds / len(preds):.1f} ms per image')


if __name__ == '__main__':
    main()
