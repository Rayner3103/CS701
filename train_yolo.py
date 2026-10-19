"""Train DINO-YOLO (YOLO12 + DINOv3 ViT features, sarbench/dino_yolo.py) on the data of prepare_yolo_data.py with the
ultralytics trainer, then score its best epoch on the local val and test splits (eval_yolo.py).
Example: python train_yolo.py --epochs 50 --batch-size 16 --name dino_yolo12s"""
import argparse
from pathlib import Path

import torch

from eval_yolo import DATA, score_run
from sarbench.dino_yolo import DinoYOLOTrainer, parameter_counts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-YOLO' / 'data.yaml',
                   help='data.yaml, or data_full.yaml to train on all of train (final submission)')
    p.add_argument('--yolo', default='yolo12s', help='YOLO12 scale; its COCO weights <yolo>.pt start the YOLO part')
    p.add_argument('--dino', default='vit_small_patch16_dinov3.lvd1689m', help='a timm DINOv3 model')
    p.add_argument('--fuse', type=int, nargs='+', default=[4, 6, 8],
                   help='YOLO12 backbone layers that get the DINO features (4 = P3/8, 6 = P4/16, 8 = P5/32)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3, help='AdamW, cosine decay to lr x 0.01')
    p.add_argument('--dino-lr-mult', type=float, default=0.1, help='the DINO ViT trains at lr x this')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--imgsz', type=int, default=512, help='at most 512 (assignment limit)')
    p.add_argument('--patience', type=int, default=20, help='stop after N epochs without a better val fitness')
    p.add_argument('--fraction', type=float, default=1.0, help='use this fraction of the train images (quick checks)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('runs/yolo'))
    p.add_argument('--name', default='dino_yolo12s')
    p.add_argument('--resume', action='store_true', help="continue from the run folder's weights/last.pt")
    p.add_argument('--no-eval', action='store_true', help='skip scoring on the local val and test splits')
    args = p.parse_args()
    if args.imgsz > 512:
        p.error('--imgsz: images go in at most 512 x 512')
    return args


def main():
    args = parse_args()
    run = args.out.resolve() / args.name
    if args.resume:
        overrides = {'resume': str(run / 'weights' / 'last.pt')}
    else:
        overrides = dict(
            model=f'{args.yolo}.yaml', pretrained=f'{args.yolo}.pt', data=str(args.data), epochs=args.epochs,
            batch=args.batch_size, imgsz=args.imgsz, optimizer='AdamW', lr0=args.lr, cos_lr=True,
            weight_decay=args.weight_decay, warmup_bias_lr=0.0, patience=args.patience, fraction=args.fraction,
            workers=args.workers, seed=args.seed, project=str(run.parent), name=run.name, exist_ok=True,
            # Augmentation: overhead images have no "up", so vertical flips too; gray images: no hue or saturation
            flipud=0.5, hsv_h=0.0, hsv_s=0.0)
    trainer = DinoYOLOTrainer(overrides=overrides, dino=args.dino, fuse=args.fuse, dino_lr_mult=args.dino_lr_mult)
    trainer.train()
    total, trainable = parameter_counts(trainer.model)
    print(f'parameters: {total / 1e6:.2f} M total, {trainable / 1e6:.2f} M trainable')
    del trainer  # frees its GPU memory (model, EMA, optimizer) for scoring
    torch.cuda.empty_cache()
    if not args.no_eval:
        score_run(run, 'best.pt', DATA, imgsz=args.imgsz, batch=args.batch_size, trainable=trainable)


if __name__ == '__main__':
    main()
