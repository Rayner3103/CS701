"""Train DINO-YOLO (YOLO12 or YOLO26 + DINOv3 ViT features, sarbench/dino_yolo.py), or the plain YOLO (--dino none), on
the data of prepare_yolo_data.py with the ultralytics trainer, then score its best epoch on the local val and test
splits (eval_yolo.py).
Example: python train_yolo.py --epochs 50 --batch-size 16 --name dino_yolo12s
         python train_yolo.py --yolo yolo26s --epochs 24 --name dino_yolo26s_ep24
         python train_yolo.py --yolo yolo26s --dino vit_base_patch16_dinov3.lvd1689m --dino-adapt lora --epochs 24 \
             --name dinob_lora_yolo26s_ep24   (a frozen ViT-B/16 + LoRA: 86 M frozen, about 12 M trainable)"""
import argparse
import re
from pathlib import Path

import torch

from eval_yolo import DATA, score_run
from sarbench.dino_yolo import DinoYOLOTrainer, parameter_counts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-YOLO' / 'data.yaml',
                   help='data.yaml, or data_full.yaml to train on all of train (final submission)')
    p.add_argument('--yolo', default='yolo12s', help='an ultralytics model config: yolo12s, yolo26s, yolo26s-p2, ...')
    p.add_argument('--pretrained', default=None,
                   help='COCO weights that start the YOLO part; default <yolo>.pt without a -p2 / -p6 suffix')
    p.add_argument('--dino', default='vit_small_patch16_dinov3.lvd1689m', help="a timm DINOv3 model, or 'none'")
    p.add_argument('--fuse', type=int, nargs='+', default=None,
                   help='backbone layers that get the DINO features; default the ones the head reads '
                        '(YOLO12: 4 6 8 = P3/8, P4/16, P5/32; YOLO26: 4 6 10)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3, help='AdamW, cosine decay to lr x 0.01')
    p.add_argument('--dino-lr-mult', type=float, default=0.1,
                   help='the DINO ViT (with --dino-adapt lora: its LoRA weights) trains at lr x this')
    p.add_argument('--dino-adapt', choices=['full', 'frozen', 'lora'], default='full',
                   help='fine-tune the DINO ViT, freeze it, or freeze it and train LoRA adapters in its blocks')
    p.add_argument('--lora-rank', type=int, default=16, help='LoRA rank (alpha = 2 x rank), with --dino-adapt lora')
    p.add_argument('--lora-targets', nargs='+', default=['attn.qkv', 'attn.proj'],
                   help='the block layers that get LoRA; add mlp.fc1 mlp.fc2 to adapt the MLPs too')
    p.add_argument('--dino-grad-ckpt', action='store_true',
                   help='gradient checkpointing in the DINO ViT: less GPU memory, slower (LoRA on a ViT-B, 10 GB)')
    p.add_argument('--cls-head', action='store_true',
                   help='classify the image with a head on the DINO features, not from the detections')
    p.add_argument('--cls-weight', type=float, default=1.0, help='weight of the image-class loss (with --cls-head)')
    p.add_argument('--rot90', type=float, default=0.0,
                   help='probability of a 90 / 180 / 270 degree rotation in training; 0.75: every multiple of 90 '
                        'degrees equally likely; 0: flips only. Give it again on --resume')
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
    if args.dino.lower() == 'none':
        args.dino = None
        if args.cls_head:
            p.error('--cls-head needs the DINO features')
        if args.dino_adapt != 'full':
            p.error('--dino-adapt needs a DINO ViT')
    if args.pretrained is None:  # the P2 / P6 variants have no COCO weights of their own: the shared layers load
        args.pretrained = re.sub(r'-p[26]$', '', args.yolo) + '.pt'
    return args


def main():
    args = parse_args()
    run = args.out.resolve() / args.name
    if args.resume:
        overrides = {'resume': str(run / 'weights' / 'last.pt')}
    else:
        overrides = dict(
            model=f'{args.yolo}.yaml', pretrained=args.pretrained, data=str(args.data), epochs=args.epochs,
            batch=args.batch_size, imgsz=args.imgsz, optimizer='AdamW', lr0=args.lr, cos_lr=True,
            weight_decay=args.weight_decay, warmup_bias_lr=0.0, patience=args.patience, fraction=args.fraction,
            workers=args.workers, seed=args.seed, project=str(run.parent), name=run.name, exist_ok=True,
            # Augmentation: overhead images have no "up", so vertical flips too; gray images: no hue or saturation
            flipud=0.5, hsv_h=0.0, hsv_s=0.0)
    trainer = DinoYOLOTrainer(overrides=overrides, dino=args.dino, fuse=args.fuse, dino_lr_mult=args.dino_lr_mult,
                              cls_head=args.cls_head, cls_weight=args.cls_weight, rot90=args.rot90,
                              dino_adapt=args.dino_adapt, lora_rank=args.lora_rank, lora_targets=args.lora_targets,
                              grad_ckpt=args.dino_grad_ckpt)
    print(f'rot90 augmentation: p = {args.rot90}')
    trainer.train()
    total, trainable = parameter_counts(trainer.model)
    print(f'parameters: {total / 1e6:.2f} M total, {trainable / 1e6:.2f} M trainable')
    del trainer  # frees its GPU memory (model, EMA, optimizer) for scoring
    torch.cuda.empty_cache()
    if not args.no_eval:
        score_run(run, 'best.pt', DATA, imgsz=args.imgsz, batch=args.batch_size, trainable=trainable)


if __name__ == '__main__':
    main()
