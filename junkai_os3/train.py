"""Train one model jointly for classification and detection, then score it on the local holdouts.
Example (the reference): python train.py --backbone vit --init pretrained --adapt lora
Every option beyond the reference is opt-in; the defaults reproduce the reference run."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from sarbench.adapters import add_adapters, add_lora, add_mona, unfreeze
from sarbench.backbones import BACKBONES, build_backbone, set_token_stride
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.training_state import (compatibility_signature, load_training_state, save_training_state,
                                     seed_everything, seed_loader_epoch, seed_worker)
from sarbench.model import REFERENCE_ANCHORS, MultiTaskModel
from sarbench.tiny import (AlignedMultiScaleRoIAlign, CascadeRoIHeads, OffsetAnchorGenerator, PretrainedPrior,
                           SameClassCopyPaste, iou_nwd_similarity)

MAX_TOTAL, MAX_TRAINABLE = 130_000_000, 40_000_000  # Codabench Terms


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--backbone', required=True, choices=BACKBONES)
    p.add_argument('--init', required=True, choices=['pretrained', 'scratch'])
    p.add_argument('--adapt', required=True, choices=['full', 'lora', 'moelora'])
    # adaptation capacity
    p.add_argument('--lora-targets', default='attn.qkv,attn.proj', help='layers inside each block to wrap with LoRA')
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--lora-alpha', type=float, default=32)
    p.add_argument('--unfreeze-last', type=int, default=0, help='fully train the last N blocks (no LoRA on them)')
    p.add_argument('--train-norms', action='store_true', help='train the backbone LayerNorm weights and biases')
    p.add_argument('--train-bias', action='store_true', help='train every bias of the backbone')
    p.add_argument('--train-patch-embed', action='store_true')
    p.add_argument('--train-ls', action='store_true', help='train layer-scale gammas')
    p.add_argument('--train-pos-embed', action='store_true', help='train the position embedding (and CLS/register tokens)')
    p.add_argument('--cnx-full-stages', default='', help="ConvNeXt: fully train the stem and these stages (e.g. '0,1,3'), LoRA on the others")
    p.add_argument('--drop-path', type=float, default=0.0, help='stochastic depth rate of the backbone')
    p.add_argument('--mona', type=int, default=0, help='hidden size of Mona adapters in every ViT block (0: off)')
    # model
    p.add_argument('--spatial-prior', type=int, default=0, help='channels of the CNN spatial-prior branch (0: off)')
    p.add_argument('--ml-levels', default='', help="multi-level neck: 4 block indices feeding strides 4..32, e.g. '2,5,8,11'")
    p.add_argument('--anchor-sizes', default=';'.join(','.join(str(s) for s in lvl) for lvl in REFERENCE_ANCHORS))
    p.add_argument('--anchor-ratios', default='0.5,1.0,2.0')
    p.add_argument('--box-head', default='mlp', choices=['mlp', 'small'])
    p.add_argument('--cls-pool', default='mean', choices=['mean', 'clsmean', 'attn'])
    p.add_argument('--roi-batch', type=int, default=512)
    p.add_argument('--cascade', action='store_true', help='3-stage Cascade R-CNN with light class-agnostic heads')
    p.add_argument('--cascade-rep', type=int, default=1024, help='FC width of each cascade stage head')
    p.add_argument('--cascade-beta', type=float, default=0.0, help='smooth-L1 beta of the cascade box loss (0: L1)')
    p.add_argument('--cascade-size-iou', action='store_true', help='size-aware IoU thresholds for the cascade stages')
    p.add_argument('--size-iou-table', default='a4', choices=['a4', 'small16'],
                   help="with --cascade-size-iou: 'a4' (relaxed up to 32 px) or 'small16' (relaxed below 16 px only)")
    p.add_argument('--roi-context', type=float, default=0.0,
                   help='cascade: also pool the RoI enlarged by this factor (>= 32 px) and add it to the box features')
    p.add_argument('--roi-context-mode', default='shared', choices=['shared', 'classification'],
                   help='cascade context feeds both tasks (shared), or classification only; regression then uses local RoI features')
    p.add_argument('--cls-weights', default='',
                   help="cascade: RoI classification loss weights by category id, e.g. '3:3,7:3' (bridge, playground)")
    p.add_argument('--partial-neg', action='store_true',
                   help="cascade: unmatched RoIs learn only 'not the image's class' (images carry one class's boxes)")
    p.add_argument('--oversample', default='',
                   help="training images drawn with these weights by image label 0..8, e.g. '2:2,6:2' (one GPU only)")
    p.add_argument('--oversample-small', type=int, default=1,
                   help='training images with a box of area < --small-px^2 (on the 512 input) appear this many times per epoch '
                        '(1: off; 3 = the rayner branch\'s os3). The epoch gets longer; originals are kept; val/test untouched')
    p.add_argument('--small-px', type=float, default=32.0, help='small box: area < small-px^2 on the 512 input (COCO: 32)')
    p.add_argument('--token-stride', type=int, default=0, help='timm ViTs: patch-embedding stride (8: overlapping patches, 4x tokens)')
    p.add_argument('--grad-ckpt', action='store_true', help='gradient checkpointing of the ViT blocks (saves memory)')
    p.add_argument('--iou-branch', type=float, default=0.0, help='loss weight of an IoU-quality branch on the last cascade stage (0: off)')
    p.add_argument('--rpn-nwd', type=float, default=0.0, help='RPN anchor matching by max(IoU, NWD) with this C (0: IoU)')
    p.add_argument('--anchor-offset', type=float, default=0.0, help='anchor centres at (i + offset) * stride')
    p.add_argument('--aligned-roi', action='store_true', help='RoIAlign with aligned=True')
    p.add_argument('--pretrained-prior', action='store_true', help='ImageNet ResNet-18 stem as the spatial prior (needs --spatial-prior 64)')
    p.add_argument('--giou', type=float, default=0.0, help='weight of a GIoU loss on the RoI boxes (0: off)')
    p.add_argument('--giou-only', action='store_true', help='the GIoU loss replaces the smooth-L1 RoI box loss')
    # optimisation
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--warmup-epochs', type=float, default=1.0, help='linear warmup length; default reproduces one epoch')
    p.add_argument('--schedule-epochs', type=int, help='cosine horizon (default: --epochs; must be >= --epochs)')
    p.add_argument('--save-training-state', action='store_true', help='atomically save complete training_state.pt each epoch')
    p.add_argument('--resume', type=Path, help='restore a trusted local training_state.pt, including AdamW and rank RNG')
    p.add_argument('--init-weights', type=Path, help='initialise the trained tensors from this model.pt / ckpt (e.g. an averaged checkpoint) with a fresh optimizer')
    p.add_argument('--stop-after-epoch', type=int, help='cleanly stop after this epoch without shortening the LR schedule')
    p.add_argument('--reproducible-loader', action='store_true',
                   help='seed Python/NumPy and restart epoch-seeded workers; required for training-state save/resume')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--wd-hygiene', action='store_true', help='no weight decay on 1-D parameters (norms, biases, gammas) and embeddings')
    p.add_argument('--loraplus', type=float, default=1.0, help='LoRA+: learning-rate multiplier for the LoRA B (up) matrices')
    p.add_argument('--full-lr', type=float, help='LR of full-rank pretrained backbone weights that train (default: --backbone-lr)')
    p.add_argument('--cheap-lr', type=float, help='LR of 1-D pretrained backbone parameters: norms, biases, gammas (default: --backbone-lr)')
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--cls-weight', type=float, default=1.0, help='weight of the classification loss')
    p.add_argument('--ema', type=float, default=0.0, help='EMA decay of the trainable weights (0: off); EMA weights are evaluated and saved')
    p.add_argument('--aug', default='flip', choices=['flip', 'd4'])
    p.add_argument('--flip-bias', type=float, default=0.0,
                   help="annotation position bias in original px (e.g. -0.3): flipped boxes are moved by twice it")
    p.add_argument('--copy-paste', type=float, default=0.0, help='probability of same-class copy-paste of small objects')
    p.add_argument('--cp-source', default='mixed', choices=['mixed', 'same_class', 'same_image'])
    p.add_argument('--cp-max-n', type=int, default=6)
    p.add_argument('--cp-max-side', type=int, default=40)
    p.add_argument('--speckle', type=float, default=0.0, help='photometric aug: max std of multiplicative speckle noise (0: off)')
    p.add_argument('--gamma-jitter', type=float, default=0.0, help='photometric aug: gamma = exp(U(-g, g)) (0: off)')
    p.add_argument('--photo-p', type=float, default=0.5, help='probability of each photometric aug per image')
    p.add_argument('--zoom-p', type=float, default=0.0)
    p.add_argument('--zoom-max', type=float, default=1.5)
    # data, evaluation, output
    p.add_argument('--train-split', default='train', help="e.g. 'train+val+test' for the final model")
    p.add_argument('--eval-splits', default='val,test', help="comma-separated; '' for none")
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--eval-batch-size', type=int, help='evaluation batch size (default: --batch-size)')
    p.add_argument('--save-every', type=int, default=0, help='save a checkpoint (trained tensors) every N epochs (0: off)')
    p.add_argument('--score-thresh', type=float, default=0.05, help='detection score threshold at evaluation')
    p.add_argument('--dump-raw', action='store_true', help='save raw per-class boxes and logits of the eval splits')
    p.add_argument('--save', default='trainable', choices=['full', 'trainable', 'none'])
    p.add_argument('--allow-over-budget', action='store_true', help='only for reference runs that break the Terms')
    p.add_argument('--compile', action='store_true', help='torch.compile each backbone block (same math, faster)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N images of every split (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--name', help='run folder name under --out (default: backbone_init_adapt)')
    args = p.parse_args(argv)
    if args.init == 'scratch' and args.adapt != 'full':
        p.error('--init scratch is only valid with --adapt full')
    if args.roi_context_mode == 'classification' and (not args.cascade or args.roi_context <= 0):
        p.error('--roi-context-mode classification requires --cascade and --roi-context > 0')
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    if args.schedule_epochs is None:
        args.schedule_epochs = args.epochs
    if args.epochs < 1 or args.schedule_epochs < args.epochs:
        p.error('--schedule-epochs must be >= --epochs, and --epochs must be positive')
    if args.oversample_small < 1:
        p.error('--oversample-small must be at least 1')
    if not 0 <= args.warmup_epochs <= args.schedule_epochs:
        p.error('--warmup-epochs must be between 0 and --schedule-epochs')
    if args.batch_size < 1 or args.eval_every < 1 or args.save_every < 0:
        p.error('batch-size/eval-every must be positive and save-every nonnegative')
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        p.error('--eval-batch-size must be positive')
    if (args.save_training_state or args.resume) and not args.reproducible_loader:
        p.error('--save-training-state/--resume require --reproducible-loader for epoch-boundary RNG recovery')
    if args.stop_after_epoch is not None:
        if not 1 <= args.stop_after_epoch <= args.epochs or not args.save_training_state:
            p.error('--stop-after-epoch must be within 1..epochs and requires --save-training-state')
    return args


def parse_weights(spec):
    """'3:3,7:2.5' -> {3: 3.0, 7: 2.5} ('' -> {})."""
    return {int(k): float(v) for k, v in (item.split(':') for item in spec.split(',') if item)}


def log(message, file):
    """Print to the console and to the run's log.txt (only the main process logs: file is None elsewhere)."""
    if file is None:
        return
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, split, train, world=1, rank=0):
    """Batches of `split`; only the training set is shuffled and augmented (copy-paste uses training images only).
    With world > 1 (DDP) the training set is split over the processes, each taking batch_size / world images."""
    reproducible = getattr(args, 'reproducible_loader', False)
    loader_options = ({'generator': torch.Generator().manual_seed(args.seed + rank),
                       'worker_init_fn': seed_worker} if reproducible else {})
    persistent = train and args.workers > 0 and not reproducible
    batch_size = args.batch_size if train else (getattr(args, 'eval_batch_size', None) or args.batch_size)
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit, aug=args.aug, zoom_p=args.zoom_p,
                           zoom_max=args.zoom_max, flip_bias=getattr(args, 'flip_bias', 0.0),
                           speckle=getattr(args, 'speckle', 0.0), gamma_jitter=getattr(args, 'gamma_jitter', 0.0),
                           photo_p=getattr(args, 'photo_p', 0.5),
                           oversample_small=getattr(args, 'oversample_small', 1), small_px=getattr(args, 'small_px', 32.0))
    if train and getattr(dataset, 'n_small', 0):
        print(f'oversample-small x{args.oversample_small}: {dataset.n_small} images with a box < {args.small_px:g}^2 px '
              f'repeated, {len(dataset.rows)} training items per epoch', flush=True)
    if train and args.copy_paste:
        dataset = SameClassCopyPaste(dataset, p=args.copy_paste, max_n=args.cp_max_n, max_side=args.cp_max_side,
                                     source=args.cp_source, seed=args.seed)
    if train and getattr(args, 'oversample', ''):
        # each image drawn with probability proportional to the weight of its label; an epoch keeps its length
        assert world == 1, '--oversample is implemented for one GPU only'
        factor = parse_weights(args.oversample)
        base = dataset.base if isinstance(dataset, SameClassCopyPaste) else dataset
        weights = [factor.get(int(r['label']), 1.0) for r in base.rows]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
        return DataLoader(dataset, args.batch_size, sampler=sampler, num_workers=args.workers, collate_fn=collate,
                          persistent_workers=persistent, drop_last=False, **loader_options)
    if train and world > 1:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
        return DataLoader(dataset, args.batch_size // world, sampler=sampler, num_workers=args.workers,
                          collate_fn=collate, persistent_workers=persistent, drop_last=False, **loader_options)
    return DataLoader(dataset, batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=persistent, drop_last=False, **loader_options)


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def build_model(args):
    """The model as configured by args (also used by predict.py, with pretrained=False plus a checkpoint)."""
    if getattr(args, 'roi_context_mode', 'shared') == 'classification' and (
            not getattr(args, 'cascade', False) or getattr(args, 'roi_context', 0.0) <= 0):
        raise ValueError('--roi-context-mode classification requires --cascade and --roi-context > 0')
    backbone = build_backbone(args.backbone, pretrained=args.init == 'pretrained', drop_path=getattr(args, 'drop_path', 0.0))
    if getattr(args, 'token_stride', 0):
        set_token_stride(backbone, args.token_stride)
    if getattr(args, 'grad_ckpt', False):
        backbone.vit.set_grad_checkpointing(True)
    if getattr(args, 'ml_levels', ''):
        backbone.ml_indices = [int(i) for i in args.ml_levels.split(',')]
        assert len(backbone.ml_indices) == 4, '--ml-levels needs 4 block indices'
    if args.adapt == 'moelora':
        add_adapters(backbone, 'moelora')
    elif args.adapt == 'lora':
        targets = tuple(t for t in args.lora_targets.split(',') if t)
        n = len(list(backbone.blocks))
        skip = set(range(n - args.unfreeze_last, n)) if args.unfreeze_last else set()
        full_stages = [int(i) for i in getattr(args, 'cnx_full_stages', '').split(',') if i]
        if full_stages:  # ConvNeXt: blocks of the fully trained stages get no LoRA
            first = 0
            for i in range(4):
                count = len(getattr(backbone.body, f'stages_{i}').blocks)
                if i in full_stages:
                    skip |= set(range(first, first + count))
                first += count
        if targets == ('attn.qkv', 'attn.proj') and args.lora_rank == 16 and args.lora_alpha == 32 and not skip:
            add_adapters(backbone, 'lora')  # the reference, unchanged
        else:
            wrapped = add_lora(backbone, targets, args.lora_rank, args.lora_alpha, skip_blocks=skip)
            expected = (n - len(skip)) * len(targets)
            if wrapped != expected and not hasattr(backbone, 'body'):  # ConvNeXt blocks have no attn layers
                raise SystemExit(f'--lora-targets wrapped {wrapped} layers, expected {expected}: check the layer names')
    if getattr(args, 'mona', 0):
        add_mona(backbone, args.mona)  # trainable (created after the backbone was frozen)
    if args.adapt != 'full':
        unfreeze(backbone, norms=args.train_norms, biases=args.train_bias, patch_embed=args.train_patch_embed,
                 layer_scale=args.train_ls, last_blocks=args.unfreeze_last, pos_embed=getattr(args, 'train_pos_embed', False))
        for i in [int(i) for i in getattr(args, 'cnx_full_stages', '').split(',') if i]:
            getattr(backbone.body, f'stages_{i}').requires_grad_(True)
            backbone.body.stem_0.requires_grad_(True)
            backbone.body.stem_1.requires_grad_(True)
    anchors = tuple(tuple(float(s) for s in lvl.split(',')) for lvl in args.anchor_sizes.split(';'))
    ratios = tuple(float(r) for r in args.anchor_ratios.split(','))
    model = MultiTaskModel(backbone, task_routing=args.adapt == 'moelora', spatial_prior=args.spatial_prior,
                           anchor_sizes=anchors, anchor_ratios=ratios, box_head=args.box_head, cls_pool=args.cls_pool,
                           roi_batch=args.roi_batch)
    model.roi_heads.giou_weight = getattr(args, 'giou', 0.0)
    model.roi_heads.giou_only = getattr(args, 'giou_only', False)
    # tiny-object options (sarbench/tiny.py)
    if getattr(args, 'anchor_offset', 0.0):
        model.rpn.anchor_generator = OffsetAnchorGenerator(anchors, (ratios,) * len(anchors), offset=args.anchor_offset)
    if getattr(args, 'rpn_nwd', 0.0):
        model.rpn.box_similarity = iou_nwd_similarity(args.rpn_nwd)
    pool = model.roi_heads.box_roi_pool
    if getattr(args, 'aligned_roi', False):
        pool = AlignedMultiScaleRoIAlign(pool.featmap_names, output_size=pool.output_size, sampling_ratio=pool.sampling_ratio)
        model.roi_heads.box_roi_pool = pool
    if getattr(args, 'cascade', False):
        assert not getattr(args, 'giou_only', False), '--giou-only is not implemented for --cascade (use --giou w to add a GIoU term)'
        model.roi_heads = CascadeRoIHeads(pool, num_classes=10, reduced=64, representation=args.cascade_rep,
                                          beta=args.cascade_beta, batch_size_per_image=args.roi_batch,
                                          size_iou=(getattr(args, 'size_iou_table', 'a4') if getattr(args, 'cascade_size_iou', False) else False),
                                          iou_weight=getattr(args, 'iou_branch', 0.0),
                                          context=getattr(args, 'roi_context', 0.0),
                                          context_mode=getattr(args, 'roi_context_mode', 'shared'),
                                          cls_weights=parse_weights(getattr(args, 'cls_weights', '')),
                                          partial_neg=getattr(args, 'partial_neg', False))
        model.roi_heads.giou_weight = getattr(args, 'giou', 0.0)
    if getattr(args, 'pretrained_prior', False):
        assert args.spatial_prior == 64, '--pretrained-prior needs --spatial-prior 64 (64/128-channel laterals)'
        model.neck.prior = PretrainedPrior()
    return model


def param_groups(model, args):
    """Trainable backbone parameters at --backbone-lr, all new modules at --lr (the reference: two groups, every
    parameter with --weight-decay). --loraplus multiplies the LR of LoRA B matrices; --wd-hygiene removes weight decay
    from 1-D parameters and embeddings. The head group stays last (the log prints its LR)."""
    hygiene, lp = getattr(args, 'wd_hygiene', False), getattr(args, 'loraplus', 1.0)
    full_lr, cheap_lr = getattr(args, 'full_lr', None), getattr(args, 'cheap_lr', None)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        backbone = name.startswith('backbone.')
        lr = args.backbone_lr if backbone else args.lr
        adapter = '.mona' in name or '.routers.' in name or name.endswith(('.down.weight', '.up.weight'))
        if backbone and not adapter:  # pretrained weights that train: full-rank (patch embed, blocks) or 1-D
            if p.ndim >= 2 and full_lr is not None:
                lr = full_lr
            if p.ndim <= 1 and cheap_lr is not None:
                lr = cheap_lr
        if backbone and lp != 1.0 and name.endswith('.up.weight'):
            lr = lr * lp
        no_wd = hygiene and (p.ndim <= 1 or name.rsplit('.', 1)[-1] in ('pos_embed', 'cls_token', 'reg_token', 'pool_query'))
        key = (not backbone, lr, 0.0 if no_wd else args.weight_decay)
        groups.setdefault(key, []).append(p)
    return [{'params': ps, 'lr': lr, 'weight_decay': wd} for (_, lr, wd), ps in sorted(groups.items(), key=lambda kv: kv[0][0])]


def compile_blocks(backbone):
    """torch.compile every block of the backbone in place (the parameters stay the same objects)."""
    container = getattr(getattr(backbone, 'vit', None), 'blocks', None)
    if container is None:
        return
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)  # 12 blocks x train/eval x batch sizes
    for i in range(len(container)):
        container[i] = torch.compile(container[i], dynamic=False)


class EMA:
    """Exponential moving average of the trainable weights (one model: the average replaces the weights)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self, step):
        d = min(self.decay, (1 + step) / (10 + step))
        torch._foreach_lerp_(self.shadow, [p.detach() for p in self.params], 1 - d)

    @torch.no_grad()
    def swap(self):
        for p, s in zip(self.params, self.shadow):
            tmp = p.detach().clone()
            p.copy_(s)
            s.copy_(tmp)


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file, args, ema=None, step0=0):
    """One pass over the training set; the loss is cross-entropy plus the four Faster R-CNN losses."""
    model.train()
    start, seen = time.time(), 0
    for step, (images, labels, targets) in enumerate(loader, 1):
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):  # bf16 has fp32's range: no loss scaling needed
            logits, _, det_losses = model(images.cuda(), targets)
            losses = {'classification': args.cls_weight * F.cross_entropy(logits.float(), labels.cuda(),
                                                                          label_smoothing=args.label_smoothing),
                      **det_losses}
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f'nan loss at epoch {epoch} step {step}: {losses}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if ema is not None:
            ema.update(step0 + step)
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} ({parts}) '
                f'lr {optimizer.param_groups[-1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0
    return step0 + len(loader)


@torch.no_grad()
def evaluate(model, loader, gt_json, raw=False):
    """Classification and COCO detection metrics on one split, the detections in COCO result format and, with
    raw=True, the raw outputs (class logits, per-class boxes and logits of every proposal) for offline study."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    dump = {'image_ids': [], 'orig_sizes': [], 'labels': [], 'cls_logits': [], 'raw_boxes': [], 'raw_logits': [],
            'raw_obj': [], 'raw_iou': [], 'raw_stage': []}
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.to(next(model.parameters()).device), return_raw=raw)
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, out, lg, yy in zip(targets, outputs, logits, y.tolist()):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size']).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
            if raw:
                dump['image_ids'].append(target['image_id'])
                dump['orig_sizes'].append(target['orig_size'])
                dump['labels'].append(yy)
                dump['cls_logits'].append(lg.float().cpu())
                dump['raw_boxes'].append(out['raw_boxes'].float().cpu())
                dump['raw_logits'].append(out['raw_logits'].half().cpu())
                dump['raw_obj'].append(out['raw_obj'].float().cpu())
                if 'raw_iou' in out:
                    dump['raw_iou'].append(out['raw_iou'].float().cpu())
                if 'raw_stage' in out:
                    dump['raw_stage'].append(out['raw_stage'].float().cpu())
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections, (dump if raw else None)


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50'))


def main():
    args = parse_args()
    # multi-GPU (torchrun): one process per GPU, the global batch stays --batch-size
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    main_proc = rank == 0
    if args.batch_size % world:
        raise ValueError('--batch-size must be divisible by the DDP world size')
    if args.reproducible_loader:
        seed_everything(args.seed)
    else:
        torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    run_dir = args.out / (args.name or f'{args.backbone}_{args.init}_{args.adapt}')
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / 'log.txt', 'a' if args.resume else 'w') if main_proc else None
    log(json.dumps(vars(args), default=str), log_file)
    if world > 1:
        log(f'DDP: {world} processes, {args.batch_size // world} images per GPU per step', log_file)

    # Data
    train_loader = make_loader(args, args.train_split, train=True, world=world, rank=rank)
    eval_splits = [s for s in args.eval_splits.split(',') if s]
    eval_loaders = {s: make_loader(args, s, train=False) for s in eval_splits}

    # Model: adapters are created on the CPU, so they are added before .cuda()
    model = build_model(args).cuda()
    if getattr(args, 'init_weights', None):
        state = torch.load(args.init_weights, map_location='cpu', weights_only=True)
        state = {k.replace('._orig_mod', ''): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise SystemExit(f'--init-weights has unexpected tensors: {unexpected[:5]}')
        log(f'Initialised {len(state)} tensors from {args.init_weights} ({len(missing)} tensors keep their pretrained/fresh values)', log_file)
    model.roi_heads.post['score_thresh'] = args.score_thresh
    if args.compile:
        compile_blocks(model.backbone)
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    log(f'parameters: {parameters:,} total, {trainable:,} trainable (+ {buffers:,} floating-point buffer values)', log_file)
    parameters_and_buffers = parameters + buffers
    if (parameters_and_buffers > MAX_TOTAL or trainable > MAX_TRAINABLE) and not args.allow_over_budget:
        raise SystemExit(f'over the Codabench budget: {parameters:,} total (max {MAX_TOTAL:,}), '
                         f'{trainable:,} trainable (max {MAX_TRAINABLE:,})')

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr, all new modules at --lr
    optimizer = torch.optim.AdamW(param_groups(model, args), lr=args.lr, weight_decay=args.weight_decay)
    # training steps go through train_model (DDP-wrapped with several GPUs); evaluation, EMA and checkpoints use model
    train_model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()]) if world > 1 else model
    warmup = round(args.warmup_epochs * len(train_loader))
    total = args.schedule_epochs * len(train_loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))
    ema = EMA(model, args.ema) if args.ema else None

    signature = compatibility_signature(args, model, train_loader, world) if (args.save_training_state or args.resume) else None
    history, start, step, first_epoch, previous_hours = [], time.time(), 0, 1, 0.0
    if args.resume:
        progress = load_training_state(args.resume, model, optimizer, schedule, ema, signature=signature, rank=rank)
        history, step = progress['history'], progress['global_step']
        first_epoch, previous_hours = progress['epoch'] + 1, progress['train_hours']
        if first_epoch > args.epochs or (args.stop_after_epoch and args.stop_after_epoch < first_epoch):
            raise ValueError('The resumed checkpoint has already reached the requested stopping epoch')
        log(f'Resumed {args.resume}: epoch {first_epoch - 1}, global step {step}', log_file)
        start = time.time()

    # Evaluation and state saving are coordinated across all DDP ranks at epoch boundaries.
    for epoch in range(first_epoch, args.epochs + 1):
        if world > 1:
            train_loader.sampler.set_epoch(epoch)
        if args.reproducible_loader:
            seed_loader_epoch(train_loader, args.seed, epoch, rank)
            for loader in eval_loaders.values():
                seed_loader_epoch(loader, args.seed, epoch, rank)
        step = train_one_epoch(train_model, train_loader, optimizer, schedule, epoch, log_file, args, ema, step)
        if epoch == args.epochs:
            break  # final evaluation/export below precedes the final training-state checkpoint
        periodic = (args.save_every and epoch % args.save_every == 0) or \
            (eval_splits and epoch % args.eval_every == 0)
        if main_proc:
            if args.save_every and epoch % args.save_every == 0:
                keep = {n for n, q in model.named_parameters() if q.requires_grad} | {n for n, _ in model.named_buffers()}
                ckpt = {k.replace('._orig_mod', ''): v for k, v in model.state_dict().items() if k in keep}
                torch.save(ckpt, run_dir / f'ckpt_e{epoch}.pt')
            if eval_splits and epoch % args.eval_every == 0:
                if ema:
                    ema.swap()
                scores = {}
                for split in eval_splits:
                    scores[split], _, _ = evaluate(model, eval_loaders[split], args.data / 'detection' / f'instances_{split}.json')
                if ema:
                    ema.swap()
                history.append({'epoch': epoch, **scores[eval_splits[0]], 'all': scores})
                log(f'epoch {epoch} ' + ' || '.join(f'{split}: {headline(m)}' for split, m in scores.items()), log_file)
        if args.save_training_state:
            save_training_state(run_dir / 'training_state.pt', model, optimizer, schedule, ema,
                                epoch=epoch, step=step, history=history,
                                train_hours=previous_hours + (time.time() - start) / 3600,
                                signature=signature, world=world, rank=rank)
        elif world > 1 and periodic:
            dist.barrier()
        if args.stop_after_epoch == epoch:
            log(f'Stopped cleanly after epoch {epoch}; resume from {run_dir / "training_state.pt"}', log_file)
            if log_file:
                log_file.close()
            if world > 1:
                dist.destroy_process_group()
            return
    train_hours = previous_hours + (time.time() - start) / 3600
    if main_proc:
        if ema:  # evaluate the last-iterate weights too (paired comparison with the EMA weights, same run)
            last = {}
            for s in eval_splits:
                m, _, raw = evaluate(model, eval_loaders[s], args.data / 'detection' / f'instances_{s}.json', raw=args.dump_raw)
                last[s] = m
                log(f'{s} (last-iterate weights): {headline(m)}', log_file)
                if raw is not None:
                    torch.save(raw, run_dir / f'raw_{s}_last.pt')
            (run_dir / 'metrics_last.json').write_text(json.dumps(last, indent=2, default=str))
            ema.swap()  # from here on the model holds the EMA weights

        # Final evaluation of the last-epoch model on every eval split
        results = {}
        for s in eval_splits:
            t0 = time.time()
            m, dets, raw = evaluate(model, eval_loaders[s], args.data / 'detection' / f'instances_{s}.json', raw=args.dump_raw)
            m['seconds'] = time.time() - t0
            results[s] = m
            log(f'{s}: {headline(m)}', log_file)
            (run_dir / f'predictions_{s}.json').write_text(json.dumps(dets))
            if raw is not None:
                torch.save(raw, run_dir / f'raw_{s}.pt')
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        log(f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)

        # Save
        summary = {'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable, 'buffer_values': buffers,
                   'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb, 'val_history': history,
                   **{s: results[s] for s in eval_splits}}
        if 'val' in results:
            summary['val'] = results['val']
        (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
        if args.save != 'none':
            state = model.state_dict()
            if args.save == 'trainable':
                keep = {n for n, p in model.named_parameters() if p.requires_grad} | {n for n, _ in model.named_buffers()}
                state = {k: v for k, v in state.items() if k in keep}
            state = {k.replace('._orig_mod', ''): v for k, v in state.items()}  # compiled blocks: plain module names
            torch.save(state, run_dir / 'model.pt')
        log('RESULT ' + json.dumps({s: {k: round(results[s][k], 4) for k in ('macro_f1', 'mAP', 'AP50')} for s in eval_splits}), log_file)
        if ema:
            ema.swap()  # restore last-iterate weights before saving optimizer-compatible training state
    if args.save_training_state:
        save_training_state(run_dir / 'training_state.pt', model, optimizer, schedule, ema,
                            epoch=args.epochs, step=step, history=history, train_hours=train_hours,
                            signature=signature, world=world, rank=rank)
    elif world > 1:
        dist.barrier()
    if log_file:
        log_file.close()
    if world > 1:
        dist.destroy_process_group()



if __name__ == '__main__':
    main()
