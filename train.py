"""Train one model jointly for classification and detection, then score it on the local holdouts.
Example (the reference): python train.py --backbone vit --init pretrained --adapt lora
Every option beyond the reference is opt-in; the defaults reproduce the reference run."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.adapters import add_adapters, add_lora, add_mona, unfreeze
from sarbench.backbones import BACKBONES, build_backbone
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import REFERENCE_ANCHORS, MultiTaskModel

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
    p.add_argument('--giou', type=float, default=0.0, help='weight of a GIoU loss on the RoI boxes (0: off)')
    p.add_argument('--giou-only', action='store_true', help='the GIoU loss replaces the smooth-L1 RoI box loss')
    # optimisation
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--wd-hygiene', action='store_true', help='no weight decay on 1-D parameters (norms, biases, gammas) and embeddings')
    p.add_argument('--loraplus', type=float, default=1.0, help='LoRA+: learning-rate multiplier for the LoRA B (up) matrices')
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--cls-weight', type=float, default=1.0, help='weight of the classification loss')
    p.add_argument('--ema', type=float, default=0.0, help='EMA decay of the trainable weights (0: off); EMA weights are evaluated and saved')
    p.add_argument('--aug', default='flip', choices=['flip', 'd4'])
    p.add_argument('--zoom-p', type=float, default=0.0)
    p.add_argument('--zoom-max', type=float, default=1.5)
    # data, evaluation, output
    p.add_argument('--train-split', default='train', help="e.g. 'train+val+test' for the final model")
    p.add_argument('--eval-splits', default='val,test', help="comma-separated; '' for none")
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
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
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    return args


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, split, train):
    """Batches of `split`; only the training set is shuffled and augmented."""
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit, aug=args.aug, zoom_p=args.zoom_p,
                           zoom_max=args.zoom_max)
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0, drop_last=False)


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def build_model(args):
    """The model as configured by args (also used by predict.py, with pretrained=False plus a checkpoint)."""
    backbone = build_backbone(args.backbone, pretrained=args.init == 'pretrained', drop_path=getattr(args, 'drop_path', 0.0))
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
            add_lora(backbone, targets, args.lora_rank, args.lora_alpha, skip_blocks=skip)
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
    return model


def param_groups(model, args):
    """Trainable backbone parameters at --backbone-lr, all new modules at --lr (the reference: two groups, every
    parameter with --weight-decay). --loraplus multiplies the LR of LoRA B matrices; --wd-hygiene removes weight decay
    from 1-D parameters and embeddings. The head group stays last (the log prints its LR)."""
    hygiene, lp = getattr(args, 'wd_hygiene', False), getattr(args, 'loraplus', 1.0)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        backbone = name.startswith('backbone.')
        lr = args.backbone_lr if backbone else args.lr
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
    dump = {'image_ids': [], 'orig_sizes': [], 'labels': [], 'cls_logits': [], 'raw_boxes': [], 'raw_logits': []}
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda(), return_raw=raw)
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
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections, (dump if raw else None)


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50'))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    run_dir = args.out / (args.name or f'{args.backbone}_{args.init}_{args.adapt}')
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / 'log.txt', 'w')
    log(json.dumps(vars(args), default=str), log_file)

    # Data
    train_loader = make_loader(args, args.train_split, train=True)
    eval_splits = [s for s in args.eval_splits.split(',') if s]
    eval_loaders = {s: make_loader(args, s, train=False) for s in eval_splits}

    # Model: adapters are created on the CPU, so they are added before .cuda()
    model = build_model(args).cuda()
    model.roi_heads.post['score_thresh'] = args.score_thresh
    if args.compile:
        compile_blocks(model.backbone)
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)
    if (parameters > MAX_TOTAL or trainable > MAX_TRAINABLE) and not args.allow_over_budget:
        raise SystemExit(f'over the Codabench budget: {parameters:,} total (max {MAX_TOTAL:,}), '
                         f'{trainable:,} trainable (max {MAX_TRAINABLE:,})')

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr, all new modules at --lr
    optimizer = torch.optim.AdamW(param_groups(model, args), lr=args.lr, weight_decay=args.weight_decay)
    warmup, total = len(train_loader), args.epochs * len(train_loader)  # one epoch of warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))
    ema = EMA(model, args.ema) if args.ema else None

    # Train, validating every --eval-every epochs on the first eval split; the reported model is the last-epoch model
    history, start, step = [], time.time(), 0
    for epoch in range(1, args.epochs + 1):
        step = train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file, args, ema, step)
        if eval_splits and (epoch % args.eval_every == 0 or epoch == args.epochs) and epoch != args.epochs:
            s = eval_splits[0]
            if ema:
                ema.swap()
            val, _, _ = evaluate(model, eval_loaders[s], args.data / 'detection' / f'instances_{s}.json')
            if ema:
                ema.swap()
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} {s}: {headline(val)}', log_file)
    train_hours = (time.time() - start) / 3600
    if ema:
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
    summary = {'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
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
        torch.save(state, run_dir / 'model.pt')
    log('RESULT ' + json.dumps({s: {k: round(results[s][k], 4) for k in ('macro_f1', 'mAP', 'AP50')} for s in eval_splits}), log_file)


if __name__ == '__main__':
    main()
