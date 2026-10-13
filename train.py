"""Train one model jointly for classification and detection, then score it on val and test.
Example: python train.py --backbone vit --init pretrained --adapt lora"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.adapters import add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MultiTaskModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--backbone', required=True, choices=['vit', 'terramind'])
    p.add_argument('--init', required=True, choices=['pretrained', 'scratch'])
    p.add_argument('--adapt', required=True, choices=['full', 'lora', 'moelora'])
    p.add_argument('--spatial-prior', action='store_true',
                   help='add a small CNN on the raw image to the neck at strides 4 and 8 (ViT-Adapter style)')
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--patience', type=int,
                   help='early stopping: stop once N validations in a row have not improved the best val Δm '
                        '(default: off, train all --epochs)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N images of every split (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--resume', action='store_true',
                   help="continue from the run folder's last_checkpoint.pt (the last completed epoch), if there is one")
    args = p.parse_args()
    if args.init == 'scratch' and args.adapt != 'full':
        p.error('--init scratch is only valid with --adapt full')
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    return args


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, split):
    """Batches of `split`; only the training set is shuffled and augmented."""
    train = split == 'train'
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit)
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0)  # training workers live across epochs


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file):
    """One pass over the training set; the loss is cross-entropy plus the four Faster R-CNN losses."""
    model.train()
    start, seen = time.time(), 0
    for step, (images, labels, targets) in enumerate(loader, 1):
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):  # bf16 has fp32's range: no loss scaling needed
            logits, _, det_losses = model(images.cuda(), targets)
            losses = {'classification': F.cross_entropy(logits, labels.cuda()), **det_losses}
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} ({parts}) '
                f'lr {optimizer.param_groups[1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0


@torch.no_grad()
def evaluate(model, loader, gt_json):
    """Classification and COCO detection metrics on one split, and the detections in COCO result format."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, out in zip(targets, outputs):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size']).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections


# The Δm reference on val (README: ViT + LoRA, mean of 3 seeds). Δm ranks the leaderboard, so it also picks the best epoch.
VAL_REFERENCE = {'macro_f1': 0.9331, 'mAP': 0.2613}


def delta_m(metrics, reference=VAL_REFERENCE):
    """Mean relative change of macro-F1 and mAP vs the reference, in %."""
    return 100 * sum((metrics[k] - ref) / ref for k, ref in reference.items()) / len(reference)


def should_stop(history, best, patience):
    """Early stopping: True once the last `patience` validations have all come after the best one."""
    return bool(patience) and best is not None and sum(h['epoch'] > best['epoch'] for h in history) >= patience


def save_checkpoint(path, **state):
    """Write to a temporary file, then rename: a crash while saving leaves the previous checkpoint intact."""
    torch.save(state, path.with_suffix('.tmp'))
    os.replace(path.with_suffix('.tmp'), path)


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50'))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = args.out / f'{args.backbone}_{args.init}_{args.adapt}{"_sp" if args.spatial_prior else ""}'
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / 'last_checkpoint.pt'  # the last completed epoch, for --resume
    checkpoint = None
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # not the GPU: no extra GPU memory on resume
    if checkpoint and checkpoint['epochs'] != args.epochs:  # the cosine schedule depends on the total
        raise SystemExit(f'{checkpoint_path} is from a run of {checkpoint["epochs"]} epochs, not --epochs {args.epochs}')
    log_file = open(run_dir / 'log.txt', 'a' if checkpoint else 'w')  # a resumed run continues its log
    log(json.dumps(vars(args), default=str), log_file)

    # Data
    train_loader, val_loader, test_loader = (make_loader(args, split) for split in ('train', 'val', 'test'))

    # Model: adapters are created on the CPU, so they are added before .cuda()
    backbone = build_backbone(args.backbone, pretrained=args.init == 'pretrained')
    if args.adapt != 'full':
        add_adapters(backbone, args.adapt)  # also freezes the backbone
    model = MultiTaskModel(backbone, task_routing=args.adapt == 'moelora', spatial_prior=args.spatial_prior).cuda()
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr, all new modules at --lr
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': args.backbone_lr}, {'params': head_params}],
                                  lr=args.lr, weight_decay=args.weight_decay)
    warmup, total = len(train_loader), args.epochs * len(train_loader)  # one epoch of warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))

    # Resume: weights, optimizer, schedule, random state and val history as they were after the checkpoint's epoch
    history, best, hours_before, first_epoch = [], None, 0.0, 1
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        schedule.load_state_dict(checkpoint['schedule'])
        torch.set_rng_state(checkpoint['rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        history, best, hours_before = checkpoint['history'], checkpoint['best'], checkpoint['train_hours']
        first_epoch = checkpoint['epoch'] + 1
        log(f'resumed from the checkpoint of epoch {checkpoint["epoch"]}', log_file)
        del checkpoint  # frees its copy of the weights

    # Train, validating every --eval-every epochs. The model with the best val Δm so far is saved as best_model.pt
    # at once, so it survives an interrupted run; the last-epoch model stays the reported one, as in the README.
    # After every epoch, last_checkpoint.pt is overwritten with everything --resume needs.
    # With --patience, training stops early; the epoch it stops at is then the "last" one. The check reads the val
    # history, so it also holds after --resume. The learning-rate schedule still follows --epochs.
    start, last_epoch, stopped_early = time.time(), first_epoch - 1, False
    for epoch in range(first_epoch, args.epochs + 1):
        if should_stop(history, best, args.patience):
            stopped_early = True
            log(f'early stopping after epoch {last_epoch}: no better val delta_m in the last {args.patience} '
                f'validations (best: epoch {best["epoch"]}, delta_m {best["val"]["delta_m"]:+.2f})', log_file)
            break
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file)
        last_epoch = epoch
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = evaluate(model, val_loader, args.data / 'detection' / 'instances_val.json')
            val['delta_m'] = delta_m(val)
            history.append({'epoch': epoch, **val})
            (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
            improved = best is None or val['delta_m'] > best['val']['delta_m']
            if improved:
                best = {'epoch': epoch, 'val': val}
                torch.save(model.state_dict(), run_dir / 'best_model.pt')
                (run_dir / 'predictions_val_best.json').write_text(json.dumps(val_detections))
            log(f'epoch {epoch} val: {headline(val)} delta_m {val["delta_m"]:+.2f}'
                f'{" (best so far, saved)" if improved else ""}', log_file)
        save_checkpoint(checkpoint_path, epoch=epoch, epochs=args.epochs, model=model.state_dict(),
                        optimizer=optimizer.state_dict(), schedule=schedule.state_dict(),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), history=history,
                        best=best, train_hours=hours_before + (time.time() - start) / 3600)
    train_hours = hours_before + (time.time() - start) / 3600

    # Evaluate on test, once per model: the last-epoch model, then the best one (unless it is the same)
    test_json = args.data / 'detection' / 'instances_test.json'
    test, test_detections = evaluate(model, test_loader, test_json)
    test['delta_m'] = delta_m(test)
    torch.save(model.state_dict(), run_dir / 'model.pt')
    if best['epoch'] == last_epoch:
        best['test'], best_test_detections = test, test_detections
    else:
        model.load_state_dict(torch.load(run_dir / 'best_model.pt'))
        best['test'], best_test_detections = evaluate(model, test_loader, test_json)
        best['test']['delta_m'] = delta_m(best['test'])
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'test (last, epoch {last_epoch}): {headline(test)}\n'
        f'test (best, epoch {best["epoch"]}): {headline(best["test"])}\n'
        f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)

    # Save
    summary = {'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
               'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'last_epoch': last_epoch, 'stopped_early': stopped_early,
               'val_history': history, 'val': history[-1], 'test': test, 'best': best}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
    (run_dir / 'predictions_test_best.json').write_text(json.dumps(best_test_detections))
    checkpoint_path.unlink(missing_ok=True)  # the run is complete: model.pt and best_model.pt hold its weights


if __name__ == '__main__':
    main()
