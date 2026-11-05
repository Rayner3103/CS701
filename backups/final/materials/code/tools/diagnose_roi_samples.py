"""Inspect Cascade training samples on a fixed, class-balanced subset of LOCAL TRAIN only.

No optimizer, backward, torch.compile, checkpoint writes, or validation/test images.
RPN/Cascade use their normal training matcher/sampler; the backbone and stochastic
layers use eval mode. This is a sampling/loss diagnostic, not an evaluation score.
The analytic dL/dlogits L1 sums below are LOCAL proxies, not backbone gradients.
"""
import argparse
from collections import Counter, defaultdict
import contextlib
import json
import os
from pathlib import Path
import random
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True, help='converted SARFact-Course-20K root')
    p.add_argument('--out', type=Path, required=True, help='new diagnostic JSON (existing files are refused)')
    p.add_argument('--per-class', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=2, help='CPU intra-op threads; data workers are always zero')
    p.add_argument('--weights', default='model.pt')
    a = p.parse_args()
    for name in ('per_class', 'batch_size', 'threads'):
        if getattr(a, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if a.out.exists():
        p.error(f'output already exists: {a.out}')
    return a


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def static_counts(dataset, indices, names):
    counts = {str(c): {'name': name, 'images': 0, 'boxes': 0, 'empty_images': 0,
                       'box_categories': {}} for c, name in names.items()}
    for i in indices:
        row = dataset.rows[i]
        category = int(row['label']) + 1
        anns = dataset.annotations[dataset.coco_images[row['image']]['id']]
        entry = counts[str(category)]
        entry['images'] += 1
        entry['boxes'] += len(anns)
        entry['empty_images'] += not anns
        for ann in anns:
            key = str(ann['category_id'])
            entry['box_categories'][key] = entry['box_categories'].get(key, 0) + 1
    return counts


def classification_terms(torch, logits, labels, weights, partial, image_categories):
    """Unreduced losses and analytic L1 d(loss_i)/d(logits_i), before batch/stage scaling."""
    logp = logits.float().log_softmax(-1)
    probabilities = logp.exp()
    row = torch.arange(len(labels), device=labels.device)
    loss = -logp[row, labels]
    gradient = probabilities.clone()
    gradient[row, labels] -= 1
    sample_weights = torch.ones_like(loss) if weights is None else weights[labels]
    loss = loss * sample_weights
    gradient = gradient * sample_weights[:, None]
    denominator = sample_weights.sum()
    if partial:
        negative = labels == 0
        pn = probabilities[negative]
        categories = image_categories[negative]
        pc = pn.gather(1, categories[:, None]).squeeze(1)
        loss[negative] = -torch.log1p(-pc.clamp(max=1 - 1e-6))
        direction = -pn
        direction[torch.arange(len(categories), device=labels.device), categories] += 1
        scale = pc / (1 - pc).clamp_min(1e-6)
        scale = torch.where(pc <= 1 - 1e-6, scale, 0.0)  # derivative of the clamp
        gradient[negative] = scale[:, None] * direction
        denominator = loss.new_tensor(max(len(labels), 1))
    return loss, gradient.abs().sum(-1), float(denominator)


class Recorder:
    """Wrap existing methods once; never call a matcher or sampler an extra time."""

    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.roi = model.roi_heads
        self.records, self.rpn_records = [], []
        self.pending = {}
        self.batch = []
        self.handles = []
        self.original_select = self.roi.select_training_samples
        self.original_anchor_match = model.rpn.assign_targets_to_anchors

    def __enter__(self):
        self.roi.select_training_samples = self.select
        self.model.rpn.assign_targets_to_anchors = self.anchor_match
        for stage, predictor in enumerate(self.roi.stage_preds):
            self.handles.append(predictor.register_forward_hook(
                lambda module, inputs, output, stage=stage: self.prediction(stage, output)))
        return self

    def __exit__(self, *exc):
        self.roi.select_training_samples = self.original_select
        self.model.rpn.assign_targets_to_anchors = self.original_anchor_match
        for handle in self.handles:
            handle.remove()

    def select(self, proposals, targets):
        result = self.original_select(proposals, targets)
        _, matched, labels, _ = result
        # Cascade appends GT boxes at every stage, then removes them before the next.
        # Coverage excluding these injected boxes is crucial to avoid a misleading 100%.
        injected = [sampled >= n for sampled, n in zip(self.roi._sampled, self.roi._n_props)]
        self.pending[self.roi._t] = (matched, labels, injected, targets)
        return result

    def anchor_match(self, anchors, targets):
        result = self.original_anchor_match(anchors, targets)
        for meta, labels in zip(self.batch, result[0]):
            self.rpn_records.append({**meta, 'positive': int((labels > 0).sum()),
                                     'negative': int((labels == 0).sum()),
                                     'ignore': int((labels < 0).sum())})
        return result

    def prediction(self, stage, output):
        torch = self.torch
        matched, label_list, injected, targets = self.pending.pop(stage)
        labels = torch.cat(label_list)
        if bool((labels < 0).any()):
            raise RuntimeError('ignored RoIs unexpectedly survived the sampler')
        logits = output[0]
        cw = self.roi.cls_weights
        cw = cw.to(logits.device) if cw is not None else None
        categories = torch.cat([lab.new_full(lab.shape, meta['category_id'])
                                for lab, meta in zip(label_list, self.batch)])
        loss, proxy, denominator = classification_terms(
            torch, logits, labels, cw, self.roi.partial_neg, categories)
        if denominator <= 0:
            raise RuntimeError('classification loss has a nonpositive normalization denominator')
        stage_weight = self.roi.stage_weights[stage]
        offset = 0
        for meta, lab, match, is_gt, target in zip(self.batch, label_list, matched, injected, targets):
            pos, neg = lab > 0, lab == 0
            n_gt = len(target['boxes'])
            hits = torch.bincount(match[pos], minlength=n_gt) if n_gt else lab.new_zeros(0)
            ordinary = pos & ~is_gt
            hits_ordinary = (torch.bincount(match[ordinary], minlength=n_gt)
                             if n_gt else lab.new_zeros(0))
            li, gi = loss[offset:offset + len(lab)], proxy[offset:offset + len(lab)]
            record = {**meta, 'stage': stage + 1, 'gt_count': n_gt,
                      'positive': int(pos.sum()), 'negative': int(neg.sum()),
                      'positive_appended_gt': int((pos & is_gt).sum()),
                      'positive_non_gt_proposal': int(ordinary.sum()),
                      'positive_per_gt': hits.cpu().tolist(),
                      'positive_non_gt_proposal_per_gt': hits_ordinary.cpu().tolist(),
                      'gt_covered': int((hits > 0).sum()),
                      'gt_covered_non_gt_proposals': int((hits_ordinary > 0).sum()),
                      'stage_weight': stage_weight, 'batch_loss_denominator': denominator}
            for name, mask in [('positive', pos), ('negative', neg)]:
                record[f'ce_{name}_sum'] = float(li[mask].sum())
                record[f'logits_grad_l1_{name}_sum'] = float(gi[mask].sum())
                record[f'training_loss_{name}_contribution'] = stage_weight * float(li[mask].sum()) / denominator
            self.records.append(record)
            offset += len(lab)


def summarize(records, names):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['category_id'], row['stage'])].append(row)
    keys = ('gt_count', 'positive', 'negative', 'positive_appended_gt', 'positive_non_gt_proposal',
            'gt_covered', 'gt_covered_non_gt_proposals', 'ce_positive_sum', 'ce_negative_sum',
            'logits_grad_l1_positive_sum', 'logits_grad_l1_negative_sum',
            'training_loss_positive_contribution', 'training_loss_negative_contribution')
    out = []
    for (category, stage), rows in sorted(grouped.items()):
        entry = {'category_id': category, 'name': names[category], 'stage': stage, 'images': len(rows)}
        entry.update({key: sum(row[key] for row in rows) for key in keys})
        for key in ('positive_per_gt', 'positive_non_gt_proposal_per_gt'):
            counts = Counter(n for row in rows for n in row[key])
            entry[f'{key}_histogram'] = dict(sorted(counts.items()))
        entry['gt_coverage'] = ratio(entry['gt_covered'], entry['gt_count'])
        entry['gt_coverage_non_gt_proposals'] = ratio(entry['gt_covered_non_gt_proposals'], entry['gt_count'])
        entry['negative_to_positive_ce_ratio'] = ratio(entry['ce_negative_sum'], entry['ce_positive_sum'])
        entry['negative_to_positive_logits_grad_l1_ratio'] = ratio(
            entry['logits_grad_l1_negative_sum'], entry['logits_grad_l1_positive_sum'])
        out.append(entry)
    return out


def main():
    a = parse_args()  # --help works without importing torch, timm, or constructing a model
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    os.environ['MKL_NUM_THREADS'] = str(a.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from torch.utils.data import DataLoader, Subset
    from predict import load_run
    from sarbench.data import SARMultiTask, collate

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            raise SystemExit('CUDA unavailable; pass --device cpu for CPU use')
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    model, saved = load_run(a.run, a.weights, device=str(device))
    if not hasattr(model.roi_heads, 'stage_preds'):
        raise SystemExit('this diagnostic requires a Cascade run')
    if getattr(saved, 'copy_paste', 0) or getattr(saved, 'zoom_p', 0):
        raise SystemExit('use a run without copy-paste/zoom so each per-GT count maps to the original annotation')
    dataset = SARMultiTask(a.data, 'train', train=True, aug=saved.aug,
                           flip_bias=getattr(saved, 'flip_bias', 0.0))
    coco = json.loads((a.data / 'detection/instances_train.json').read_text())
    names = {c['id']: c['name'] for c in coco['categories']}
    groups = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        groups[int(row['label']) + 1].append(i)
    rng = random.Random(a.seed)
    selected = []
    for category in sorted(groups):
        selected.extend(rng.sample(groups[category], min(a.per_class, len(groups[category]))))
    rng.shuffle(selected)
    if not selected:
        raise SystemExit('training dataset is empty')
    metadata = []
    for index in selected:
        row = dataset.rows[index]
        image_id = dataset.coco_images[row['image']]['id']
        anns = dataset.annotations[image_id]
        if not anns or any(ann['category_id'] != int(row['label']) + 1 for ann in anns):
            raise SystemExit('expected nonempty, single-category annotations matching each training image label')
        metadata.append({'image_id': image_id, 'file_name': row['image'],
                         'category_id': int(row['label']) + 1,
                         'annotation_ids_in_gt_order': [ann.get('id') for ann in anns]})
    loader = DataLoader(Subset(dataset, selected), batch_size=a.batch_size, num_workers=0,
                        shuffle=False, collate_fn=collate)
    model.train()
    model.backbone.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)):
            module.eval()
    vit = getattr(model.backbone, 'vit', None)
    if vit is not None and hasattr(vit, 'set_grad_checkpointing'):
        vit.set_grad_checkpointing(False)
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    # Reset after model construction, which consumes RNG, to keep sampling/augmentation seed explicit.
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    losses_by_batch = []
    with Recorder(model, torch) as recorder, torch.no_grad():
        offset = 0
        for batch_index, (images, _, targets) in enumerate(loader):
            recorder.batch = metadata[offset:offset + len(images)]
            record_start = len(recorder.records)
            targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]
            amp = (torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda'
                   else contextlib.nullcontext())
            with amp:
                _, _, losses = model(images.to(device), targets)
            if recorder.pending:
                raise RuntimeError('a Cascade predictor hook did not consume its sampled labels')
            if any(not bool(torch.isfinite(value)) for value in losses.values()):
                raise RuntimeError('non-finite training loss during diagnostic')
            for stage in range(len(model.roi_heads.stage_preds)):
                rows = [r for r in recorder.records[record_start:] if r['stage'] == stage + 1]
                reconstructed = sum(r['training_loss_positive_contribution'] +
                                    r['training_loss_negative_contribution'] for r in rows)
                actual = float(losses[f'loss_classifier_s{stage}'])
                if abs(reconstructed - actual) > 1e-5 * max(1.0, abs(actual)):
                    raise RuntimeError('classification contribution mismatch: this run may use an unsupported '
                                       'loss reduction; diagnose the original AA run instead')
            losses_by_batch.append({'batch': batch_index, 'images': len(images),
                                    'losses': {key: float(value) for key, value in losses.items()}})
            offset += len(images)
            if offset % 16 == 0 or offset == len(selected):
                print(f'diagnosed {offset}/{len(selected)} training images', flush=True)
    unchanged = all(parameter._version == versions[name] for name, parameter in model.named_parameters())
    buffers_unchanged = all(torch.equal(buffers[name], value) for name, value in model.named_buffers())
    if not unchanged or not buffers_unchanged:
        raise RuntimeError('model parameter mutation version or buffer values changed during diagnostic')
    summary = summarize(recorder.records, names)
    rpn_summary = []
    for category in sorted(groups):
        rows = [row for row in recorder.rpn_records if row['category_id'] == category]
        rpn_summary.append({'category_id': category, 'name': names[category], 'images': len(rows),
                            **{key: sum(row[key] for row in rows) for key in ('positive', 'negative', 'ignore')}})
    result = {
        'schema_version': 1, 'run': str(a.run.resolve()), 'data': str(a.data.resolve()), 'weights': a.weights,
        'split': 'train', 'seed': a.seed, 'per_class_limit': a.per_class, 'batch_size': a.batch_size,
        'device': str(device), 'cpu_threads': a.threads, 'data_workers': 0,
        'sampling': 'fixed per-class sample without replacement; original RPN/Cascade matcher and sampler',
        'preprocessing': {'training_augmentation': saved.aug, 'flip_bias': getattr(saved, 'flip_bias', 0.0),
                          'size': dataset.size, 'backbone_mode': 'eval', 'rpn_and_roi_mode': 'train'},
        'loss': 'partial-negative CE' if model.roi_heads.partial_neg else 'weighted cross-entropy',
        'interpretation': [
            'Sampling/loss diagnostic at fixed trained weights, not AP/F1 or an unbiased dataset-wide estimate.',
            'Per-class image counts are balanced; raw totals do not estimate natural training class exposure.',
            'positive_per_gt includes GT proposals appended by Cascade; the non_gt variant excludes them.',
            'ce_*_sum is unreduced and includes optional class weights; training_loss_* uses batch denominator and stage weight.',
            'logits_grad_l1_* sums analytic per-RoI L1 d(loss_i)/d(logits_i) before batch/stage scaling.',
            'This proxy is NOT backbone gradient magnitude, gradient conflict, or a measured parameter update.',
            'RPN counts describe all anchors after matching, BEFORE the RPN positive/negative subsampler.',
        ],
        'no_optimizer_or_backward': True, 'torch_compile': False,
        'parameter_mutation_versions_unchanged': unchanged, 'buffer_values_unchanged': buffers_unchanged,
        'static_train_counts': static_counts(dataset, range(len(dataset)), names),
        'static_selected_counts': static_counts(dataset, selected, names),
        'selected_images': metadata, 'cascade_summary': summary, 'cascade_per_image': recorder.records,
        'rpn_match_summary': rpn_summary, 'rpn_match_per_image': recorder.rpn_records,
        'training_losses_by_batch': losses_by_batch,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print('class          stage  pos    neg    GT covered / without appended GT   CE neg/pos   logits-L1 neg/pos')
    for row in summary:
        fmt = lambda value: 'n/a' if value is None else f'{value:.2f}'
        print(f'{row["name"]:14} {row["stage"]:>3} {row["positive"]:>6} {row["negative"]:>6}  '
              f'{fmt(row["gt_coverage"]):>6} / {fmt(row["gt_coverage_non_gt_proposals"]):<6} '
              f'{fmt(row["negative_to_positive_ce_ratio"]):>16} '
              f'{fmt(row["negative_to_positive_logits_grad_l1_ratio"]):>19}')
    print(f'wrote {a.out}; logits-L1 is a local proxy, not a backbone-gradient measurement')


if __name__ == '__main__':
    main()
