"""Tiny-object components (adapted from the research sweep's tinydet_components.py, reviewed and wired into train.py):
NWD-based RPN assignment, anchors at cell centres, aligned RoIAlign, a 3-stage Cascade R-CNN with light class-agnostic
heads, same-class copy-paste augmentation and an ImageNet ResNet-18 spatial prior. All opt-in; see train.py flags."""
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign, box_iou, roi_align
from torchvision.ops import boxes as box_ops
from torchvision.ops.poolers import _convert_to_roi_format, _filter_input, _setup_scales

from .postprocess import detections_from_raw

# ----------------------------------------------------------------------------------------------- RPN assignment
def nwd(boxes1, boxes2, C=8.0):
    """Normalized Gaussian Wasserstein distance (Wang et al. 2021), XYXY [N,4] x [M,4] -> [N,M] in (0, 1]."""
    c1, c2 = (boxes1[:, :2] + boxes1[:, 2:]) / 2, (boxes2[:, :2] + boxes2[:, 2:]) / 2
    h1, h2 = (boxes1[:, 2:] - boxes1[:, :2]) / 2, (boxes2[:, 2:] - boxes2[:, :2]) / 2
    a, b = torch.cat([c1, h1], 1).float(), torch.cat([c2, h2], 1).float()
    d = torch.cdist(a, b, compute_mode='donot_use_mm_for_euclid_dist')  # exact; the matmul path is TF32 on GPUs
    return torch.exp(-d / C)


def iou_nwd_similarity(C=8.0):
    """For RegionProposalNetwork.box_similarity ((gt [G,4], anchors [A,4]) -> [G,A]). max(IoU, NWD): objects
    >= ~32 px behave exactly as with IoU, objects < 16 px get ~2-5x more positive anchors (C=8 on this data)."""
    def sim(gt, anchors):
        return torch.maximum(box_iou(gt, anchors), nwd(gt, anchors, C))
    return sim


class OffsetAnchorGenerator(AnchorGenerator):
    """torchvision puts anchor centres at i * stride (cell corners); offset=0.5 puts them at (i + 0.5) * stride,
    the centre of the image region that ViTDet pyramid cell i covers (2 px at stride 4, i.e. 1/4 of a 8 px object)."""

    def __init__(self, sizes, aspect_ratios, offset=0.5):
        super().__init__(sizes, aspect_ratios)
        self.offset = offset

    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        for (gh, gw), (sh, sw), base in zip(grid_sizes, strides, self.cell_anchors):
            sx = (torch.arange(gw, dtype=torch.float32, device=base.device) + self.offset) * sw
            sy = (torch.arange(gh, dtype=torch.float32, device=base.device) + self.offset) * sh
            yy, xx = torch.meshgrid(sy, sx, indexing='ij')
            shifts = torch.stack((xx.reshape(-1), yy.reshape(-1), xx.reshape(-1), yy.reshape(-1)), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base.view(1, -1, 4)).reshape(-1, 4))
        return anchors


# ----------------------------------------------------------------------------------------------- RoIAlign
class AlignedMultiScaleRoIAlign(MultiScaleRoIAlign):
    """MultiScaleRoIAlign with aligned=True (torchvision's default is the legacy aligned=False)."""

    def forward(self, x, boxes, image_shapes):
        feats = _filter_input(x, self.featmap_names)
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(feats, image_shapes, self.canonical_scale,
                                                         self.canonical_level)
        rois = _convert_to_roi_format(boxes)
        kw = dict(output_size=self.output_size, sampling_ratio=self.sampling_ratio, aligned=True)
        if len(feats) == 1:
            return roi_align(feats[0], rois, spatial_scale=self.scales[0], **kw)
        levels = self.map_levels(boxes)
        out = feats[0].new_zeros((len(rois), feats[0].shape[1]) + tuple(self.output_size))
        for lvl, (f, s) in enumerate(zip(feats, self.scales)):
            idx = torch.where(levels == lvl)[0]
            out[idx] = roi_align(f, rois[idx], spatial_scale=s, **kw).to(out.dtype)
        return out


# ----------------------------------------------------------------------------------------------- Cascade R-CNN
class LightBoxHead(nn.Module):
    """1x1 conv 256 -> reduced, then two FC layers (the repo's SmallBoxHead; reduced=64 -> 4.3 M params)."""

    def __init__(self, in_channels=256, reduced=64, resolution=7, representation=1024, context=False):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, reduced, 1), nn.GELU())
        # context=True: the RoI features of an enlarged box go through their own 1x1 conv and are added (+16 K params)
        self.ctx_reduce = nn.Sequential(nn.Conv2d(in_channels, reduced, 1), nn.GELU()) if context else None
        self.fc6 = nn.Linear(reduced * resolution * resolution, representation)
        self.fc7 = nn.Linear(representation, representation)

    def forward(self, x, ctx=None):
        x = self.reduce(x)
        if ctx is not None:
            x = x + self.ctx_reduce(ctx)
        return F.relu(self.fc7(F.relu(self.fc6(x.flatten(1)))))


def context_boxes(boxes, image_shape, factor, min_side=32.0):
    """XYXY boxes enlarged about their centres: each side becomes max(factor * side, min_side), clipped to the image."""
    c = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = ((boxes[:, 2:] - boxes[:, :2]) * factor).clamp_min(min_side) / 2
    return box_ops.clip_boxes_to_image(torch.cat([c - half, c + half], 1), image_shape)


class AgnosticPredictor(nn.Module):
    """num_classes scores (incl. background) and ONE class-agnostic box, repeated per class so that torchvision's
    box coder, fastrcnn_loss and the repo's detections_from_raw work unchanged."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, 4)
        self.num_classes = num_classes
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, x):
        x = x.flatten(1)
        return self.cls_score(x), self.bbox_pred(x).repeat(1, self.num_classes)


def box_loss(box_regression, labels, regression_targets, beta):
    """torchvision's fastrcnn box loss with a configurable smooth-L1 beta (0 -> L1, the detectron2/mmdet default)."""
    pos = torch.where(labels > 0)[0]
    pred = box_regression.reshape(len(labels), -1, 4)[pos, labels[pos]]
    if beta > 0:
        loss = F.smooth_l1_loss(pred, regression_targets[pos], beta=beta, reduction='sum')
    else:
        loss = F.l1_loss(pred, regression_targets[pos], reduction='sum')
    return loss / max(labels.numel(), 1)


# size-aware IoU thresholds per cascade stage: (max box side at the 512 input, (stage-1, stage-2, stage-3 thresholds))
SIZE_IOU = ((8, (0.40, 0.45, 0.50)), (16, (0.45, 0.50, 0.55)), (32, (0.50, 0.55, 0.65)), (float('inf'), (0.5, 0.6, 0.7)))
# 'small16': relax only below 16 px (A4 gained on <16 px but lost ~2 AP on 32-96 px objects)
SIZE_IOU_TABLES = {'a4': SIZE_IOU,
                   'small16': ((8, (0.40, 0.45, 0.50)), (16, (0.45, 0.50, 0.60)), (float('inf'), (0.5, 0.6, 0.7)))}


class CascadeRoIHeads(RoIHeads):
    """Cascade R-CNN (Cai & Vasconcelos, CVPR 2018) with the eval interface of the repo's RoIHeadsPlus
    (image_probs fusion, return_raw, self.post). Stages: IoU 0.5 / 0.6 / 0.7, box-coder weights (10,10,5,5) /
    (20,20,10,10) / (30,30,15,15), loss weights 1 / 0.5 / 0.25, class-agnostic regression; at test time the
    class probabilities of the three stages are averaged and the boxes come from the last stage.
    Params with LightBoxHead(reduced=64): 3 x 4.28 M = 12.8 M (the reference TwoMLPHead alone is 13.9 M)."""

    post = dict(score_thresh=0.05, nms_thresh=0.5, per_class=None, total=100, fusion='none', gamma=1.0, topk=9,
                obj_pow=0.0, nms='hard', soft_sigma=0.75, shift=0.0, quality_pow=0.0, stage_box='last')

    def __init__(self, box_roi_pool, num_classes=10, reduced=64, representation=1024, ious=(0.5, 0.6, 0.7),
                 coder_weights=((10., 10., 5., 5.), (20., 20., 10., 10.), (30., 30., 15., 15.)),
                 stage_weights=(1.0, 0.5, 0.25), beta=1 / 9, batch_size_per_image=512, positive_fraction=0.25,
                 size_iou=False, iou_weight=0.0, context=0.0, cls_weights=None):
        """size_iou: per-GT IoU thresholds by box size (SIZE_IOU) so tiny objects get positives at every stage.
        iou_weight > 0: an IoU-quality branch on the last stage (BCE to the IoU of the decoded box with its GT); at test
        time its prediction can re-score boxes (post['quality_pow']).
        context > 0: every stage also pools the RoI enlarged by this factor (at least 32 px) and adds it to the box
        features (context_boxes). cls_weights: {category id: weight} for the RoI classification loss (others 1)."""
        heads = nn.ModuleList(LightBoxHead(256, reduced, 7, representation, context=context > 0) for _ in ious)
        preds = nn.ModuleList(AgnosticPredictor(representation, num_classes) for _ in ious)
        super().__init__(box_roi_pool, heads[0], preds[0], fg_iou_thresh=ious[0], bg_iou_thresh=ious[0],
                         batch_size_per_image=batch_size_per_image, positive_fraction=positive_fraction,
                         bbox_reg_weights=coder_weights[0], score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)
        del self.box_head, self.box_predictor  # the stages own the heads (no duplicate state_dict keys)
        self.stage_heads, self.stage_preds = heads, preds
        self.matchers = [det_utils.Matcher(t, t, allow_low_quality_matches=False) for t in ious]
        self.coders = [det_utils.BoxCoder(w) for w in coder_weights]
        self.stage_weights, self.beta = stage_weights, beta
        self.post = dict(CascadeRoIHeads.post)
        self.size_iou, self.iou_weight, self._t = size_iou, iou_weight, 0
        self.size_table = SIZE_IOU_TABLES[size_iou if isinstance(size_iou, str) else 'a4'] if size_iou else None
        self.context = context
        self.cls_weights = None  # a plain tensor, not a buffer: no state_dict key, not in the parameter budget
        if cls_weights:
            self.cls_weights = torch.ones(num_classes)
            for c, w in cls_weights.items():
                self.cls_weights[c] = w
        if iou_weight:
            self.iou_head = nn.Linear(representation, 1)
            nn.init.normal_(self.iou_head.weight, std=0.01)
            nn.init.zeros_(self.iou_head.bias)

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        if not self.size_iou:
            return super().assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        matched_idxs, labels = [], []
        base = self.proposal_matcher.high_threshold
        for props, gtb, gtl in zip(proposals, gt_boxes, gt_labels):
            if gtb.numel() == 0:
                matched = torch.zeros((props.shape[0],), dtype=torch.int64, device=props.device)
                lab = torch.zeros((props.shape[0],), dtype=torch.int64, device=props.device)
            else:
                iou = box_ops.box_iou(gtb, props)
                side = ((gtb[:, 2] - gtb[:, 0]).clamp_min(0) * (gtb[:, 3] - gtb[:, 1]).clamp_min(0)).sqrt()
                thr = torch.empty_like(side)
                for limit, ts in reversed(self.size_table):
                    thr[side < limit] = ts[self._t]
                quality = (iou * (base / thr)[:, None]).clamp(max=1.0)  # a GT of size s passes at IoU >= thr(s)
                m = self.proposal_matcher(quality)
                matched = m.clamp(min=0)
                lab = gtl[matched].to(dtype=torch.int64)
                lab[m == self.proposal_matcher.BELOW_LOW_THRESHOLD] = 0
                lab[m == self.proposal_matcher.BETWEEN_THRESHOLDS] = -1
            matched_idxs.append(matched)
            labels.append(lab)
        return matched_idxs, labels

    def add_gt_proposals(self, proposals, gt_boxes):  # rows >= _n_props are the appended GT boxes
        self._n_props = [len(p) for p in proposals]
        return super().add_gt_proposals(proposals, gt_boxes)

    def subsample(self, labels):  # sorted indices of the sampled rows, per image
        self._sampled = super().subsample(labels)
        return self._sampled

    def forward(self, features, proposals, image_shapes, targets=None, image_probs=None, return_raw=False,
                objectness=None):
        losses, probs, quality, stage_boxes = {}, [], None, []
        last = len(self.stage_heads) - 1
        for t, (head, pred) in enumerate(zip(self.stage_heads, self.stage_preds)):
            self.proposal_matcher, self.box_coder = self.matchers[t], self.coders[t]  # used by select_training_samples
            self._t = t
            if self.training:
                proposals, _, labels, reg_targets = self.select_training_samples(proposals, targets)
                is_gt = [smp >= n for smp, n in zip(self._sampled, self._n_props)]  # mmdet's pos_is_gt
            ctx = None
            if self.context:
                ctx_props = [context_boxes(p, s, self.context) for p, s in zip(proposals, image_shapes)]
                ctx = self.box_roi_pool(features, ctx_props, image_shapes)
            feat = head(self.box_roi_pool(features, proposals, image_shapes), ctx)
            cls, reg = pred(feat)
            if self.training:
                labels, reg_targets = torch.cat(labels), torch.cat(reg_targets)
                w = self.stage_weights[t]
                cw = self.cls_weights.to(cls.device) if self.cls_weights is not None else None
                losses[f'loss_classifier_s{t}'] = w * F.cross_entropy(cls, labels, weight=cw)
                losses[f'loss_box_reg_s{t}'] = w * box_loss(reg, labels, reg_targets, self.beta)
                if self.iou_weight and t == last:  # IoU-quality branch: predict the IoU of the decoded box with its GT
                    pos = torch.where(labels > 0)[0]
                    props = torch.cat(proposals)[pos]
                    pred_b = self.box_coder.decode_single(reg[pos, :4].detach().float(), props)
                    gt_b = self.box_coder.decode_single(reg_targets[pos].float(), props)
                    lt, rb = torch.max(pred_b[:, :2], gt_b[:, :2]), torch.min(pred_b[:, 2:], gt_b[:, 2:])
                    inter = (rb - lt).clamp(min=0).prod(1)
                    area = lambda b: (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1)
                    iou_t = (inter / (area(pred_b) + area(gt_b) - inter).clamp_min(1e-6)).clamp(0, 1)
                    logit = self.iou_head(feat[pos]).squeeze(1).float()
                    losses['loss_iou'] = self.iou_weight * (F.binary_cross_entropy_with_logits(logit, iou_t)
                                                            if len(pos) else logit.sum() * 0)
            else:
                probs.append(F.softmax(cls.float(), -1))
                if self.iou_weight and t == last:
                    quality = torch.sigmoid(self.iou_head(feat).squeeze(1).float())
            if not self.training:
                stage_boxes.append(proposals)  # the input boxes of stage t (RPN proposals for t = 0)
            if t < len(self.stage_heads) - 1:  # refine the boxes for the next stage, without gradient
                n = [len(p) for p in proposals]
                new = self.box_coder.decode(reg[:, :4].detach(), proposals)[:, 0]
                proposals = [box_ops.clip_boxes_to_image(b, s) for b, s in zip(new.split(n), image_shapes)]
                if self.training:  # drop the GT boxes appended at this stage: the next stage appends them again
                    proposals = [pr[~g] for pr, g in zip(proposals, is_gt)]
        if self.training:
            return [], losses
        logits = torch.stack(probs).mean(0).clamp_min(1e-12).log()  # softmax(log p) = p downstream
        boxes_all = self.box_coder.decode(reg, proposals)  # [sum N, C + 1, 4], identical boxes per class
        results = []
        for i, (b, lg) in enumerate(zip(boxes_all.split([len(p) for p in proposals]), logits.split([len(p) for p in proposals]))):
            pi = image_probs[i] if image_probs is not None else None
            obj = objectness[i] if objectness is not None else None
            q = quality.split([len(p) for p in proposals])[i] if quality is not None else None
            stages = torch.stack([sb[i] for sb in stage_boxes])  # [stages, N, 4] stage input boxes
            boxes, scores, labels = detections_from_raw(b[:, 1:], lg, pi, image_size=image_shapes[i], objectness=obj,
                                                        quality=q, stage_inputs=stages, **self.post)
            out = {'boxes': boxes, 'scores': scores, 'labels': labels}
            if return_raw:
                out['raw_boxes'], out['raw_logits'] = b[:, 1:], lg
                if obj is not None:
                    out['raw_obj'] = obj
                if q is not None:
                    out['raw_iou'] = q
                out['raw_stage'] = stages
            results.append(out)
        return results, {}



# ----------------------------------------------------------------------------------------------- copy-paste
def _keep_mask(boxes, j, x1, y1, x2, y2):
    """[y2-y1, x2-x1] bool: False on window pixels outside boxes[j] that another GT box touches (so a pasted patch
    carries no unlabelled fragment of a neighbouring object)."""
    ys = torch.arange(y1, y2, dtype=torch.float32)[:, None]
    xs = torch.arange(x1, x2, dtype=torch.float32)[None, :]

    def touch(b):
        return (xs + 1 > b[0]) & (xs < b[2]) & (ys + 1 > b[1]) & (ys < b[3])

    other = torch.zeros(y2 - y1, x2 - x1, dtype=torch.bool)
    for k, o in enumerate(boxes):
        if k != j:
            other |= touch(o)
    return ~(other & ~touch(boxes[j]))


class SameClassCopyPaste(Dataset):
    """Wraps the repo's SARMultiTask (train=True). With probability p, pastes n ~ U{1..max_n} small objects
    (side <= max_side px at the 512 input) cut from OTHER training images of the SAME class into the image, at random
    places that do not touch existing boxes; a feathered alpha (ramp over `margin` px) hides the seams. The image
    class is unchanged, so the one-class-per-image rule of the data still holds. Patches are cached once, in memory."""

    def __init__(self, base, p=0.5, max_n=6, max_side=40, margin=3, max_bank_per_class=3000, seed=0,
                 source='same_class'):
        """source: 'same_class' (patches from other images of the same class, cached bank), 'same_image'
        (duplicate the image's own small objects elsewhere in the same image: background statistics and
        speckle stay consistent; no bank needed) or 'mixed' (50/50 per pasted object)."""
        self.base, self.p, self.max_n, self.margin = base, p, max_n, margin
        self.source, self.max_side = source, max_side
        rng = random.Random(seed)
        self.bank = {}
        if source == 'same_image':
            return
        aug, zoom = base.aug, base.zoom_p
        base.aug, base.zoom_p = 'none', 0.0  # deterministic, un-augmented pass to cut the patches
        flips = base.transform.transforms
        base.transform.transforms = [t for t in flips if 'Flip' not in type(t).__name__]
        order = list(range(len(base)))
        rng.shuffle(order)
        for i in order:
            img, label, tgt = base[i]
            for j, b in enumerate(tgt['boxes']):
                w, h = float(b[2] - b[0]), float(b[3] - b[1])
                if max(w, h) > max_side or min(w, h) < 2:
                    continue
                lst = self.bank.setdefault(label, [])
                if len(lst) >= max_bank_per_class:
                    continue
                x1, y1 = int(max(b[0] - margin, 0)), int(max(b[1] - margin, 0))
                x2, y2 = int(min(b[2] + margin + 1, img.shape[-1])), int(min(b[3] + margin + 1, img.shape[-2]))
                patch = img[:, y1:y2, x1:x2].clone()
                inner = torch.tensor([b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1])  # box inside the patch
                lst.append((i, patch, inner, _keep_mask(tgt['boxes'], j, x1, y1, x2, y2)))
        base.aug, base.zoom_p, base.transform.transforms = aug, zoom, flips

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label, tgt = self.base[i]
        boxes, labels = tgt['boxes'], tgt['labels']
        H, W = img.shape[-2:]
        own = []  # this image's own small objects (for 'same_image' / 'mixed')
        if self.source != 'same_class':
            m = self.margin
            for j, b in enumerate(boxes):
                w, h = float(b[2] - b[0]), float(b[3] - b[1])
                if max(w, h) <= self.max_side and min(w, h) >= 2:
                    x1, y1 = int(max(b[0] - m, 0)), int(max(b[1] - m, 0))
                    x2, y2 = int(min(b[2] + m + 1, W)), int(min(b[3] + m + 1, H))
                    own.append((-1, img[:, y1:y2, x1:x2].clone(), torch.tensor([b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1]),
                                _keep_mask(boxes, j, x1, y1, x2, y2)))
        if random.random() >= self.p or not (self.bank.get(label) or own):
            return img, label, tgt
        img = img.clone()
        new_boxes = []
        for _ in range(random.randint(1, self.max_n)):
            use_own = own and (self.source == 'same_image' or not self.bank.get(label) or
                               (self.source == 'mixed' and random.random() < 0.5))
            if not use_own and not self.bank.get(label):
                break
            src, patch, inner, keep = random.choice(own) if use_own else random.choice(self.bank[label])
            if src == i:
                continue
            k = random.randrange(4)  # random quarter turn of the patch (exact)
            for _ in range(k):
                ph, pw = patch.shape[-2:]
                inner = torch.stack([inner[1], pw - inner[2], inner[3], pw - inner[0]])
                patch = torch.rot90(patch, 1, dims=(-2, -1))
                keep = torch.rot90(keep, 1, dims=(-2, -1))
            ph, pw = patch.shape[-2:]
            for _try in range(10):
                x0, y0 = random.randint(0, W - pw), random.randint(0, H - ph)
                cand = inner + torch.tensor([x0, y0, x0, y0], dtype=inner.dtype)
                region = torch.tensor([[x0, y0, x0 + pw, y0 + ph]], dtype=torch.float32)
                others = torch.cat([boxes] + ([torch.stack(new_boxes)] if new_boxes else [])).float()
                if len(others) and box_iou(region, others).max() > 0:
                    continue
                yy = torch.arange(ph, dtype=torch.float32)[:, None]
                xx = torch.arange(pw, dtype=torch.float32)[None, :]
                d = torch.minimum(torch.minimum(yy + 1, ph - yy), torch.minimum(xx + 1, pw - xx))
                alpha = (d / (self.margin + 1)).clamp(max=1.0) * keep  # feathered border; neighbours masked out
                img[:, y0:y0 + ph, x0:x0 + pw] = alpha * patch + (1 - alpha) * img[:, y0:y0 + ph, x0:x0 + pw]
                new_boxes.append(cand.float())
                break
        if new_boxes:
            tgt = dict(tgt)
            tgt['boxes'] = torch.cat([boxes, torch.stack(new_boxes)])
            tgt['labels'] = torch.cat([labels, labels.new_full((len(new_boxes),), label + 1)])
        return img, label, tgt



# ----------------------------------------------------------------------------------------------- pretrained prior
class PretrainedPrior(nn.Module):
    """ImageNet ResNet-18 stem + layer1 + layer2 (frozen BatchNorm) as a drop-in for the repo's SpatialPrior:
    image [B, 1, H, W] -> ([B, 64, H/4, W/4], [B, 128, H/8, W/8]); 0.68 M trainable parameters. The RGB conv1 weights
    are summed into one gray channel, as timm does for in_chans=1. Use: build the model with --spatial-prior 64 (that
    creates the zero-initialised laterals 64 -> 192 and 128 -> 384), then set model.neck.prior = PretrainedPrior()."""

    def __init__(self):
        super().__init__()
        import torchvision
        from torchvision.ops.misc import FrozenBatchNorm2d
        r = torchvision.models.resnet18(weights='IMAGENET1K_V1', norm_layer=FrozenBatchNorm2d)
        conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        conv1.weight.data.copy_(r.conv1.weight.data.sum(1, keepdim=True))
        self.stride4 = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.stride8 = r.layer2
        self.channels = (64, 128)

    def forward(self, images):
        c4 = self.stride4(images)
        return c4, self.stride8(c4)
