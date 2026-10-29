"""A Cascade R-CNN box head (Cai & Vasconcelos, 2018) on torchvision's RoIHeads machinery, with light stage heads."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops


class StageHead(nn.Module):
    """RoI features [N, C, 7, 7] -> (class logits [N, num_classes + 1], class-agnostic box deltas [N, 4]).
    Two 3 x 3 convs, a 1 x 1 reduction to `reduce` channels and one fc: about 4.4 M parameters, where torchvision's
    TwoMLPHead has 13.9 M (three of those would break the 40 M trainable limit)."""

    def __init__(self, num_classes, channels=256, convs=2, reduce=64, hidden=1024, size=7):
        super().__init__()
        layers = []
        for _ in range(convs):
            layers += [nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels), nn.GELU()]
        layers += [nn.Conv2d(channels, reduce, 1), nn.GELU(), nn.Flatten(), nn.Linear(reduce * size * size, hidden),
                   nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.cls = nn.Linear(hidden, num_classes + 1)  # + 1: background
        self.reg = nn.Linear(hidden, 4)
        nn.init.normal_(self.cls.weight, std=0.01)  # mmdetection's Cascade R-CNN initialisation
        nn.init.normal_(self.reg.weight, std=0.001)
        nn.init.zeros_(self.cls.bias)
        nn.init.zeros_(self.reg.bias)

    def forward(self, x):
        x = self.body(x)
        return self.cls(x), self.reg(x)


class CascadeRoIHeads(nn.Module):
    """Stage t samples from the boxes stage t - 1 refined, at a higher IoU threshold, and regresses with tighter box
    coder weights. Training: the stages' losses, weighted 1, 0.5, 0.25 (keys loss_classifier, loss_box_reg, then
    with _s2, _s3). Inference: class probabilities averaged over the stages, boxes from the last stage.

    Each stage is a torchvision RoIHeads, used only for its sampling, box coder and post-processing. With
    checkpoint=True the stage heads recompute their activations in the backward pass: the same losses and gradients
    for less memory (the 3 stages hold ~4 GB of activations at batch size 8), at some speed."""

    def __init__(self, box_roi_pool, num_classes=9, iou_thresholds=(0.5, 0.6, 0.7),
                 reg_weights=((10., 10., 5., 5.), (20., 20., 10., 10.), (30., 30., 15., 15.)),
                 stage_weights=(1., 0.5, 0.25), score_thresh=0.05, nms_thresh=0.5, detections_per_img=100,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.stage_weights = stage_weights
        self.stages = nn.ModuleList(
            RoIHeads(box_roi_pool, StageHead(num_classes), nn.Identity(), fg_iou_thresh=iou, bg_iou_thresh=iou,
                     batch_size_per_image=512, positive_fraction=0.25, bbox_reg_weights=weights,
                     score_thresh=score_thresh, nms_thresh=nms_thresh, detections_per_img=detections_per_img)
            for iou, weights in zip(iou_thresholds, reg_weights))

    def refine(self, stage, proposals, deltas, image_shapes, targets=None):
        """The boxes `deltas` move `proposals` to, clipped to the image. Training: the ground-truth boxes that
        select_training_samples added as proposals are dropped (the next stage adds them again), and so are boxes
        under 1 px. Inference keeps every box, so the stages' scores stay aligned."""
        refined = stage.box_coder.decode(deltas.detach(), proposals).squeeze(1)
        refined = refined.split([len(p) for p in proposals])
        out = []
        for i, (boxes, shape) in enumerate(zip(refined, image_shapes)):
            boxes = box_ops.clip_boxes_to_image(boxes, shape)
            if targets is not None:
                is_gt = (proposals[i][:, None] == targets[i]['boxes'][None]).all(-1).any(1)
                big = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
                boxes = boxes[big & ~is_gt]
            out.append(boxes.detach())
        return out

    def forward(self, features, proposals, image_shapes, targets=None):
        losses, probs = {}, []
        for t, stage in enumerate(self.stages):
            labels = regression_targets = None
            if self.training:
                proposals, _, labels, regression_targets = stage.select_training_samples(proposals, targets)
            pooled = stage.box_roi_pool(features, proposals, image_shapes)
            if self.checkpoint and self.training:
                logits, deltas = checkpoint(stage.box_head, pooled, use_reentrant=False)
            else:
                logits, deltas = stage.box_head(pooled)
            deltas = deltas.float()
            if self.training:
                # the loss indexes the deltas of each box's class: all classes share the same 4 numbers
                cls_loss, reg_loss = fastrcnn_loss(logits, deltas.repeat(1, self.num_classes + 1), labels,
                                                   regression_targets)
                suffix = '' if t == 0 else f'_s{t + 1}'
                losses[f'loss_classifier{suffix}'] = self.stage_weights[t] * cls_loss
                losses[f'loss_box_reg{suffix}'] = self.stage_weights[t] * reg_loss
            else:
                probs.append(logits.softmax(-1))
            if t < len(self.stages) - 1:
                proposals = self.refine(stage, proposals, deltas, image_shapes, targets if self.training else None)
        if self.training:
            return [], losses
        mean_logits = torch.stack(probs).mean(0).clamp_min(1e-12).log()  # it softmaxes them back to the mean probs
        boxes, scores, labels = stage.postprocess_detections(mean_logits, deltas.repeat(1, self.num_classes + 1),
                                                             proposals, image_shapes)
        return [{'boxes': b, 'labels': l, 'scores': s} for b, s, l in zip(boxes, scores, labels)], {}
