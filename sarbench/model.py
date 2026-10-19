"""The multi-task model: one shared backbone feeds a classification head and a Faster R-CNN detector.
With the default arguments it is the reference model; every addition is opt-in."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign, generalized_box_iou_loss
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .adapters import set_task
from .postprocess import detections_from_raw

REFERENCE_ANCHORS = ((8, 11), (16, 23), (32, 45), (64, 91), (128, 181))


def conv_gn(c_in, c_out, stride=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(32, c_out),
                         nn.GELU())


class SpatialPrior(nn.Module):
    """A small CNN on the raw image, after ViT-Adapter's spatial prior module (Chen et al., 2023): real pixel detail
    at strides 4 and 8, where a stride-16 ViT can only upsample. image [B, 1, H, W] ->
    ([B, c, H/4, W/4], [B, 2c, H/8, W/8])."""

    def __init__(self, c=64):
        super().__init__()
        self.stride4 = nn.Sequential(conv_gn(1, c, stride=2), conv_gn(c, c), conv_gn(c, c),
                                     nn.MaxPool2d(3, stride=2, padding=1))
        self.stride8 = conv_gn(c, 2 * c, stride=2)
        self.channels = (c, 2 * c)

    def forward(self, images):
        c4 = self.stride4(images)
        return c4, self.stride8(c4)


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the single stride-16 ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64.
    With spatial_prior, SpatialPrior features are added (through zero-initialised 1x1 convs) to the upsampled maps at
    strides 4 and 8, so training starts from exactly the plain neck."""

    def __init__(self, dim=768, channels=256, spatial_prior=0):
        super().__init__()
        self.prior = None
        if spatial_prior:
            with torch.random.fork_rng(devices=[]):  # same init of everything else with and without the prior
                self.prior = SpatialPrior(spatial_prior)
                self.lateral = nn.ModuleList([nn.Conv2d(self.prior.channels[0], dim // 4, 1),
                                              nn.Conv2d(self.prior.channels[1], dim // 2, 1)])
            for conv in self.lateral:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GroupNorm(32, dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),  # stride 4
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 8
            nn.Identity(),  # stride 16
            nn.MaxPool2d(2),  # stride 32
        ])
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.GroupNorm(32, channels),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels))
            for c in (dim // 4, dim // 2, dim, dim))

    def forward(self, x, images=None, maps=None):
        """x: the last block's map [B, dim, H/16, W/16]; maps: optionally one map per level (multi-level neck:
        the shallowest block feeds stride 4, the last one stride 32); images: needed only with the spatial prior."""
        scaled = [rescale(m) for rescale, m in zip(self.rescale, maps)] if maps else [rescale(x) for rescale in self.rescale]
        if self.prior is not None:
            for i, (prior, lateral) in enumerate(zip(self.prior(images), self.lateral)):
                scaled[i] = scaled[i] + lateral(prior)
        levels = [project(s) for s, project in zip(scaled, self.project)]
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class HierarchicalNeck(nn.Module):
    """FPN over the four maps of a hierarchical backbone (strides 4-32) + max-pool level (stride 64)."""

    def __init__(self, in_channels, channels=256):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(list(in_channels), channels, extra_blocks=LastLevelMaxPool())

    def forward(self, maps, images=None):
        return self.fpn(OrderedDict((str(i), m) for i, m in enumerate(maps)))


class SmallBoxHead(nn.Module):
    """A lighter RoI box head: 1x1 conv 256 -> c, then two FC layers (fewer parameters than TwoMLPHead's 12.8 M fc6)."""

    def __init__(self, in_channels=256, reduced=64, resolution=7, representation=1024):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, reduced, 1), nn.GELU())
        self.fc6 = nn.Linear(reduced * resolution * resolution, representation)
        self.fc7 = nn.Linear(representation, representation)

    def forward(self, x):
        x = self.reduce(x).flatten(1)
        return F.relu(self.fc7(F.relu(self.fc6(x))))


class RPNKeepScores(RegionProposalNetwork):
    """torchvision's RPN; also keeps the objectness (sigmoid) of the proposals it returns, for re-scoring."""

    last_scores = None

    def filter_proposals(self, *args, **kwargs):
        boxes, scores = super().filter_proposals(*args, **kwargs)
        self.last_scores = scores
        return boxes, scores


class RoIHeadsPlus(RoIHeads):
    """torchvision RoIHeads; at eval it scores boxes with sarbench.postprocess (the default settings reproduce
    torchvision's postprocess_detections exactly) and can also return the raw per-class boxes and logits."""

    post = dict(score_thresh=0.05, nms_thresh=0.5, per_class=None, total=100, fusion='none', gamma=1.0, topk=9,
                obj_pow=0.0, nms='hard', soft_sigma=0.75)
    giou_weight = 0.0  # > 0: add a GIoU loss on the positive RoIs' decoded boxes (tighter boxes for tiny objects)
    giou_only = False  # True: the GIoU loss replaces the smooth-L1 box loss

    def forward(self, features, proposals, image_shapes, targets=None, image_probs=None, return_raw=False,
                objectness=None):
        if self.training and not self.giou_weight:
            return super().forward(features, proposals, image_shapes, targets)
        if self.training:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
            box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
            class_logits, box_regression = self.box_predictor(box_features)
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            labels, props = torch.cat(labels), torch.cat(proposals)
            targets_d = torch.cat(regression_targets)
            pos = torch.where(labels > 0)[0]
            deltas = box_regression.reshape(len(labels), -1, 4)[pos, labels[pos]]
            pred = self.box_coder.decode_single(deltas, props[pos])
            gt = self.box_coder.decode_single(targets_d[pos], props[pos])
            loss_giou = self.giou_weight * generalized_box_iou_loss(pred, gt, reduction='sum') / max(labels.numel(), 1)
            losses = {'loss_classifier': loss_classifier, 'loss_giou': loss_giou}
            if not self.giou_only:
                losses['loss_box_reg'] = loss_box_reg
            return [], losses
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        boxes_all = self.box_coder.decode(box_regression, proposals)  # [sum N, C + 1, 4]
        sizes = [len(p) for p in proposals]
        results = []
        for i, (b, logit) in enumerate(zip(boxes_all.split(sizes), class_logits.split(sizes))):
            probs = image_probs[i] if image_probs is not None else None
            obj = objectness[i] if objectness is not None else None
            boxes, scores, labels = detections_from_raw(b[:, 1:], logit, probs, image_size=image_shapes[i],
                                                        objectness=obj, **self.post)
            out = {'boxes': boxes, 'scores': scores, 'labels': labels}
            if return_raw:
                out['raw_boxes'], out['raw_logits'] = b[:, 1:], logit
                if obj is not None:
                    out['raw_obj'] = obj
            results.append(out)
        return results, {}


class MultiTaskModel(nn.Module):
    """backbone -> classification logits and Faster R-CNN detections.
    cls_pool: 'mean' (reference: mean token -> LayerNorm -> Linear), 'clsmean' (CLS/first prefix token
    concatenated with the mean token), 'attn' (a learned query attends over the tokens; concatenated with the mean)."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, spatial_prior=0, anchor_sizes=REFERENCE_ANCHORS,
                 anchor_ratios=(0.5, 1.0, 2.0), box_head='mlp', cls_pool='mean', rpn_post_nms_test=1000,
                 roi_batch=512):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        self.hierarchical = getattr(backbone, 'hierarchical', False)
        self.cls_pool = cls_pool = 'mean' if self.hierarchical else cls_pool  # ConvNeXt: mean of the last map
        dim = backbone.embed_dim
        feat = dim * (2 if cls_pool in ('clsmean', 'attn') else 1)
        if cls_pool == 'attn':
            self.pool_query = nn.Parameter(torch.zeros(1, 1, dim))
            self.pool_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.cls_head = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, num_classes))
        if self.hierarchical:
            self.neck = HierarchicalNeck(backbone.channels)
        else:
            self.neck = SimpleFeaturePyramid(dim, spatial_prior=spatial_prior)
        anchors = AnchorGenerator(tuple(tuple(s) for s in anchor_sizes), (tuple(anchor_ratios),) * len(anchor_sizes))
        n_anchors = len(anchor_sizes[0]) * len(anchor_ratios)
        self.rpn = RPNKeepScores(
            anchors, RPNHead(256, n_anchors), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=rpn_post_nms_test),
            post_nms_top_n=dict(training=2000, testing=rpn_post_nms_test), nms_thresh=0.7)
        head = TwoMLPHead(256 * 7 * 7, 1024) if box_head == 'mlp' else SmallBoxHead()
        self.roi_heads = RoIHeadsPlus(
            MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2),
            head, FastRCNNPredictor(1024, num_classes + 1),  # + 1: background
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=roi_batch, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)
        self.roi_heads.post = dict(RoIHeadsPlus.post)  # per-instance copy, set by train.py / predict.py

    def pool(self, tokens):
        mean = tokens.mean(dim=1)
        if self.cls_pool == 'mean':
            return mean
        if self.cls_pool == 'clsmean':
            return torch.cat([self.backbone.prefix_tokens[:, 0], mean], dim=1)
        q = self.pool_query.expand(len(tokens), -1, -1).to(tokens.dtype)
        attended, _ = self.pool_attn(q, tokens, tokens, need_weights=False)
        return torch.cat([attended[:, 0], mean], dim=1)

    def forward(self, images, targets=None, return_raw=False):
        """images [B, 1, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the four Faster R-CNN losses (training mode))."""
        set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
        out = self.backbone(images)
        if self.hierarchical:
            logits = self.cls_head(out[-1].mean(dim=(2, 3)))
            features = self.neck(out, images)
        else:
            logits = self.cls_head(self.pool(out))
            if self.task_routing:
                set_task(self.backbone, 1)
                out = self.backbone(images)
            p = self.backbone.patch_size
            h, w = images.shape[-2] // p, images.shape[-1] // p  # one token per p x p patch
            maps = getattr(self.backbone, 'intermediates', None) if getattr(self.backbone, 'ml_indices', None) else None
            features = self.neck(out.transpose(1, 2).unflatten(2, (h, w)), images, maps)
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            if self.training:
                detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
            else:
                image_probs = logits.float().softmax(dim=1)
                detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes,
                                                        image_probs=image_probs, return_raw=return_raw,
                                                        objectness=self.rpn.last_scores)
        return logits, detections, {**rpn_losses, **roi_losses}
