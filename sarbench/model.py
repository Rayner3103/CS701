"""The multi-task model: one shared ViT backbone feeds a classification head and a Faster R-CNN detector."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign

from .adapters import set_task


def conv_gn(c_in, c_out, stride=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(32, c_out),
                         nn.GELU())


class SpatialPrior(nn.Module):
    """A small CNN on the raw image, after ViT-Adapter's spatial prior module (Chen et al., 2023): real pixel detail
    at strides 4 and 8, where the ViT's stride-16 tokens can only be upsampled. image [B, 1, H, W] ->
    ([B, 64, H/4, W/4], [B, 128, H/8, W/8])."""

    def __init__(self, in_chans=1):
        super().__init__()
        self.stride4 = nn.Sequential(conv_gn(in_chans, 64, stride=2), conv_gn(64, 64), conv_gn(64, 64),
                                     nn.MaxPool2d(3, stride=2, padding=1))
        self.stride8 = conv_gn(64, 128, stride=2)

    def forward(self, images):
        c4 = self.stride4(images)
        return c4, self.stride8(c4)


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the single stride-16 ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64.
    With spatial_prior=True, SpatialPrior features are added to the upsampled maps at strides 4 and 8."""

    def __init__(self, dim=768, channels=256, spatial_prior=False):
        super().__init__()
        self.prior = None
        if spatial_prior:
            # A forked RNG: building the prior leaves the global random state untouched, so with the same seed every
            # other weight, the data order and the flips are the same with and without the prior.
            with torch.random.fork_rng(devices=[]):
                self.prior = SpatialPrior()
                self.lateral = nn.ModuleList([nn.Conv2d(64, dim // 4, 1), nn.Conv2d(128, dim // 2, 1)])
            for conv in self.lateral:  # zero: training starts from exactly the plain ViTDet neck
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

    def forward(self, x, images=None):
        """x: the ViT map [B, dim, H/16, W/16]; images [B, 1, H, W] are needed only with the spatial prior."""
        scaled = [rescale(x) for rescale in self.rescale]
        if self.prior is not None:
            for i, (prior, lateral) in enumerate(zip(self.prior(images), self.lateral)):
                scaled[i] = scaled[i] + lateral(prior)
        levels = [project(s) for s, project in zip(scaled, self.project)]
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and Faster R-CNN detections."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, spatial_prior=False):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        self.neck = SimpleFeaturePyramid(backbone.embed_dim, spatial_prior=spatial_prior)
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location.
        anchors = AnchorGenerator(((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), ((0.5, 1.0, 2.0),) * 5)
        # Everything else is torchvision's Faster R-CNN default.
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(256, 6), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000), nms_thresh=0.7)
        self.roi_heads = RoIHeads(
            MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2),
            TwoMLPHead(256 * 7 * 7, 1024), FastRCNNPredictor(1024, num_classes + 1),  # + 1: background
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)

    def forward(self, images, targets=None):
        """images [B, 1, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the four Faster R-CNN losses (training mode))."""
        set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
        tokens = self.backbone(images)
        logits = self.cls_head(tokens.mean(dim=1))
        if self.task_routing:
            set_task(self.backbone, 1)
            tokens = self.backbone(images)
        h, w = images.shape[-2] // 16, images.shape[-1] // 16  # one token per 16 x 16 patch
        features = self.neck(tokens.transpose(1, 2).unflatten(2, (h, w)), images)  # tokens as a [B, 768, h, w] map
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}
