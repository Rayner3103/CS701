"""DINO-YOLO: YOLO12 or YOLO26 whose backbone also gets the features of a DINOv3 ViT run on the same image, after the
DINOv3 + YOLOv12 design of Sompote/DINOV3-YOLOV12, built on the official ultralytics package. DinoYOLOTrainer with
dino=None trains the plain YOLO (a baseline), with the same limits check and protocol.
The image class comes from an optional classification head on the DINO features (cls_head=True), or else from the
detections: every box of an image has the image's class (image_class)."""
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

# Assignment limits (README 1.2): frozen parameters count towards the total
MAX_TOTAL, MAX_TRAINABLE = 130e6, 40e6


def neck_inputs(cfg):
    """The backbone layers whose outputs the head reads: the last one and those it concatenates. YOLO12: 4, 6, 8
    (P3/8, P4/16, P5/32); YOLO26: 4, 6, 10; YOLO26-p2: 2, 4, 6, 10 (P2/4 too)."""
    n = len(cfg['backbone'])
    read = {j for f, *_ in cfg['head'] for j in ([f] if isinstance(f, int) else f) if 0 <= j < n}
    return sorted(read | {n - 1})


class DinoYOLO(DetectionModel):
    """YOLO12 / YOLO26 + a DINOv3 ViT. The ViT's stride-16 map is resized to the outputs of the backbone layers `fuse`
    (default: the ones the head reads, neck_inputs), projected by 1x1 convs and added to them. The convs start at
    zero, so training starts from exactly the plain (COCO-pretrained) YOLO.
    cls_head=True adds an image classifier: LayerNorm + Linear on the ViT's CLS token and mean patch token (the DINOv2
    linear probe), trained with a multi-label loss of weight cls_weight (loss)."""

    def __init__(self, cfg='yolo12s.yaml', ch=3, nc=None, verbose=True, dino='vit_small_patch16_dinov3.lvd1689m',
                 fuse=None, pretrained_dino=True, cls_head=False, cls_weight=1.0):
        super().__init__(cfg, ch=ch, nc=nc, verbose=False)
        # A checkpoint stores self.yaml, and resume and predict rebuild the model from it: it carries the DINO settings
        dino = self.yaml.setdefault('dino', dino)
        fuse = self.yaml.setdefault('dino_fuse', list(fuse) if fuse else neck_inputs(self.yaml))
        cls_head = self.yaml.setdefault('dino_cls_head', cls_head)
        self.yaml.setdefault('dino_cls_weight', cls_weight)
        channels = self._output_channels(fuse, ch)
        self.dino = timm.create_model(dino, pretrained=pretrained_dino, num_classes=0, dynamic_img_size=True)
        mean, std = self.dino.pretrained_cfg['mean'], self.dino.pretrained_cfg['std']
        self.register_buffer('dino_mean', torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('dino_std', torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.dino_proj = nn.ModuleDict({str(i): nn.Conv2d(self.dino.embed_dim, c, 1) for i, c in channels.items()})
        for conv in self.dino_proj.values():
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
        d = self.dino.embed_dim
        self.cls_head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, self.yaml['nc'])) if cls_head else None
        if verbose:
            self.info()

    def _output_channels(self, layers, ch):
        """{layer index: its output channels}, from one forward pass of the plain YOLO."""
        channels, hooks = {}, []
        for i in layers:
            hooks.append(self.model[i].register_forward_hook(lambda m, _, out, i=i: channels.__setitem__(i, out.shape[1])))
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self._predict_once(torch.zeros(1, ch, 64, 64))
        self.train(was_training)
        for h in hooks:
            h.remove()
        return channels

    def dino_features(self, x):
        """images [B, 3, H, W] in [0, 1] (H, W multiples of 16) -> the last ViT block: patch tokens as a map
        [B, embed_dim, H/16, W/16] and the CLS token [B, embed_dim]."""
        x = (x - self.dino_mean) / self.dino_std
        tokens, prefix = self.dino.forward_intermediates(x, indices=1, norm=True, output_fmt='NCHW',
                                                         intermediates_only=True, return_prefix_tokens=True)[0]
        return tokens, prefix[:, 0]

    def _predict_once(self, x, profile=False, embed=None):
        """DetectionModel._predict_once (ultralytics 8.4.170) with the DINO features added after the `fuse` layers.
        With the classification head, its logits [B, nc] are left in self.cls_logits for loss() and predict()."""
        dino = None  # stays None while __init__ builds the YOLO
        if hasattr(self, 'dino_proj'):
            dino, cls_token = self.dino_features(x)
            if getattr(self, 'cls_head', None) is not None:  # getattr: models saved before the head have no attribute
                self.cls_logits = self.cls_head(torch.cat([cls_token, dino.mean((2, 3))], 1))
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            if dino is not None and str(m.i) in self.dino_proj:
                # bilinear: x2 up to P3, as is at P4, the mean of 2 x 2 at P5
                x = x + self.dino_proj[str(m.i)](F.interpolate(dino, size=x.shape[-2:], mode='bilinear'))
            y.append(x if m.i in self.save else None)  # save output
            if m.i in embed:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """The detection loss, plus cls_weight x the image-class loss when there is a classification head. Mosaic puts
        up to four images into one, so the target is every class with a box in it (multi-label, binary cross-entropy);
        after close_mosaic that is the image's one class."""
        if getattr(self, 'cls_head', None) is None:
            return super().loss(batch, preds)
        if preds is None:
            preds = self.forward(batch['img'])
        loss, items = super().loss(batch, preds)
        logits = self.__dict__.pop('cls_logits').float()  # pop: the graph is not kept on the module
        target = torch.zeros_like(logits)
        target[batch['batch_idx'].long(), batch['cls'].view(-1).long()] = 1
        img_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='sum') / len(logits)
        return loss.sum() + self.yaml['dino_cls_weight'] * img_loss * len(logits), {**items, 'img_loss': img_loss.detach()}


def parameter_counts(model):
    """(total, trainable) parameters."""
    params = list(model.parameters())
    return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad)


def check_limits(model):
    total, trainable = parameter_counts(model)
    if total > MAX_TOTAL or trainable > MAX_TRAINABLE:
        raise ValueError(f'{total / 1e6:.1f} M parameters ({trainable / 1e6:.1f} M trainable): over the limits of '
                         f'{MAX_TOTAL / 1e6:.0f} M ({MAX_TRAINABLE / 1e6:.0f} M trainable)')
    return total, trainable


class DinoYOLOTrainer(DetectionTrainer):
    """DetectionTrainer that builds DinoYOLO, checks the parameter limits, and gives the DINO ViT dino_lr_mult x the
    learning rate: a pretrained ViT is fine-tuned gently (the reference full fine-tuning uses 2e-5 vs 1e-4).
    dino=None builds the plain YOLO DetectionModel instead."""

    def __init__(self, *args, dino='vit_small_patch16_dinov3.lvd1689m', fuse=None, dino_lr_mult=0.1,
                 cls_head=False, cls_weight=1.0, **kwargs):
        # before super(): it may build the model
        self.dino, self.fuse, self.dino_lr_mult, self.cls_head, self.cls_weight = dino, fuse, dino_lr_mult, cls_head, cls_weight
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # On resume cfg is the checkpoint's yaml: its 'dino' (None for a plain YOLO) wins over the trainer's
        dino = cfg.get('dino', self.dino) if isinstance(cfg, dict) else self.dino
        verbose = verbose and RANK == -1
        if dino is None:
            model = DetectionModel(cfg, ch=self.data['channels'], nc=self.data['nc'], verbose=verbose)
            model.yaml['dino'] = None
        else:
            model = DinoYOLO(cfg, ch=self.data['channels'], nc=self.data['nc'], verbose=verbose, dino=dino,
                             fuse=self.fuse, cls_head=self.cls_head, cls_weight=self.cls_weight)
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)  # COCO-pretrained YOLO12 (all but the class outputs), or a checkpoint to resume
        check_limits(model)
        return model

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        if not hasattr(unwrap_model(model), 'dino'):  # a plain YOLO
            return optimizer
        dino = {id(p) for p in unwrap_model(model).dino.parameters()}
        for group in list(optimizer.param_groups):
            moved = [p for p in group['params'] if id(p) in dino]
            if moved:
                group['params'] = [p for p in group['params'] if id(p) not in dino]
                # a new param_group name: the warmup treats 'bias' groups specially
                optimizer.add_param_group({**{k: v for k, v in group.items() if k != 'params'}, 'params': moved,
                                           'lr': group['lr'] * self.dino_lr_mult,
                                           'param_group': f"dino_{group.get('param_group')}"})
        return optimizer


def image_class(classes, scores, logits=None, num_classes=9):
    """The image class: the classification head's argmax if the model has one (logits), else read from the detections
    (classes 0..8, scores): the class with the highest total score. An image without detections gets class 0."""
    if logits is not None:
        return int(logits.argmax())
    return int(np.bincount(classes, weights=scores, minlength=num_classes).argmax())


def predict(model, paths, imgsz=512, batch=16, conf=0.001, iou=0.7, max_det=100):
    """Yield (boxes XYXY in original pixels [N, 4], classes 0..8 [N], scores [N], class logits [nc] or None without a
    classification head) per image, in the order of `paths`. conf 0.001 and 100 boxes per image: as COCO mAP is
    computed. No test-time augmentation."""
    for start in range(0, len(paths), batch):  # ultralytics runs a list of paths as one batch: one forward pass
        results = model.predict(paths[start:start + batch], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                                verbose=False)
        logits = getattr(model.predictor.model.backend.model, 'cls_logits', None)  # left by that forward pass
        for i, r in enumerate(results):
            b = r.boxes
            yield (b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy(),
                   None if logits is None else logits[i].float().cpu().numpy())
