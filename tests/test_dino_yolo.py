"""sarbench/dino_yolo.py: the parameter limits, the zero-initialised DINO fusion (training starts from the plain YOLO12
or YOLO26), the DINO learning-rate groups, the classification head, the frozen / LoRA DINO, the plain-YOLO trainer and
the image class. No data, no downloads (random DINO weights); runs on the CPU."""
import random

import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel, yaml_model_load

from ultralytics.utils.instance import Instances

from sarbench.adapters import LoRA
from sarbench.dino_yolo import (MAX_TOTAL, MAX_TRAINABLE, DinoYOLO, DinoYOLOTrainer, RandomRot90, check_limits,
                                image_class, neck_inputs, parameter_counts)

YOLOS = ['yolo12s.yaml', 'yolo26s.yaml']
VIT_B = 'vit_base_patch16_dinov3.lvd1689m'


def build(cfg='yolo12s.yaml'):
    return DinoYOLO(cfg, nc=9, verbose=False, pretrained_dino=False)


def test_the_dino_features_go_to_the_backbone_layers_the_head_reads():
    assert neck_inputs(yaml_model_load('yolo12s.yaml')) == [4, 6, 8]
    assert neck_inputs(yaml_model_load('yolo26s.yaml')) == [4, 6, 10]  # 10: C2PSA after SPPF, the P5/32 output
    assert neck_inputs(yaml_model_load('yolo26s-p2.yaml')) == [2, 4, 6, 10]
    assert set(build('yolo26s.yaml').dino_proj) == {'4', '6', '10'}


@pytest.mark.parametrize('cfg', YOLOS)
def test_within_the_parameter_limits(cfg):
    total, trainable = parameter_counts(build(cfg))
    assert total <= MAX_TOTAL and trainable <= MAX_TRAINABLE
    assert 30e6 < total < 33e6  # DINOv3 ViT-S/16 21.6 M + YOLO12-s 9.3 M / YOLO26-s 10.0 M + the 1x1 projections


@pytest.mark.parametrize('cfg', YOLOS)
def test_fusion_starts_as_the_plain_yolo_and_then_uses_the_dino_features(cfg):
    model = build(cfg).eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        plain = DetectionModel._predict_once(model, x)[0]
        assert torch.equal(model(x)[0], plain)
        for conv in model.dino_proj.values():  # random, not constant: LayerNorm-ed DINO channels sum to 0
            torch.nn.init.normal_(conv.weight, std=0.01)
        assert not torch.allclose(model(x)[0], plain)


def test_dino_settings_travel_with_the_model_yaml():
    model = DinoYOLO(build().yaml, nc=9, verbose=False, pretrained_dino=False, fuse=(6,))
    assert model.yaml['dino_fuse'] == [4, 6, 8] and set(model.dino_proj) == {'4', '6', '8'}


def test_dino_parameters_get_their_own_lower_learning_rate():
    model = build()
    trainer = DinoYOLOTrainer.__new__(DinoYOLOTrainer)  # only build_optimizer is used: no data, no run folder
    trainer.dino_lr_mult, trainer.args, trainer.data = 0.1, type('Args', (), {'lr0': 1e-3})(), {'nc': 9}
    trainer.unfreeze_lr_mult = 0.02
    optimizer = trainer.build_optimizer(model, 'AdamW', lr=1e-3)
    dino = {id(p) for p in model.dino.parameters()}
    for group in optimizer.param_groups:
        in_dino = {id(p) in dino for p in group['params']}
        assert len(in_dino) <= 1  # no group mixes DINO and YOLO parameters
        if in_dino == {True}:
            assert group['lr'] == 1e-4 and group['param_group'].startswith('dino_')
    # every trainable parameter (not the fixed DFL conv of YOLO12)
    assert sum(len(g['params']) for g in optimizer.param_groups) == len([p for p in model.parameters() if p.requires_grad])


def build_vit_b(adapt, **kwargs):
    return DinoYOLO('yolo26s.yaml', nc=9, verbose=False, pretrained_dino=False, dino=VIT_B, dino_adapt=adapt, **kwargs)


@pytest.mark.parametrize('adapt, targets, lora', [('frozen', ('attn.qkv', 'attn.proj'), 0),
                                                  # 12 blocks x rank 16 x ((768 + 2304) + (768 + 768)), as train.py
                                                  ('lora', ('attn.qkv', 'attn.proj'), 884_736),
                                                  # + 12 x 16 x 2 x (768 + 3072) for mlp.fc1, mlp.fc2
                                                  ('lora', ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'), 2_359_296)])
def test_a_frozen_vit_b_counts_towards_the_total_but_only_lora_trains(adapt, targets, lora):
    model = build_vit_b(adapt, lora_targets=targets)
    total, trainable = parameter_counts(model)
    vit = sum(p.numel() for p in model.dino.parameters())
    assert total <= MAX_TOTAL and trainable <= MAX_TRAINABLE
    assert 85e6 < vit - lora < 86e6 and total > vit + 10e6  # ViT-B/16 + YOLO26-s 10 M
    assert sum(p.numel() for p in model.dino.parameters() if p.requires_grad) == lora
    assert trainable == total - vit + lora  # all of the YOLO and the fusion convs train
    assert all(n.endswith(('.down.weight', '.up.weight')) for n, p in model.dino.named_parameters() if p.requires_grad)


def test_lora_starts_as_the_frozen_vit():
    torch.manual_seed(0)
    frozen = build_vit_b('frozen').eval()
    torch.manual_seed(0)  # the same YOLO and ViT weights: the LoRA layers are drawn after them
    lora = build_vit_b('lora').eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(frozen.dino_features(x)[0], lora.dino_features(x)[0], atol=1e-5)


def test_a_frozen_vit_keeps_no_graph_and_lora_gets_gradients():
    x = torch.rand(1, 3, 64, 64)
    assert not build_vit_b('frozen').dino_features(x)[0].requires_grad
    model = build_vit_b('lora')
    for m in model.dino.modules():
        if isinstance(m, LoRA):
            torch.nn.init.normal_(m.up.weight, std=0.01)
    model.dino_features(x)[0].sum().backward()
    assert model.dino.blocks[0].attn.qkv.down.weight.grad.abs().sum() > 0
    assert model.dino.blocks[0].attn.qkv.base.weight.grad is None


def test_lora_settings_travel_with_the_model_yaml():
    model = build_vit_b('lora', lora_rank=8, lora_targets=('mlp.fc1',))
    rebuilt = DinoYOLO(model.yaml, nc=9, verbose=False, pretrained_dino=False)  # as on resume and predict
    assert isinstance(rebuilt.dino.blocks[0].mlp.fc1, LoRA) and rebuilt.dino.blocks[0].mlp.fc1.down.out_features == 8
    assert not isinstance(rebuilt.dino.blocks[0].attn.qkv, LoRA)
    assert parameter_counts(rebuilt) == parameter_counts(model)


def test_the_trainer_freezes_the_vit_again_and_leaves_it_out_of_the_optimizer():
    model = build_vit_b('lora')
    model.requires_grad_(True)  # as ultralytics _setup_train does before building the optimizer
    trainer = DinoYOLOTrainer.__new__(DinoYOLOTrainer)
    trainer.dino_lr_mult, trainer.args, trainer.data = 0.1, type('Args', (), {'lr0': 1e-3})(), {'nc': 9}
    trainer.unfreeze_lr_mult = 0.02
    optimizer = trainer.build_optimizer(model, 'AdamW', lr=1e-3)
    in_optimizer = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert in_optimizer == {id(p) for p in model.parameters() if p.requires_grad}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) < MAX_TRAINABLE
    lora = {id(p) for m in model.dino.modules() if isinstance(m, LoRA) for p in m.parameters() if p.requires_grad}
    for group in optimizer.param_groups:
        if any(id(p) in lora for p in group['params']):
            assert group['lr'] == 1e-4 and group['param_group'].startswith('dino_')


BLOCK = 7_087_104  # one ViT-B/16 block
NORM = 1_536


def test_frozen_vit_with_unfrozen_last_blocks_trains_them_and_the_norm():
    frozen = build_vit_b('frozen')
    model = build_vit_b('frozen', unfreeze=2)
    extra = parameter_counts(model)[1] - parameter_counts(frozen)[1]
    assert extra == 2 * BLOCK + NORM
    assert parameter_counts(model)[0] == parameter_counts(frozen)[0]
    assert all(p.requires_grad for p in model.dino.blocks[-1].parameters())
    assert not any(p.requires_grad for p in model.dino.blocks[-3].parameters())
    assert model.dino.norm.weight.requires_grad


def test_unfrozen_blocks_get_gradients_and_the_frozen_prefix_none():
    model = build_vit_b('frozen', unfreeze=2)
    model.dino_features(torch.rand(1, 3, 64, 64))[0].sum().backward()
    assert model.dino.blocks[-1].attn.qkv.weight.grad.abs().sum() > 0
    assert model.dino.blocks[-2].mlp.fc1.weight.grad.abs().sum() > 0
    assert model.dino.blocks[0].attn.qkv.weight.grad is None  # frozen prefix: no graph


def test_lora_goes_only_on_the_blocks_before_the_unfrozen_ones():
    model = build_vit_b('lora', unfreeze=2)
    for i, blk in enumerate(model.dino.blocks):
        assert isinstance(blk.attn.qkv, LoRA) == (i < 10) and isinstance(blk.attn.proj, LoRA) == (i < 10)
    lora = sum(p.numel() for m in model.dino.modules() if isinstance(m, LoRA) for p in m.parameters() if p.requires_grad)
    assert lora == 884_736 * 10 // 12
    assert sum(p.numel() for p in model.dino.parameters() if p.requires_grad) == lora + 2 * BLOCK + NORM
    assert all(p.requires_grad for p in model.dino.blocks[-1].attn.qkv.parameters())


def test_unfreeze_over_the_trainable_limit_or_with_full_adapt_is_refused():
    with pytest.raises(ValueError, match='over the limits'):
        check_limits(build_vit_b('frozen', unfreeze=5))  # 10.7 M + 5 x 7.1 M > 40 M
    check_limits(build_vit_b('frozen', unfreeze=4))
    with pytest.raises(ValueError, match='dino_unfreeze'):
        build_vit_b('full', unfreeze=1)
    with pytest.raises(ValueError, match='dino_unfreeze'):
        build_vit_b('frozen', unfreeze=13)


def test_unfreeze_settings_travel_with_the_model_yaml():
    model = build_vit_b('lora', unfreeze=3)
    rebuilt = DinoYOLO(model.yaml, nc=9, verbose=False, pretrained_dino=False)  # as on resume and predict
    assert rebuilt.yaml['dino_unfreeze'] == 3 and parameter_counts(rebuilt) == parameter_counts(model)
    del model.yaml['dino_unfreeze']  # a model saved before dino_unfreeze
    old = DinoYOLO(model.yaml, nc=9, verbose=False, pretrained_dino=False)
    assert old.yaml['dino_unfreeze'] == 0 and old.unfrozen_parameters() == []


def test_unfrozen_blocks_train_at_their_own_learning_rate_and_lora_at_the_other():
    model = build_vit_b('lora', unfreeze=2)
    model.requires_grad_(True)
    trainer = DinoYOLOTrainer.__new__(DinoYOLOTrainer)
    trainer.dino_lr_mult, trainer.unfreeze_lr_mult = 0.1, 0.02
    trainer.args, trainer.data = type('Args', (), {'lr0': 1e-3})(), {'nc': 9}
    optimizer = trainer.build_optimizer(model, 'AdamW', lr=1e-3)
    unfrozen = {id(p) for p in model.unfrozen_parameters()}
    lora = {id(p) for m in model.dino.modules() if isinstance(m, LoRA) for p in m.parameters()}
    seen = []
    for group in optimizer.param_groups:
        ids = {id(p) for p in group['params']}
        seen += [id(p) for p in group['params']]
        if ids & unfrozen:
            assert ids <= unfrozen and group['lr'] == pytest.approx(2e-5) and group['param_group'].startswith('dino_unf')
        if ids & lora:
            assert ids <= lora and group['lr'] == pytest.approx(1e-4)
    assert len(seen) == len(set(seen)) == len([p for p in model.parameters() if p.requires_grad])
    assert unfrozen <= set(seen)


def test_image_class_is_the_class_with_the_highest_total_score():
    assert image_class(np.array([7, 2, 2]), np.array([0.9, 0.5, 0.5])) == 2
    assert image_class(np.array([], dtype=int), np.array([])) == 0


def batch_of_two():
    return {'img': torch.rand(2, 3, 64, 64), 'cls': torch.tensor([[0.], [3.], [3.]]), 'batch_idx': torch.tensor([0., 1., 1.]),
            'bboxes': torch.tensor([[.5, .5, .3, .3], [.3, .3, .2, .2], [.7, .7, .2, .2]])}


@pytest.mark.parametrize('cfg', YOLOS)
def test_classification_head_adds_an_image_loss_that_trains_the_head(cfg):
    model = DinoYOLO(cfg, nc=9, verbose=False, pretrained_dino=False, cls_head=True)
    model.args = get_cfg()  # the loss hyperparameters, as the trainer sets them
    loss, items = model.loss(batch_of_two())  # YOLO26: the end-to-end loss (one-to-many + one-to-one)
    assert {'box_loss', 'cls_loss', 'img_loss'} <= set(items) and loss.ndim == 0
    loss.backward()
    assert model.cls_head[1].weight.grad.abs().sum() > 0
    assert 'cls_logits' not in model.__dict__  # the graph is not kept on the module
    model.eval()
    with torch.no_grad():
        model(torch.rand(1, 3, 64, 64))
    assert model.cls_logits.shape == (1, 9)
    assert image_class(np.array([1]), np.array([0.9]), logits=np.eye(9)[4]) == 4


def test_models_saved_before_the_classification_head_still_run():
    model = build()
    del model.__dict__['cls_head']  # as unpickled from a checkpoint of the first version
    model.args = get_cfg()
    assert 'img_loss' not in model.loss(batch_of_two())[1]
    model.eval()
    with torch.no_grad():
        model(torch.rand(1, 3, 64, 64))
    assert not hasattr(model, 'cls_logits')


def test_without_dino_the_trainer_builds_the_plain_yolo_also_on_resume():
    trainer = DinoYOLOTrainer.__new__(DinoYOLOTrainer)  # only get_model is used: no data, no run folder
    trainer.args, trainer.data = get_cfg(), {'channels': 3, 'nc': 9, 'names': {i: str(i) for i in range(9)}}
    trainer.dino, trainer.fuse, trainer.cls_head, trainer.cls_weight = None, None, False, 1.0
    trainer.dino_adapt, trainer.lora_rank, trainer.lora_targets, trainer.grad_ckpt = 'full', 16, ('attn.qkv', 'attn.proj'), False
    trainer.unfreeze = 0
    model = trainer.get_model('yolo26s.yaml', verbose=False)
    assert type(model) is DetectionModel and model.yaml['dino'] is None
    trainer.dino = 'vit_small_patch16_dinov3.lvd1689m'  # the default of train_yolo.py on --resume
    assert type(trainer.get_model(model.yaml, verbose=False)) is DetectionModel  # the checkpoint's yaml wins


@pytest.mark.parametrize('normalized', [False, True])
def test_rot90_turns_the_boxes_with_the_image(normalized):
    """A white box on a black, non-square image: after every rotation the box still covers exactly the white pixels."""
    h, w = 40, 60
    seen = set()
    for seed in range(20):
        img = np.zeros((h, w, 3), np.uint8)
        img[5:15, 10:30] = 255  # x 10..30, y 5..15
        box = np.array([[20., 10., 20., 10.]])  # xywh
        if normalized:
            box /= [w, h, w, h]
        labels = {'img': img, 'instances': Instances(box.copy(), normalized=normalized, bbox_format='xywh')}
        random.seed(seed)
        out = RandomRot90(p=1.0)(labels)
        img, inst = out['img'], out['instances']
        oh, ow = img.shape[:2]
        seen.add((oh, ow))
        cx, cy, bw, bh = inst.bboxes[0] * ([ow, oh, ow, oh] if normalized else 1)
        x1, y1, x2, y2 = (round(v) for v in (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        ys, xs = np.nonzero(img[..., 0])
        assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (x1, y1, x2, y2)
    assert seen == {(h, w), (w, h)}  # 180 and 90 / 270 degrees all came up
