"""sarbench/dino_yolo.py: the parameter limits, the zero-initialised DINO fusion (training starts from the plain YOLO12),
the DINO learning-rate groups and the image class read from the detections. No data, no downloads (random DINO weights);
runs on the CPU."""
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel

from sarbench.dino_yolo import MAX_TOTAL, MAX_TRAINABLE, DinoYOLO, DinoYOLOTrainer, image_class, parameter_counts


def build():
    return DinoYOLO('yolo12s.yaml', nc=9, verbose=False, pretrained_dino=False)


def test_within_the_parameter_limits():
    total, trainable = parameter_counts(build())
    assert total <= MAX_TOTAL and trainable <= MAX_TRAINABLE
    assert 30e6 < total < 33e6  # DINOv3 ViT-S/16 21.6 M + YOLO12-s 9.3 M + the 1x1 projections


def test_fusion_starts_as_the_plain_yolo12_and_then_uses_the_dino_features():
    model = build().eval()
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
    optimizer = trainer.build_optimizer(model, 'AdamW', lr=1e-3)
    dino = {id(p) for p in model.dino.parameters()}
    for group in optimizer.param_groups:
        in_dino = {id(p) in dino for p in group['params']}
        assert len(in_dino) <= 1  # no group mixes DINO and YOLO parameters
        if in_dino == {True}:
            assert group['lr'] == 1e-4 and group['param_group'].startswith('dino_')
    assert sum(len(g['params']) for g in optimizer.param_groups) == len(list(model.parameters()))


def test_image_class_is_the_class_with_the_highest_total_score():
    assert image_class(np.array([7, 2, 2]), np.array([0.9, 0.5, 0.5])) == 2
    assert image_class(np.array([], dtype=int), np.array([])) == 0
