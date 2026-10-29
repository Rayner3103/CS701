"""sarbench/cascade.py: stage losses and gradients, inference outputs, box refinement, and the parameter budget."""
import pytest
import torch
from torchvision.ops import MultiScaleRoIAlign

from sarbench.adapters import LORA_TARGETS, add_adapters, adapter_kwargs
from sarbench.backbones import build_backbone
from sarbench.cascade import CascadeRoIHeads
from sarbench.model import MultiTaskModel
from test_model import first_train_images, forward


def model(**kwargs):
    return MultiTaskModel(build_backbone('vit', pretrained=False), task_routing=False, cascade=True, **kwargs).cuda()


def test_training_returns_weighted_losses_for_every_stage_and_all_stages_get_gradients():
    net = model().train()
    images, _, targets = first_train_images(2)
    _, detections, losses = forward(net, images, targets)
    assert detections == []
    stage_losses = {f'loss_{k}{s}' for k in ('classifier', 'box_reg') for s in ('', '_s2', '_s3')}
    assert set(losses) == stage_losses | {'loss_objectness', 'loss_rpn_box_reg'}
    assert all(loss.dtype == torch.float32 and loss.isfinite() for loss in losses.values())
    sum(losses.values()).backward()
    assert all(p.grad is not None for stage in net.roi_heads.stages for p in stage.box_head.parameters())


def test_inference_returns_valid_detections():
    net = model().eval()
    images, _, _ = first_train_images(2)
    with torch.no_grad():
        _, detections, losses = forward(net, images)
    assert losses == {} and len(detections) == 2
    for d in detections:
        assert set(d) == {'boxes', 'labels', 'scores'} and len(d['boxes']) <= 100
        assert ((d['scores'] >= 0) & (d['scores'] <= 1)).all() and ((d['labels'] >= 1) & (d['labels'] <= 9)).all()
        assert (d['boxes'] >= 0).all() and (d['boxes'] <= 512).all()


def test_refine_with_zero_deltas_keeps_boxes_but_drops_added_ground_truth_and_tiny_boxes_in_training():
    heads = CascadeRoIHeads(MultiScaleRoIAlign(['0'], 7, 2))
    gt = torch.tensor([[10., 10., 50., 60.]])
    proposals = [torch.tensor([[10., 10., 50., 60.], [0., 0., 30., 30.], [5., 5., 5.5, 5.5], [100., 100., 140., 120.]])]
    zero = torch.zeros(4, 4)
    kept = heads.refine(heads.stages[0], proposals, zero, [(512, 512)], [{'boxes': gt}])[0]
    assert torch.equal(kept, proposals[0][[1, 3]])  # the ground-truth copy (row 0) and the 0.5 px box (row 2) go
    every = heads.refine(heads.stages[0], proposals, zero, [(512, 512)])[0]  # inference: nothing is dropped
    assert torch.allclose(every, proposals[0], atol=1e-3)


def test_refine_clips_to_the_image():
    heads = CascadeRoIHeads(MultiScaleRoIAlign(['0'], 7, 2))
    refined = heads.refine(heads.stages[0], [torch.tensor([[480., 480., 600., 600.]])], torch.zeros(1, 4), [(512, 512)])
    assert refined[0].max() <= 512


def test_cascade_with_lora_is_within_the_parameter_limits():
    backbone = build_backbone('dinov3', pretrained=False)
    add_adapters(backbone, 'lora')
    net = MultiTaskModel(backbone, task_routing=False, cascade=True)
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert total <= 130e6 and trainable <= 40e6
    assert trainable < 21.0e6  # the single TwoMLPHead model trains 21.0 M


@pytest.mark.parametrize('targets, trainable', [('attn', 884_736), ('attn+mlp', 884_736 + 12 * 2 * (768 * 16 + 16 * 3072))])
def test_lora_targets(targets, trainable):
    backbone = build_backbone('dinov3', pretrained=False)
    add_adapters(backbone, 'lora', **adapter_kwargs({'adapt': 'lora', 'lora_targets': targets}))
    assert sum(p.numel() for p in backbone.parameters() if p.requires_grad) == trainable


def test_old_runs_without_the_new_keys_get_the_old_lora():
    assert adapter_kwargs({'adapt': 'lora'}) == {'targets': LORA_TARGETS['attn'], 'rank': 16, 'alpha': 32}
    assert adapter_kwargs({'adapt': 'moelora'}) == {} and adapter_kwargs({'adapt': 'full'}) == {}


def test_checkpointing_the_stage_heads_changes_neither_losses_nor_gradients():
    plain = model().train()
    checkpointed = model().train()
    checkpointed.load_state_dict(plain.state_dict())
    checkpointed.roi_heads.checkpoint = True
    images, _, targets = first_train_images(2)
    grads = []
    for net in (plain, checkpointed):
        torch.manual_seed(0)  # the same proposals are sampled
        _, _, losses = forward(net, images, targets)
        sum(losses.values()).backward()
        grads.append(({k: v.item() for k, v in losses.items()},
                      [p.grad.clone() for p in net.roi_heads.parameters() if p.grad is not None]))
    assert grads[0][0] == pytest.approx(grads[1][0], rel=1e-4)
    assert len(grads[0][1]) == len(grads[1][1]) > 0
    assert all(torch.allclose(a, b, rtol=1e-3, atol=1e-6) for a, b in zip(grads[0][1], grads[1][1]))
