"""CPU checks for classification-only RoI context; no backbone, data or downloads needed."""
import json

import pytest
import torch
from torch import nn

from sarbench.tiny import AgnosticPredictor, CascadeRoIHeads


class ControlledPool(nn.Module):
    """Supply independent local/context features for small proposals and their enlarged RoIs."""

    def forward(self, features, boxes, image_shapes):
        boxes = torch.cat(boxes)
        enlarged = ((boxes[:, 2:] - boxes[:, :2]).amax(1) >= 24)[:, None, None, None]
        return torch.where(enlarged, features['context'], features['local'])


def inputs(context_value=1.0):
    features = {'local': torch.ones(1, 256, 7, 7),
                'context': torch.full((1, 256, 7, 7), context_value)}
    proposals = [torch.tensor([[20., 20., 28., 28.], [42., 42., 50., 50.]])]
    targets = [{'boxes': torch.tensor([[20., 20., 28., 28.]]), 'labels': torch.tensor([1])}]
    return features, proposals, [(64, 64)], targets


def make_heads(mode=None, **kwargs):
    options = dict(num_classes=3, reduced=2, representation=8, context=3.0,
                   batch_size_per_image=8, positive_fraction=0.5, iou_weight=0.5)
    options.update(kwargs)
    if mode is not None:
        options['context_mode'] = mode
    model = CascadeRoIHeads(ControlledPool(), **options)
    # Positive hidden activations make the sensitivity checks deterministic, including all three stages.
    with torch.no_grad():
        for head, pred in zip(model.stage_heads, model.stage_preds):
            for conv in (head.reduce[0], head.ctx_reduce[0]):
                conv.weight.fill_(1 / 256)
                conv.bias.fill_(0.2)
            head.fc6.weight.fill_(0.01)
            head.fc6.bias.fill_(0.2)
            head.fc7.weight.fill_(0.05)
            head.fc7.bias.fill_(0.2)
            pred.cls_score.weight.copy_(torch.arange(1, 4)[:, None].expand(3, 8) * 0.01)
            pred.bbox_pred.weight.fill_(0.001)
        model.iou_head.weight.fill_(0.01)
    return model


def evaluate(model, context_value=1.0):
    features, proposals, shapes, _ = inputs(context_value)
    with torch.no_grad():
        output, losses = model.eval()(features, proposals, shapes, return_raw=True)
    assert losses == {}
    assert len(output) == 1
    assert all(value.isfinite().all() for value in output[0].values())
    return output[0]


def test_predictor_default_preserves_legacy_regression():
    pred = AgnosticPredictor(8, 3)
    features = torch.randn(4, 8, 1, 1)
    scores, boxes = pred(features)
    torch.testing.assert_close(scores, pred.cls_score(features.flatten(1)), rtol=0, atol=0)
    torch.testing.assert_close(boxes, pred.bbox_pred(features.flatten(1)).repeat(1, 3), rtol=0, atol=0)


def test_default_shared_and_classification_have_compatible_state():
    legacy = make_heads()
    shared, classification = make_heads('shared'), make_heads('classification')
    assert legacy.context_mode == 'shared'
    state = legacy.state_dict()
    for model in (shared, classification):
        assert set(model.state_dict()) == set(state)
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in legacy.parameters())
        model.load_state_dict(state, strict=True)
    before, after = evaluate(legacy), evaluate(shared)
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['shared', 'classification'])
def test_context_changes_classification_but_only_shared_changes_boxes_and_quality(mode):
    model = make_heads(mode)
    predictions = []
    hooks = [pred.register_forward_hook(lambda module, args, result:
                                       predictions.append(tuple(t.detach().clone() for t in result)))
             for pred in model.stage_preds]
    try:
        original = evaluate(model, 1.0)
        changed = evaluate(model, 3.0)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(predictions) == 6
    for (cls_before, reg_before), (cls_after, reg_after) in zip(predictions[:3], predictions[3:]):
        assert not torch.allclose(cls_before, cls_after)
        if mode == 'classification':
            torch.testing.assert_close(reg_before, reg_after, rtol=0, atol=0)
        else:
            assert not torch.allclose(reg_before, reg_after)
    assert not torch.allclose(original['raw_logits'], changed['raw_logits'])
    for field in ('raw_boxes', 'raw_stage', 'raw_iou'):
        if mode == 'classification':
            torch.testing.assert_close(original[field], changed[field], rtol=0, atol=0)
        else:
            assert not torch.allclose(original[field], changed[field])


@pytest.mark.parametrize('partial_neg', [False, True])
@pytest.mark.parametrize('loss_kind', ['box', 'quality', 'classification'])
def test_training_gradients_keep_local_losses_out_of_context(loss_kind, partial_neg):
    model = make_heads('classification', partial_neg=partial_neg).train()
    features, proposals, shapes, targets = inputs()
    detections, losses = model(features, proposals, shapes, targets)
    assert detections == []
    assert len(losses) == 7
    assert all(loss.ndim == 0 and loss.isfinite() for loss in losses.values())
    if loss_kind == 'box':
        loss = sum(value for key, value in losses.items() if key.startswith('loss_box_reg'))
    elif loss_kind == 'quality':
        loss = losses['loss_iou']
    else:
        loss = sum(value for key, value in losses.items() if key.startswith('loss_classifier'))
    loss.backward()
    for head in model.stage_heads:
        gradients = [parameter.grad for parameter in head.ctx_reduce.parameters()]
        if loss_kind == 'classification':
            assert all(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
        else:
            assert all(gradient is None or torch.count_nonzero(gradient) == 0 for gradient in gradients)
    assert model.stage_heads[-1].reduce[0].weight.grad.abs().sum() > 0
    if loss_kind == 'box':
        assert all(pred.bbox_pred.weight.grad.abs().sum() > 0 for pred in model.stage_preds)


@pytest.mark.parametrize('mode,context', [('invalid', 3), ('classification', 0), ('classification', -1)])
def test_invalid_context_configuration_is_rejected(mode, context):
    with pytest.raises(ValueError, match='context'):
        CascadeRoIHeads(ControlledPool(), context=context, context_mode=mode)


CLI_REQUIRED = ['--backbone', 'vit', '--init', 'scratch', '--adapt', 'full']


@pytest.mark.parametrize('flags', [
    ['--roi-context', '3', '--roi-context-mode', 'classification'],
    ['--cascade', '--roi-context-mode', 'classification'],
    ['--cascade', '--roi-context', '-1', '--roi-context-mode', 'classification'],
    ['--cascade', '--roi-context', '3', '--roi-context-mode', 'invalid'],
])
def test_cli_rejects_invalid_context_combinations(flags):
    import train

    with pytest.raises(SystemExit) as error:
        train.parse_args(CLI_REQUIRED + flags)
    assert error.value.code == 2


@pytest.mark.parametrize('mode', ['legacy', 'classification'])
def test_cli_mode_survives_checkpoint_restore_without_backbone(tmp_path, monkeypatch, mode):
    import predict
    import train

    class DetectorShell(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.roi_heads = nn.Module()
            self.roi_heads.box_roi_pool = ControlledPool()

    # Exercise the real Cascade wiring and predict.load_run while excluding the large pretrained model.
    monkeypatch.setattr(train, 'build_backbone', lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(train, 'MultiTaskModel', DetectorShell)
    flags = CLI_REQUIRED + ['--cascade', '--roi-context', '3', '--cascade-rep', '8', '--roi-batch', '8']
    if mode != 'legacy':
        flags += ['--roi-context-mode', mode]
    args = train.parse_args(flags)
    model = train.build_model(args)
    saved = vars(args).copy()
    if mode == 'legacy':
        saved.pop('roi_context_mode')  # Existing checkpoints predate this flag.
    (tmp_path / 'metrics.json').write_text(json.dumps({'args': saved}, default=str))
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    restored, restored_args = predict.load_run(tmp_path, device='cpu')
    expected = 'shared' if mode == 'legacy' else mode
    assert restored_args.roi_context_mode == restored.roi_heads.context_mode == expected
    assert next(restored.parameters()).device.type == 'cpu'
    original_output, restored_output = evaluate(model.roi_heads), evaluate(restored.roi_heads)
    for name in original_output:
        torch.testing.assert_close(original_output[name], restored_output[name], rtol=0, atol=0)
