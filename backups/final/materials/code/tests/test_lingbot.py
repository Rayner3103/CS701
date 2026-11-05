"""CPU regression checks for the pinned LingBot-B integration and LoRA handoff."""
import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from sarbench.adapters import LoRA, add_lora, unfreeze
from sarbench.lingbot import (
    CheckpointedLingBotViT, LingBotViT, base_config, load_pretrained_gray,
)
from sarbench.vendor.lingbot_vision.layers import LinearKMaskedBias
from sarbench.vendor.lingbot_vision.vit import LingBotVisionTransformer


def small_config():
    config = base_config(img_size=32)
    config.update(embed_dim=48, depth=2, num_heads=3, ffn_ratio=2)
    return config


def small_encoder():
    encoder = CheckpointedLingBotViT(**small_config())
    encoder.init_weights()
    return encoder


def adapted_small_encoder():
    encoder = small_encoder()
    assert add_lora(encoder, ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'), 4, 8) == 8
    unfreeze(encoder, norms=True)
    for module in encoder.modules():
        if isinstance(module, LoRA):
            nn.init.normal_(module.up.weight, std=0.02)
    return encoder


def objective(output):
    tokens = output['x_norm_patchtokens']
    # A nonconstant loss across channels makes norm and adapter gradients observable.
    weights = torch.linspace(-0.7, 0.9, tokens.shape[-1], dtype=tokens.dtype)
    return (tokens * weights).sum() / tokens.numel()


def test_adapter_forward_matches_unmodified_upstream_encoder():
    adapted = small_encoder().eval()
    reference = LingBotVisionTransformer(**small_config()).eval()
    reference.load_state_dict(adapted.state_dict(), strict=True)
    image = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        expected = reference.forward_features(image)
        actual = adapted.forward_features(image)
    for key in ('x_norm_patchtokens', 'x_norm_clstoken', 'x_storage_tokens', 'x_prenorm'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_downstream_rope_has_no_stochastic_coordinate_augmentation():
    model = small_encoder().train()
    assert model.rope_embed.shift_coords is None
    assert model.rope_embed.jitter_coords is None
    assert model.rope_embed.rescale_coords is None
    image = torch.randn(1, 1, 32, 32)
    first = model.forward_features(image)['x_norm_patchtokens']
    second = model.forward_features(image)['x_norm_patchtokens']
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_lora_preserves_masked_key_bias_and_zero_initialized_function():
    layer = LinearKMaskedBias(12, 36, bias=True)
    with torch.no_grad():
        layer.bias_mask.fill_(1)
        layer.bias_mask[12:24].zero_()
    lora = LoRA(layer, rank=4, alpha=8)
    value = torch.randn(2, 5, 12)
    expected = layer(value)
    torch.testing.assert_close(lora(value), expected, rtol=0, atol=0)
    assert lora.in_features == 12 and lora.out_features == 36
    with torch.no_grad():
        lora.base.bias[12:24].fill_(10000)
    torch.testing.assert_close(lora(value), expected, rtol=0, atol=0)
    assert 'base.bias_mask' in lora.state_dict()


def test_lora_still_preserves_ordinary_linear_layers_used_by_dinov3():
    original = nn.Linear(12, 24)
    image = torch.randn(2, 5, 12)
    wrapped = LoRA(original, rank=4, alpha=8)
    torch.testing.assert_close(wrapped(image), original(image), rtol=0, atol=0)
    assert set(wrapped.state_dict()) == {'base.weight', 'base.bias', 'down.weight', 'up.weight'}


def test_checkpointing_keeps_output_gradients_and_frozen_input_support():
    eager = adapted_small_encoder().train()
    recomputed = copy.deepcopy(eager)
    recomputed.set_grad_checkpointing(True)
    image = torch.randn(2, 1, 32, 32)  # no input gradient and frozen patch embedding
    assert not image.requires_grad and not eager.patch_embed.proj.weight.requires_grad
    outputs = [model.forward_features(image) for model in (eager, recomputed)]
    torch.testing.assert_close(outputs[0]['x_norm_patchtokens'], outputs[1]['x_norm_patchtokens'], rtol=0, atol=0)
    for output in outputs:
        objective(output).backward()
    other = dict(recomputed.named_parameters())
    for name, parameter in eager.named_parameters():
        gradient = other[name].grad
        if parameter.requires_grad:
            assert parameter.grad is not None and gradient is not None, name
            assert torch.isfinite(parameter.grad).all() and torch.isfinite(gradient).all(), name
            torch.testing.assert_close(parameter.grad, gradient, rtol=1e-5, atol=1e-8, msg=name)
        else:
            assert parameter.grad is None and gradient is None, name


def test_compiled_block_graph_with_checkpointing_has_no_unused_trainable_parameters():
    model = adapted_small_encoder().train()
    model.set_grad_checkpointing(True)
    # CPU eager backend checks Dynamo graph capture; the queued GPU smoke tests Inductor.
    model.blocks = nn.ModuleList(torch.compile(block, backend='eager', fullgraph=True)
                                 for block in model.blocks)
    objective(model.forward_features(torch.randn(2, 1, 32, 32))).backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_checkpoint_hash_is_checked_before_loading(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'not the published LingBot weights')
    with pytest.raises(ValueError, match='SHA256'):
        load_pretrained_gray(small_encoder(), checkpoint)


def test_full_bp7_parameter_budgets_and_existing_dinov3_are_unchanged(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import train
    from run_bp7_full_stop130 import training_command

    command = training_command('/python', tmp_path / 'data', tmp_path / 'suite')
    args = train.parse_args(command[command.index('train.py') + 1:])
    args.init = 'scratch'  # architecture-only construction, then the unchanged LoRA recipe
    expected = {'dinov3': (125076960, 39867360, 16),
                'lingbot_b': (125105376, 39867360, 27664)}
    for name, budget in expected.items():
        args.backbone = name
        with torch.device('meta'):
            model = train.build_model(args)
        actual = (sum(p.numel() for p in model.parameters()),
                  sum(p.numel() for p in model.parameters() if p.requires_grad),
                  sum(b.numel() for b in model.buffers() if b.is_floating_point()))
        assert actual == budget
        assert actual[0] + actual[2] <= 130000000 and actual[1] <= 40000000
        assert sum(isinstance(module, LoRA) for module in model.modules()) == 48


def test_official_weights_load_strictly_and_gray_matches_replicated_rgb():
    checkpoint = Path('/mnt/hdd/junkaichen/hf-cache/hub/models--robbyant--lingbot-vision-vit-base/'
                      'snapshots/f606f8c6c4002234ea68038f4d7c7cf57da96dfa/model.pt')
    if not checkpoint.is_file():
        pytest.skip('Pinned official checkpoint is not cached; this test never downloads')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if 'model' in state:
        state = state['model']
    if 'backbone' in state:
        state = state['backbone']
    state = {name.removeprefix('backbone.'): tensor for name, tensor in state.items()}
    reference = LingBotVisionTransformer(**base_config(img_size=64, in_chans=3)).eval()
    reference.load_state_dict(state, strict=True)
    gray = CheckpointedLingBotViT(**base_config(img_size=64)).eval()
    load_pretrained_gray(gray, checkpoint)
    torch.testing.assert_close(gray.patch_embed.proj.weight,
                               reference.patch_embed.proj.weight.sum(dim=1, keepdim=True), rtol=0, atol=0)
    image = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        expected = reference.forward_features(image.repeat(1, 3, 1, 1))
        actual = gray.forward_features(image)
    assert actual['x_norm_patchtokens'].shape == (1, 16, 768)
    assert actual['x_storage_tokens'].shape == (1, 4, 768)
    for key in ('x_norm_patchtokens', 'x_norm_clstoken', 'x_storage_tokens'):
        assert torch.isfinite(actual[key]).all()
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-4, atol=2e-4)
