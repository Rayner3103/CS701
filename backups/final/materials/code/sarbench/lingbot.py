"""Pinned LingBot-Vision-B/16 for the existing grayscale SAR training recipe.

The official encoder implementation is vendored unmodified. This adapter adds
non-reentrant block checkpointing and exposes patch tokens to the existing neck.
RGB patch weights are summed exactly as timm adapts the DINOv3 baseline; inputs
keep the existing SAR mean/std. Training-time RoPE coordinate augmentation is
disabled, matching the baseline downstream recipe rather than upstream pretraining.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .vendor.lingbot_vision.vit import LingBotVisionTransformer

HF_REPO = 'robbyant/lingbot-vision-vit-base'
HF_REVISION = 'f606f8c6c4002234ea68038f4d7c7cf57da96dfa'
WEIGHTS_SHA256 = '783dfb59014c34e9f0013db60bf6f5cc3c6b0604eb9528fb5df65e80769c5825'


def base_config(img_size=512, in_chans=1, drop_path=0.0):
    """Official ViT-B configuration, with downstream RoPE augmentation disabled."""
    return dict(img_size=img_size, patch_size=16, in_chans=in_chans, embed_dim=768,
                depth=12, num_heads=12, ffn_ratio=4, qkv_bias=True, proj_bias=True,
                ffn_bias=True, drop_path_rate=drop_path, layerscale_init=1e-5,
                norm_layer='layernormbf16', ffn_layer='mlp', n_storage_tokens=4,
                mask_k_bias=True, untie_cls_and_patch_norms=False,
                untie_global_and_local_cls_norm=False, pos_embed_rope_base=100.0,
                pos_embed_rope_min_period=None, pos_embed_rope_max_period=None,
                pos_embed_rope_normalize_coords='separate', pos_embed_rope_shift_coords=None,
                pos_embed_rope_jitter_coords=None, pos_embed_rope_rescale_coords=None,
                pos_embed_rope_dtype='fp32')


class CheckpointedLingBotViT(LingBotVisionTransformer):
    """Single-image-stream downstream encoder with unchanged upstream block math."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grad_checkpointing = False
        self.num_prefix_tokens = 1 + self.n_storage_tokens

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = bool(enable)

    def forward_features(self, x, masks=None):
        if not isinstance(x, torch.Tensor):
            raise TypeError('The SAR encoder expects one image tensor, not multi-crop inputs')
        if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
            raise ValueError('The pinned ViT-B checkpoint uses a shared final norm')
        x, (height, width) = self.prepare_tokens_with_masks(x, masks)
        # Deterministic fp32 RoPE is shared by all blocks/recomputations.
        if any(value is not None for value in (self.rope_embed.shift_coords,
                                               self.rope_embed.jitter_coords,
                                               self.rope_embed.rescale_coords)):
            raise ValueError('RoPE coordinate augmentation is disabled for this comparison')
        rope = self.rope_embed(H=height, W=width)
        for block in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                # Non-reentrant mode also works when frozen patch tokens have no
                # input gradients but the block contains trainable LoRA weights.
                x = checkpoint(block, x, rope, use_reentrant=False, preserve_rng_state=True)
            else:
                x = block(x, rope)
        normed = self.norm(x)
        return dict(x_norm_clstoken=normed[:, 0], x_storage_tokens=normed[:, 1:self.num_prefix_tokens],
                    x_norm_patchtokens=normed[:, self.num_prefix_tokens:], x_prenorm=x, masks=masks)


def load_pretrained_gray(model, path):
    """Verify the pinned official checkpoint, sum RGB kernels, and strictly load."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise ValueError('LingBot weights do not match the pinned official SHA256')
    state = torch.load(path, map_location='cpu', weights_only=True)
    # The pinned Base release wraps its tensors in {"model": state_dict}.
    # The official loader also documents raw / {"backbone": ...} variants.
    for wrapper in ('model', 'backbone'):
        if isinstance(state, dict) and wrapper in state:
            state = state[wrapper]
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                              for k, v in state.items()):
        raise ValueError('Expected a backbone tensor state dict')
    normalized = {key.removeprefix('backbone.'): value for key, value in state.items()}
    if len(normalized) != len(state):
        raise ValueError('Checkpoint key normalization produced duplicate names')
    weight = normalized['patch_embed.proj.weight']
    if weight.shape != (768, 3, 16, 16):
        raise ValueError(f'Unexpected official RGB patch kernel: {weight.shape}')
    normalized['patch_embed.proj.weight'] = weight.sum(dim=1, keepdim=True)
    model.load_state_dict(normalized, strict=True)


class LingBotViT(nn.Module):
    """LingBot-Vision-B/16 -> normalized [B, (H/16)*(W/16), 768] tokens."""

    ml_indices = None

    def __init__(self, pretrained, img_size=512, drop_path=0.0):
        super().__init__()
        self.vit = CheckpointedLingBotViT(**base_config(img_size, 1, drop_path))
        self.vit.init_weights()  # initializes masked QKV bias and all special tokens
        if pretrained:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=HF_REPO, filename='model.pt', revision=HF_REVISION)
            load_pretrained_gray(self.vit, path)
        self.embed_dim, self.patch_size = 768, 16

    @property
    def blocks(self):
        return self.vit.blocks

    def forward(self, x):
        if self.ml_indices:
            raise ValueError('LingBot multi-level outputs are not part of this backbone comparison')
        output = self.vit.forward_features(x)
        self.prefix_tokens = torch.cat((output['x_norm_clstoken'].unsqueeze(1), output['x_storage_tokens']), dim=1)
        return output['x_norm_patchtokens']
