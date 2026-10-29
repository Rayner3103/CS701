"""The backbones, built for 512 px input. Plain ViTs map images [B, 1, 512, 512] -> patch tokens [B, N, D];
the ConvNeXt maps them to four feature maps at strides 4, 8, 16 and 32."""
import timm
import torch
from torch import nn

try:  # as in the reference: this import also switches fp32 matmuls to TF32
    from terratorch.registry import BACKBONE_REGISTRY
except ImportError:  # terratorch is only needed for the TerraMind backbone; keep its TF32 side effect
    BACKBONE_REGISTRY = None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def forward_tokens(backbone, x):
    """The final normed tokens; with backbone.ml_indices set, also keep the normed outputs of those blocks as
    [B, D, h, w] maps in backbone.intermediates (shallower blocks keep finer spatial detail)."""
    if backbone.ml_indices:
        tokens, backbone.intermediates = backbone.vit.forward_intermediates(
            x, indices=list(backbone.ml_indices), norm=True, output_fmt='NCHW')
        return tokens
    return backbone.vit.forward_features(x)


class ViT(nn.Module):
    """timm ViT-B/16 pretrained on ImageNet-21k."""

    def __init__(self, pretrained: bool, drop_path=0.0):
        super().__init__()
        # in_chans=1: timm sums the RGB patch-embedding weights; img_size=512: it resamples the position embedding
        extra = {'drop_path_rate': drop_path} if drop_path else {}  # the reference passes nothing
        self.vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrained,
                                     img_size=512, in_chans=1, num_classes=0, **extra)
        self.embed_dim = 768
        self.patch_size = 16

    @property
    def blocks(self):
        return self.vit.blocks

    ml_indices = None  # block indices whose (normed) outputs feed the pyramid levels; None: last block only

    def forward(self, x):
        tokens = forward_tokens(self, x)  # [B, 1 + N, 768], final norm applied
        self.prefix_tokens = tokens[:, :self.vit.num_prefix_tokens]  # CLS, for poolings that use it
        return tokens[:, self.vit.num_prefix_tokens:]  # drop the CLS token


class TimmViT(nn.Module):
    """Any timm plain ViT (DINOv3, DINOv2, EVA-02, CLIP, ...) at img_size px with a 1-channel patch embedding.
    The prefix tokens (CLS, registers) are dropped from the output and kept in self.prefix_tokens."""

    def __init__(self, name: str, pretrained: bool, img_size=512, drop_path=0.0):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained, img_size=img_size, in_chans=1, num_classes=0,
                                     drop_path_rate=drop_path)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[0]

    @property
    def blocks(self):
        return self.vit.blocks

    ml_indices = None

    def forward(self, x):
        tokens = forward_tokens(self, x)  # final norm applied
        n = self.vit.num_prefix_tokens
        self.prefix_tokens = tokens[:, :n]
        return tokens[:, n:]


def set_token_stride(backbone, stride):
    """Overlapping patch embedding: the pretrained p x p patch kernel slides with this stride (padded by (p - stride) / 2),
    so a 512 px image gives (512 / stride)^2 tokens, each still seeing a p x p window of the image (dense ViT
    descriptors, Amir et al. 2021). DINOv3's RoPE adapts to the larger token grid; no parameter changes."""
    proj = backbone.vit.patch_embed.proj
    p = proj.kernel_size[0]
    assert (p - stride) % 2 == 0 and stride <= p
    proj.stride, proj.padding = (stride, stride), ((p - stride) // 2, (p - stride) // 2)
    backbone.token_stride = stride


class TimmConvNeXt(nn.Module):
    """A timm ConvNeXt as a feature extractor: four maps at strides 4, 8, 16, 32 (hierarchical: no neck upsampling
    needed for small objects)."""

    hierarchical = True

    def __init__(self, name: str, pretrained: bool, drop_path=0.0):
        super().__init__()
        self.body = timm.create_model(name, pretrained=pretrained, in_chans=1, features_only=True,
                                      out_indices=(0, 1, 2, 3), drop_path_rate=drop_path)
        self.channels = self.body.feature_info.channels()
        self.embed_dim = self.channels[-1]
        # With terratorch imported (TF32 settings), cuDNN rejects the depthwise conv on the channels-last-strided
        # tensors that ConvNeXt's permuted LayerNorms produce (CUDNN_STATUS_BAD_PARAM): give it contiguous input.
        for m in self.body.modules():
            if isinstance(m, nn.Conv2d) and m.groups > 1:
                m.register_forward_pre_hook(lambda module, args: (args[0].contiguous(),))

    @property
    def blocks(self):  # features_only renames the stages (stages_0, ...): collect the blocks by type, in order
        return [m for m in self.body.modules() if type(m).__name__ == 'ConvNeXtBlock']

    def forward(self, x):
        return self.body(x)


class TerraMind(nn.Module):
    """TerraMind-1.0-base (IBM/ESA) through its Sentinel-1 GRD input."""

    def __init__(self, pretrained: bool):
        super().__init__()
        if BACKBONE_REGISTRY is None:
            raise ImportError('terratorch is required for the TerraMind backbone')
        self.terramind = BACKBONE_REGISTRY.build('terratorch_terramind_v1_base', pretrained=pretrained,
                                                 modalities=['S1GRD'])
        self.embed_dim = 768
        self.patch_size = 16

    @property
    def blocks(self):
        return self.terramind.encoder

    def forward(self, x):
        per_block = self.terramind({'S1GRD': x.repeat(1, 2, 1, 1)})  # gray image as both VV and VH
        return per_block[-1]  # last block's tokens [B, N, 768]; terratorch already applied encoder_norm


TIMM_VITS = {
    'dinov3': 'vit_base_patch16_dinov3.lvd1689m',
    'dinov3s': 'vit_small_plus_patch16_dinov3.lvd1689m',  # SwiGLU MLP (mlp.fc1_g / fc1_x / fc2)
    'dinov3_vits': 'vit_small_patch16_dinov3.lvd1689m',
    'dinov2': 'vit_base_patch14_reg4_dinov2.lvd142m',
    'eva02': 'eva02_base_patch14_224.mim_in22k',
    'clip': 'vit_base_patch16_clip_224.openai',
}
TIMM_CONVNEXTS = {
    'convnext_dinov3': 'convnext_base.dinov3_lvd1689m',
    'convnext_s_dinov3': 'convnext_small.dinov3_lvd1689m',
    'convnext_t_dinov3': 'convnext_tiny.dinov3_lvd1689m',
    'convnext_in22k': 'convnext_base.fb_in22k',
}
BACKBONES = ['vit', 'terramind', *TIMM_VITS, *TIMM_CONVNEXTS]


def build_backbone(name, pretrained, img_size=512, drop_path=0.0):
    """name: one of BACKBONES. pretrained=False builds the same architecture with random weights."""
    if name == 'vit':
        return ViT(pretrained, drop_path)
    if name == 'terramind':
        return TerraMind(pretrained)
    if name in TIMM_VITS:
        return TimmViT(TIMM_VITS[name], pretrained, img_size, drop_path)
    return TimmConvNeXt(TIMM_CONVNEXTS[name], pretrained, drop_path)
