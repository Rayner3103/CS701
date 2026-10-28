"""LoRA and MoE-LoRA adapters for the attention layers of a frozen backbone."""
import torch.nn.functional as F
from torch import nn


class LoRA(nn.Module):
    """y = W x + (alpha / r) * B A x, where W is the wrapped layer (frozen by add_adapters)."""

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)  # A
        self.up = nn.Linear(rank, base.out_features, bias=False)  # B
        nn.init.zeros_(self.up.weight)  # B = 0: training starts from exactly the pretrained layer
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


class MoELoRA(nn.Module):
    """y = W x + (alpha / r) * sum_e g_e(x) B_e A_e x, with gates g(x) = softmax(router[task](x)) per token.

    The experts' A_e are stacked in `down` and their B_e side by side in `up`, so gating each
    expert's r hidden units and applying `up` sums the experts.
    """

    def __init__(self, base: nn.Linear, experts=4, rank=4, alpha=8, tasks=2):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, experts * rank, bias=False)  # A_1 ... A_E
        self.up = nn.Linear(experts * rank, base.out_features, bias=False)  # B_1 ... B_E
        nn.init.zeros_(self.up.weight)
        self.routers = nn.ModuleList(nn.Linear(base.in_features, experts, bias=False) for _ in range(tasks))
        self.rank = rank
        self.scale = alpha / rank
        self.task = 0  # which router to use; see set_task

    def forward(self, x):
        gates = self.routers[self.task](x).softmax(dim=-1)
        gates = gates.repeat_interleave(self.rank, dim=-1)  # expert e's gate on each of its r hidden units
        return self.base(x) + self.scale * self.up(self.down(x) * gates)


class AdaptFormerBlock(nn.Module):
    """A DINOv3 (timm EvaBlock) block with an AdaptFormer adapter (Chen et al., 2022) parallel to its MLP branch:
    x = x + g1 * attn(norm1(x)); x = x + g2 * mlp(norm2(x)) + s * up(relu(down(norm2(x)))).

    The adapter sits outside LayerScale (g2): the pretrained g2 grows from ~0.1 in the first blocks to ~10 in the
    last, so inside it the adapter would be damped early and amplified late. It reads norm2(x), the MLP's own input.
    """

    def __init__(self, block, bottleneck=64, scale=0.1):
        super().__init__()
        self.block = block
        self.down = nn.Linear(block.norm2.normalized_shape[0], bottleneck)
        self.up = nn.Linear(bottleneck, block.norm2.normalized_shape[0])
        nn.init.zeros_(self.up.weight)  # up = 0: training starts from exactly the pretrained block
        nn.init.zeros_(self.up.bias)
        self.scale = scale

    def forward(self, x, **kwargs):  # kwargs (rope, attn_mask, ...) differ between timm versions: pass them on
        b = self.block
        x = x + b.drop_path1(b.gamma_1 * b.attn(b.norm1(x), **kwargs))
        h = b.norm2(x)
        return x + b.drop_path2(b.gamma_2 * b.mlp(h)) + self.scale * self.up(F.relu(self.down(h)))


def add_adaptformer(backbone, bottleneck=64, scale=0.1, freeze=True):
    """Freeze the backbone (unless freeze=False: it is already frozen, with adapters on top that must keep training),
    then add a parallel AdaptFormer adapter to the MLP of every block (DINOv3 blocks only)."""
    if freeze:
        backbone.requires_grad_(False)
    for i, blk in enumerate(backbone.blocks):
        if not hasattr(blk, 'gamma_2'):
            raise ValueError('adaptformer needs DINOv3 (LayerScale) blocks')
        backbone.blocks[i] = AdaptFormerBlock(blk, bottleneck, scale)


def add_adapters(backbone, kind, targets=('attn.qkv', 'attn.proj'), blocks=None, **kwargs):
    """Freeze the backbone, then wrap the linear layers `targets` of every block, or only of `blocks` (kind: 'lora' or
    'moelora'; kwargs: the adapter's, e.g. rank and alpha). Default: attn.qkv and attn.proj; 'mlp.fc1', 'mlp.fc2'
    adapt the MLP too. kind 'adaptformer' instead adds a parallel adapter to every block's MLP (add_adaptformer; kwargs:
    bottleneck, scale); 'adaptformer_lora' adds both: LoRA on the attention and AdaptFormer on the MLP."""
    if kind == 'adaptformer':
        return add_adaptformer(backbone, **kwargs)
    if kind == 'adaptformer_lora':
        add_adapters(backbone, 'lora', targets=targets, blocks=blocks)  # first: AdaptFormer wraps the blocks
        return add_adaptformer(backbone, freeze=False, **kwargs)
    adapter = {'lora': LoRA, 'moelora': MoELoRA}[kind]
    backbone.requires_grad_(False)
    for blk in backbone.blocks if blocks is None else blocks:
        for target in targets:
            parent, name = blk.get_submodule(target.rpartition('.')[0]), target.rpartition('.')[2]
            setattr(parent, name, adapter(getattr(parent, name), **kwargs))


def set_task(module, task):
    """Route every MoELoRA inside `module` through the router of `task` (0 = classification, 1 = detection)."""
    for layer in module.modules():
        if isinstance(layer, MoELoRA):
            layer.task = task


def adapter_kwargs(args):
    """The add_adapters keyword arguments a run's saved args (train.py's) imply: only AdaptFormer has any."""
    return {'bottleneck': args.get('bottleneck', 64)} if args['adapt'].startswith('adaptformer') else {}
