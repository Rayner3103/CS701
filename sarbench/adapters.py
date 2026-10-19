"""LoRA and MoE-LoRA adapters for the attention layers of a frozen backbone."""
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


def add_adapters(backbone, kind):
    """Freeze the backbone, then wrap attn.qkv and attn.proj of every block (kind: 'lora' or 'moelora')."""
    adapter = {'lora': LoRA, 'moelora': MoELoRA}[kind]
    backbone.requires_grad_(False)
    for blk in backbone.blocks:
        blk.attn.qkv = adapter(blk.attn.qkv)
        blk.attn.proj = adapter(blk.attn.proj)


def add_lora(backbone, targets=('attn.qkv', 'attn.proj'), rank=16, alpha=32, skip_blocks=()):
    """Freeze the backbone, then wrap the nn.Linear layers named by `targets` (paths inside a block, e.g. 'mlp.fc1')
    of every block with LoRA(rank, alpha). Blocks whose index is in skip_blocks are left alone (e.g. blocks that are
    fully fine-tuned instead). Returns the number of wrapped layers."""
    backbone.requires_grad_(False)
    wrapped = 0
    for i, blk in enumerate(backbone.blocks):
        if i in skip_blocks:
            continue
        for path in targets:
            parent_path, _, attr = path.rpartition('.')
            parent = blk.get_submodule(parent_path) if parent_path else blk
            base = getattr(parent, attr, None)
            if isinstance(base, nn.Linear):
                setattr(parent, attr, LoRA(base, rank, alpha))
                wrapped += 1
    return wrapped


def unfreeze(backbone, norms=False, biases=False, patch_embed=False, layer_scale=False, last_blocks=0):
    """Make more of a frozen backbone trainable: LayerNorm weights, all biases, the patch embedding, layer-scale
    gammas, and/or every weight of the last `last_blocks` blocks."""
    for name, p in backbone.named_parameters():
        leaf = name.rsplit('.', 1)[-1]
        if norms and ('norm' in name.lower()) and leaf in ('weight', 'bias'):
            p.requires_grad_(True)
        if biases and leaf == 'bias':
            p.requires_grad_(True)
        if patch_embed and ('patch_embed' in name or 'stem' in name):
            p.requires_grad_(True)
        if layer_scale and (leaf in ('gamma', 'scale') or '.ls1.' in name or '.ls2.' in name or 'gamma_' in leaf):
            p.requires_grad_(True)
    if last_blocks:
        for blk in list(backbone.blocks)[-last_blocks:]:
            blk.requires_grad_(True)


def set_task(module, task):
    """Route every MoELoRA inside `module` through the router of `task` (0 = classification, 1 = detection)."""
    for layer in module.modules():
        if isinstance(layer, MoELoRA):
            layer.task = task
