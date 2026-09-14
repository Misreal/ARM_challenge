"""A patch-4 Vision Transformer for 32x32 CIFAR-100.

Why a transformer is in a repo of CNNs: every bundled model so far loses under
0.1 top-1 points to uniform INT8, which leaves mixed precision nothing to
recover. The search cannot be shown to work or fail on models where the
quantity it optimizes is already zero. Attention blocks are the standard
counter-example -- per-tensor INT8 on QKV/projection MatMuls is genuinely
lossy, and LayerNorm/Softmax/GELU sit outside ONNX Runtime's default
quantizable op set and stay FP32 whatever the config says. That mix is the
point: it is the regime the CNNs cannot reach.

Patch 4 rather than the usual 16: at 32x32 a patch-16 grid is 2x2 = 4 tokens,
which is not a sequence. Patch 4 gives an 8x8 grid, 64 tokens, and keeps the
whole model at the project's (N, 3, 32, 32) input contract -- no resize, no
change to the Pi data bundle, no change to `INPUT_SHAPE` in
`src/export_onnx.py`.

Every operation that carries weights lives inside a named `nn.Module`, and the
attention math is written as module calls rather than functional one-liners
wherever a functional call would emit an unscoped node. `src/quant/groups.py`
maps ONNX nodes to blocks purely by their module path, raises on any unnamed
node, and rejects a graph whose nodes fall mostly into `_unscoped`. A
transformer written the idiomatic way -- `F.scaled_dot_product_attention`, bare
`.transpose()`/`.reshape()` chains -- produces exactly that rejected graph.
Hence `Attention` splitting QKV through named Linears and `nn.Softmax` as a
module attribute, which read as verbose PyTorch and are deliberate.

Grouping depth is 2 (`blocks.0`, `blocks.1`, ...), declared in `GROUP_DEPTH`
alongside MobileNetV2's, because depth 1 would put every block under one
`blocks` container and trip `MAX_GROUP_SHARE`.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from src.models.registry import register_model


class PatchEmbed(nn.Module):
    """Image -> token sequence, as one strided convolution.

    A Conv2d with kernel == stride == patch size is the patch projection: each
    output position sees exactly one non-overlapping tile. Written as a conv
    rather than an unfold/reshape because a conv is one named, quantizable node
    where the unfold path is several unscoped ones.
    """

    def __init__(self, patch_size: int = 4, in_channels: int = 3, dim: int = 192) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # (N, dim, H/p, W/p) -> (N, tokens, dim). flatten(2) then transpose is
        # two nodes; a reshape to an inferred shape would emit the
        # Shape/Gather/Unsqueeze cluster that custom_cnn's comment warns about.
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention with QKV as three separate named Linears.

    The fused single-Linear QKV that every reference implementation uses is one
    matmul instead of three, but it forces a reshape-and-unbind to split the
    heads, and those land unscoped. Three Linears cost a little latency and buy
    a graph whose every weight-carrying node maps to `attn.q`, `attn.k`,
    `attn.v` or `attn.proj`. Sensitivity analysis reads that map, so the
    trade is worth making here even though it would not be in production.
    """

    def __init__(self, dim: int, num_heads: int = 3, attn_dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        query = self._split_heads(self.q(x))
        key = self._split_heads(self.k(x))
        value = self._split_heads(self.v(x))

        # Explicit matmul-softmax-matmul rather than
        # F.scaled_dot_product_attention: the fused op exports as a single
        # opaque node on some opsets and as an unscoped subgraph on others, and
        # neither gives the analyzer anything to attribute damage to.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        weights = self.attn_drop(self.softmax(scores))
        context = torch.matmul(weights, value)

        context = context.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(context)


class Mlp(nn.Module):
    """The feed-forward half of a block: expand, GELU, project back."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x)).

    Pre-norm rather than the original post-norm because post-norm needs
    warmup to train at all at this depth, and the quantization story is the
    experiment here -- the training recipe should not also be a variable.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTCifar(nn.Module):
    """ViT-Tiny's width and depth, re-tiled for 32x32.

    dim 192 / depth 12 / 3 heads is ViT-Tiny (and DeiT-Tiny) exactly; only the
    patch size and the resulting token count differ. Keeping the width means
    the quantization behaviour is comparable to the published tiny models even
    though the weights are trained here, on this project's own split.
    """

    def __init__(
        self,
        num_classes: int = 100,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError(f"image_size {image_size} is not divisible by patch_size {patch_size}")
        grid = image_size // patch_size
        num_patches = grid * grid

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=3, dim=dim)
        # A learned class token and learned (not sinusoidal) position
        # embeddings: both are parameters added to the token stream, so they
        # fold into the first LayerNorm's input range rather than appearing as
        # their own quantizable nodes.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            OrderedDict(
                (
                    str(index),
                    Block(
                        dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                    ),
                )
                for index in range(depth)
            )
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        # expand() rather than repeat(): a view, so it adds no copy node.
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # Classify from the class token. Slicing [:, 0] keeps one Gather node;
        # mean-pooling the sequence would be an extra ReduceMean for no gain.
        return self.head(x[:, 0])


@register_model("vit_cifar")
def build_vit_cifar(num_classes: int = 100, **kwargs: object) -> nn.Module:
    """~5.4M parameters, ~21 MB FP32 -- ViT-Tiny's budget at CIFAR resolution.

    Trained from scratch on this project's own train split rather than adapted
    from published CIFAR-100 weights: every public checkpoint was fit on the
    full 50k train set, which contains this project's `optval` images, and a
    model that has seen the evaluation split reports quantization damage it did
    not actually suffer.
    """
    return ViTCifar(num_classes=num_classes, **kwargs)  # type: ignore[arg-type]
