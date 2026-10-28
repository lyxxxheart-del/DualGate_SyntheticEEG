"""Self-contained U-ViT, adapted from the official baofff/U-ViT ``libs/uvit.py``.

The upstream file (kept here as ``uvit_official.py`` for reference) cannot be
imported as-shipped in this repo because it:

* does ``from .timm import trunc_normal_, Mlp`` -- a *relative* import against a
  ``timm.py`` that DDF-Net does not ship, and a package context that this
  directory does not provide;
* does ``import einops`` (not installed here);
* assumes a *square* image in ``UViT.__init__`` (``num_patches =
  (img_size // patch_size) ** 2``) and in ``unpatchify`` (``h = w =
  int(x.shape[1] ** .5)``), which cannot represent DDF-Net's interpolated EEG
  image of ``166 x 600``.

This module reproduces the same architecture and ``forward(x, timesteps,
y=None) -> [B, in_chans, H, W]`` contract with three changes:

* ``trunc_normal_`` and ``Mlp`` are defined inline (no timm dependency);
* ``patchify``/``unpatchify``/attention use plain ``torch`` (no einops);
* ``img_size`` accepts ``(H, W)`` and ``unpatchify`` reconstructs the exact
  (possibly non-square) ``H x W`` grid.

Use exactly as the upstream::

    from uvit import UViT
    model = UViT(img_size=(166, 600), patch_size=2, in_chans=1, ...)
    out = model(x, timesteps)  # x: [B, 1, 166, 600] -> out: same shape

Note: the transformer attention cost is O(num_patches^2). For ``166 x 600`` the
only patch size that divides both sides is 2, giving 24,900 patches -- too many
for a practical CPU forward pass. This is the reason DDF-Net's Stage 1 diffusion
actually uses ``UNetLidar`` (a conv net) and imports ``UViT`` only as an unused
"optional" symbol.
"""

import math

import torch
import torch.nn as nn


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # timm.models.layers.weight_init.trunc_normal_ (inline to drop the timm dep)
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        import warnings
        warnings.warn(
            "mean is more than 2 std from [a, b] in trunc_normal_; the "
            "distribution of values may be incorrect.", stacklevel=2)
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Mlp(nn.Module):
    # timm.models.layers.mlp.Mlp (minimal inline copy)
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])],
                              dim=-1)
    return embedding


def patchify(imgs, patch_size):
    """[B, C, H, W] -> [B, (H/p)*(W/p), p*p*C] (pure torch, non-square)."""
    B, C, H, W = imgs.shape
    x = imgs.reshape(B, C, H // patch_size, patch_size,
                     W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)  # B, h, w, p, p, C
    return x.reshape(B, (H // patch_size) * (W // patch_size),
                     patch_size * patch_size * C)


def unpatchify(x, H, W, patch_size, channels):
    """[B, (H/p)*(W/p), p*p*C] -> [B, C, H, W] (pure torch, non-square)."""
    B, L, D = x.shape
    h, w = H // patch_size, W // patch_size
    assert L == h * w, f"patch count {L} != {h}*{w}"
    assert D == patch_size * patch_size * channels
    x = x.reshape(B, h, w, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)  # B, C, h, p, w, p
    return x.reshape(B, channels, H, W)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads,
                                  C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, L, D
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, L, D
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"{H}x{W} not divisible by patch_size {self.patch_size}"
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False,
                 num_classes=-1, use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        if isinstance(img_size, (tuple, list)):
            H_img, W_img = img_size
        else:
            H_img = W_img = img_size
        self.img_size = (H_img, W_img)
        self.patch_size = patch_size

        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim)
        num_patches = (H_img // patch_size) * (W_img // patch_size)

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                  use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
            use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                  skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3,
                                     padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(self, x, timesteps, y=None):
        B, C_in, H, W = x.shape
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(
            timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, H, W, self.patch_size, self.in_chans)
        x = self.final_layer(x)
        return x
