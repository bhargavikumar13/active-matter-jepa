"""
model.py — Spatiotemporal Vision Transformer for active_matter

Architecture overview
---------------------
Input : (B, T, C, H, W) = (B, 16, 11, 224, 224)

1. PatchEmbed3D
   Splits the video into non-overlapping 3-D "tubelets" of shape
   (t_patch, h_patch, w_patch) = (2, 16, 16), then linearly projects
   each tubelet to embed_dim.

   Number of tokens:
     n_t = T / t_patch = 16 / 2  = 8
     n_h = H / h_patch = 224 / 16 = 14
     n_w = W / w_patch = 224 / 16 = 14
     N   = n_t * n_h * n_w        = 1568 tokens per sample

2. VisionTransformer3D (Encoder)
   Adds learned 3-D positional embeddings, then passes tokens through
   a stack of standard transformer blocks.
   Default: embed_dim=192, depth=4, num_heads=3 → ~3.1M params total.

3. Predictor  (used only during JEPA pre-training)
   A narrower transformer that takes *context* token embeddings +
   learnable mask tokens and predicts the embeddings of *target* tokens.
   Default: predictor_embed_dim=96, depth=4, num_heads=3 → ~0.8M params.

Parameter budget
----------------
  Encoder  : ~3.1M
  Predictor: ~0.8M
  Total    : ~3.9M  (well under the 100M limit)

VRAM note
---------
  With B=8 and 1568 tokens the attention matrix is 1568×1568 per head.
  Keep batch size small (4–8) when training on a single GPU.
"""

import math
import torch
import torch.nn as nn
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# 1. Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """
    Tubelet embedding: splits (T, C, H, W) into non-overlapping 3-D patches
    and projects each patch to embed_dim via a single linear layer.

    Parameters
    ----------
    in_channels  : number of physical channels (11 for active_matter)
    t_patch      : temporal patch size (tubelets span this many frames)
    h_patch      : spatial patch height
    w_patch      : spatial patch width
    embed_dim    : output embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 11,
        t_patch: int = 2,
        h_patch: int = 16,
        w_patch: int = 16,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.t_patch  = t_patch
        self.h_patch  = h_patch
        self.w_patch  = w_patch
        self.embed_dim = embed_dim

        patch_dim = in_channels * t_patch * h_patch * w_patch   # 11*2*16*16 = 5632
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Parameters
        ----------
        x : (B, T, C, H, W)

        Returns
        -------
        tokens : (B, N, embed_dim)   N = n_t * n_h * n_w
        grid   : (n_t, n_h, n_w)     token grid shape (useful for masking)
        """
        B, T, C, H, W = x.shape
        tp, hp, wp = self.t_patch, self.h_patch, self.w_patch

        assert T % tp == 0, f"T={T} must be divisible by t_patch={tp}"
        assert H % hp == 0, f"H={H} must be divisible by h_patch={hp}"
        assert W % wp == 0, f"W={W} must be divisible by w_patch={wp}"

        n_t = T // tp
        n_h = H // hp
        n_w = W // wp

        # (B, T, C, H, W)
        # → reshape into (B, n_t, tp, C, n_h, hp, n_w, wp)
        x = x.reshape(B, n_t, tp, C, n_h, hp, n_w, wp)
        # → (B, n_t, n_h, n_w, tp, C, hp, wp)
        x = x.permute(0, 1, 4, 6, 2, 3, 5, 7).contiguous()
        # → (B, N, patch_dim)
        x = x.reshape(B, n_t * n_h * n_w, tp * C * hp * wp)

        tokens = self.norm(self.proj(x))   # (B, N, embed_dim)
        return tokens, (n_t, n_h, n_w)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transformer blocks
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class MLP(nn.Module):
    """Two-layer feed-forward block."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention → residual,
       LayerNorm → MLP → residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = Attention(embed_dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio, proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Positional embedding
# ─────────────────────────────────────────────────────────────────────────────

def build_3d_sincos_pos_embed(
    embed_dim: int,
    grid: tuple[int, int, int],
    temperature: float = 10000.0,
) -> torch.Tensor:
    """
    Sine-cosine positional embedding for a 3-D token grid.

    Parameters
    ----------
    embed_dim : must be divisible by 6 (2 dims per axis × 3 axes)
    grid      : (n_t, n_h, n_w)

    Returns
    -------
    pos_embed : (1, N, embed_dim)
    """
    n_t, n_h, n_w = grid
    assert embed_dim % 6 == 0, "embed_dim must be divisible by 6 for 3-D sin-cos"
    d = embed_dim // 6          # dims per axis

    def sin_cos(n, d):
        pos   = torch.arange(n, dtype=torch.float32)
        omega = torch.arange(d, dtype=torch.float32) / d
        omega = 1.0 / (temperature ** omega)          # (d,)
        out   = pos[:, None] * omega[None, :]         # (n, d)
        return torch.cat([out.sin(), out.cos()], dim=-1)  # (n, 2d)

    et = sin_cos(n_t, d)   # (n_t, 2d)
    eh = sin_cos(n_h, d)   # (n_h, 2d)
    ew = sin_cos(n_w, d)   # (n_w, 2d)

    # broadcast over the 3-D grid
    et = et[:, None, None, :].expand(n_t, n_h, n_w, -1)   # (n_t,n_h,n_w, 2d)
    eh = eh[None, :, None, :].expand(n_t, n_h, n_w, -1)
    ew = ew[None, None, :, :].expand(n_t, n_h, n_w, -1)

    pos = torch.cat([et, eh, ew], dim=-1)              # (n_t,n_h,n_w, 6d)
    pos = pos.reshape(1, n_t * n_h * n_w, embed_dim)  # (1, N, D)
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# 4. Encoder
# ─────────────────────────────────────────────────────────────────────────────

class VisionTransformer3D(nn.Module):
    """
    Spatiotemporal ViT encoder.

    Default settings match the project baseline:
      embed_dim=192, depth=4, num_heads=3  →  ~3.1M parameters

    The encoder is used in two modes:
      • Full forward  : encode all N tokens (during linear probe / supervised)
      • Masked forward: encode only context tokens (during SSL pre-training)
        Pass a boolean mask of shape (B, N) where True = keep (context).

    Parameters
    ----------
    in_channels        : physical channels in the input (11)
    t_patch, h_patch,
    w_patch            : tubelet dimensions
    embed_dim          : transformer width
    depth              : number of transformer blocks
    num_heads          : attention heads (embed_dim must be divisible)
    mlp_ratio          : MLP hidden size = embed_dim * mlp_ratio
    pos_embed_type     : 'sincos' (fixed) or 'learned'
    T, H, W            : expected input dimensions (for pos embed precomputation)
    """

    def __init__(
        self,
        in_channels: int = 11,
        t_patch: int = 2,
        h_patch: int = 16,
        w_patch: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pos_embed_type: str = 'sincos',
        T: int = 16,
        H: int = 224,
        W: int = 224,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(in_channels, t_patch, h_patch, w_patch, embed_dim)

        # Grid dimensions (fixed for this dataset)
        n_t = T // t_patch
        n_h = H // h_patch
        n_w = W // w_patch
        self.grid = (n_t, n_h, n_w)
        N = n_t * n_h * n_w   # 1568

        # Positional embedding
        if pos_embed_type == 'sincos':
            assert embed_dim % 6 == 0, \
                "sincos pos embed needs embed_dim divisible by 6"
            pos = build_3d_sincos_pos_embed(embed_dim, self.grid)
            self.register_buffer('pos_embed', pos)   # (1, N, D) — not a param
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, N, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (B, T, C, H, W)
        keep_mask : (B, N) bool, True = keep this token.
                    If None, all tokens are processed (full forward).

        Returns
        -------
        tokens : (B, N', embed_dim)
                 N' = N if keep_mask is None, else number of True entries per sample.
        """
        tokens, _ = self.patch_embed(x)          # (B, N, D)
        tokens = tokens + self.pos_embed          # add positional info

        if keep_mask is not None:
            # Select only the context tokens for each sample in the batch.
            # Assumes the same number of kept tokens per sample (required for
            # batched training — the masking strategy must ensure this).
            B, N, D = tokens.shape
            tokens = tokens[keep_mask].reshape(B, -1, D)  # (B, N_ctx, D)

        for block in self.blocks:
            tokens = block(tokens)

        return self.norm(tokens)   # (B, N', D)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Predictor (JEPA)
# ─────────────────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """
    JEPA predictor: maps context embeddings → predicted target embeddings.

    Architecture
    ------------
    • Projects context tokens from encoder embed_dim → predictor_embed_dim.
    • Replaces target positions with learned mask tokens (also projected).
    • Runs a small transformer over context + mask tokens.
    • Projects back to encoder embed_dim.

    This keeps the predictor lightweight (~0.8M params with the defaults)
    so the encoder is forced to learn all the useful information.

    Parameters
    ----------
    encoder_embed_dim    : embed_dim of the VisionTransformer3D encoder
    predictor_embed_dim  : internal width of the predictor (narrower)
    depth                : number of transformer blocks in the predictor
    num_heads            : attention heads
    num_tokens           : total number of patch tokens N (1568 by default)
    """

    def __init__(
        self,
        encoder_embed_dim: int = 192,
        predictor_embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        num_tokens: int = 1568,   # 8 * 14 * 14
    ):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim

        # Project encoder output → predictor width
        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Learnable mask token (one per target position)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding for ALL tokens (context + target) in predictor space
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, predictor_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_embed_dim)

        # Project back to encoder embed_dim for loss computation
        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context_tokens : (B, N_ctx, encoder_embed_dim)
            Encoded context tokens from the encoder.
        context_mask   : (B, N) bool, True = context position
        target_mask    : (B, N) bool, True = target position

        Returns
        -------
        pred_tokens : (B, N_tgt, encoder_embed_dim)
            Predicted embeddings at target positions.
        """
        B = context_tokens.shape[0]
        N = self.pos_embed.shape[1]

        # Project context tokens into predictor space
        ctx = self.input_proj(context_tokens)    # (B, N_ctx, pred_dim)

        # Expand mask tokens for target positions
        N_tgt = target_mask[0].sum().item()
        mask_tokens = self.mask_token.expand(B, int(N_tgt), -1)  # (B, N_tgt, pred_dim)

        # Build full sequence: place context and mask tokens at their positions
        # and add positional embeddings.
        full = torch.zeros(B, N, self.predictor_embed_dim, device=ctx.device, dtype=ctx.dtype)
        full[context_mask] = ctx.reshape(-1, self.predictor_embed_dim)
        full[target_mask]  = mask_tokens.reshape(-1, self.predictor_embed_dim).to(ctx.dtype)
        full = full + self.pos_embed    # (B, N, pred_dim)

        # Run predictor transformer over all positions
        for block in self.blocks:
            full = block(full)
        full = self.norm(full)

        # Extract and project target positions
        pred = full[target_mask].reshape(B, int(N_tgt), self.predictor_embed_dim)
        return self.output_proj(pred)   # (B, N_tgt, encoder_embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full JEPA model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class JEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture for active_matter.

    Components
    ----------
    • context_encoder  : VisionTransformer3D  (trained via gradient)
    • target_encoder   : VisionTransformer3D  (updated via EMA — no gradients)
    • predictor        : Predictor

    Training loop (handled in scripts/train.py)
    -------------------------------------------
    1. Sample context and target masks from the token grid.
    2. context_encoder encodes context tokens.
    3. predictor predicts target token embeddings from context.
    4. target_encoder encodes target tokens (no grad, EMA weights).
    5. Loss = mean cosine similarity (or L2) between predicted and target embeddings.
    6. Update context_encoder + predictor via backprop.
    7. Update target_encoder via EMA: θ_t ← τ·θ_t + (1-τ)·θ_c

    Parameters
    ----------
    encoder_kwargs   : dict passed to VisionTransformer3D
    predictor_kwargs : dict passed to Predictor
    ema_momentum     : starting EMA momentum τ (typically annealed 0.996 → 1.0)
    """

    def __init__(
        self,
        encoder_kwargs: dict | None = None,
        predictor_kwargs: dict | None = None,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        enc_kw  = encoder_kwargs   or {}
        pred_kw = predictor_kwargs or {}

        self.context_encoder = VisionTransformer3D(**enc_kw)
        self.target_encoder  = VisionTransformer3D(**enc_kw)
        self.predictor       = Predictor(
            encoder_embed_dim=self.context_encoder.embed_dim,
            **pred_kw,
        )

        # Target encoder is EMA — copy weights, disable gradients
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None):
        """EMA update of target encoder. Call once per training step."""
        tau = momentum if momentum is not None else self.ema_momentum
        for p_c, p_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_t.data.mul_(tau).add_(p_c.data, alpha=1.0 - tau)

    def forward(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x            : (B, T, C, H, W)
        context_mask : (B, N) bool
        target_mask  : (B, N) bool

        Returns
        -------
        pred_embeds   : (B, N_tgt, embed_dim)  — predictor output
        target_embeds : (B, N_tgt, embed_dim)  — target encoder output (no grad)
        """
        # Encode context tokens (with gradient)
        ctx_tokens = self.context_encoder(x, keep_mask=context_mask)  # (B, N_ctx, D)

        # Predict target embeddings
        pred_embeds = self.predictor(ctx_tokens, context_mask, target_mask)

        # Encode target tokens with EMA encoder (no gradient)
        with torch.no_grad():
            tgt_tokens = self.target_encoder(x, keep_mask=target_mask)  # (B, N_tgt, D)

        return pred_embeds, tgt_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module, name: str = '') -> int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name or model.__class__.__name__}: {n:,} trainable params  ({n/1e6:.2f}M)")
    return n


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    B, T, C, H, W = 2, 16, 11, 224, 224

    # ── Build model ──────────────────────────────────────────────────────────
    encoder_cfg = dict(
        in_channels=11,
        t_patch=2, h_patch=16, w_patch=16,
        embed_dim=192, depth=4, num_heads=3,
        T=T, H=H, W=W,
    )
    predictor_cfg = dict(
        predictor_embed_dim=96,
        depth=4,
        num_heads=3,
        num_tokens=8 * 14 * 14,  # 1568
    )

    print("=" * 50)
    print("Parameter counts")
    print("=" * 50)
    model = JEPA(encoder_cfg, predictor_cfg).to(device)
    n_ctx  = count_parameters(model.context_encoder, 'context_encoder')
    n_pred = count_parameters(model.predictor,       'predictor')
    print(f"  {'total (trainable)':<25}: {(n_ctx+n_pred):,}  ({(n_ctx+n_pred)/1e6:.2f}M)")
    assert (n_ctx + n_pred) < 100_000_000, "Model exceeds 100M parameter limit!"
    print("  Parameter limit check passed ✓\n")

    # ── Forward pass ─────────────────────────────────────────────────────────
    print("=" * 50)
    print("Forward pass")
    print("=" * 50)
    x = torch.randn(B, T, C, H, W, device=device)

    N = 8 * 14 * 14  # 1568 total tokens
    # Simple mask: first 60% = context, next 20% = target
    context_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    target_mask  = torch.zeros(B, N, dtype=torch.bool, device=device)
    context_mask[:, :940] = True
    target_mask[:,  940:1254] = True

    pred, tgt = model(x, context_mask, target_mask)
    print(f"  Input shape          : {tuple(x.shape)}")
    print(f"  Total tokens (N)     : {N}")
    print(f"  Context tokens       : {context_mask[0].sum().item()}")
    print(f"  Target tokens        : {target_mask[0].sum().item()}")
    print(f"  pred_embeds shape    : {tuple(pred.shape)}")
    print(f"  target_embeds shape  : {tuple(tgt.shape)}")

    # Simple L2 loss
    loss = torch.nn.functional.mse_loss(pred, tgt)
    loss.backward()
    print(f"  Loss                 : {loss.item():.4f}")
    print(f"  Backward pass        : ✓\n")

    # ── EMA update ───────────────────────────────────────────────────────────
    model.update_target_encoder()
    print("  EMA update           : ✓")

    # ── VRAM ─────────────────────────────────────────────────────────────────
    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak VRAM (B={B})     : {mem:.2f} GB")
