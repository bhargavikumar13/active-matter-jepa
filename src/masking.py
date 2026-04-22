"""
masking.py — Spatiotemporal masking for JEPA pre-training

The token grid for active_matter is:
    n_t=8  (temporal)  × n_h=14 (height) × n_w=14 (width)  =  1568 tokens

JEPA masking strategy
---------------------
For each sample we produce:
    • context_mask : (N,) bool  — tokens the encoder sees
    • target_mask  : (N,) bool  — tokens the predictor must reconstruct

Design goals
~~~~~~~~~~~~
1. Target blocks are spatiotemporally *contiguous* — the model must predict
   coherent physical regions, not scattered random pixels.
2. Context and target are *disjoint* — no information leakage.
3. Enough context is kept so the encoder has something to work with.

Strategy: Multi-block target masking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sample M random 3-D bounding boxes on the (n_t, n_h, n_w) grid.
All tokens inside any box → target.
All remaining tokens    → context  (with optional random dropout).

Typical values (matching the I-JEPA / Video-JEPA papers):
    M            = 4   target blocks
    target_scale = (0.15, 0.30)  fraction of tokens per block
    target_ratio = (0.75, 1.50)  spatial aspect ratio (h/w)
    context_keep = 0.90          fraction of non-target tokens to keep

With M=4, scale~0.2:
    target  ≈ 4 × 0.2 × 1568 ≈ 314  tokens  (~20% of N, less due to overlap)
    context ≈ 0.90 × (1568 − target)         (~70% of N)
"""

import math
import random
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def grid_to_idx(
    t0: int, t1: int,
    h0: int, h1: int,
    w0: int, w1: int,
    n_t: int, n_h: int, n_w: int,
) -> torch.Tensor:
    """
    Return a 1-D tensor of flat token indices for the 3-D bounding box
    [t0:t1, h0:h1, w0:w1] on the (n_t, n_h, n_w) grid.
    Flat index = t * n_h * n_w + h * n_w + w.
    """
    ts = torch.arange(t0, t1)
    hs = torch.arange(h0, h1)
    ws = torch.arange(w0, w1)

    tt, hh, ww = torch.meshgrid(ts, hs, ws, indexing='ij')
    idx = tt * (n_h * n_w) + hh * n_w + ww
    return idx.reshape(-1)


def sample_block(
    n_t: int, n_h: int, n_w: int,
    scale: tuple[float, float] = (0.15, 0.30),
    ratio: tuple[float, float] = (0.75, 1.50),
) -> tuple[int, int, int, int, int, int]:
    """
    Sample a random 3-D bounding box on the token grid.

    The spatial area of the block is drawn uniformly from
    [scale[0]*N, scale[1]*N], then the aspect ratio (h/w) is drawn
    from [ratio[0], ratio[1]], and t is drawn proportionally.

    Returns (t0, t1, h0, h1, w0, w1).
    """
    N = n_t * n_h * n_w

    for _ in range(20):   # retry if block doesn't fit
        target_area = random.uniform(scale[0], scale[1]) * N
        ar = random.uniform(ratio[0], ratio[1])           # h / w

        # Solve: n_h_blk / n_w_blk = ar,  n_t_blk * n_h_blk * n_w_blk = target_area
        # Choose n_t proportional to the time fraction
        t_frac = random.uniform(0.5, 1.0)
        n_t_blk = max(1, round(t_frac * n_t))
        spatial  = target_area / n_t_blk

        n_w_blk = max(1, round(math.sqrt(spatial / ar)))
        n_h_blk = max(1, round(ar * n_w_blk))

        if n_t_blk <= n_t and n_h_blk <= n_h and n_w_blk <= n_w:
            t0 = random.randint(0, n_t - n_t_blk)
            h0 = random.randint(0, n_h - n_h_blk)
            w0 = random.randint(0, n_w - n_w_blk)
            return t0, t0 + n_t_blk, h0, h0 + n_h_blk, w0, w0 + n_w_blk

    # Fallback: small centre block
    t0, h0, w0 = 0, n_h // 4, n_w // 4
    return t0, min(t0 + 2, n_t), h0, min(h0 + 4, n_h), w0, min(w0 + 4, n_w)


# ─────────────────────────────────────────────────────────────────────────────
# Main masking function
# ─────────────────────────────────────────────────────────────────────────────

def sample_jepa_masks(
    batch_size: int,
    n_t: int = 8,
    n_h: int = 14,
    n_w: int = 14,
    num_target_blocks: int = 4,
    target_scale: tuple[float, float] = (0.15, 0.30),
    target_ratio: tuple[float, float] = (0.75, 1.50),
    context_keep_ratio: float = 0.90,
    device: torch.device | str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample context and target masks for a batch.

    Strategy
    --------
    1. Sample `num_target_blocks` 3-D blocks; union them → target set.
    2. Context = all non-target tokens, then randomly drop
       (1 - context_keep_ratio) of them.
    3. Pad/trim so every sample in the batch has the same number of
       context tokens and the same number of target tokens (required for
       batched matrix ops in the predictor).

    Parameters
    ----------
    batch_size         : B
    n_t, n_h, n_w      : token grid dimensions
    num_target_blocks  : M — number of target bounding boxes
    target_scale       : (min, max) fraction of N per block
    target_ratio       : (min, max) spatial aspect ratio h/w
    context_keep_ratio : fraction of non-target tokens kept as context
    device             : torch device

    Returns
    -------
    context_mask : (B, N) bool
    target_mask  : (B, N) bool

    Note: N_ctx and N_tgt are the same for all samples in the batch.
    The actual counts depend on the random draws; they are printed during
    the first call in train.py for verification.
    """
    N = n_t * n_h * n_w

    context_masks = []
    target_masks  = []
    n_ctx_list    = []
    n_tgt_list    = []

    for _ in range(batch_size):
        # ── 1. Build target mask ──────────────────────────────────────────
        tgt_idx = set()
        for _ in range(num_target_blocks):
            t0, t1, h0, h1, w0, w1 = sample_block(n_t, n_h, n_w, target_scale, target_ratio)
            idx = grid_to_idx(t0, t1, h0, h1, w0, w1, n_t, n_h, n_w)
            tgt_idx.update(idx.tolist())

        tgt_mask = torch.zeros(N, dtype=torch.bool)
        tgt_mask[list(tgt_idx)] = True

        # ── 2. Build context mask ─────────────────────────────────────────
        non_tgt_idx = (~tgt_mask).nonzero(as_tuple=True)[0]   # all non-target
        n_keep = max(1, round(context_keep_ratio * len(non_tgt_idx)))
        perm   = torch.randperm(len(non_tgt_idx))[:n_keep]
        ctx_idx = non_tgt_idx[perm]

        ctx_mask = torch.zeros(N, dtype=torch.bool)
        ctx_mask[ctx_idx] = True

        context_masks.append(ctx_mask)
        target_masks.append(tgt_mask)
        n_ctx_list.append(ctx_mask.sum().item())
        n_tgt_list.append(tgt_mask.sum().item())

    # ── 3. Make N_ctx and N_tgt uniform across the batch ─────────────────
    # Take the minimum so no sample needs padding (we just drop a few tokens)
    min_ctx = int(min(n_ctx_list))
    min_tgt = int(min(n_tgt_list))

    final_ctx = []
    final_tgt = []

    for ctx_mask, tgt_mask in zip(context_masks, target_masks):
        # Trim context to min_ctx
        ctx_idx = ctx_mask.nonzero(as_tuple=True)[0]
        if len(ctx_idx) > min_ctx:
            keep = ctx_idx[torch.randperm(len(ctx_idx))[:min_ctx]]
            ctx_mask = torch.zeros(N, dtype=torch.bool)
            ctx_mask[keep] = True

        # Trim target to min_tgt
        tgt_idx = tgt_mask.nonzero(as_tuple=True)[0]
        if len(tgt_idx) > min_tgt:
            keep = tgt_idx[torch.randperm(len(tgt_idx))[:min_tgt]]
            tgt_mask = torch.zeros(N, dtype=torch.bool)
            tgt_mask[keep] = True

        final_ctx.append(ctx_mask)
        final_tgt.append(tgt_mask)

    context_mask = torch.stack(final_ctx).to(device)   # (B, N)
    target_mask  = torch.stack(final_tgt).to(device)   # (B, N)

    return context_mask, target_mask


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    B = 4
    n_t, n_h, n_w = 8, 14, 14
    N = n_t * n_h * n_w

    ctx_mask, tgt_mask = sample_jepa_masks(
        batch_size=B,
        n_t=n_t, n_h=n_h, n_w=n_w,
        num_target_blocks=4,
        target_scale=(0.15, 0.30),
        context_keep_ratio=0.90,
    )

    print("=" * 50)
    print("Mask statistics (per sample)")
    print("=" * 50)
    for i in range(B):
        n_ctx = ctx_mask[i].sum().item()
        n_tgt = tgt_mask[i].sum().item()
        overlap = (ctx_mask[i] & tgt_mask[i]).sum().item()
        print(
            f"  Sample {i}: "
            f"context={n_ctx} ({100*n_ctx/N:.1f}%)  "
            f"target={n_tgt} ({100*n_tgt/N:.1f}%)  "
            f"overlap={overlap}  "
            f"unmasked={N - n_ctx - n_tgt} ({100*(N-n_ctx-n_tgt)/N:.1f}%)"
        )

    assert (ctx_mask & tgt_mask).sum() == 0, "Context and target overlap!"
    print("\nNo context/target overlap ✓")

    # ── Visualise one sample as a spatial grid (collapsed over time) ──────
    sample = 0
    ctx = ctx_mask[sample].reshape(n_t, n_h, n_w)
    tgt = tgt_mask[sample].reshape(n_t, n_h, n_w)

    # Show the middle time step
    t_mid = n_t // 2
    grid  = torch.zeros(n_h, n_w)
    grid[ctx[t_mid]] = 0.5   # context = grey
    grid[tgt[t_mid]] = 1.0   # target  = white
    # 0 = dropped context (dark)

    fig, axes = plt.subplots(1, n_t, figsize=(n_t * 2, 2.5))
    for t in range(n_t):
        g = torch.zeros(n_h, n_w)
        g[ctx[t]] = 0.5
        g[tgt[t]] = 1.0
        axes[t].imshow(g.numpy(), vmin=0, vmax=1, cmap='gray', interpolation='nearest')
        axes[t].set_title(f't={t}', fontsize=8)
        axes[t].axis('off')

    fig.suptitle(
        'Token mask  |  dark=dropped  grey=context  white=target',
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig('mask_visualisation.png', dpi=120)
    print("\nSaved mask_visualisation.png")
