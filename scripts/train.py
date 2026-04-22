"""
scripts/train.py — JEPA pre-training for active_matter

Usage
-----
# Single GPU
python scripts/train.py --config configs/jepa.yaml

# Resume after preemption (same command — script detects checkpoint)
python scripts/train.py --config configs/jepa.yaml

# Override config values on the command line
python scripts/train.py --config configs/jepa.yaml training.lr=3e-4 training.batch_size=4

Checkpoint / restart
--------------------
Checkpoints are saved to cfg.training.checkpoint_dir every
cfg.training.save_every epochs, and always at the end of each epoch.
On preemption (SIGTERM from Slurm), the signal handler saves an
emergency checkpoint before the process exits.
The script resumes automatically from the latest checkpoint if one exists.

WandB
-----
Set WANDB_API_KEY in your environment or run `wandb login` before training.
Disable with --no-wandb.
"""

import argparse
import os
import random
import signal
import sys
import time
import math

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DotDict, load_config
from src.dataset import ActiveMatterDataset, build_dataloaders
from src.model   import JEPA, VisionTransformer3D
from src.masking import sample_jepa_masks


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = 'smooth_l1',
) -> torch.Tensor:
    """
    Compute JEPA loss between predicted and target embeddings.

    Parameters
    ----------
    pred, target : (B, N_tgt, D)
    loss_type    : 'smooth_l1' | 'mse' | 'cosine'
    """
    if loss_type == 'smooth_l1':
        return nn.functional.smooth_l1_loss(pred, target)
    elif loss_type == 'mse':
        return nn.functional.mse_loss(pred, target)
    elif loss_type == 'cosine':
        # 1 - mean cosine similarity (higher sim = lower loss)
        pred_n   = nn.functional.normalize(pred,   dim=-1)
        target_n = nn.functional.normalize(target, dim=-1)
        return 1.0 - (pred_n * target_n).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_ema_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """Anneal EMA momentum from start → end over training."""
    progress = step / max(1, total_steps)
    return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic write — safe if preempted mid-save
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path: str, model: JEPA, optimizer: optim.Optimizer, scaler: GradScaler):
    ckpt = torch.load(path, map_location='cpu')
    model.context_encoder.load_state_dict(ckpt['context_encoder'])
    model.target_encoder.load_state_dict(ckpt['target_encoder'])
    model.predictor.load_state_dict(ckpt['predictor'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    print(f"  Resumed from epoch {ckpt['epoch']}  (step {ckpt['step']})")
    return ckpt['epoch'], ckpt['step'], ckpt.get('best_val_loss', float('inf'))


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.endswith('.pt') and not f.endswith('.tmp')
    ])
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else None


# ─────────────────────────────────────────────────────────────────────────────
# Representation collapse check
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def check_collapse(encoder: VisionTransformer3D, loader: DataLoader, device: torch.device, n_batches: int = 10) -> dict:
    """
    Check for representation collapse by computing:
      - std of embedding dimensions (should be > 0.1, collapse → ~0)
      - average cosine similarity between samples (collapse → ~1)
    """
    encoder.eval()
    all_embeds = []
    for i, (frames, _) in enumerate(loader):
        if i >= n_batches:
            break
        frames = frames.to(device)
        embeds = encoder(frames)           # (B, N, D)
        embeds = embeds.mean(dim=1)        # (B, D) — mean pool over tokens
        all_embeds.append(nn.functional.normalize(embeds, dim=-1).cpu())

    all_embeds = torch.cat(all_embeds, dim=0)   # (M, D)

    # Std across samples per dimension
    std_per_dim = all_embeds.std(dim=0).mean().item()

    # Mean pairwise cosine similarity (sample 64 pairs)
    idx = torch.randperm(len(all_embeds))[:64]
    sims = (all_embeds[idx] * all_embeds[idx.roll(1)]).sum(dim=-1)
    mean_cos_sim = sims.mean().item()

    encoder.train()
    return {'embed_std': std_per_dim, 'mean_cos_sim': mean_cos_sim}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: DotDict, use_wandb: bool):

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"  Seed: {cfg.training.seed} | deterministic: True | benchmark: False")

    # ── Data ──────────────────────────────────────────────────────────────────
    stats = None
    if os.path.exists(cfg.data.stats_path):
        with open(cfg.data.stats_path) as f:
            raw = yaml.safe_load(f)
        print(f"Loaded channel stats from {cfg.data.stats_path}")
        print(f"  Raw stats keys: {list(raw.keys())}")

        # Parse the dataset-provided stats.yaml.
        # Channel order must match dataset.py stacking order:
        #   concentration (1) -> velocity (2) -> D (4) -> E (4)  =  11 channels
        #
        # stats.yaml structure:
        #   concentration : scalar float
        #   velocity      : [vx, vy]
        #   D             : [[D00, D01], [D10, D11]]
        #   E             : [[E00, E01], [E10, E11]]

        def _flatten(v) -> list:
            """Recursively flatten nested lists/scalars to flat list of floats."""
            if isinstance(v, list):
                out = []
                for item in v:
                    out.extend(_flatten(item))
                return out
            return [float(v)]

        def _parse_field_dict(d: dict) -> list:
            """Extract 11 channels in correct order from a mean/std sub-dict."""
            return (
                _flatten(d['concentration'])  # 1 ch
                + _flatten(d['velocity'])     # 2 ch
                + _flatten(d['D'])            # 4 ch
                + _flatten(d['E'])            # 4 ch
            )

        if isinstance(raw['mean'], dict):
            mean_list = _parse_field_dict(raw['mean'])
            std_list  = _parse_field_dict(raw['std'])
        else:
            mean_list = [float(x) for x in raw['mean']]
            std_list  = [float(x) for x in raw['std']]
        if len(mean_list) != 11 or len(std_list) != 11:
            print(f"  Warning: expected 11 channels, got "
                  f"mean={len(mean_list)} std={len(std_list)}. "
                  f"Skipping normalisation.")
            stats = None
        else:
            stats = {'mean': mean_list, 'std': std_list}
            print(f"  mean: {[f'{v:.4f}' for v in mean_list]}")
            print(f"  std : {[f'{v:.4f}' for v in std_list]}")
    else:
        print(f"Warning: stats file not found at {cfg.data.stats_path}. "
              f"Run scripts/compute_stats.py first.")

    train_loader, val_loader, _ = build_dataloaders(
        root_dir    = cfg.data.root_dir,
        stats       = stats,
        clip_len    = cfg.data.clip_len,
        spatial_size= cfg.data.spatial_size,
        batch_size  = cfg.training.batch_size,
        num_workers = cfg.training.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder_cfg = dict(
        in_channels  = cfg.model.in_channels,
        t_patch      = cfg.model.t_patch,
        h_patch      = cfg.model.h_patch,
        w_patch      = cfg.model.w_patch,
        embed_dim    = cfg.model.embed_dim,
        depth        = cfg.model.encoder_depth,
        num_heads    = cfg.model.num_heads,
        mlp_ratio    = cfg.model.mlp_ratio,
        T            = cfg.data.clip_len,
        H            = cfg.data.spatial_size,
        W            = cfg.data.spatial_size,
    )
    predictor_cfg = dict(
        predictor_embed_dim = cfg.model.predictor_embed_dim,
        depth               = cfg.model.predictor_depth,
        num_heads           = cfg.model.num_heads,
        num_tokens          = (cfg.data.clip_len // cfg.model.t_patch) *
                              (cfg.data.spatial_size // cfg.model.h_patch) *
                              (cfg.data.spatial_size // cfg.model.w_patch),
    )

    model = JEPA(encoder_cfg, predictor_cfg, ema_momentum=cfg.model.ema_start).to(device)

    n_enc  = sum(p.numel() for p in model.context_encoder.parameters() if p.requires_grad)
    n_pred = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
    print(f"\nModel parameters: encoder={n_enc/1e6:.2f}M  predictor={n_pred/1e6:.2f}M  "
          f"total={((n_enc+n_pred)/1e6):.2f}M")

    # ── Optimizer & scaler ────────────────────────────────────────────────────
    # Only context encoder + predictor are optimised (target encoder = EMA)
    params = list(model.context_encoder.parameters()) + \
             list(model.predictor.parameters())

    optimizer = optim.AdamW(
        params,
        lr           = cfg.training.lr,
        weight_decay = cfg.training.weight_decay,
        betas        = (0.9, 0.95),
    )
    scaler = GradScaler('cuda', enabled=cfg.training.use_amp)

    # ── LR schedule ───────────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps     = cfg.training.epochs * steps_per_epoch
    warmup_steps    = cfg.training.warmup_epochs * steps_per_epoch

    # ── Checkpoint ────────────────────────────────────────────────────────────
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    start_epoch    = 0
    global_step    = 0
    best_val_loss  = float('inf')

    latest_ckpt = find_latest_checkpoint(cfg.training.checkpoint_dir)
    if latest_ckpt:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            latest_ckpt, model, optimizer, scaler
        )
        start_epoch += 1   # resume from the next epoch

    # ── SIGTERM handler (Slurm preemption) ────────────────────────────────────
    def _sigterm_handler(signum, frame):
        print("\n[SIGTERM] Saving emergency checkpoint...")
        save_checkpoint({
            'epoch'          : start_epoch,
            'step'           : global_step,
            'context_encoder': model.context_encoder.state_dict(),
            'target_encoder' : model.target_encoder.state_dict(),
            'predictor'      : model.predictor.state_dict(),
            'optimizer'      : optimizer.state_dict(),
            'scaler'         : scaler.state_dict(),
            'best_val_loss'  : best_val_loss,
            'cfg'            : dict(cfg),
        }, os.path.join(cfg.training.checkpoint_dir, 'emergency.pt'))
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── WandB ─────────────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.init(
            project = cfg.logging.wandb_project,
            name    = cfg.logging.run_name,
            config  = dict(cfg),
            resume  = 'allow',
        )

    # ── Masking config ────────────────────────────────────────────────────────
    n_t = cfg.data.clip_len      // cfg.model.t_patch
    n_h = cfg.data.spatial_size  // cfg.model.h_patch
    n_w = cfg.data.spatial_size  // cfg.model.w_patch

    mask_kwargs = dict(
        n_t                = n_t,
        n_h                = n_h,
        n_w                = n_w,
        num_target_blocks  = cfg.masking.num_target_blocks,
        target_scale       = tuple(cfg.masking.target_scale),
        target_ratio       = tuple(cfg.masking.target_ratio),
        context_keep_ratio = cfg.masking.context_keep_ratio,
        device             = device,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    print(f"\nStarting training from epoch {start_epoch} / {cfg.training.epochs}\n")

    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_loss  = 0.0
        epoch_start = time.time()

        for batch_idx, (frames, _) in enumerate(train_loader):
            # frames : (B, T, C, H, W)  labels not used during SSL
            frames = frames.to(device, non_blocking=True)
            B      = frames.shape[0]

            # ── LR update ────────────────────────────────────────────────────
            lr = get_lr(global_step, warmup_steps, total_steps,
                        cfg.training.lr, cfg.training.min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # ── EMA momentum update ───────────────────────────────────────────
            ema_tau = get_ema_momentum(global_step, total_steps,
                                       cfg.model.ema_start, cfg.model.ema_end)

            # ── Sample masks ─────────────────────────────────────────────────
            ctx_mask, tgt_mask = sample_jepa_masks(batch_size=B, **mask_kwargs)

            # ── Forward + loss ────────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=cfg.training.use_amp):
                pred_embeds, tgt_embeds = model(frames, ctx_mask, tgt_mask)
                loss = jepa_loss(pred_embeds, tgt_embeds, cfg.training.loss_type)

            # ── Backward ──────────────────────────────────────────────────────
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, cfg.training.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # ── EMA update ────────────────────────────────────────────────────
            model.update_target_encoder(momentum=ema_tau)

            # ── Logging ───────────────────────────────────────────────────────
            loss_val    = loss.item()
            epoch_loss += loss_val
            global_step += 1

            if batch_idx % cfg.logging.log_every == 0:
                print(
                    f"  Epoch {epoch:03d} [{batch_idx:4d}/{steps_per_epoch}]  "
                    f"loss={loss_val:.4f}  lr={lr:.2e}  ema_tau={ema_tau:.5f}"
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss'    : loss_val,
                        'train/lr'      : lr,
                        'train/ema_tau' : ema_tau,
                        'train/step'    : global_step,
                    }, step=global_step)

        # ── End of epoch ──────────────────────────────────────────────────────
        avg_loss    = epoch_loss / steps_per_epoch
        epoch_time  = time.time() - epoch_start
        print(f"\nEpoch {epoch:03d} done | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s")

        # ── Collapse check (every N epochs) ───────────────────────────────────
        if epoch % cfg.logging.collapse_check_every == 0:
            collapse_stats = check_collapse(model.context_encoder, val_loader, device)
            print(
                f"  Collapse check — "
                f"embed_std={collapse_stats['embed_std']:.4f}  "
                f"mean_cos_sim={collapse_stats['mean_cos_sim']:.4f}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    'collapse/embed_std'    : collapse_stats['embed_std'],
                    'collapse/mean_cos_sim' : collapse_stats['mean_cos_sim'],
                }, step=global_step)

            if collapse_stats['embed_std'] < 0.01:
                print("  WARNING: possible representation collapse (embed_std < 0.01)")

        # ── Validation loss ───────────────────────────────────────────────────
        val_loss = evaluate(model, val_loader, device, mask_kwargs, cfg)
        print(f"  Val loss: {val_loss:.4f}")
        if use_wandb:
            import wandb
            wandb.log({'val/loss': val_loss, 'epoch': epoch}, step=global_step)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # ── Save checkpoint ───────────────────────────────────────────────────
        state = {
            'epoch'          : epoch,
            'step'           : global_step,
            'context_encoder': model.context_encoder.state_dict(),
            'target_encoder' : model.target_encoder.state_dict(),
            'predictor'      : model.predictor.state_dict(),
            'optimizer'      : optimizer.state_dict(),
            'scaler'         : scaler.state_dict(),
            'best_val_loss'  : best_val_loss,
            'cfg'            : dict(cfg),
        }

        # Always save latest
        save_checkpoint(state, os.path.join(cfg.training.checkpoint_dir, 'latest.pt'))

        # Save periodic checkpoint
        if epoch % cfg.training.save_every == 0:
            save_checkpoint(state, os.path.join(
                cfg.training.checkpoint_dir, f'epoch_{epoch:04d}.pt'
            ))

        # Save best
        if is_best:
            save_checkpoint(state, os.path.join(cfg.training.checkpoint_dir, 'best.pt'))

        print()

    print("Training complete.")
    if use_wandb:
        import wandb
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: JEPA,
    loader: DataLoader,
    device: torch.device,
    mask_kwargs: dict,
    cfg: DotDict,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = min(len(loader), cfg.training.val_batches)

    for i, (frames, _) in enumerate(loader):
        if i >= n_batches:
            break
        frames = frames.to(device, non_blocking=True)
        B      = frames.shape[0]

        ctx_mask, tgt_mask = sample_jepa_masks(batch_size=B, **mask_kwargs)
        pred, tgt = model(frames, ctx_mask, tgt_mask)
        total_loss += jepa_loss(pred, tgt, cfg.training.loss_type).item()

    model.train()
    return total_loss / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JEPA pre-training for active_matter')
    parser.add_argument('--config',   default='configs/jepa.yaml', help='Path to config YAML')
    parser.add_argument('--no-wandb', action='store_true',         help='Disable WandB logging')
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    train(cfg, use_wandb=not args.no_wandb)
