"""
scripts/supervised.py — End-to-end supervised baseline for active_matter

Trains a VisionTransformer3D encoder + linear head end-to-end to predict
alpha and zeta directly from the input frames. This serves as an upper-bound
comparison in the report against the frozen JEPA representations.

Key differences from the SSL pipeline:
  • Labels ARE used during training (supervised)
  • Encoder is NOT frozen — gradients flow through the full model
  • Single linear head on top of mean-pooled encoder output
  • Evaluated with same MSE metric as linear probe for fair comparison

Usage
-----
python scripts/supervised.py --config configs/jepa.yaml

# Resume after preemption
python scripts/supervised.py --config configs/jepa.yaml
"""

import argparse
import os
import signal
import sys
import time
import math

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DotDict, load_config
from src.dataset import build_dataloaders
from src.model   import VisionTransformer3D




# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int,
           base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)
    print(f"  Checkpoint saved → {path}")


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.endswith('.pt') and not f.endswith('.tmp')
    ])
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else None


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: DotDict, use_wandb: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)

    # ── Stats ──────────────────────────────────────────────────────────────
    stats = None
    if os.path.exists(cfg.data.stats_path):
        with open(cfg.data.stats_path) as f:
            raw = yaml.safe_load(f)

        def _flatten(v) -> list:
            if isinstance(v, list):
                out = []
                for item in v:
                    out.extend(_flatten(item))
                return out
            return [float(v)]

        def _parse_field_dict(d):
            return (
                _flatten(d['concentration'])
                + _flatten(d['velocity'])
                + _flatten(d['D'])
                + _flatten(d['E'])
            )

        mean_list = _parse_field_dict(raw['mean'])
        std_list  = _parse_field_dict(raw['std'])
        if len(mean_list) == 11:
            stats = {'mean': mean_list, 'std': std_list}

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir     = cfg.data.root_dir,
        stats        = stats,
        clip_len     = cfg.data.clip_len,
        spatial_size = cfg.data.spatial_size,
        batch_size   = cfg.supervised.batch_size,
        num_workers  = cfg.training.num_workers,
    )

    # ── Label stats for normalisation (train split only) ──────────────────
    print("Computing label stats...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    all_labels  = torch.cat(all_labels, dim=0)
    label_mean  = all_labels.mean(dim=0).to(device)
    label_std   = all_labels.std(dim=0).to(device)
    print(f"  alpha: mean={label_mean[0]:.4f}  std={label_std[0]:.4f}")
    print(f"  zeta:  mean={label_mean[1]:.4f}  std={label_std[1]:.4f}")

    # ── Model ──────────────────────────────────────────────────────────────
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
    encoder = VisionTransformer3D(**encoder_cfg).to(device)
    head    = nn.Linear(cfg.model.embed_dim, 2).to(device)
    nn.init.trunc_normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)

    n_params = sum(p.numel() for p in encoder.parameters()) + \
               sum(p.numel() for p in head.parameters())
    print(f"\nModel parameters: {n_params/1e6:.2f}M  (encoder + head)")

    # ── Optimizer & scaler ────────────────────────────────────────────────
    params    = list(encoder.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(
        params,
        lr           = cfg.supervised.lr,
        weight_decay = cfg.supervised.weight_decay,
        betas        = (0.9, 0.95),
    )
    scaler = GradScaler('cuda', enabled=cfg.training.use_amp)

    # ── LR schedule ───────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps     = cfg.supervised.epochs * steps_per_epoch
    warmup_steps    = cfg.supervised.warmup_epochs * steps_per_epoch

    # ── Checkpoint dir ────────────────────────────────────────────────────
    ckpt_dir = os.path.join(
        os.path.dirname(cfg.training.checkpoint_dir),
        'supervised'
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch   = 0
    global_step   = 0
    best_val_mse  = float('inf')

    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        ckpt = torch.load(latest_ckpt, map_location='cpu')
        encoder.load_state_dict(ckpt['encoder'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch  = ckpt['epoch'] + 1
        global_step  = ckpt['step']
        best_val_mse = ckpt.get('best_val_mse', float('inf'))
        print(f"  Resumed from epoch {ckpt['epoch']}  (step {global_step})")

    # ── SIGTERM handler ───────────────────────────────────────────────────
    def _sigterm_handler(signum, frame):
        print("\n[SIGTERM] Saving emergency checkpoint...")
        save_checkpoint({
            'epoch'      : start_epoch,
            'step'       : global_step,
            'encoder'    : encoder.state_dict(),
            'head'       : head.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'scaler'     : scaler.state_dict(),
            'best_val_mse': best_val_mse,
            'label_mean' : label_mean.cpu(),
            'label_std'  : label_std.cpu(),
        }, os.path.join(ckpt_dir, 'emergency.pt'))
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── WandB ─────────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.init(
            project = cfg.logging.wandb_project,
            name    = f"{cfg.logging.run_name}-supervised",
            config  = dict(cfg),
            resume  = 'allow',
        )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nStarting supervised training from epoch {start_epoch} / {cfg.supervised.epochs}\n")

    for epoch in range(start_epoch, cfg.supervised.epochs):
        encoder.train()
        head.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Normalise labels
            labels_norm = (labels - label_mean) / (label_std + 1e-8)

            # LR update
            lr = get_lr(global_step, warmup_steps, total_steps,
                        cfg.supervised.lr, cfg.supervised.min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=cfg.training.use_amp):
                tokens = encoder(frames)          # (B, N, D)
                feats  = tokens.mean(dim=1)       # (B, D) mean pool
                preds  = head(feats)              # (B, 2)
                loss   = nn.functional.mse_loss(preds, labels_norm)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            global_step += 1

            if batch_idx % cfg.logging.log_every == 0:
                print(
                    f"  Epoch {epoch:03d} [{batch_idx:4d}/{steps_per_epoch}]  "
                    f"loss={loss.item():.4f}  lr={lr:.2e}"
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        'supervised/train_loss': loss.item(),
                        'supervised/lr'        : lr,
                        'supervised/step'      : global_step,
                    }, step=global_step)

        avg_loss   = epoch_loss / steps_per_epoch
        epoch_time = time.time() - epoch_start

        # ── Validation ───────────────────────────────────────────────────
        val_mse_norm, val_mse_alpha, val_mse_zeta = evaluate(
            encoder, head, val_loader, label_mean, label_std, device, cfg
        )
        print(
            f"\nEpoch {epoch:03d} done | avg_loss={avg_loss:.4f} | "
            f"val_mse_norm={val_mse_norm:.4f} "
            f"(α={val_mse_alpha:.4f} ζ={val_mse_zeta:.4f}) | "
            f"time={epoch_time:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log({
                'supervised/val_mse_norm' : val_mse_norm,
                'supervised/val_mse_alpha': val_mse_alpha,
                'supervised/val_mse_zeta' : val_mse_zeta,
                'supervised/epoch'        : epoch,
            }, step=global_step)

        is_best = val_mse_norm < best_val_mse
        if is_best:
            best_val_mse = val_mse_norm

        state = {
            'epoch'       : epoch,
            'step'        : global_step,
            'encoder'     : encoder.state_dict(),
            'head'        : head.state_dict(),
            'optimizer'   : optimizer.state_dict(),
            'scaler'      : scaler.state_dict(),
            'best_val_mse': best_val_mse,
            'label_mean'  : label_mean.cpu(),
            'label_std'   : label_std.cpu(),
        }
        save_checkpoint(state, os.path.join(ckpt_dir, 'latest.pt'))
        if is_best:
            save_checkpoint(state, os.path.join(ckpt_dir, 'best.pt'))
        if epoch % cfg.training.save_every == 0:
            save_checkpoint(state, os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pt'))
        print()

    # ── Final test evaluation ─────────────────────────────────────────────
    best_ckpt = torch.load(os.path.join(ckpt_dir, 'best.pt'), map_location='cpu')
    encoder.load_state_dict(best_ckpt['encoder'])
    head.load_state_dict(best_ckpt['head'])

    test_mse_norm, test_mse_alpha, test_mse_zeta = evaluate(
        encoder, head, test_loader, label_mean, label_std, device, cfg
    )

    print(f"\n{'='*55}")
    print(f"Supervised baseline — test results")
    print(f"{'='*55}")
    print(f"  MSE (normalised, combined) : {test_mse_norm:.4f}")
    print(f"  MSE alpha (normalised)     : {test_mse_alpha:.4f}")
    print(f"  MSE zeta  (normalised)     : {test_mse_zeta:.4f}")

    results = {
        'mse_normalised'      : test_mse_norm,
        'mse_alpha_normalised': test_mse_alpha,
        'mse_zeta_normalised' : test_mse_zeta,
    }
    with open(os.path.join(ckpt_dir, 'supervised_results.yaml'), 'w') as f:
        yaml.dump(results, f)
    print(f"\nResults saved → {ckpt_dir}/supervised_results.yaml")

    if use_wandb:
        import wandb
        wandb.log({
            'supervised/test_mse_norm' : test_mse_norm,
            'supervised/test_mse_alpha': test_mse_alpha,
            'supervised/test_mse_zeta' : test_mse_zeta,
        })
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    encoder    : VisionTransformer3D,
    head       : nn.Linear,
    loader     : DataLoader,
    label_mean : torch.Tensor,
    label_std  : torch.Tensor,
    device     : torch.device,
    cfg        : DotDict,
) -> tuple[float, float, float]:
    encoder.eval()
    head.eval()
    total_mse = total_alpha = total_zeta = 0.0
    n_batches = min(len(loader), cfg.supervised.val_batches)

    for i, (frames, labels) in enumerate(loader):
        if i >= n_batches:
            break
        frames      = frames.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)
        labels_norm = (labels - label_mean) / (label_std + 1e-8)

        tokens = encoder(frames)
        feats  = tokens.mean(dim=1)
        preds  = head(feats)

        total_mse   += nn.functional.mse_loss(preds, labels_norm).item()
        total_alpha += nn.functional.mse_loss(preds[:, 0], labels_norm[:, 0]).item()
        total_zeta  += nn.functional.mse_loss(preds[:, 1], labels_norm[:, 1]).item()

    encoder.train()
    head.train()
    return (
        total_mse   / n_batches,
        total_alpha / n_batches,
        total_zeta  / n_batches,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised baseline for active_matter')
    parser.add_argument('--config',   default='configs/jepa.yaml')
    parser.add_argument('--no-wandb', action='store_true')
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)

    # Inject supervised defaults if not in config
    if 'supervised' not in cfg:
        cfg['supervised'] = {}
    defaults = {
        'epochs'       : 100,
        'lr'           : 3.0e-4,
        'min_lr'       : 1.0e-6,
        'weight_decay' : 0.05,
        'warmup_epochs': 10,
        'batch_size'   : 8,
        'val_batches'  : 20,
    }
    for k, v in defaults.items():
        if k not in cfg['supervised']:
            cfg['supervised'][k] = v

    train(cfg, use_wandb=not args.no_wandb)
