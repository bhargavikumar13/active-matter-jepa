"""
scripts/probe_sweep.py — L2 regularization sweep for linear probe

Sweeps weight_decay over a log-spaced grid and reports val/test MSE
for each value. Identifies the best regularization strength and saves
the best probe weights.

Usage
-----
python scripts/probe_sweep.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --no-wandb

# Custom sweep values
python scripts/probe_sweep.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --weight_decays 1e-5 1e-4 1e-3 1e-2 1e-1 \
    --no-wandb
"""

import argparse
import os
import sys

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DotDict, load_config
from src.dataset import build_dataloaders
from src.model   import VisionTransformer3D






def load_channel_stats(stats_path):
    with open(stats_path) as f:
        raw = yaml.safe_load(f)

    def _flatten(v):
        if isinstance(v, list):
            out = []
            for item in v:
                out.extend(_flatten(item))
            return out
        return [float(v)]

    def _parse(d):
        return (
            _flatten(d['concentration'])
            + _flatten(d['velocity'])
            + _flatten(d['D'])
            + _flatten(d['E'])
        )
    return _parse(raw['mean']), _parse(raw['std'])


@torch.no_grad()
def extract_features(encoder, loader, device, pool='mean'):
    encoder.eval()
    all_feats, all_labels = [], []
    for frames, labels in loader:
        frames = frames.to(device)
        tokens = encoder(frames)
        feats  = tokens.mean(dim=1) if pool == 'mean' else tokens[:, 0, :]
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def train_probe(
    train_feats, train_labels_norm,
    val_feats, val_labels_norm,
    embed_dim, lr, weight_decay, epochs, batch_size, device
):
    """Train probe with given weight_decay, return best val MSE and weights."""
    train_feats       = train_feats.to(device)
    train_labels_norm = train_labels_norm.to(device)
    val_feats         = val_feats.to(device)
    val_labels_norm   = val_labels_norm.to(device)

    ds = TensorDataset(train_feats, train_labels_norm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    head = nn.Linear(embed_dim, 2).to(device)
    nn.init.trunc_normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)

    optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    best_mse   = float('inf')
    best_state = None

    for epoch in range(epochs):
        head.train()
        for fb, lb in dl:
            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(head(fb), lb).backward()
            optimizer.step()
        scheduler.step()

        head.eval()
        with torch.no_grad():
            mse = nn.functional.mse_loss(head(val_feats), val_labels_norm).item()
        if mse < best_mse:
            best_mse   = mse
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    return best_mse, head


def main(cfg, checkpoint_path, weight_decays, use_wandb):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Sweep weight_decays: {weight_decays}")

    # Stats + data
    channel_mean, channel_std = load_channel_stats(cfg.data.stats_path)
    stats = {'mean': channel_mean, 'std': channel_std}

    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir     = cfg.data.root_dir,
        stats        = stats,
        clip_len     = cfg.data.clip_len,
        spatial_size = cfg.data.spatial_size,
        batch_size   = cfg.probe.extract_batch_size,
        num_workers  = cfg.training.num_workers,
    )

    # Encoder
    encoder = VisionTransformer3D(
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
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(ckpt['context_encoder'])
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    print(f"Loaded encoder (epoch {ckpt['epoch']})")

    # Extract features once
    pool = getattr(cfg.probe, 'pool', 'mean')
    print("\nExtracting features...")
    train_feats, train_labels = extract_features(encoder, train_loader, device, pool)
    val_feats,   val_labels   = extract_features(encoder, val_loader,   device, pool)
    test_feats,  test_labels  = extract_features(encoder, test_loader,  device, pool)

    # Label stats
    label_mean = train_labels.mean(dim=0)
    label_std  = train_labels.std(dim=0)
    train_norm = (train_labels - label_mean) / (label_std + 1e-8)
    val_norm   = (val_labels   - label_mean) / (label_std + 1e-8)
    test_norm  = (test_labels  - label_mean) / (label_std + 1e-8)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'weight_decay':<15} {'val_MSE':<12} {'alpha':<10} {'zeta':<10} {'test_MSE'}")
    print(f"{'='*60}")

    best_wd       = None
    best_val_mse  = float('inf')
    best_head     = None
    sweep_results = []

    for wd in weight_decays:
        val_mse, head = train_probe(
            train_feats, train_norm,
            val_feats,   val_norm,
            embed_dim    = cfg.model.embed_dim,
            lr           = cfg.probe.lr,
            weight_decay = wd,
            epochs       = cfg.probe.epochs,
            batch_size   = cfg.probe.batch_size,
            device       = device,
        )

        # Per-parameter val MSE
        head.eval()
        with torch.no_grad():
            vp = head(val_feats.to(device))
            vn = val_norm.to(device)
            val_alpha = nn.functional.mse_loss(vp[:, 0], vn[:, 0]).item()
            val_zeta  = nn.functional.mse_loss(vp[:, 1], vn[:, 1]).item()

            # Test MSE
            tp = head(test_feats.to(device))
            tn = test_norm.to(device)
            test_mse = nn.functional.mse_loss(tp, tn).item()

        print(f"{wd:<15.0e} {val_mse:<12.4f} {val_alpha:<10.4f} {val_zeta:<10.4f} {test_mse:.4f}")

        sweep_results.append({
            'weight_decay': wd,
            'val_mse': val_mse,
            'val_alpha': val_alpha,
            'val_zeta': val_zeta,
            'test_mse': test_mse,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_wd      = wd
            best_head    = head

    print(f"\nBest weight_decay: {best_wd} (val MSE = {best_val_mse:.4f})")

    # ── Final evaluation with best probe ─────────────────────────────────────
    best_head.eval()
    with torch.no_grad():
        tp = best_head(test_feats.to(device))
        tn = test_norm.to(device)
        test_mse   = nn.functional.mse_loss(tp, tn).item()
        test_alpha = nn.functional.mse_loss(tp[:, 0], tn[:, 0]).item()
        test_zeta  = nn.functional.mse_loss(tp[:, 1], tn[:, 1]).item()

    print(f"\n{'='*55}")
    print(f"Best probe test results (wd={best_wd})")
    print(f"{'='*55}")
    print(f"  MSE (normalised, combined) : {test_mse:.4f}")
    print(f"  MSE alpha (normalised)     : {test_alpha:.4f}")
    print(f"  MSE zeta  (normalised)     : {test_zeta:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir   = os.path.join(cfg.training.checkpoint_dir, 'probe')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]

    torch.save(best_head.state_dict(),
               os.path.join(out_dir, f'probe_sweep_best__{ckpt_stem}.pt'))

    results = {
        'best_weight_decay': best_wd,
        'best_val_mse': best_val_mse,
        'test': {'mse_combined': test_mse, 'mse_alpha': test_alpha, 'mse_zeta': test_zeta},
        'sweep': sweep_results,
    }
    out_path = os.path.join(out_dir, f'probe_sweep_results__{ckpt_stem}.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(results, f)
    print(f"\nResults saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',        required=True)
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--weight_decays', nargs='+', type=float,
                        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument('--no-wandb',      action='store_true')
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)

    if 'probe' not in cfg:
        cfg['probe'] = {}
    for k, v in {'epochs': 100, 'lr': 1e-3, 'weight_decay': 1e-4,
                 'batch_size': 256, 'extract_batch_size': 8, 'pool': 'mean'}.items():
        if k not in cfg['probe']:
            cfg['probe'][k] = v

    main(cfg, args.checkpoint, args.weight_decays, use_wandb=not args.no_wandb)
