"""
scripts/probe_ensemble_checkpoints.py — Ensemble probes from multiple epochs

Trains linear probes on checkpoints from different training epochs
(e.g., epoch 80, 90, best) and averages their test predictions.

Different epochs capture slightly different representation quality
and ensembling them often gives 2-3% improvement.

Usage
-----
python scripts/probe_ensemble_checkpoints.py \
    --config configs/jepa_run5.yaml \
    --checkpoints checkpoints/jepa/epoch_0080.pt \
                  checkpoints/jepa/epoch_0090.pt \
                  checkpoints/jepa/best.pt \
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


def load_encoder(ckpt_path, cfg, device):
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
    ckpt = torch.load(ckpt_path, map_location='cpu')
    encoder.load_state_dict(ckpt['context_encoder'])
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder, ckpt.get('epoch', '?')


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


def train_probe(train_feats, train_norm, val_feats, val_norm,
                embed_dim, cfg, device):
    train_feats = train_feats.to(device)
    train_norm  = train_norm.to(device)
    val_feats   = val_feats.to(device)
    val_norm    = val_norm.to(device)

    ds = TensorDataset(train_feats, train_norm)
    dl = DataLoader(ds, batch_size=cfg.probe.batch_size, shuffle=True)

    head = nn.Linear(embed_dim, 2).to(device)
    nn.init.trunc_normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)

    optimizer = optim.Adam(head.parameters(), lr=cfg.probe.lr,
                           weight_decay=cfg.probe.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.probe.epochs, eta_min=cfg.probe.lr * 0.01
    )

    best_mse, best_state = float('inf'), None
    for epoch in range(cfg.probe.epochs):
        head.train()
        for fb, lb in dl:
            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(head(fb), lb).backward()
            optimizer.step()
        scheduler.step()
        head.eval()
        with torch.no_grad():
            mse = nn.functional.mse_loss(head(val_feats), val_norm).item()
        if mse < best_mse:
            best_mse   = mse
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    return best_mse, head


def main(cfg, checkpoint_paths, use_wandb):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoints: {checkpoint_paths}")

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

    pool = getattr(cfg.probe, 'pool', 'mean')
    all_test_preds = []
    results        = []

    for ckpt_path in checkpoint_paths:
        print(f"\n── Checkpoint: {os.path.basename(ckpt_path)} ──")
        encoder, epoch = load_encoder(ckpt_path, cfg, device)
        print(f"  epoch {epoch}")

        print("  Extracting features...")
        train_feats, train_labels = extract_features(encoder, train_loader, device, pool)
        val_feats,   val_labels   = extract_features(encoder, val_loader,   device, pool)
        test_feats,  test_labels  = extract_features(encoder, test_loader,  device, pool)

        label_mean = train_labels.mean(dim=0)
        label_std  = train_labels.std(dim=0)
        train_norm = (train_labels - label_mean) / (label_std + 1e-8)
        val_norm   = (val_labels   - label_mean) / (label_std + 1e-8)
        test_norm  = (test_labels  - label_mean) / (label_std + 1e-8)

        val_mse, head = train_probe(
            train_feats, train_norm,
            val_feats,   val_norm,
            embed_dim = cfg.model.embed_dim,
            cfg       = cfg,
            device    = device,
        )

        # Individual test MSE
        head.eval()
        with torch.no_grad():
            tp = head(test_feats.to(device))
            tn = test_norm.to(device)
            test_mse   = nn.functional.mse_loss(tp, tn).item()
            test_alpha = nn.functional.mse_loss(tp[:, 0], tn[:, 0]).item()
            test_zeta  = nn.functional.mse_loss(tp[:, 1], tn[:, 1]).item()

        print(f"  Val MSE: {val_mse:.4f} | Test MSE: {test_mse:.4f} "
              f"(α={test_alpha:.4f}, ζ={test_zeta:.4f})")

        all_test_preds.append(tp.cpu())
        results.append({
            'checkpoint': ckpt_path,
            'epoch': epoch,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'test_alpha': test_alpha,
            'test_zeta': test_zeta,
        })

        del encoder
        torch.cuda.empty_cache()

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Ensemble ({len(checkpoint_paths)} probes)")
    print(f"{'='*55}")

    ensemble_pred = torch.stack(all_test_preds, dim=0).mean(dim=0)

    # Use label stats from last checkpoint (same across all since same train set)
    label_mean = train_labels.mean(dim=0)
    label_std  = train_labels.std(dim=0)
    test_norm  = (test_labels - label_mean) / (label_std + 1e-8)

    ep = ensemble_pred.to(device)
    tn = test_norm.to(device)
    ens_mse   = nn.functional.mse_loss(ep, tn).item()
    ens_alpha = nn.functional.mse_loss(ep[:, 0], tn[:, 0]).item()
    ens_zeta  = nn.functional.mse_loss(ep[:, 1], tn[:, 1]).item()

    print(f"  MSE (normalised, combined) : {ens_mse:.4f}")
    print(f"  MSE alpha (normalised)     : {ens_alpha:.4f}")
    print(f"  MSE zeta  (normalised)     : {ens_zeta:.4f}")

    # Compare to best single model
    best_single = min(results, key=lambda x: x['test_mse'])
    print(f"\n  Best single model: {os.path.basename(best_single['checkpoint'])} "
          f"(test={best_single['test_mse']:.4f})")
    print(f"  Ensemble improvement: "
          f"{100*(best_single['test_mse']-ens_mse)/best_single['test_mse']:+.1f}%")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = os.path.join(cfg.training.checkpoint_dir, 'probe')
    os.makedirs(out_dir, exist_ok=True)

    final_results = {
        'ensemble': {
            'n_checkpoints': len(checkpoint_paths),
            'test_mse_combined': float(ens_mse),
            'test_mse_alpha': float(ens_alpha),
            'test_mse_zeta': float(ens_zeta),
        },
        'individual': results,
    }
    out_path = os.path.join(out_dir, 'probe_ensemble_checkpoints_results.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(final_results, f)
    print(f"\nResults saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      required=True)
    parser.add_argument('--checkpoints', required=True, nargs='+',
                        help='List of checkpoint paths to ensemble')
    parser.add_argument('--no-wandb',    action='store_true')
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)

    if 'probe' not in cfg:
        cfg['probe'] = {}
    for k, v in {'epochs': 100, 'lr': 1e-3, 'weight_decay': 1e-4,
                 'batch_size': 256, 'extract_batch_size': 8, 'pool': 'mean'}.items():
        if k not in cfg['probe']:
            cfg['probe'][k] = v

    main(cfg, args.checkpoints, use_wandb=not args.no_wandb)
