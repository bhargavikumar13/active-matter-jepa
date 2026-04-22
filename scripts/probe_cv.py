"""
scripts/probe_cv.py — 5-fold cross-validation linear probe

Splits the training set into 5 folds. For each fold:
  - Trains a linear probe on 4 folds
  - Evaluates on the held-out fold

Benefits:
  - More reliable MSE estimates (reduces variance from single val split)
  - 5 trained probes that can be ensembled at test time
  - Better hyperparameter selection

Usage
-----
python scripts/probe_cv.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --no-wandb

# Ensemble predictions on test set using all 5 folds
python scripts/probe_cv.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --ensemble_test \
    --no-wandb
"""

import argparse
import os
import sys

import yaml
import numpy as np
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
    train_feats, train_norm,
    val_feats, val_norm,
    embed_dim, cfg, device
):
    train_feats = train_feats.to(device)
    train_norm  = train_norm.to(device)
    val_feats   = val_feats.to(device)
    val_norm    = val_norm.to(device)

    ds = TensorDataset(train_feats, train_norm)
    dl = DataLoader(ds, batch_size=cfg.probe.batch_size, shuffle=True)

    head = nn.Linear(embed_dim, 2).to(device)
    nn.init.trunc_normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)

    optimizer = optim.Adam(
        head.parameters(),
        lr=cfg.probe.lr,
        weight_decay=cfg.probe.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.probe.epochs, eta_min=cfg.probe.lr * 0.01
    )

    best_mse   = float('inf')
    best_state = None

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


def main(cfg, checkpoint_path, n_folds, ensemble_test, use_wandb):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Folds: {n_folds}")

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

    # Extract features
    pool = getattr(cfg.probe, 'pool', 'mean')
    print("\nExtracting features...")
    train_feats, train_labels = extract_features(encoder, train_loader, device, pool)
    val_feats,   val_labels   = extract_features(encoder, val_loader,   device, pool)
    test_feats,  test_labels  = extract_features(encoder, test_loader,  device, pool)

    # Label stats from full train set
    label_mean = train_labels.mean(dim=0)
    label_std  = train_labels.std(dim=0)

    print(f"\nLabel stats:")
    print(f"  alpha: mean={label_mean[0]:.4f} std={label_std[0]:.4f}")
    print(f"  zeta:  mean={label_mean[1]:.4f} std={label_std[1]:.4f}")

    # Normalise
    train_norm = (train_labels - label_mean) / (label_std + 1e-8)
    val_norm   = (val_labels   - label_mean) / (label_std + 1e-8)
    test_norm  = (test_labels  - label_mean) / (label_std + 1e-8)

    N = len(train_feats)
    fold_size = N // n_folds
    indices   = torch.randperm(N)

    # ── Cross-validation ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'Fold':<6} {'val_MSE':<12} {'alpha':<10} {'zeta':<10}")
    print(f"{'='*55}")

    fold_scores = []
    fold_heads  = []

    for fold in range(n_folds):
        # Split indices
        val_idx   = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = torch.cat([
            indices[:fold * fold_size],
            indices[(fold + 1) * fold_size:]
        ])

        fold_train_feats  = train_feats[train_idx]
        fold_train_norm   = train_norm[train_idx]
        fold_val_feats    = train_feats[val_idx]
        fold_val_norm     = train_norm[val_idx]
        fold_val_labels   = train_labels[val_idx]

        best_mse, head = train_probe(
            fold_train_feats, fold_train_norm,
            fold_val_feats,   fold_val_norm,
            embed_dim = cfg.model.embed_dim,
            cfg       = cfg,
            device    = device,
        )

        # Per-parameter MSE on fold val
        head.eval()
        with torch.no_grad():
            vp = head(fold_val_feats.to(device))
            vn = fold_val_norm.to(device)
            mse_alpha = nn.functional.mse_loss(vp[:, 0], vn[:, 0]).item()
            mse_zeta  = nn.functional.mse_loss(vp[:, 1], vn[:, 1]).item()

        print(f"{fold+1:<6} {best_mse:<12.4f} {mse_alpha:<10.4f} {mse_zeta:.4f}")
        fold_scores.append({'mse': best_mse, 'alpha': mse_alpha, 'zeta': mse_zeta})
        fold_heads.append(head)

    mean_mse   = np.mean([s['mse']   for s in fold_scores])
    mean_alpha = np.mean([s['alpha'] for s in fold_scores])
    mean_zeta  = np.mean([s['zeta']  for s in fold_scores])
    std_mse    = np.std( [s['mse']   for s in fold_scores])

    print(f"\nCV summary ({n_folds} folds):")
    print(f"  Mean val MSE : {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"  Mean alpha   : {mean_alpha:.4f}")
    print(f"  Mean zeta    : {mean_zeta:.4f}")

    # ── Ensemble test predictions ─────────────────────────────────────────────
    if ensemble_test:
        print(f"\n── Ensemble test predictions ({n_folds} probes) ──")
        test_preds = []
        for head in fold_heads:
            head.eval()
            with torch.no_grad():
                preds = head(test_feats.to(device)).cpu()
            test_preds.append(preds)

        ensemble_pred = torch.stack(test_preds, dim=0).mean(dim=0)
        tn = test_norm.to(device)
        ep = ensemble_pred.to(device)

        test_mse   = nn.functional.mse_loss(ep, tn).item()
        test_alpha = nn.functional.mse_loss(ep[:, 0], tn[:, 0]).item()
        test_zeta  = nn.functional.mse_loss(ep[:, 1], tn[:, 1]).item()

        print(f"  MSE (normalised, combined) : {test_mse:.4f}")
        print(f"  MSE alpha (normalised)     : {test_alpha:.4f}")
        print(f"  MSE zeta  (normalised)     : {test_zeta:.4f}")

    # Also evaluate val MSE on the held-out val split (not part of CV folds)
    print(f"\n── Held-out val set evaluation (single best-fold probe) ──")
    best_fold = int(np.argmin([s['mse'] for s in fold_scores]))
    best_head = fold_heads[best_fold]
    best_head.eval()
    with torch.no_grad():
        vp = best_head(val_feats.to(device))
        vn = val_norm.to(device)
        val_mse   = nn.functional.mse_loss(vp, vn).item()
        val_alpha = nn.functional.mse_loss(vp[:, 0], vn[:, 0]).item()
        val_zeta  = nn.functional.mse_loss(vp[:, 1], vn[:, 1]).item()

    print(f"  Best fold: {best_fold + 1}")
    print(f"  Val MSE (combined) : {val_mse:.4f}")
    print(f"  Val alpha          : {val_alpha:.4f}")
    print(f"  Val zeta           : {val_zeta:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir   = os.path.join(cfg.training.checkpoint_dir, 'probe')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]

    for i, head in enumerate(fold_heads):
        torch.save(head.state_dict(),
                   os.path.join(out_dir, f'probe_cv_fold{i+1}__{ckpt_stem}.pt'))

    results = {
        'n_folds': n_folds,
        'cv_mean_mse':   float(mean_mse),
        'cv_std_mse':    float(std_mse),
        'cv_mean_alpha': float(mean_alpha),
        'cv_mean_zeta':  float(mean_zeta),
        'fold_scores':   fold_scores,
    }
    if ensemble_test:
        results['ensemble_test'] = {
            'mse_combined': float(test_mse),
            'mse_alpha': float(test_alpha),
            'mse_zeta': float(test_zeta),
        }

    out_path = os.path.join(out_dir, f'probe_cv_results__{ckpt_stem}.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(results, f)
    print(f"\nResults saved → {out_path}")
    print(f"Fold weights  → {out_dir}/probe_cv_fold*__{ckpt_stem}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',        required=True)
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--n_folds',       type=int, default=5)
    parser.add_argument('--ensemble_test', action='store_true')
    parser.add_argument('--no-wandb',      action='store_true')
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)

    if 'probe' not in cfg:
        cfg['probe'] = {}
    for k, v in {'epochs': 100, 'lr': 1e-3, 'weight_decay': 1e-4,
                 'batch_size': 256, 'extract_batch_size': 8, 'pool': 'mean'}.items():
        if k not in cfg['probe']:
            cfg['probe'][k] = v

    main(cfg, args.checkpoint, args.n_folds,
         ensemble_test=args.ensemble_test,
         use_wandb=not args.no_wandb)
