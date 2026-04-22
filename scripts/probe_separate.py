"""
scripts/probe_separate.py — Separate linear probes for alpha and zeta

Instead of predicting [alpha, zeta] jointly with one probe, trains
two independent probes — one per parameter. This is especially useful
for zeta which requires finer spatial structure than alpha.

Usage
-----
python scripts/probe_separate.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
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



# ─────────────────────────────────────────────────────────────────────────────
# Stats loading
# ─────────────────────────────────────────────────────────────────────────────

def load_channel_stats(stats_path: str):
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


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(encoder, loader, device, pool='mean'):
    encoder.eval()
    all_feats, all_labels = [], []
    for frames, labels in loader:
        frames = frames.to(device)
        tokens = encoder(frames)
        if pool == 'mean':
            feats = tokens.mean(dim=1)
        else:
            feats = tokens[:, 0, :]
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Single-output linear probe
# ─────────────────────────────────────────────────────────────────────────────

def train_single_probe(
    train_feats  : torch.Tensor,
    train_labels : torch.Tensor,   # (N,) single parameter
    val_feats    : torch.Tensor,
    val_labels   : torch.Tensor,
    label_mean   : float,
    label_std    : float,
    embed_dim    : int,
    cfg          : DotDict,
    device       : torch.device,
    param_name   : str,
) -> nn.Linear:
    """Train a single nn.Linear(embed_dim, 1) for one parameter."""

    # Normalise
    train_norm = (train_labels - label_mean) / (label_std + 1e-8)
    val_norm   = (val_labels   - label_mean) / (label_std + 1e-8)

    train_feats = train_feats.to(device)
    val_feats   = val_feats.to(device)
    train_norm  = train_norm.to(device).unsqueeze(1)   # (N, 1)
    val_norm    = val_norm.to(device).unsqueeze(1)

    ds = TensorDataset(train_feats, train_norm)
    dl = DataLoader(ds, batch_size=cfg.probe.batch_size, shuffle=True)

    head = nn.Linear(embed_dim, 1).to(device)
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

    print(f"\nTraining probe for {param_name} ({cfg.probe.epochs} epochs)...")

    for epoch in range(cfg.probe.epochs):
        head.train()
        for fb, lb in dl:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(head(fb), lb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        head.eval()
        with torch.no_grad():
            val_pred = head(val_feats)
            mse = nn.functional.mse_loss(val_pred, val_norm).item()

        if mse < best_mse:
            best_mse   = mse
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg.probe.epochs} | "
                  f"val_mse_norm={mse:.4f}")

    head.load_state_dict(best_state)
    print(f"  Best val MSE ({param_name}): {best_mse:.4f}")
    return head


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg, checkpoint_path, use_wandb):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Stats
    channel_mean, channel_std = load_channel_stats(cfg.data.stats_path)
    stats = {'mean': channel_mean, 'std': channel_std}

    # Data
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
    print(f"Loaded encoder from {checkpoint_path}  (epoch {ckpt['epoch']})")

    # Extract features
    pool = getattr(cfg.probe, 'pool', 'mean')
    print("\nExtracting features...")
    train_feats, train_labels = extract_features(encoder, train_loader, device, pool)
    val_feats,   val_labels   = extract_features(encoder, val_loader,   device, pool)
    test_feats,  test_labels  = extract_features(encoder, test_loader,  device, pool)

    # Label stats from train
    label_mean = train_labels.mean(dim=0)   # (2,)
    label_std  = train_labels.std(dim=0)    # (2,)

    print(f"\nLabel stats:")
    print(f"  alpha: mean={label_mean[0]:.4f} std={label_std[0]:.4f}")
    print(f"  zeta:  mean={label_mean[1]:.4f} std={label_std[1]:.4f}")

    # ── Train separate probes ────────────────────────────────────────────────
    results = {}
    heads   = {}

    for i, param in enumerate(['alpha', 'zeta']):
        head = train_single_probe(
            train_feats, train_labels[:, i],
            val_feats,   val_labels[:, i],
            label_mean   = label_mean[i].item(),
            label_std    = label_std[i].item(),
            embed_dim    = cfg.model.embed_dim,
            cfg          = cfg,
            device       = device,
            param_name   = param,
        )
        heads[param] = head

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    for split_name, feats, labels in [
        ('val',  val_feats,  val_labels),
        ('test', test_feats, test_labels),
    ]:
        alpha_pred_norm, zeta_pred_norm = [], []

        for param, idx in [('alpha', 0), ('zeta', 1)]:
            head = heads[param]
            head.eval()
            f = feats.to(device)
            with torch.no_grad():
                pred = head(f).squeeze(1)   # (N,)
            # Re-normalise for MSE comparison
            true_norm = (labels[:, idx].to(device) - label_mean[idx].to(device)) \
                        / (label_std[idx].to(device) + 1e-8)
            mse = nn.functional.mse_loss(pred, true_norm).item()

            if param == 'alpha':
                alpha_pred_norm = pred.cpu()
                alpha_true_norm = true_norm.cpu()
                mse_alpha = mse
            else:
                zeta_pred_norm = pred.cpu()
                zeta_true_norm = true_norm.cpu()
                mse_zeta = mse

        mse_combined = (mse_alpha + mse_zeta) / 2

        print(f"Separate probe results — {split_name}")
        print(f"{'='*55}")
        print(f"  MSE (normalised, combined) : {mse_combined:.4f}")
        print(f"  MSE alpha (normalised)     : {mse_alpha:.4f}")
        print(f"  MSE zeta  (normalised)     : {mse_zeta:.4f}")
        results[split_name] = {
            'mse_combined': mse_combined,
            'mse_alpha': mse_alpha,
            'mse_zeta': mse_zeta,
        }

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir  = os.path.join(cfg.training.checkpoint_dir, 'probe')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]

    for param, head in heads.items():
        torch.save(head.state_dict(),
                   os.path.join(out_dir, f'probe_separate_{param}__{ckpt_stem}.pt'))

    with open(os.path.join(out_dir, f'probe_separate_results__{ckpt_stem}.yaml'), 'w') as f:
        yaml.dump(results, f)

    print(f"\nResults saved → {out_dir}/probe_separate_results__{ckpt_stem}.yaml")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--no-wandb',   action='store_true')
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)

    if 'probe' not in cfg:
        cfg['probe'] = {}
    for k, v in {'epochs': 100, 'lr': 1e-3, 'weight_decay': 1e-4,
                 'batch_size': 256, 'extract_batch_size': 8, 'pool': 'mean'}.items():
        if k not in cfg['probe']:
            cfg['probe'][k] = v

    main(cfg, args.checkpoint, use_wandb=not args.no_wandb)
