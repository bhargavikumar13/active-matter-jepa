"""
scripts/probe.py — Linear probing evaluation for active_matter

Freezes the pre-trained context encoder and trains a single linear layer
to predict the normalised physical parameters alpha (α) and zeta (ζ).

Usage
-----
# After pre-training
python scripts/probe.py --config configs/jepa.yaml --checkpoint checkpoints/jepa/best.pt

# Override any config value
python scripts/probe.py --config configs/jepa.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    probe.lr=1e-3 \
    probe.epochs=50

Rules (from project spec)
--------------------------
• Encoder weights are FROZEN — no gradient flows through the backbone.
• Only a single nn.Linear layer is allowed as the regression head.
• Labels are z-score normalised (mean=0, std=1) before regression.
• Evaluation metric is MSE on the normalised labels.
• Both alpha and zeta are predicted jointly (output dim = 2).
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
from src.dataset import ActiveMatterDataset, build_dataloaders
from src.model   import VisionTransformer3D



# ─────────────────────────────────────────────────────────────────────────────
# Label normalisation
# ─────────────────────────────────────────────────────────────────────────────

def compute_label_stats(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of [alpha, zeta] over the training set."""
    all_labels = []
    for _, labels in loader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)   # (N, 2)
    return all_labels.mean(dim=0), all_labels.std(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    encoder: VisionTransformer3D,
    loader: DataLoader,
    device: torch.device,
    pool: str = 'mean',
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pass all samples through the frozen encoder and return pooled features.

    Parameters
    ----------
    pool : 'mean' | 'cls_like'
        'mean'     — average pool over all tokens → (B, D)
        'cls_like' — use the first token (token at position 0) → (B, D)

    Returns
    -------
    features : (N, D)
    labels   : (N, 2)   [alpha, zeta] — raw (un-normalised)
    """
    encoder.eval()
    all_feats  = []
    all_labels = []

    for frames, labels in loader:
        frames = frames.to(device)
        tokens = encoder(frames)              # (B, N_tokens, D)

        if pool == 'mean':
            feats = tokens.mean(dim=1)        # (B, D)
        elif pool == 'cls_like':
            feats = tokens[:, 0, :]           # (B, D)
        else:
            raise ValueError(f"Unknown pool: {pool}")

        all_feats.append(feats.cpu())
        all_labels.append(labels)

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe training
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_probe(
    train_feats : torch.Tensor,
    train_labels: torch.Tensor,
    val_feats   : torch.Tensor,
    val_labels  : torch.Tensor,
    label_mean  : torch.Tensor,
    label_std   : torch.Tensor,
    embed_dim   : int,
    cfg         : DotDict,
    device      : torch.device,
    use_wandb   : bool,
) -> nn.Linear:
    """
    Train a single nn.Linear layer on top of frozen features.

    Labels are z-score normalised for training, then de-normalised
    for reporting MSE in the original units as well.
    """
    # Normalise labels
    train_labels_norm = (train_labels - label_mean) / (label_std + 1e-8)
    val_labels_norm   = (val_labels   - label_mean) / (label_std + 1e-8)

    # Move everything to device
    train_feats       = train_feats.to(device)
    train_labels_norm = train_labels_norm.to(device)
    val_feats         = val_feats.to(device)
    val_labels_norm   = val_labels_norm.to(device)
    label_mean        = label_mean.to(device)
    label_std         = label_std.to(device)

    # Build TensorDatasets for mini-batch training
    train_ds = TensorDataset(train_feats, train_labels_norm)
    train_dl = DataLoader(train_ds, batch_size=cfg.probe.batch_size, shuffle=True)

    # Single linear layer — the only thing allowed by the project rules
    head = nn.Linear(embed_dim, 2).to(device)
    nn.init.trunc_normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)

    optimizer = optim.Adam(head.parameters(), lr=cfg.probe.lr,
                           weight_decay=cfg.probe.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.probe.epochs, eta_min=cfg.probe.lr * 0.01
    )

    best_val_mse = float('inf')
    best_state   = None

    print(f"\nTraining linear probe for {cfg.probe.epochs} epochs...")
    print(f"  Train samples : {len(train_feats)}")
    print(f"  Val samples   : {len(val_feats)}")
    print(f"  Feature dim   : {embed_dim}\n")

    for epoch in range(cfg.probe.epochs):
        # ── Train ────────────────────────────────────────────────────────────
        head.train()
        epoch_loss = 0.0
        for feats_batch, labels_batch in train_dl:
            optimizer.zero_grad(set_to_none=True)
            preds = head(feats_batch)
            loss  = nn.functional.mse_loss(preds, labels_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────────
        head.eval()
        with torch.no_grad():
            val_preds_norm = head(val_feats)

            # MSE on normalised labels (main metric for the report)
            val_mse_norm = nn.functional.mse_loss(
                val_preds_norm, val_labels_norm
            ).item()

            # Per-parameter MSE on normalised labels
            val_mse_alpha_norm = nn.functional.mse_loss(
                val_preds_norm[:, 0], val_labels_norm[:, 0]
            ).item()
            val_mse_zeta_norm = nn.functional.mse_loss(
                val_preds_norm[:, 1], val_labels_norm[:, 1]
            ).item()

            # De-normalise for interpretable MSE in original units
            val_preds_raw = val_preds_norm * label_std + label_mean
            val_mse_alpha = nn.functional.mse_loss(
                val_preds_raw[:, 0], val_labels[:, 0].to(device)
            ).item()
            val_mse_zeta = nn.functional.mse_loss(
                val_preds_raw[:, 1], val_labels[:, 1].to(device)
            ).item()

        if val_mse_norm < best_val_mse:
            best_val_mse = val_mse_norm
            best_state   = {k: v.clone() for k, v in head.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{cfg.probe.epochs} | "
                f"train_loss={epoch_loss/len(train_dl):.4f} | "
                f"val_mse_norm={val_mse_norm:.4f} "
                f"(α={val_mse_alpha_norm:.4f}, ζ={val_mse_zeta_norm:.4f}) | "
                f"val_mse_raw α={val_mse_alpha:.4f} ζ={val_mse_zeta:.4f}"
            )

        if use_wandb:
            import wandb
            wandb.log({
                'probe/train_loss'      : epoch_loss / len(train_dl),
                'probe/val_mse_norm'    : val_mse_norm,
                'probe/val_mse_alpha_norm': val_mse_alpha_norm,
                'probe/val_mse_zeta_norm' : val_mse_zeta_norm,
                'probe/epoch'           : epoch,
            })

    # Restore best weights
    head.load_state_dict(best_state)
    print(f"\nBest val MSE (normalised): {best_val_mse:.4f}")
    return head


# ─────────────────────────────────────────────────────────────────────────────
# Final evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_probe(
    head        : nn.Linear,
    feats       : torch.Tensor,
    labels      : torch.Tensor,
    label_mean  : torch.Tensor,
    label_std   : torch.Tensor,
    split_name  : str,
    device      : torch.device,
) -> dict:
    """Evaluate the linear probe on a given split."""
    head.eval()
    feats      = feats.to(device)
    label_mean = label_mean.to(device)
    label_std  = label_std.to(device)

    labels_norm = (labels.to(device) - label_mean) / (label_std + 1e-8)

    preds_norm = head(feats)
    preds_raw  = preds_norm * label_std + label_mean

    mse_norm  = nn.functional.mse_loss(preds_norm, labels_norm).item()
    mse_alpha = nn.functional.mse_loss(
        preds_norm[:, 0], labels_norm[:, 0]
    ).item()
    mse_zeta  = nn.functional.mse_loss(
        preds_norm[:, 1], labels_norm[:, 1]
    ).item()

    results = {
        'split'              : split_name,
        'mse_normalised'     : mse_norm,
        'mse_alpha_normalised': mse_alpha,
        'mse_zeta_normalised' : mse_zeta,
    }

    print(f"\n{'='*55}")
    print(f"Linear probe results — {split_name}")
    print(f"{'='*55}")
    print(f"  MSE (normalised, combined) : {mse_norm:.4f}")
    print(f"  MSE alpha (normalised)     : {mse_alpha:.4f}")
    print(f"  MSE zeta  (normalised)     : {mse_zeta:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: DotDict, checkpoint_path: str, use_wandb: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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
        batch_size   = cfg.probe.extract_batch_size,
        num_workers  = cfg.training.num_workers,
    )

    # ── Encoder ────────────────────────────────────────────────────────────
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

    # Load pre-trained weights
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(ckpt['context_encoder'])
    print(f"Loaded encoder from {checkpoint_path}  (epoch {ckpt['epoch']})")

    # Freeze encoder completely
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    print("Encoder frozen ✓")

    # ── Extract features ───────────────────────────────────────────────────
    print("\nExtracting features...")
    pool = cfg.probe.get('pool', 'mean') if hasattr(cfg.probe, 'get') else 'mean'

    print("  train split...")
    train_feats, train_labels = extract_features(encoder, train_loader, device, pool)
    print("  val split...")
    val_feats,   val_labels   = extract_features(encoder, val_loader,   device, pool)
    print("  test split...")
    test_feats,  test_labels  = extract_features(encoder, test_loader,  device, pool)

    print(f"\nFeature shapes:")
    print(f"  train : {train_feats.shape}  labels: {train_labels.shape}")
    print(f"  val   : {val_feats.shape}  labels: {val_labels.shape}")
    print(f"  test  : {test_feats.shape}  labels: {test_labels.shape}")

    # ── Label statistics (computed on train split only) ────────────────────
    label_mean, label_std = compute_label_stats(
        DataLoader(
            torch.utils.data.TensorDataset(train_feats, train_labels),
            batch_size=256
        )
    )
    print(f"\nLabel stats (train):")
    print(f"  alpha  mean={label_mean[0]:.4f}  std={label_std[0]:.4f}")
    print(f"  zeta   mean={label_mean[1]:.4f}  std={label_std[1]:.4f}")

    # ── WandB ──────────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.init(
            project = cfg.logging.wandb_project,
            name    = f"{cfg.logging.run_name}-probe",
            config  = dict(cfg),
            resume  = 'allow',
        )

    # ── Train linear probe ─────────────────────────────────────────────────
    head = train_linear_probe(
        train_feats, train_labels,
        val_feats,   val_labels,
        label_mean,  label_std,
        embed_dim  = cfg.model.embed_dim,
        cfg        = cfg,
        device     = device,
        use_wandb  = use_wandb,
    )

    # ── Evaluate on val and test ───────────────────────────────────────────
    val_results  = evaluate_probe(head, val_feats,  val_labels,  label_mean, label_std, 'val',  device)
    test_results = evaluate_probe(head, test_feats, test_labels, label_mean, label_std, 'test', device)

    if use_wandb:
        import wandb
        wandb.log({
            'probe/test_mse_norm'         : test_results['mse_normalised'],
            'probe/test_mse_alpha_norm'   : test_results['mse_alpha_normalised'],
            'probe/test_mse_zeta_norm'    : test_results['mse_zeta_normalised'],
        })
        wandb.finish()

    # ── Save probe weights and results ─────────────────────────────────────
    out_dir = os.path.join(cfg.training.checkpoint_dir, 'probe')
    os.makedirs(out_dir, exist_ok=True)

    # Build a unique filename based on checkpoint name and pool strategy
    # e.g. best__mean, epoch_0040__cls_like
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    run_tag   = f"{ckpt_stem}__{pool}"

    torch.save({
        'head'        : head.state_dict(),
        'label_mean'  : label_mean,
        'label_std'   : label_std,
        'embed_dim'   : cfg.model.embed_dim,
        'pool'        : pool,
        'val_results' : val_results,
        'test_results': test_results,
    }, os.path.join(out_dir, f'linear_probe__{run_tag}.pt'))

    # Save results as yaml for easy reading
    results_yaml = {
        'checkpoint'    : checkpoint_path,
        'pool'          : pool,
        'run_tag'       : run_tag,
        'val'           : val_results,
        'test'          : test_results,
    }
    out_path = os.path.join(out_dir, f'probe_results__{run_tag}.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(results_yaml, f, default_flow_style=False)

    print(f"\nResults saved → {out_path}")
    print(f"Probe weights  → {out_dir}/linear_probe__{run_tag}.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear probing for active_matter')
    parser.add_argument('--config',     required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', required=True, help='Path to pre-trained checkpoint')
    parser.add_argument('--no-wandb',   action='store_true')
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)

    # Inject probe defaults if not in config
    if 'probe' not in cfg:
        cfg['probe'] = {}
    probe_defaults = {
        'epochs'            : 100,
        'lr'                : 1e-3,
        'weight_decay'      : 1e-4,
        'batch_size'        : 256,
        'extract_batch_size': 8,
        'pool'              : 'mean',
    }
    for k, v in probe_defaults.items():
        if k not in cfg['probe']:
            cfg['probe'][k] = v

    main(cfg, args.checkpoint, use_wandb=not args.no_wandb)
