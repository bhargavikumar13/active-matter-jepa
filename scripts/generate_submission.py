"""
scripts/generate_submission.py — Generate Kaggle submission CSV

Loads a trained JEPA encoder + linear probe and runs inference on the
test set, outputting unnormalised alpha and zeta predictions.

Supports:
  - Single checkpoint inference
  - Ensemble of multiple checkpoints (average predictions)
  - Test-time augmentation (TTA) with multiple random crops
  - Isotonic regression calibration on val set (2-3% free gain)

Usage
-----
# Single checkpoint
python scripts/generate_submission.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --probe checkpoints/jepa/probe/linear_probe__best__mean.pt \
    --output submissions/submission_run5.csv

# Ensemble Run 3 + Run 4 + Run 5 with TTA and calibration
python scripts/generate_submission.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
                  checkpoints/jepa_run4/best.pt \
                  checkpoints/jepa_run3/best.pt \
    --probe checkpoints/jepa/probe/linear_probe__best__mean.pt \
            checkpoints/jepa_run4/probe/linear_probe__best__mean.pt \
            checkpoints/jepa_run3/probe/linear_probe__best__mean.pt \
    --config2 configs/jepa_run4.yaml configs/jepa_run3.yaml \
    --tta 5 \
    --calibrate \
    --output submissions/submission_ensemble_tta_cal.csv

# Val set evaluation (verify numbers match paper)
python scripts/generate_submission.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --probe checkpoints/jepa/probe/linear_probe__best__mean.pt \
    --split val \
    --output submissions/val_check.csv
"""

import argparse
import os
import sys
import csv

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DotDict, load_config
from src.dataset import ActiveMatterDataset
from src.model import VisionTransformer3D


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────
# Load encoder
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(
    checkpoint_path: str,
    cfg: DotDict,
    device: torch.device,
) -> VisionTransformer3D:
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    epoch = ckpt.get('epoch', 'unknown')
    print(f"  Loaded encoder from {checkpoint_path}  (epoch {epoch})")

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
    )
    encoder.load_state_dict(ckpt['context_encoder'])
    encoder.eval()
    encoder.to(device)
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# Load linear probe
# ─────────────────────────────────────────────────────────────────────────────

def load_probe(
    probe_path: str,
    embed_dim: int,
    device: torch.device,
) -> nn.Linear:
    head  = nn.Linear(embed_dim, 2)
    saved = torch.load(probe_path, map_location='cpu')
    # probe.py saves a dict with 'head' key; handle both formats
    if isinstance(saved, dict) and 'head' in saved:
        state = saved['head']
    else:
        state = saved
    head.load_state_dict(state)
    head.eval()
    head.to(device)
    print(f"  Loaded probe from {probe_path}")
    return head


# ─────────────────────────────────────────────────────────────────────────────
# Label stats
# ─────────────────────────────────────────────────────────────────────────────

def get_label_stats(
    train_loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)
    mean = all_labels.mean(dim=0)
    std  = all_labels.std(dim=0)
    print(f"  Label stats — alpha: mean={mean[0]:.4f} std={std[0]:.4f}")
    print(f"                zeta:  mean={mean[1]:.4f} std={std[1]:.4f}")
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Inference (single model, with optional TTA)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    encoder    : VisionTransformer3D,
    probe      : nn.Linear,
    loader     : DataLoader,
    label_mean : torch.Tensor,
    label_std  : torch.Tensor,
    device     : torch.device,
    pool       : str = 'mean',
    tta_crops  : int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return unnormalised predictions.

    For TTA the DataLoader must have augment=True so each pass
    sees a different random crop.

    Returns
    -------
    pred_alpha : (N,)  unnormalised
    pred_zeta  : (N,)  unnormalised
    """
    encoder.eval()
    probe.eval()

    label_mean = label_mean.to(device)
    label_std  = label_std.to(device)

    all_crops = []

    for crop_idx in range(tta_crops):
        if tta_crops > 1:
            print(f"    TTA crop {crop_idx + 1}/{tta_crops}...")
        crop_preds = []

        for frames, _ in loader:
            frames    = frames.to(device)
            tokens    = encoder(frames)                       # (B, N, D)

            if pool == 'mean':
                feats = tokens.mean(dim=1)                    # (B, D)
            elif pool == 'cls_like':
                feats = tokens[:, 0, :]                       # (B, D)
            else:
                raise ValueError(f"Unknown pool: {pool}")

            pred_norm = probe(feats)                          # (B, 2)
            pred_raw  = pred_norm * label_std + label_mean    # (B, 2)
            crop_preds.append(pred_raw.cpu())

        all_crops.append(torch.cat(crop_preds, dim=0))        # (N, 2)

    # Average TTA crops
    preds      = torch.stack(all_crops, dim=0).mean(dim=0)    # (N, 2)
    pred_alpha = preds[:, 0].numpy()
    pred_zeta  = preds[:, 1].numpy()

    return pred_alpha, pred_zeta


# ─────────────────────────────────────────────────────────────────────────────
# Isotonic regression calibration
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibration(
    val_pred_alpha : np.ndarray,
    val_pred_zeta  : np.ndarray,
    val_true_alpha : np.ndarray,
    val_true_zeta  : np.ndarray,
):
    """
    Fit isotonic regression calibrators on val set predictions.
    Isotonic regression fits a monotone mapping from predicted to true,
    correcting systematic bias especially at extremes of the range.
    Returns calibrator objects to apply to test predictions.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        print("  WARNING: sklearn not available, skipping calibration")
        return None, None

    cal_alpha = IsotonicRegression(out_of_bounds='clip').fit(
        val_pred_alpha, val_true_alpha
    )
    cal_zeta = IsotonicRegression(out_of_bounds='clip').fit(
        val_pred_zeta, val_true_zeta
    )

    # Report improvement on val set
    raw_mse_alpha = float(np.mean((val_pred_alpha - val_true_alpha) ** 2))
    raw_mse_zeta  = float(np.mean((val_pred_zeta  - val_true_zeta)  ** 2))
    cal_mse_alpha = float(np.mean(
        (cal_alpha.predict(val_pred_alpha) - val_true_alpha) ** 2
    ))
    cal_mse_zeta = float(np.mean(
        (cal_zeta.predict(val_pred_zeta) - val_true_zeta) ** 2
    ))

    print(f"  Calibration on val set (raw units):")
    print(f"    alpha MSE: {raw_mse_alpha:.4f} -> {cal_mse_alpha:.4f}  "
          f"({100*(raw_mse_alpha-cal_mse_alpha)/raw_mse_alpha:+.1f}%)")
    print(f"    zeta  MSE: {raw_mse_zeta:.4f} -> {cal_mse_zeta:.4f}  "
          f"({100*(raw_mse_zeta-cal_mse_zeta)/raw_mse_zeta:+.1f}%)")

    return cal_alpha, cal_zeta


# ─────────────────────────────────────────────────────────────────────────────
# MSE reporting (normalised)
# ─────────────────────────────────────────────────────────────────────────────

def report_mse(
    pred_alpha    : np.ndarray,
    pred_zeta     : np.ndarray,
    true_alpha    : np.ndarray,
    true_zeta     : np.ndarray,
    label_mean_np : np.ndarray,
    label_std_np  : np.ndarray,
    tag           : str = '',
):
    alpha_norm_pred = (pred_alpha - label_mean_np[0]) / label_std_np[0]
    zeta_norm_pred  = (pred_zeta  - label_mean_np[1]) / label_std_np[1]
    alpha_norm_true = (true_alpha - label_mean_np[0]) / label_std_np[0]
    zeta_norm_true  = (true_zeta  - label_mean_np[1]) / label_std_np[1]

    mse_alpha = float(np.mean((alpha_norm_pred - alpha_norm_true) ** 2))
    mse_zeta  = float(np.mean((zeta_norm_pred  - zeta_norm_true)  ** 2))
    mse_comb  = (mse_alpha + mse_zeta) / 2

    prefix = f"[{tag}] " if tag else ""
    print(f"  {prefix}MSE combined (norm) : {mse_comb:.4f}")
    print(f"  {prefix}MSE alpha    (norm) : {mse_alpha:.4f}")
    print(f"  {prefix}MSE zeta     (norm) : {mse_zeta:.4f}")
    return mse_comb, mse_alpha, mse_zeta


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate Kaggle submission CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config',     required=True,
                        help='Primary config YAML (dataset + model 0)')
    parser.add_argument('--checkpoint', required=True, nargs='+',
                        help='Encoder checkpoint(s). Multiple = ensemble.')
    parser.add_argument('--probe',      required=True, nargs='+',
                        help='Linear probe weight(s). Must match --checkpoint.')
    parser.add_argument('--config2',    default=None, nargs='*',
                        help='Configs for checkpoints 2, 3, ... '
                             '(if different architecture). '
                             'Omit to reuse --config for all.')
    parser.add_argument('--output',     default='submission.csv',
                        help='Output CSV path')
    parser.add_argument('--tta',        type=int, default=1,
                        help='TTA crops (1 = disabled, 5 recommended)')
    parser.add_argument('--pool',       default='mean',
                        choices=['mean', 'cls_like'])
    parser.add_argument('--split',      default='test',
                        choices=['test', 'val'],
                        help='Split to run inference on')
    parser.add_argument('--calibrate',  action='store_true',
                        help='Apply isotonic regression calibration '
                             'fitted on val set')
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    if len(args.checkpoint) != len(args.probe):
        raise ValueError(
            f"--checkpoint ({len(args.checkpoint)}) and "
            f"--probe ({len(args.probe)}) must have the same length"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print(f"  JEPA Submission Generator")
    print(f"{'='*55}")
    print(f"  Device        : {device}")
    print(f"  Split         : {args.split}")
    print(f"  Ensemble size : {len(args.checkpoint)}")
    print(f"  TTA crops     : {args.tta}")
    print(f"  Calibrate     : {args.calibrate}")
    print(f"  Output        : {args.output}")

    # ── Load primary config ───────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── Channel stats ─────────────────────────────────────────────────────────
    # stats.yaml stores values nested by field (concentration, velocity, D, E)
    # We must flatten them in the correct channel order: 1 + 2 + 4 + 4 = 11
    with open(cfg.data.stats_path) as f:
        raw = yaml.safe_load(f)

    def _flatten(v) -> list:
        if isinstance(v, list):
            out = []
            for item in v:
                out.extend(_flatten(item))
            return out
        return [float(v)]

    def _parse_field_dict(d: dict) -> list:
        return (
            _flatten(d['concentration'])  # 1 ch
            + _flatten(d['velocity'])     # 2 ch
            + _flatten(d['D'])            # 4 ch
            + _flatten(d['E'])            # 4 ch
        )

    channel_mean = _parse_field_dict(raw['mean'])
    channel_std  = _parse_field_dict(raw['std'])

    assert len(channel_mean) == 11 and len(channel_std) == 11, \
        f"Expected 11 channels, got mean={len(channel_mean)} std={len(channel_std)}"
    print(f"  Channel stats loaded: 11 channels")

    # ── Build loaders ─────────────────────────────────────────────────────────
    print(f"\nBuilding dataloaders...")

    def make_loader(split: str, augment: bool):
        ds = ActiveMatterDataset(
            data_dir     = os.path.join(cfg.data.root_dir, split),
            clip_len     = cfg.data.clip_len,
            spatial_size = cfg.data.spatial_size,
            mean         = channel_mean,
            std          = channel_std,
            augment      = augment,
        )
        loader = DataLoader(
            ds,
            batch_size  = cfg.probe.extract_batch_size,
            shuffle     = False,
            num_workers = cfg.training.num_workers,
            pin_memory  = True,
        )
        return loader, ds

    train_loader,  train_ds  = make_loader('train',      augment=False)
    # Note: validation directory on disk is "valid" not "val"
    split_dir = "valid" if args.split == "val" else args.split
    target_loader, target_ds = make_loader(split_dir,  augment=(args.tta > 1))

    # Val loader for calibration (only needed when running on test split)
    if args.calibrate and args.split == 'test':
        val_loader, val_ds = make_loader('valid', augment=(args.tta > 1))
        print(f"  train  : {len(train_ds)}")
        print(f"  val    : {len(val_ds)}")
        print(f"  test   : {len(target_ds)}")
    else:
        val_loader = target_loader
        val_ds     = target_ds
        print(f"  train  : {len(train_ds)}")
        print(f"  {args.split:<6} : {len(target_ds)}")

    # ── Label stats ───────────────────────────────────────────────────────────
    print("\nComputing label stats from training set...")
    label_mean, label_std = get_label_stats(train_loader)
    label_mean_np = label_mean.numpy()
    label_std_np  = label_std.numpy()

    # ── Collect ground truth labels ───────────────────────────────────────────
    true_alpha, true_zeta = [], []
    for _, labels in target_loader:
        true_alpha.extend(labels[:, 0].numpy())
        true_zeta.extend(labels[:, 1].numpy())
    true_alpha = np.array(true_alpha)
    true_zeta  = np.array(true_zeta)

    val_true_alpha, val_true_zeta = [], []
    if args.calibrate and args.split == 'test':
        for _, labels in val_loader:
            val_true_alpha.extend(labels[:, 0].numpy())
            val_true_zeta.extend(labels[:, 1].numpy())
    val_true_alpha = np.array(val_true_alpha)
    val_true_zeta  = np.array(val_true_zeta)

    # ── Per-model inference ───────────────────────────────────────────────────
    all_alpha_preds     = []
    all_zeta_preds      = []
    all_val_alpha_preds = []
    all_val_zeta_preds  = []

    for i, (ckpt_path, probe_path) in enumerate(
            zip(args.checkpoint, args.probe)):

        print(f"\n── Model {i+1}/{len(args.checkpoint)} "
              f"({os.path.basename(ckpt_path)}) ──")

        # Pick config for this model
        if i == 0:
            model_cfg = cfg
        elif args.config2 and (i - 1) < len(args.config2):
            model_cfg = load_config(args.config2[i - 1])
            print(f"  Config: {args.config2[i-1]}")
        else:
            model_cfg = cfg

        encoder = load_encoder(ckpt_path, model_cfg, device)
        probe   = load_probe(probe_path, model_cfg.model.embed_dim, device)

        # Inference on target split
        print(f"  Inferring on {args.split} (TTA={args.tta})...")
        pred_alpha, pred_zeta = predict(
            encoder, probe, target_loader,
            label_mean, label_std, device,
            pool=args.pool, tta_crops=args.tta,
        )
        all_alpha_preds.append(pred_alpha)
        all_zeta_preds.append(pred_zeta)

        # Inference on val split for calibration
        if args.calibrate and args.split == 'test':
            print(f"  Inferring on val for calibration (TTA={args.tta})...")
            val_pred_alpha, val_pred_zeta = predict(
                encoder, probe, val_loader,
                label_mean, label_std, device,
                pool=args.pool, tta_crops=args.tta,
            )
            all_val_alpha_preds.append(val_pred_alpha)
            all_val_zeta_preds.append(val_pred_zeta)

        del encoder, probe
        torch.cuda.empty_cache()

    # ── Ensemble: average across models ──────────────────────────────────────
    print(f"\n── Ensemble ({len(args.checkpoint)} models) ──")
    final_alpha = np.mean(all_alpha_preds, axis=0)
    final_zeta  = np.mean(all_zeta_preds,  axis=0)

    report_mse(final_alpha, final_zeta, true_alpha, true_zeta,
               label_mean_np, label_std_np, tag='before calibration')

    # ── Isotonic calibration ──────────────────────────────────────────────────
    if args.calibrate:
        print(f"\n── Isotonic Calibration ──")

        if args.split == 'val':
            print("  WARNING: fitting and evaluating on same split "
                  "(in-sample). Use --split test for real submissions.")
            cal_pred_alpha = final_alpha
            cal_pred_zeta  = final_zeta
            cal_true_alpha = true_alpha
            cal_true_zeta  = true_zeta
        else:
            cal_pred_alpha = np.mean(all_val_alpha_preds, axis=0)
            cal_pred_zeta  = np.mean(all_val_zeta_preds,  axis=0)
            cal_true_alpha = val_true_alpha
            cal_true_zeta  = val_true_zeta

        cal_alpha, cal_zeta = fit_calibration(
            cal_pred_alpha, cal_pred_zeta,
            cal_true_alpha, cal_true_zeta,
        )

        if cal_alpha is not None:
            final_alpha = cal_alpha.predict(final_alpha)
            final_zeta  = cal_zeta.predict(final_zeta)
            print(f"\n  After calibration on {args.split}:")
            report_mse(final_alpha, final_zeta, true_alpha, true_zeta,
                       label_mean_np, label_std_np, tag='after calibration')

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Final — {args.split} set")
    print(f"{'='*55}")
    report_mse(final_alpha, final_zeta, true_alpha, true_zeta,
               label_mean_np, label_std_np)
    print(f"  Ensemble : {len(args.checkpoint)} models")
    print(f"  TTA      : {args.tta} crops")
    print(f"  Calibrated: {args.calibrate}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'alpha', 'zeta'])
        for idx, (a, z) in enumerate(zip(final_alpha, final_zeta)):
            writer.writerow([idx, f'{a:.6f}', f'{z:.6f}'])

    print(f"\nSubmission saved -> {args.output}")
    print(f"  Rows    : {len(target_ds)}")
    print(f"  Columns : id, alpha, zeta (unnormalised original units)")


if __name__ == '__main__':
    main()
