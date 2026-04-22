"""
scripts/eval_knn.py — kNN regression evaluation for active_matter

Evaluates pre-trained encoder representations using k-Nearest Neighbours
regression to predict the normalised physical parameters alpha (α) and
zeta (ζ).

Usage
-----
python scripts/eval_knn.py \
    --config configs/jepa.yaml \
    --checkpoint checkpoints/jepa/best.pt

# Sweep over k values
python scripts/eval_knn.py \
    --config configs/jepa.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --k 1 5 10 20 50

Rules (from project spec)
--------------------------
• No model weights are updated — pure inference only.
• kNN regression on frozen encoder features.
• Labels are z-score normalised (same stats as linear probe).
• Metric is MSE on normalised labels.
• Both alpha and zeta are evaluated.
"""

import argparse
import os
import sys

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DotDict, load_config
from src.dataset import build_dataloaders
from src.model   import VisionTransformer3D



# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (same as probe.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    encoder: VisionTransformer3D,
    loader,
    device: torch.device,
    pool: str = 'mean',
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    all_feats  = []
    all_labels = []
    for frames, labels in loader:
        frames = frames.to(device)
        tokens = encoder(frames)              # (B, N_tokens, D)
        if pool == 'mean':
            feats = tokens.mean(dim=1)        # (B, D)
        elif pool == 'cls_like':
            feats = tokens[:, 0, :]
        else:
            raise ValueError(f"Unknown pool: {pool}")
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# kNN regression
# ─────────────────────────────────────────────────────────────────────────────

def knn_regression(
    train_feats : torch.Tensor,
    train_labels: torch.Tensor,
    query_feats : torch.Tensor,
    query_labels: torch.Tensor,
    label_mean  : torch.Tensor,
    label_std   : torch.Tensor,
    k           : int,
    batch_size  : int = 256,
) -> dict:
    """
    Weighted kNN regression using cosine similarity as the kernel.

    For each query point:
      1. Find the k nearest neighbours in the train set by cosine similarity.
      2. Predict as the similarity-weighted average of their (normalised) labels.
      3. Compute MSE against the normalised query labels.

    All computations are done on CPU to avoid OOM — features are typically
    small enough (N × 192) that this is fast.

    Parameters
    ----------
    train_feats  : (N_train, D)
    train_labels : (N_train, 2)  raw labels [alpha, zeta]
    query_feats  : (N_query, D)
    query_labels : (N_query, 2)  raw labels
    label_mean   : (2,)
    label_std    : (2,)
    k            : number of neighbours

    Returns
    -------
    dict with MSE metrics
    """
    # Normalise labels
    train_labels_norm = (train_labels - label_mean) / (label_std + 1e-8)
    query_labels_norm = (query_labels - label_mean) / (label_std + 1e-8)

    # L2-normalise features for cosine similarity via dot product
    train_feats_n = F.normalize(train_feats, dim=-1)   # (N_train, D)
    query_feats_n = F.normalize(query_feats, dim=-1)   # (N_query, D)

    N_query = query_feats_n.shape[0]
    preds   = torch.zeros(N_query, 2)

    # Process queries in batches to avoid building an N_query × N_train matrix
    for start in range(0, N_query, batch_size):
        end   = min(start + batch_size, N_query)
        q_bat = query_feats_n[start:end]               # (bs, D)

        # Cosine similarity: (bs, N_train)
        sim = q_bat @ train_feats_n.T

        # Top-k neighbours
        topk_sim, topk_idx = sim.topk(k, dim=-1)       # (bs, k) each

        # Softmax weights over similarities for smooth weighting
        weights = topk_sim.softmax(dim=-1)              # (bs, k)

        # Gather neighbour labels: (bs, k, 2)
        neighbour_labels = train_labels_norm[topk_idx]  # (bs, k, 2)

        # Weighted sum: (bs, 2)
        preds[start:end] = (weights.unsqueeze(-1) * neighbour_labels).sum(dim=1)

    # MSE on normalised labels
    mse_combined = F.mse_loss(preds, query_labels_norm).item()
    mse_alpha    = F.mse_loss(preds[:, 0], query_labels_norm[:, 0]).item()
    mse_zeta     = F.mse_loss(preds[:, 1], query_labels_norm[:, 1]).item()

    return {
        'k'                   : k,
        'mse_normalised'      : mse_combined,
        'mse_alpha_normalised': mse_alpha,
        'mse_zeta_normalised' : mse_zeta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: DotDict, checkpoint_path: str, k_values: list[int], use_wandb: bool):
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
    extract_bs = cfg.probe.extract_batch_size if hasattr(cfg, 'probe') else 8
    pool       = cfg.probe.pool if hasattr(cfg, 'probe') else 'mean'

    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir     = cfg.data.root_dir,
        stats        = stats,
        clip_len     = cfg.data.clip_len,
        spatial_size = cfg.data.spatial_size,
        batch_size   = extract_bs,
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
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(ckpt['context_encoder'])
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    print(f"Loaded encoder from {checkpoint_path}  (epoch {ckpt['epoch']})")
    print("Encoder frozen ✓")

    # ── Extract features ───────────────────────────────────────────────────
    print("\nExtracting features...")
    print("  train...")
    train_feats, train_labels = extract_features(encoder, train_loader, device, pool)
    print("  val...")
    val_feats,   val_labels   = extract_features(encoder, val_loader,   device, pool)
    print("  test...")
    test_feats,  test_labels  = extract_features(encoder, test_loader,  device, pool)

    print(f"\nFeatures extracted:")
    print(f"  train : {train_feats.shape}")
    print(f"  val   : {val_feats.shape}")
    print(f"  test  : {test_feats.shape}")

    # ── Label stats (train only) ───────────────────────────────────────────
    label_mean = train_labels.mean(dim=0)
    label_std  = train_labels.std(dim=0)
    print(f"\nLabel stats (train split):")
    print(f"  alpha  mean={label_mean[0]:.4f}  std={label_std[0]:.4f}")
    print(f"  zeta   mean={label_mean[1]:.4f}  std={label_std[1]:.4f}")

    # ── WandB ──────────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.init(
            project = cfg.logging.wandb_project,
            name    = f"{cfg.logging.run_name}-knn",
            config  = dict(cfg),
        )

    # ── kNN sweep ──────────────────────────────────────────────────────────
    all_results = {'val': [], 'test': []}

    print(f"\n{'='*55}")
    print(f"kNN regression — sweeping k = {k_values}")
    print(f"{'='*55}")

    for k in k_values:
        print(f"\n  k = {k}")

        # Val
        val_res = knn_regression(
            train_feats, train_labels,
            val_feats,   val_labels,
            label_mean,  label_std,
            k=k,
        )
        print(
            f"    val  — MSE_norm={val_res['mse_normalised']:.4f}  "
            f"α={val_res['mse_alpha_normalised']:.4f}  "
            f"ζ={val_res['mse_zeta_normalised']:.4f}"
        )
        all_results['val'].append(val_res)

        # Test
        test_res = knn_regression(
            train_feats, train_labels,
            test_feats,  test_labels,
            label_mean,  label_std,
            k=k,
        )
        print(
            f"    test — MSE_norm={test_res['mse_normalised']:.4f}  "
            f"α={test_res['mse_alpha_normalised']:.4f}  "
            f"ζ={test_res['mse_zeta_normalised']:.4f}"
        )
        all_results['test'].append(test_res)

        if use_wandb:
            import wandb
            wandb.log({
                f'knn/val_mse_norm_k{k}'  : val_res['mse_normalised'],
                f'knn/test_mse_norm_k{k}' : test_res['mse_normalised'],
            })

    # ── Best k summary ─────────────────────────────────────────────────────
    best_val = min(all_results['val'],  key=lambda x: x['mse_normalised'])
    best_test = [r for r in all_results['test'] if r['k'] == best_val['k']][0]

    print(f"\n{'='*55}")
    print(f"Best k = {best_val['k']}  (by val MSE)")
    print(f"{'='*55}")
    print(f"  val  MSE_norm={best_val['mse_normalised']:.4f}  "
          f"α={best_val['mse_alpha_normalised']:.4f}  "
          f"ζ={best_val['mse_zeta_normalised']:.4f}")
    print(f"  test MSE_norm={best_test['mse_normalised']:.4f}  "
          f"α={best_test['mse_alpha_normalised']:.4f}  "
          f"ζ={best_test['mse_zeta_normalised']:.4f}")

    if use_wandb:
        import wandb
        wandb.log({
            'knn/best_k'                 : best_val['k'],
            'knn/best_val_mse_norm'      : best_val['mse_normalised'],
            'knn/best_test_mse_norm'     : best_test['mse_normalised'],
            'knn/best_test_mse_alpha_norm': best_test['mse_alpha_normalised'],
            'knn/best_test_mse_zeta_norm' : best_test['mse_zeta_normalised'],
        })
        wandb.finish()

    # ── Save results ───────────────────────────────────────────────────────
    out_dir = os.path.join(cfg.training.checkpoint_dir, 'probe')
    os.makedirs(out_dir, exist_ok=True)

    # Build unique filename based on checkpoint name and pool strategy
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    run_tag   = f"{ckpt_stem}__{pool}"

    results_yaml = {
        'checkpoint' : checkpoint_path,
        'pool'       : pool,
        'run_tag'    : run_tag,
        'k_values'   : k_values,
        'val'        : all_results['val'],
        'test'       : all_results['test'],
        'best_k'     : best_val['k'],
        'best_val'   : best_val,
        'best_test'  : best_test,
    }
    out_path = os.path.join(out_dir, f'knn_results__{run_tag}.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(results_yaml, f, default_flow_style=False)

    print(f"\nResults saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kNN evaluation for active_matter')
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--k', nargs='+', type=int,
                        default=[1, 5, 10, 20, 50],
                        help='k values to sweep (default: 1 5 10 20 50)')
    parser.add_argument('--no-wandb', action='store_true')
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)

    # Inject probe defaults if not in config
    if 'probe' not in cfg:
        cfg['probe'] = {}
    if 'extract_batch_size' not in cfg['probe']:
        cfg['probe']['extract_batch_size'] = 8
    if 'pool' not in cfg['probe']:
        cfg['probe']['pool'] = 'mean'

    main(cfg, args.checkpoint, args.k, use_wandb=not args.no_wandb)
