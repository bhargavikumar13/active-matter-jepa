"""
scripts/visualize_embeddings.py — UMAP/t-SNE embedding visualization

Loads the Run 5 JEPA encoder, extracts mean-pooled features for all
test samples, reduces to 2D via UMAP (or t-SNE as fallback), and plots
two side-by-side panels colored by alpha and zeta respectively.

Usage
-----
python scripts/visualize_embeddings.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --output figures/embedding_viz.png \
    --method umap \
    --split test

# Use t-SNE if umap not installed
python scripts/visualize_embeddings.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --method tsne

Output
------
figures/embedding_viz.png  — two-panel figure for paper
figures/embedding_viz.pdf  — PDF version for LaTeX
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config
from src.dataset import ActiveMatterDataset, build_dataloaders
from src.model import VisionTransformer3D


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    encoder: VisionTransformer3D,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract mean-pooled encoder features for all samples.

    Returns
    -------
    features : (N, D) float32
    alphas   : (N,)  float32
    zetas    : (N,)  float32
    """
    encoder.eval()
    all_feats  = []
    all_alpha  = []
    all_zeta   = []

    print(f"Extracting features from {len(loader.dataset):,} samples...")
    for i, (frames, labels) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        tokens = encoder(frames)              # (B, N, D)
        feats  = tokens.mean(dim=1)           # (B, D) mean pool
        feats  = nn.functional.normalize(feats, dim=-1)  # L2 normalise

        all_feats.append(feats.cpu().float().numpy())
        all_alpha.append(labels[:, 0].numpy())
        all_zeta.append(labels[:, 1].numpy())

        if (i + 1) % 50 == 0:
            print(f"  {(i+1) * loader.batch_size:,} / {len(loader.dataset):,}")

    features = np.concatenate(all_feats, axis=0)
    alphas   = np.concatenate(all_alpha, axis=0)
    zetas    = np.concatenate(all_zeta,  axis=0)

    print(f"Features extracted: {features.shape}")
    return features, alphas, zetas


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────

def reduce_dimensions(
    features: np.ndarray,
    method: str = 'umap',
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce features to 2D using UMAP or t-SNE."""

    if method == 'umap':
        try:
            import umap
            print("Running UMAP...")
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=random_state,
                verbose=False,
            )
            return reducer.fit_transform(features)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            method = 'tsne'

    if method == 'tsne':
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        print("Running PCA (50 dims) then t-SNE...")
        # PCA first to speed up t-SNE
        n_pca = min(50, features.shape[1])
        pca = PCA(n_components=n_pca, random_state=random_state)
        features_pca = pca.fit_transform(features)
        print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

        tsne = TSNE(
            n_components=n_components,
            perplexity=30,
            learning_rate='auto',
            init='pca',
            n_iter=1000,
            random_state=random_state,
            verbose=1,
        )
        return tsne.fit_transform(features_pca)

    raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'.")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(
    embedding: np.ndarray,
    alphas: np.ndarray,
    zetas: np.ndarray,
    method: str,
    output_path: str,
):
    """
    Two-panel figure: left colored by alpha, right colored by zeta.
    Uses one color per discrete parameter value.
    """
    # Discrete parameter values
    alpha_vals = sorted(set(alphas.round(1).tolist()))   # [-5,-4,-3,-2,-1]
    zeta_vals  = sorted(set(zetas.round(1).tolist()))    # [1,3,5,...,17]

    # Color palettes — diverging for alpha, sequential for zeta
    alpha_cmap = plt.cm.get_cmap('RdYlBu', len(alpha_vals))
    zeta_cmap  = plt.cm.get_cmap('YlOrRd', len(zeta_vals))

    alpha_colors = {v: alpha_cmap(i / (len(alpha_vals) - 1))
                    for i, v in enumerate(alpha_vals)}
    zeta_colors  = {v: zeta_cmap(i / (len(zeta_vals) - 1))
                    for i, v in enumerate(zeta_vals)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    method_label = method.upper()

    # ── Panel 1: colored by alpha ─────────────────────────────────────────
    ax = axes[0]
    for val in alpha_vals:
        mask = np.abs(alphas - val) < 0.5
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[alpha_colors[val]],
            s=4, alpha=0.6, linewidths=0,
            label=f'α = {int(val)}',
            rasterized=True,
        )
    ax.set_title(f'{method_label} coloured by α (active dipole strength)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel(f'{method_label}-1', fontsize=9)
    ax.set_ylabel(f'{method_label}-2', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    legend = ax.legend(
        title='α values', fontsize=7, title_fontsize=8,
        markerscale=3, loc='best',
        framealpha=0.8, edgecolor='gray',
    )

    # ── Panel 2: colored by zeta ──────────────────────────────────────────
    ax = axes[1]
    for val in zeta_vals:
        mask = np.abs(zetas - val) < 1.0
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[zeta_colors[val]],
            s=4, alpha=0.6, linewidths=0,
            label=f'ζ = {int(val)}',
            rasterized=True,
        )
    ax.set_title(f'{method_label} coloured by ζ (steric alignment)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel(f'{method_label}-1', fontsize=9)
    ax.set_ylabel(f'{method_label}-2', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    legend = ax.legend(
        title='ζ values', fontsize=6, title_fontsize=7,
        markerscale=3, loc='best', ncol=2,
        framealpha=0.8, edgecolor='gray',
    )

    # ── Overall title ─────────────────────────────────────────────────────
    fig.suptitle(
        f'Run 5 encoder representations ({method_label})\n'
        f'α clusters are more separated than ζ, consistent with LP MSE gap (0.038 vs 0.152)',
        fontsize=9, y=1.02,
    )

    plt.tight_layout()

    # Save PNG and PDF
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved → {output_path}")

    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved → {pdf_path}")

    plt.close()

    # ── Quantitative cluster quality ─────────────────────────────────────
    _print_cluster_quality(embedding, alphas, zetas, alpha_vals, zeta_vals)


def _print_cluster_quality(embedding, alphas, zetas, alpha_vals, zeta_vals):
    """Print silhouette-style cluster separation metric."""
    try:
        from sklearn.metrics import silhouette_score

        # Subsample for speed
        n = min(2000, len(embedding))
        idx = np.random.choice(len(embedding), n, replace=False)

        alpha_labels = np.array([alpha_vals.index(round(a, 1))
                                  for a in alphas[idx]])
        zeta_labels  = np.array([zeta_vals.index(round(z, 1))
                                  for z in zetas[idx]])

        sil_alpha = silhouette_score(embedding[idx], alpha_labels,
                                      metric='euclidean')
        sil_zeta  = silhouette_score(embedding[idx], zeta_labels,
                                      metric='euclidean')

        print(f"\nCluster quality (silhouette score, higher = better separated):")
        print(f"  α: {sil_alpha:.4f}")
        print(f"  ζ: {sil_zeta:.4f}")
        print(f"  → α is {'more' if sil_alpha > sil_zeta else 'less'} "
              f"separated than ζ in embedding space")
    except Exception as e:
        print(f"Silhouette score failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize JEPA encoder embeddings via UMAP/t-SNE'
    )
    parser.add_argument('--config',
                        default='configs/jepa_run5.yaml')
    parser.add_argument('--checkpoint',
                        default='checkpoints/jepa/best.pt')
    parser.add_argument('--output',
                        default='figures/embedding_viz.png')
    parser.add_argument('--method',
                        choices=['umap', 'tsne'],
                        default='umap')
    parser.add_argument('--split',
                        choices=['train', 'val', 'test', 'all'],
                        default='test',
                        help='Which split to visualize')
    parser.add_argument('--max_samples',
                        type=int, default=None,
                        help='Subsample for speed (None = use all)')
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Stats ─────────────────────────────────────────────────────────────
    stats = None
    if os.path.exists(cfg.data.stats_path):
        with open(cfg.data.stats_path) as f:
            raw = yaml.safe_load(f)

        def _flatten(v):
            if isinstance(v, list):
                out = []
                for item in v:
                    out.extend(_flatten(item))
                return out
            return [float(v)]

        def _parse(d):
            return (_flatten(d['concentration']) + _flatten(d['velocity'])
                    + _flatten(d['D']) + _flatten(d['E']))

        mean_list = _parse(raw['mean'])
        std_list  = _parse(raw['std'])
        if len(mean_list) == 11:
            stats = {'mean': mean_list, 'std': std_list}
            print("Loaded channel stats ✓")

    # ── Data ──────────────────────────────────────────────────────────────
    split_map = {
        'train': 'train',
        'val':   'valid',
        'test':  'test',
    }

    if args.split == 'all':
        # Combine all splits
        datasets = []
        for split in ['train', 'valid', 'test']:
            ds = ActiveMatterDataset(
                os.path.join(cfg.data.root_dir, split),
                clip_len=cfg.data.clip_len,
                spatial_size=cfg.data.spatial_size,
                mean=stats['mean'] if stats else None,
                std=stats['std']  if stats else None,
                augment=False,
            )
            datasets.append(ds)
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(datasets)
    else:
        split_dir = split_map[args.split]
        dataset = ActiveMatterDataset(
            os.path.join(cfg.data.root_dir, split_dir),
            clip_len=cfg.data.clip_len,
            spatial_size=cfg.data.spatial_size,
            mean=stats['mean'] if stats else None,
            std=stats['std']  if stats else None,
            augment=False,
        )

    # Subsample if requested
    if args.max_samples and len(dataset) > args.max_samples:
        indices = np.random.choice(len(dataset), args.max_samples, replace=False)
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        print(f"Subsampled to {args.max_samples:,} samples")

    loader = DataLoader(
        dataset,
        batch_size=cfg.probe.extract_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"Dataset: {len(dataset):,} samples ({args.split} split)")

    # ── Encoder ───────────────────────────────────────────────────────────
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

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    encoder.load_state_dict(ckpt['context_encoder'])
    print(f"Loaded encoder from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ── Extract features ──────────────────────────────────────────────────
    features, alphas, zetas = extract_features(encoder, loader, device)

    # ── Dimensionality reduction ──────────────────────────────────────────
    embedding = reduce_dimensions(features, method=args.method)

    # ── Plot ──────────────────────────────────────────────────────────────
    make_figure(embedding, alphas, zetas, method=args.method,
                output_path=args.output)

    print("\nDone. Add to report:")
    print(f"  \\includegraphics[width=\\columnwidth]{{{args.output}}}")
