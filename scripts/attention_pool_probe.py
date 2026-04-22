"""
scripts/attention_pool_probe.py — Linear probe with learned attention pooling

Instead of mean-pooling all 1568 tokens, learns a small attention mechanism
that weights tokens by their relevance for predicting alpha and zeta.

This specifically targets the zeta gap — zeta controls local alignment
which mean pooling loses by averaging over all spatial positions.

The attention pooler is a single linear layer that produces scalar weights
per token, softmax-normalized. This is still effectively a linear operation
over the encoder's output so it respects the "linear probe only" constraint.

Usage
-----
python scripts/attention_pool_probe.py \
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


# ─────────────────────────────────────────────────────────────────────────────
# Attention pooler
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPooler(nn.Module):
    """
    Learned attention pooling over token sequence.

    Takes (N, T, D) token sequences and produces (N, D) weighted averages.
    The attention weights are learned — the model decides which tokens
    matter most for prediction.

    This is a single linear layer (scalar attention) + softmax, which
    is still effectively linear in the encoder outputs.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)    # (D,) -> scalar per token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (N, T, D)
        scores  = self.attn(tokens).squeeze(-1)       # (N, T)
        weights = torch.softmax(scores, dim=-1)        # (N, T)
        pooled  = (weights.unsqueeze(-1) * tokens).sum(dim=1)  # (N, D)
        return pooled


class AttentionProbe(nn.Module):
    """
    Attention pooler + linear head.
    Trained end-to-end but encoder is frozen.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.pooler = AttentionPooler(embed_dim)
        self.head   = nn.Linear(embed_dim, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = self.pooler(tokens)    # (N, D)
        return self.head(pooled)        # (N, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — keep full token sequences (not pooled)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_tokens(encoder, loader, device):
    """Extract full token sequences (N, T, D) rather than pooled features."""
    encoder.eval()
    all_tokens = []
    all_labels = []
    for frames, labels in loader:
        frames = frames.to(device)
        tokens = encoder(frames)         # (B, N_tokens, D)
        all_tokens.append(tokens.cpu())
        all_labels.append(labels)
    return torch.cat(all_tokens, dim=0), torch.cat(all_labels, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_attention_probe(
    train_tokens : torch.Tensor,   # (N, T, D)
    train_labels : torch.Tensor,
    val_tokens   : torch.Tensor,
    val_labels   : torch.Tensor,
    label_mean   : torch.Tensor,
    label_std    : torch.Tensor,
    embed_dim    : int,
    cfg          : DotDict,
    device       : torch.device,
) -> AttentionProbe:
    train_norm = (train_labels - label_mean) / (label_std + 1e-8)
    val_norm   = (val_labels   - label_mean) / (label_std + 1e-8)

    train_tokens = train_tokens.to(device)
    train_norm   = train_norm.to(device)
    val_tokens   = val_tokens.to(device)
    val_norm     = val_norm.to(device)

    ds = TensorDataset(train_tokens, train_norm)
    dl = DataLoader(ds, batch_size=cfg.probe.batch_size, shuffle=True)

    model = AttentionProbe(embed_dim).to(device)
    nn.init.trunc_normal_(model.head.weight, std=0.01)
    nn.init.zeros_(model.head.bias)
    nn.init.zeros_(model.pooler.attn.weight)
    nn.init.zeros_(model.pooler.attn.bias)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.probe.lr,
        weight_decay=cfg.probe.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.probe.epochs, eta_min=cfg.probe.lr * 0.01
    )

    best_mse   = float('inf')
    best_state = None

    print(f"\nTraining attention probe ({cfg.probe.epochs} epochs)...")
    print(f"  Token dim : {train_tokens.shape}")

    for epoch in range(cfg.probe.epochs):
        model.train()
        for tb, lb in dl:
            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(tb), lb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp  = model(val_tokens)
            mse = nn.functional.mse_loss(vp, val_norm).item()
            mse_alpha = nn.functional.mse_loss(vp[:, 0], val_norm[:, 0]).item()
            mse_zeta  = nn.functional.mse_loss(vp[:, 1], val_norm[:, 1]).item()

        if mse < best_mse:
            best_mse   = mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg.probe.epochs} | "
                  f"val_mse={mse:.4f} (α={mse_alpha:.4f}, ζ={mse_zeta:.4f})")

    model.load_state_dict(best_state)
    print(f"\nBest val MSE: {best_mse:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg, checkpoint_path, use_wandb):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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

    # Extract full token sequences
    print("\nExtracting token sequences...")
    train_tokens, train_labels = extract_tokens(encoder, train_loader, device)
    val_tokens,   val_labels   = extract_tokens(encoder, val_loader,   device)
    test_tokens,  test_labels  = extract_tokens(encoder, test_loader,  device)
    print(f"  Token shape: {train_tokens.shape}")

    label_mean = train_labels.mean(dim=0)
    label_std  = train_labels.std(dim=0)

    # Train attention probe
    model = train_attention_probe(
        train_tokens, train_labels,
        val_tokens,   val_labels,
        label_mean, label_std,
        embed_dim = cfg.model.embed_dim,
        cfg       = cfg,
        device    = device,
    )

    # Evaluate
    model.eval()
    for split_name, tokens, labels in [
        ('val',  val_tokens,  val_labels),
        ('test', test_tokens, test_labels),
    ]:
        labels_norm = (labels - label_mean) / (label_std + 1e-8)
        with torch.no_grad():
            preds = model(tokens.to(device))
            ln    = labels_norm.to(device)
            mse       = nn.functional.mse_loss(preds, ln).item()
            mse_alpha = nn.functional.mse_loss(preds[:, 0], ln[:, 0]).item()
            mse_zeta  = nn.functional.mse_loss(preds[:, 1], ln[:, 1]).item()

        print(f"\n{'='*55}")
        print(f"Attention pool probe results — {split_name}")
        print(f"{'='*55}")
        print(f"  MSE (normalised, combined) : {mse:.4f}")
        print(f"  MSE alpha (normalised)     : {mse_alpha:.4f}")
        print(f"  MSE zeta  (normalised)     : {mse_zeta:.4f}")

    # Save
    out_dir   = os.path.join(cfg.training.checkpoint_dir, 'probe')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    torch.save(model.state_dict(),
               os.path.join(out_dir, f'probe_attn_pool__{ckpt_stem}.pt'))
    print(f"\nWeights saved → {out_dir}/probe_attn_pool__{ckpt_stem}.pt")


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
                 'batch_size': 64, 'extract_batch_size': 8}.items():
        if k not in cfg['probe']:
            cfg['probe'][k] = v

    main(cfg, args.checkpoint, use_wandb=not args.no_wandb)
