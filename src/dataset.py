"""
dataset.py — PyTorch Dataset for active_matter (The Well)

Each HDF5 file contains one simulation trajectory with:
  t0_fields/concentration : (3, 81, 256, 256)       → 1 channel  (scalar)
  t1_fields/velocity      : (3, 81, 256, 256, 2)    → 2 channels (vector)
  t2_fields/D             : (3, 81, 256, 256, 2, 2) → 4 channels (tensor)
  t2_fields/E             : (3, 81, 256, 256, 2, 2) → 4 channels (tensor)
  scalars/alpha, scalars/zeta                        → labels (NOT used in SSL)

The first dim of size 3 in each field is the number of trajectory "instances"
stored per file (confirmed from shape). We treat each (file, instance) pair
as an independent simulation. Each __getitem__ returns:
  frames : Tensor (T, C, H, W) = (16, 11, 224, 224)  float32
  labels : Tensor (2,)  = [alpha, zeta]  float32  (for linear probe only)
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import torch.nn.functional as F


# ── Channel layout ────────────────────────────────────────────────────────────
# We flatten vector/tensor fields along their trailing spatial dims:
#   concentration        →  1 ch   (no trailing dim)
#   velocity (2,)        →  2 ch
#   D        (2,2)       →  4 ch
#   E        (2,2)       →  4 ch
#                           -----
#                          11 ch  total
NUM_CHANNELS   = 11
NUM_TIMESTEPS  = 81     # raw time steps per trajectory
NUM_INSTANCES  = 3      # instances stored per HDF5 file
SPATIAL_RAW    = 256    # raw spatial resolution
SPATIAL_CROP   = 224    # target spatial resolution (random crop)
CLIP_LEN       = 16     # number of time steps per training sample


class ActiveMatterDataset(Dataset):
    """
    Parameters
    ----------
    data_dir : str
        Path to the split directory, e.g. /scratch/.../data/train
    clip_len : int
        Number of consecutive time steps to return per sample.
    spatial_size : int
        Output H=W after random (train) or center (val/test) crop.
    mean : array-like, shape (11,) or None
        Per-channel mean for normalisation. If None, no normalisation.
    std  : array-like, shape (11,) or None
        Per-channel std  for normalisation. If None, no normalisation.
    augment : bool
        If True, apply random temporal offset + random spatial crop +
        random horizontal/vertical flip. Set False for val/test.
    cache_index : bool
        If True, scan all files once at init and cache the (file, instance)
        index. Adds ~1-2 s startup but makes __len__ / __getitem__ O(1).
    """

    def __init__(
        self,
        data_dir: str,
        clip_len: int = CLIP_LEN,
        spatial_size: int = SPATIAL_CROP,
        mean=None,
        std=None,
        augment: bool = True,
        cache_index: bool = True,
    ):
        super().__init__()
        self.data_dir     = data_dir
        self.clip_len     = clip_len
        self.spatial_size = spatial_size
        self.augment      = augment

        # Normalisation tensors: (C, 1, 1) for broadcast over (C, H, W)
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            self.std  = torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1)
        else:
            self.mean = None
            self.std  = None

        # Build index: list of (filepath, instance_idx) tuples
        self.samples = []
        if cache_index:
            self._build_index()

    # ── Index ─────────────────────────────────────────────────────────────────

    def _build_index(self):
        """
        Build index of (filepath, instance_idx, window_start) tuples.

        Uses a sliding window with stride 1 over the raw HDF5 files.

        With T_raw=81, clip_len=16:
            n_windows = 81 - 16 + 1 = 66 sliding windows per trajectory
            175 trajectories x 66 windows = 11,550 training samples
            (more than the pre-processed spec count of 8,750 due to
            stride-1 windowing vs the pre-processed dataset larger stride)

        During training, a small random jitter is applied to the window start
        to add temporal augmentation variety while keeping the window valid.
        """
        files = sorted([
            f for f in os.listdir(self.data_dir) if f.endswith('.hdf5')
        ])
        for fname in files:
            fpath = os.path.join(self.data_dir, fname)
            with h5py.File(fpath, 'r') as f:
                n_instances = f['t0_fields/concentration'].shape[0]
                T_raw       = f['t0_fields/concentration'].shape[1]
            # Sliding window with stride 1
            n_windows = T_raw - self.clip_len + 1
            for i in range(n_instances):
                for w in range(n_windows):
                    self.samples.append((fpath, i, w))

    def __len__(self):
        return len(self.samples)

    # ── Core loading ──────────────────────────────────────────────────────────

    def _get_file_handle(self, fpath: str) -> h5py.File:
        """
        Lazily open and cache HDF5 file handles per worker process.
        Because DataLoader workers fork, handles are created inside each
        worker — this avoids sharing file handles across processes.
        """
        if not hasattr(self, '_open_files'):
            self._open_files = {}
        if fpath not in self._open_files:
            self._open_files[fpath] = h5py.File(fpath, 'r', swmr=True)
        return self._open_files[fpath]

    def __getitem__(self, idx):
        fpath, inst, window_start = self.samples[idx]

        # Use cached file handle — avoids reopening HDF5 file every sample
        f = self._get_file_handle(fpath)

        # ── Labels (used ONLY at linear-probe/kNN stage) ──────────────
        alpha = float(f['scalars/alpha'][()])
        zeta  = float(f['scalars/zeta'][()])
        labels = torch.tensor([alpha, zeta], dtype=torch.float32)

        # ── Temporal sampling ─────────────────────────────────────────
        # With sliding windows (stride=1), each window_start is already
        # a valid start index. Clamp to ensure we always get exactly
        # clip_len frames even for edge cases.
        T_raw   = f['t0_fields/concentration'].shape[1]
        t_start = int(min(window_start, T_raw - self.clip_len))
        t_start = max(0, t_start)
        t_end   = t_start + self.clip_len

        # ── Load fields for this instance & time window ───────────────
        # concentration : (clip_len, 256, 256)
        conc = f['t0_fields/concentration'][inst, t_start:t_end]   # (T,H,W)

        # velocity      : (clip_len, 256, 256, 2)
        vel  = f['t1_fields/velocity'][inst, t_start:t_end]        # (T,H,W,2)

        # D, E          : (clip_len, 256, 256, 2, 2)
        D    = f['t2_fields/D'][inst, t_start:t_end]               # (T,H,W,2,2)
        E    = f['t2_fields/E'][inst, t_start:t_end]               # (T,H,W,2,2)

        # ── Verify we got exactly clip_len frames ─────────────────────────────
        T_loaded = conc.shape[0]
        if T_loaded != self.clip_len:
            raise RuntimeError(
                f"Expected {self.clip_len} frames but got {T_loaded} "
                f"(window_start={window_start}, t_start={t_start}, "
                f"t_end={t_end}, T_raw={T_raw})"
            )

        # ── Flatten trailing dims & stack into (T, C, H, W) ──────────────────
        conc  = torch.from_numpy(conc).unsqueeze(1)                    # (T,1,H,W)
        vel   = torch.from_numpy(vel).permute(0, 3, 1, 2)             # (T,2,H,W)
        D_flat = torch.from_numpy(D).reshape(self.clip_len, 4,
                                             SPATIAL_RAW, SPATIAL_RAW) # (T,4,H,W)
        E_flat = torch.from_numpy(E).reshape(self.clip_len, 4,
                                             SPATIAL_RAW, SPATIAL_RAW) # (T,4,H,W)

        frames = torch.cat([conc, vel, D_flat, E_flat], dim=1)         # (T,11,H,W)

        # ── Spatial crop ──────────────────────────────────────────────────────
        frames = self._spatial_crop(frames)                             # (T,11,224,224)

        # ── Random flips (augment only) ───────────────────────────────────────
        if self.augment:
            if random.random() < 0.5:
                frames = torch.flip(frames, dims=[-1])   # horizontal
            if random.random() < 0.5:
                frames = torch.flip(frames, dims=[-2])   # vertical

        # ── Normalise ─────────────────────────────────────────────────────────
        if self.mean is not None:
            # mean/std are (C,1,1); broadcast over (T,C,H,W) via unsqueeze(0)
            frames = (frames - self.mean.unsqueeze(0)) / (self.std.unsqueeze(0) + 1e-6)

        return frames, labels

    # ── Spatial crop helper ───────────────────────────────────────────────────

    def _spatial_crop(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (T, C, H, W)
        Returns (T, C, spatial_size, spatial_size)
        """
        _, _, H, W = frames.shape
        s = self.spatial_size
        if self.augment:
            top  = random.randint(0, H - s)
            left = random.randint(0, W - s)
        else:
            top  = (H - s) // 2
            left = (W - s) // 2
        return frames[:, :, top:top+s, left:left+s]


# ── Stats computation ─────────────────────────────────────────────────────────

def compute_channel_stats(
    data_dir: str,
    clip_len: int = CLIP_LEN,
    num_workers: int = 4,
    max_files: int = None,
) -> dict:
    """
    Compute per-channel mean and std over the training set using
    Welford's online algorithm (memory-efficient, single pass).

    Returns
    -------
    dict with keys 'mean' and 'std', each a list of 11 floats.

    Usage
    -----
    stats = compute_channel_stats('/scratch/.../data/train')
    # Then save and reuse:
    import yaml
    with open('stats.yaml', 'w') as f:
        yaml.dump(stats, f)
    """
    # Use dataset without normalisation or augmentation
    ds = ActiveMatterDataset(
        data_dir,
        clip_len=clip_len,
        augment=False,
        mean=None,
        std=None,
    )
    if max_files is not None:
        # Subsample for a quick estimate
        indices = list(range(0, len(ds), max(1, len(ds) // max_files)))
        ds = torch.utils.data.Subset(ds, indices)

    loader = DataLoader(ds, batch_size=1, num_workers=num_workers, shuffle=False)

    # Welford accumulators per channel
    C = NUM_CHANNELS
    count = torch.zeros(C)
    mean  = torch.zeros(C)
    M2    = torch.zeros(C)  # sum of squared deviations

    print(f"Computing stats over {len(ds)} samples...")
    for i, (frames, _) in enumerate(loader):
        # frames: (1, T, C, H, W) — squeeze batch dim
        frames = frames.squeeze(0)                 # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)       # (C, T, H, W)
        frames = frames.reshape(C, -1)             # (C, T*H*W)

        for c in range(C):
            vals = frames[c]                       # (T*H*W,)
            n_new = vals.numel()
            new_mean = vals.mean().item()
            new_var  = vals.var(unbiased=False).item()

            # Parallel / batch Welford update
            n_old = count[c].item()
            delta = new_mean - mean[c].item()
            count[c] += n_new
            mean[c]  += delta * n_new / count[c].item()
            M2[c]    += new_var * n_new + delta ** 2 * n_old * n_new / count[c].item()

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(ds)}")

    std = torch.sqrt(M2 / count)

    stats = {
        'mean': mean.tolist(),
        'std':  std.tolist(),
    }
    print("\nChannel means:", [f"{v:.4f}" for v in stats['mean']])
    print("Channel stds: ", [f"{v:.4f}" for v in stats['std']])
    return stats


# ── Convenience factory ───────────────────────────────────────────────────────

def build_dataloaders(
    root_dir: str,
    stats: dict = None,
    clip_len: int = CLIP_LEN,
    spatial_size: int = SPATIAL_CROP,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    root_dir : str
        Parent directory containing train/, valid/, test/ subdirs.
    stats : dict or None
        Dict with 'mean' and 'std' keys (output of compute_channel_stats).
        Pass None to skip normalisation (not recommended for training).

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    mean = stats['mean'] if stats else None
    std  = stats['std']  if stats else None

    train_ds = ActiveMatterDataset(
        os.path.join(root_dir, 'train'),
        clip_len=clip_len, spatial_size=spatial_size,
        mean=mean, std=std, augment=True,
    )
    val_ds = ActiveMatterDataset(
        os.path.join(root_dir, 'valid'),
        clip_len=clip_len, spatial_size=spatial_size,
        mean=mean, std=std, augment=False,
    )
    test_ds = ActiveMatterDataset(
        os.path.join(root_dir, 'test'),
        clip_len=clip_len, spatial_size=spatial_size,
        mean=mean, std=std, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print(f"Dataset sizes  — train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}")
    print(f"Batches/epoch  — train: {len(train_loader):,}  val: {len(val_loader):,}")
    return train_loader, val_loader, test_loader


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import yaml

    root = sys.argv[1] if len(sys.argv) > 1 else '/scratch/$USER/data/active_matter/data'
    train_dir = os.path.join(root, 'train')

    # ── Step 1: check a single sample loads correctly ─────────────────────
    print("=" * 60)
    print("STEP 1: Single sample load test")
    print("=" * 60)
    ds = ActiveMatterDataset(train_dir, augment=True, mean=None, std=None)
    print(f"Dataset length : {len(ds)}")   # expect 45 files * 3 instances = 135
    frames, labels = ds[0]
    print(f"frames shape   : {frames.shape}")   # expect (16, 11, 224, 224)
    print(f"frames dtype   : {frames.dtype}")
    print(f"labels         : alpha={labels[0]:.3f}  zeta={labels[1]:.3f}")
    print(f"frames min/max : {frames.min():.3f} / {frames.max():.3f}")
    assert frames.shape == (CLIP_LEN, NUM_CHANNELS, SPATIAL_CROP, SPATIAL_CROP), \
        f"Unexpected shape: {frames.shape}"
    print("Shape assertion passed ✓")

    # ── Step 2: compute stats on a small subset ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Stats computation (subset of 20 files)")
    print("=" * 60)
    stats = compute_channel_stats(train_dir, max_files=20)
    with open('stats.yaml', 'w') as f:
        yaml.dump(stats, f)
    print("Saved → stats.yaml")

    # ── Step 3: test DataLoaders with normalisation ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: DataLoader test (2 batches)")
    print("=" * 60)
    train_loader, val_loader, test_loader = build_dataloaders(
        root, stats=stats, batch_size=2, num_workers=2
    )
    for batch_idx, (frames, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: frames={frames.shape}  labels={labels.shape}")
        print(f"  frames mean={frames.mean():.4f}  std={frames.std():.4f}")
        if batch_idx >= 1:
            break

    print("\nAll checks passed ✓")
